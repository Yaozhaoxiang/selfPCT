from torchvision.transforms import ToTensor

from dataset import ModelNetDataLoader, CustomDataLoader
from omegaconf import OmegaConf
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf
import time
import torchvision.models as models

from utils.visualizer import plot_confusion_matrix
from utils.weight import compute_class_weights
from torch.utils.tensorboard import SummaryWriter


def test(model, loader, criterion, num_class=7):
    """
    在测试集上评估模型，并返回准确率、损失以及所有预测和标签。
    """
    # --- 初始化指标 ---
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    test_loss = 0.0  # <--- 新增：初始化总损失

    # --- 用于收集所有标签 ---
    all_targets = []
    all_preds = []

    # --- 优化点：将模型评估模式的切换移到循环外 ---
    model.eval()

    for j, data in tqdm(enumerate(loader), total=len(loader), desc="Testing"):
        points, target = data
        target = target[:, 0]
        # 注意：根据您的模型输入，可能需要下面这行
        # points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        # 前向传播
        pred = model(points)

        # --- 新增：计算并累加损失 ---
        loss = criterion(pred, target.long())
        test_loss += loss.item()
        # -----------------------------

        pred_choice = pred.data.max(1)[1]

        # --- 收集每个批次的标签 ---
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(pred_choice.cpu().numpy())

        # --- 计算准确率 (您的原始逻辑保持不变) ---
        for cat in np.unique(target.cpu()):
            # 修正一个潜在的小问题：确保 target[target==cat] 不为空
            mask = target == cat
            if mask.sum() > 0:
                classacc = pred_choice[mask].eq(target[mask].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[mask].size(0))
                class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size(0)))

    # --- 计算最终平均指标 ---
    class_acc[:, 2] = np.divide(class_acc[:, 0], class_acc[:, 1], out=np.zeros_like(class_acc[:, 0]),
                                where=class_acc[:, 1] != 0)
    avg_class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    avg_test_loss = test_loss / len(loader)  # 计算平均损失

    # 返回所有需要的指标
    return instance_acc, avg_class_acc, avg_test_loss, all_preds, all_targets


@hydra.main(config_path='config', config_name='cls', version_base="1.1")
def main(args):
    omegaconf.OmegaConf.set_struct(args, False) # 允许后续动态给 args 添加新的属性或字段

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # print(args.pretty())
    print(OmegaConf.to_yaml(args))

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path('D:\\yzx\\self_data_v2')
    # DATA_PATH = hydra.utils.to_absolute_path('H:\BaiduNetdiskDownload\modelnet40\modelnet40_normal_resampled')


    # TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)

    TRAIN_DATASET = CustomDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    TEST_DATASET = CustomDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    args.num_class = 7
    args.input_dim = 6 if args.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.') # 拷贝模型代码

    # 类名必须是 PointTransformerCls
    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()
    class_weights = compute_class_weights(TRAIN_DATASET)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )  # 交叉熵损失函数
    print(f"模型所在设备: {next(classifier.parameters()).device}")

    try:
        model = models.resnet18()
        checkpoint = torch.load('best_model.pth', weights_only=True)
        model.load_state_dict(checkpoint)
        # checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    global_epoch = 0  # 全局epoch计数器
    global_step = 0  # 全局训练步数计数器
    best_instance_acc = 0.0  # 记录最高的实例平均准确率
    best_class_acc = 0.0  # 记录最高的类别平均准确率
    best_epoch = 0  # 记录最佳准确率出现在哪个epoch
    mean_correct = []  # 用于计算每个epoch的平均正确率

    # --- TensorBoard 初始化 ---
    # 创建一个 writer 对象，日志将保存在 'runs/cls_experiment_1' 类似的文件夹中

    hydra_run_dir = os.getcwd()
    log_dir = os.path.join(hydra_run_dir, "runs", f"exp_{time.strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)

    # --- 记录模型图 ---
    # 从 DataLoader 中取一个批次的数据作为模型的示例输入
    sample_points, _ = next(iter(trainDataLoader))
    writer.add_graph(classifier, sample_points.cuda())  #

    # --- 记录超参数 ---
    # 将您的配置 (args/cfg) 和最佳准确率一起记录
    # 这里我们先定义一个空的度量字典，在训练结束后填充
    hparams = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'model': args.model.name
    }

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        epoch_train_loss = 0.0

        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # 数据增强
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)

            target = target[:, 0]
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad() # 梯度清零

            pred = classifier(points) # 前向传播
            loss = criterion(pred, target.long()) # 计算损失
            epoch_train_loss += loss.item()

            loss.backward() # 反向传播
            optimizer.step() # 更新权重

            # 性能指标计算
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            global_step += 1

            # --- TensorBoard 记录点 ---
            # 每10个step记录一次训练损失，避免日志过于频繁
            if global_step % 10 == 0:
                writer.add_scalar('Loss/Train', loss.item(), global_step)
            
        scheduler.step() # 更新学习率
        # 计算并打印当前 epoch 在训练集上的平均准确率。
        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)

        # --- 新增：计算并记录Epoch级别的平均训练损失 ---
        avg_train_loss = epoch_train_loss / len(trainDataLoader)
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)

        # --- TensorBoard 记录点 --- 平均训练准确率，学习率
        writer.add_scalar('Accuracy/Train', train_instance_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        mean_correct = []  # 清空列表，为下一个epoch做准备

        with torch.no_grad():
            instance_acc, class_acc, validation_loss, all_preds, all_targets = test(
                classifier,
                testDataLoader,
                criterion,
                num_class=args.num_class
            )
            writer.add_scalar('Loss/Validation', validation_loss, epoch)
            writer.add_scalar('Accuracy/Validation', instance_acc, epoch)

            # 当找到最佳模型时，生成并记录混淆矩阵
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

                # --- 新增：生成并记录混淆矩阵 ---
                logger.info('Generating and logging confusion matrix...')
                # 从数据集中获取类别名称列表
                class_names = TRAIN_DATASET.cat
                # 调用绘图函数
                cm_image = plot_confusion_matrix(all_targets, all_preds, class_names)
                # 将 PIL Image 转换为 Tensor
                cm_tensor = ToTensor()(cm_image)
                # 写入 TensorBoard
                writer.add_image('ConfusionMatrix/best_model', cm_tensor, global_step=epoch)
                # --------------------------------

                logger.info('Save model...')
                # ... (保存模型的代码) ...

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            # --- TensorBoard 记录点 --- 测试准确率
            writer.add_scalar('Accuracy/test_instance', instance_acc, epoch)
            writer.add_scalar('Accuracy/test_class_avg', class_acc, epoch)

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
    # 在训练循环结束后
    # 记录超参数和对应的最佳性能指标
    writer.add_hparams(hparams, {
        'best_instance_accuracy': best_instance_acc,
        'best_class_accuracy': best_class_acc
    })
    writer.close()
    logger.info('End of training...')

if __name__ == '__main__':
    main()