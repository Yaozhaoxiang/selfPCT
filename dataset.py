import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
import json


class ModelNetDataLoader(Dataset):
    # root	数据集根目录，如 modelnet40_normal_resampled/
    # npoint	每个样本保留的点数（如 1024）
    # split	    'train' 或 'test'
    # uniform	是否使用 FPS（Farthest Point Sampling）
    # normal_channel    是否使用法向量
    # cache_size    内存缓存的数据数量，避免反复读取硬盘，提高效率
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.normal_channel = normal_channel

        # 加载类别信息:
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt') # 加载模型类别名（共 40 个）；
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat)))) # 类别名到整数标签的映射 {'airplane': 0, 'bathtub': 1, ..., 'xbox': 39}


        # 加载训练/测试集划分信息:
        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] # 拿到每一行
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        # 构建核心数据路径列表 Self.datapath
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]  # shape_names列表中的每个元素对应shape_ids[split]中样本 ID 的类别名称。
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))


        # 初始化缓存:
        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)  # 返回数据集中样本总数

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]  # 获取类别编号
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            # 归一化 (Normalization):
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


# 建议重命名此类
class CustomDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=False, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'shape_names.txt')  # <<< 修改点 1
        self.normal_channel = normal_channel

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train.txt'))]  # <<< 修改点 2
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test.txt'))]  # <<< 修改点 3

        assert (split == 'train' or split == 'test')
        # 后面的代码完全不需要修改，因为您的文件命名和文件夹结构与它的逻辑是兼容的！
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size
        self.cache = {}

    # __len__ 和 __getitem__ 方法完全不需要修改，可以直接复制过来
    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1]).astype(np.float32)

            # 模型中已经有了 knn，所有这里只是对数据进行将采样。样本大概280000 -> 15000 参考net40
            Npoint = 15000
            if point_set.shape[0] > Npoint:
                # 使用最简单高效的随机采样
                # replace=False 确保不会重复采样同一个点
                choice_indices = np.random.choice(point_set.shape[0], Npoint, replace=False)
                point_set = point_set[choice_indices, :]

            # 如果点数不足，可以进行重复采样以补足
            elif point_set.shape[0] < Npoint:
                choice_indices = np.random.choice(point_set.shape[0], Npoint, replace=True)
                point_set = point_set[choice_indices, :]


            # 假设您的数据没有法向量，如果有点云是6列，这部分逻辑就需要保留
            # 如果您的.txt就是 x,y,z三列，那么normal_channel可以一直设为False
            # 采样点
            if self.uniform:
                # 这里需要您提供 farthest_point_sample 函数
                # point_set = farthest_point_sample(point_set, Self.npoints)
                # 暂时先用普通采样代替
                indices = np.random.choice(point_set.shape[0], self.npoints, replace=True)
                point_set = point_set[indices, :]
            else:
                point_set = point_set[0:self.npoints, :]

            # 归一化 (需要您提供 pc_normalize 函数)
            # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(Self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(Self.seg_classes.keys()):
        #     print(cat, Self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    data = CustomDataLoader('E:/data/self_data', split='train', uniform=False, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)