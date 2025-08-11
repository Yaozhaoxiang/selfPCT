import pandas as pd
import matplotlib.pyplot as plt

import sys
import matplotlib.pyplot as plt

TRAIN_CSV_PATH = 'D:\\yzx\\res_imag\\v1\\train.csv'
VALIDATION_CSV_PATH = 'D:\\yzx\\res_imag\\v1\\validation.csv'

# --- 2. 加载数据 ---
try:
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_val = pd.read_csv(VALIDATION_CSV_PATH)
    print("CSV 文件加载成功！")
except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。")
    print("请确保文件名正确，并且脚本与 CSV 文件位于同一目录下，或者提供了正确的路径。")
    sys.exit() # 退出脚本

# --- 3. 创建并绘制图表 ---
# 设置一个更美观的绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
# 创建一个图形对象，设置图像大小，使其更适合查看或插入论文
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制训练准确率曲线
# 'Step' 列通常对应 epoch
# 'Value' 列是准确率的值
ax.plot(df_train['Step'], df_train['Value'],
        label='Train Accuracy',
        color='#1f77b4',  # 使用更专业的蓝色
        linestyle='-',    # 实线
        linewidth=2)      # 线条宽度

# 绘制验证准确率曲线
ax.plot(df_val['Step'], df_val['Value'],
        label='Validation Accuracy',
        color='#ff7f0e',  # 使用更专业的橙色
        linestyle='--',   # 虚线，以示区别
        linewidth=2)

# --- 4. 美化图表，添加标签和标题 ---
# 设置图表标题，并增加字体大小
# ax.set_title('训练与验证准确率对比', fontsize=16)
# 设置X轴和Y轴的标签
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)

# 设置Y轴的范围，例如从0.5到1.0，以更好地突出变化
# ax.set_ylim(0.5, 1.02) # 如果需要，可以取消此行的注释

# 添加图例，并设置最佳显示位置
ax.legend(fontsize=11)

# 调整刻度标签的字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 优化布局，防止标签重叠
plt.tight_layout()


# --- 5. 显示并保存图像 ---
# 显示图像
plt.show()

# 将图像保存为高分辨率文件，适合用于报告或论文
# dpi=300 保证了图像的清晰度
fig.savefig('D:\\yzx\\res_imag\\v1\\accuracy_comparison.png', dpi=300)
print("图像已保存为 'accuracy_comparison.png'")