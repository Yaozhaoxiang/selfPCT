# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- 1. 配置区：请在此处修改您的Loss文件名 ---
TRAIN_LOSS_CSV_PATH = 'D:\\yzx\\res_imag\\v1\\loss_train.csv'
VALIDATION_LOSS_CSV_PATH = 'D:\\yzx\\res_imag\\v1\\loss_validation.csv'
# ----------------------------------------------------


# --- 2. 加载数据 ---
try:
    df_train = pd.read_csv(TRAIN_LOSS_CSV_PATH)
    df_val = pd.read_csv(VALIDATION_LOSS_CSV_PATH)
    print("Loss CSV 文件加载成功！")
except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。")
    print("请确保文件名正确，并且脚本与 CSV 文件位于同一目录下，或者提供了正确的路径。")
    sys.exit()

# --- 3. 创建并绘制图表 ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制训练损失曲线
ax.plot(df_train['Step'], df_train['Value'],
        label='Train Loss',
        color='#1f77b4',
        linestyle='-',
        linewidth=2)

# 绘制验证损失曲线
ax.plot(df_val['Step'], df_val['Value'],
        label='Validation Loss',
        color='#ff7f0e',
        linestyle='--',
        linewidth=2)

# --- 4. 美化图表，添加标签和标题 ---
# ax.set_title('训练与验证损失对比', fontsize=16)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.legend(fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Y轴通常开始
ax.set_ylim(bottom=0.7)

plt.tight_layout()

# --- 5. 显示并保存图像 ---
plt.show()
fig.savefig('D:\\yzx\\res_imag\\v1\\loss_comparison.png', dpi=300) # <--- 修改这里
print("图像已保存为 'loss_comparison.png'")