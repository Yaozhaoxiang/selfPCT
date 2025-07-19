import os
import shutil

# 源文件夹路径
source_root = 'E:\data\data'  # 根据实际路径修改
# 目标文件夹路径
target_folder = 'E:\data\self'

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 统计编号
file_index = 1

# 遍历源文件夹中的所有子文件夹
for subfolder in sorted(os.listdir(source_root)):
    subfolder_path = os.path.join(source_root, subfolder)
    if os.path.isdir(subfolder_path):
        # 遍历子文件夹中的所有txt文件
        for filename in sorted(os.listdir(subfolder_path)):
            if filename.endswith('.txt'):
                source_file = os.path.join(subfolder_path, filename)
                new_filename = f'std_{file_index:03d}.txt'
                target_file = os.path.join(target_folder, new_filename)
                shutil.copy2(source_file, target_file)
                print(f'Copied {source_file} → {target_file}')
                file_index += 1

print('✅ 所有文件已复制并重命名完成。')
