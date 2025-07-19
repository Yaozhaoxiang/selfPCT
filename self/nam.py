import os
from pathlib import Path

# --- 配置区域 ---
# 请将 'std' 替换为您需要重命名的文件夹的实际名称
TARGET_DIR_NAME = "std"

# 设置序号的位数（例如，3 表示 001, 002...; 4 表示 0001, 0002...）
PADDING_WIDTH = 3


# --- 配置结束 ---


def batch_rename_files(target_dir_name: str, padding: int):
    """
    批量重命名指定文件夹下的 .txt 文件。

    Args:
        target_dir_name (str): 目标文件夹的名称。
        padding (int): 序号的补零位数。
    """
    # 使用 pathlib 获取文件夹路径，这是一种更现代、更健壮的方式
    target_path = Path(target_dir_name)

    # 1. 安全检查：确保文件夹存在
    if not target_path.is_dir():
        print(f"错误：文件夹 '{target_dir_name}' 不存在！请检查名称或脚本位置。")
        return

    # 2. 获取所有 .txt 文件，并按字母顺序排序
    # 排序可以确保每次运行脚本时，文件的重命名顺序都是一致的
    print(f"正在扫描文件夹 '{target_dir_name}' 中的 .txt 文件...")
    txt_files = sorted(list(target_path.glob("*.txt")))

    if not txt_files:
        print("文件夹中没有找到任何 .txt 文件。")
        return

    print(f"找到 {len(txt_files)} 个文件，准备开始重命名...")

    # 3. 循环遍历并重命名
    counter = 1
    for old_file_path in txt_files:
        # 构建新的文件名，例如： "std_001.txt"
        # f-string 中的 :0{padding}d 语法会自动补零
        new_filename = f"{target_path.name}_{counter:0{padding}d}.txt"
        new_file_path = target_path / new_filename

        # 如果新文件名与旧文件名相同，则跳过（避免对已处理过的文件再次处理）
        if old_file_path == new_file_path:
            print(f"跳过已命名文件: {old_file_path.name}")
            counter += 1
            continue

        # 执行重命名
        try:
            old_file_path.rename(new_file_path)
            print(f"成功: '{old_file_path.name}'  ->  '{new_filename}'")
        except Exception as e:
            print(f"重命名 '{old_file_path.name}' 时出错: {e}")

        counter += 1

    print("\n所有文件重命名完毕！")


if __name__ == '__main__':
    # 运行主函数
    batch_rename_files(TARGET_DIR_NAME, PADDING_WIDTH)