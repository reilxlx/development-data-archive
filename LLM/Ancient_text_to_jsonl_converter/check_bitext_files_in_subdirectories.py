import os
from pathlib import Path

def analyze_directory(root_dir):
    root_path = Path(root_dir)
    
    total_branches = 0
    total_bitext_files = 0
    total_size_bytes = 0
    folders_without_bitext = []

    for current_dir, dirs, files in os.walk(root_path):
        current_path = Path(current_dir)
        
        # 检查是否为最终的子文件夹（不包含其他文件夹）
        if not any(current_path.joinpath(d).is_dir() for d in dirs):
            total_branches += 1
            
            if 'bitext.txt' in files:
                total_bitext_files += 1
                file_path = current_path / 'bitext.txt'
                total_size_bytes += file_path.stat().st_size
            else:
                folders_without_bitext.append(str(current_path.relative_to(root_path)))

    # 将字节转换为 MB
    total_size_mb = total_size_bytes / (1024 * 1024)

    return total_branches, total_bitext_files, total_size_mb, folders_without_bitext

# 使用示例
root_directory = './your_root_directory'
branches, bitext_files, size_mb, missing_bitext = analyze_directory(root_directory)

print(f"最终子文件夹总数: {branches}")
print(f"bitext.txt 文件总数: {bitext_files}")
print(f"bitext.txt 文件总大小: {size_mb:.2f} MB")

if missing_bitext:
    print("\n以下文件夹没有 bitext.txt 文件:")
    for folder in missing_bitext:
        print(folder)
else:
    print("\n所有最终子文件夹都包含 bitext.txt 文件。")
