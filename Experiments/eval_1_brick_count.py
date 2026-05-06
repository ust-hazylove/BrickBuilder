import os
import json
import glob
import numpy as np
from tqdm import tqdm  

def calculate_brick_counts(dataset_root):
    # 定义四个目标文件夹
    target_folders = [
        "0_init",           # Init (1x1 Naive)
        "1_greedy",         # Baseline 1 (Image2Lego/Greedy)
        "2_legolization",   # Baseline 2 (Legolization Optimization)
        "3_ours"            # Ours (RL + Greedy Merge)
    ]

    print("\n📊 Metric 1: Average Brick Count Evaluation")
    print("=" * 60)
    print(f"{'Method':<20} | {'Avg Brick Count':<15} | {'Min':<8} | {'Max':<8}")
    print("-" * 60)

    results = {}

    for folder_name in target_folders:
        folder_path = os.path.join(dataset_root, folder_name)
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"{folder_name:<20} | {'[Missing]':<15} | -        | -")
            continue

        # 获取所有 json 文件
        files = glob.glob(os.path.join(folder_path, "*.json"))
        counts = []

        # === 修改点：添加 tqdm 进度条 ===
        # desc: 进度条左边的描述文字
        # unit: 单位名称
        # leave: True 表示跑完保留进度条，False 表示跑完清除
        if not files:
            print(f"{folder_name:<20} | {'[Empty]':<15} | -        | -")
            continue

        for fpath in tqdm(files, desc=f"Processing {folder_name}", unit="file", leave=False):
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    
                    # 兼容不同格式：直接是list 或 包含在dict中
                    if isinstance(data, list):
                        num_bricks = len(data)
                    elif isinstance(data, dict) and "bricks" in data:
                        num_bricks = len(data["bricks"])
                    else:
                        num_bricks = 0
                    
                    counts.append(num_bricks)
            except Exception as e:
                # 使用 tqdm.write 来避免打印打断进度条
                # tqdm.write(f"Error reading {fpath}: {e}")
                pass

        # 计算统计量
        if counts:
            avg_val = np.mean(counts)
            min_val = np.min(counts)
            max_val = np.max(counts)
            
            print(f"{folder_name:<20} | {avg_val:<15.2f} | {min_val:<8} | {max_val:<8}")
            results[folder_name] = avg_val
        else:
            print(f"{folder_name:<20} | {'[Empty]':<15} | -        | -")

    print("=" * 60)
    print("✅ Done.")

if __name__ == "__main__":

    DATASET_ROOT = "benchmark_data"
    
    calculate_brick_counts(DATASET_ROOT)