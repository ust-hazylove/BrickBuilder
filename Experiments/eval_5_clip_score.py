import os
import json
import glob
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg') # 后台渲染，不显示窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import sys

# 尝试导入 transformers
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("❌ Error: 'transformers' library not found.")
    print("Please install it: pip install transformers torch")
    sys.exit(1)

# ==========================================
# 配置
# ==========================================
SAMPLE_SIZE = 500       # 抽样数量 (渲染很慢，建议不要全跑)
RANDOM_SEED = 42
IMAGE_SIZE = 224        # CLIP 输入尺寸
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 文本提示模板
# 我们希望生成的积木看起来像乐高，并且像那个物体
PROMPT_TEMPLATE = "A photo of a {class_name} made of lego bricks"

# ==========================================
# 1. 轻量级渲染器 (复刻自 ldr_stability)
# ==========================================
def render_bricks_to_image(bricks, save_path, dpi=100):
    """
    使用 Matplotlib 将积木列表渲染为图片
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色映射 (简单的一致性颜色)
    # 0_init=灰色, 1_greedy=蓝色, 2_legolization=绿色, 3_ours=红色
    # 这里为了 CLIP 识别，统一用一种显眼的颜色，或者根据 method 区分
    # 为了公平，我们统一用 "乐高红" 或 "经典黄"，或者保留 source 的 color id
    
    xs, ys, zs, dxs, dys, dzs, colors = [], [], [], [], [], [], []
    
    for b in bricks:
        # 坐标转换: JSON (x, y, z, sx, sy) -> Plot (x, z, y)
        # 注意：matplotlib 的 bar3d 是 (x, y, z, dx, dy, dz)
        # 我们的 z 是高度。
        x = b['x']
        z = b['y'] # depth
        y = b['z'] # height
        sx = b.get('sx', 1)
        sy = b.get('sy', 1) # depth size
        h = 1.0 # height
        
        # 简单的遮挡剔除/降采样 (可选，为了速度)
        
        xs.append(x)
        ys.append(z)
        zs.append(y)
        dxs.append(sx)
        dys.append(sy)
        dzs.append(h)
        
        # 颜色处理
        c_id = b.get('color', 1)
        if c_id == 0: c = '#A0A0A0' # Grey (Init)
        elif c_id == 1: c = '#1E90FF' # Blue (Greedy)
        elif c_id == 2: c = '#32CD32' # Green (Lego)
        elif c_id == 3: c = '#DC143C' # Red (Ours)
        else: c = '#FFD700' # Gold
        colors.append(c)

    # 绘制
    ax.bar3d(xs, ys, zs, dxs, dys, dzs, color=colors, shade=True, edgecolor='none', alpha=1.0)
    
    # 设置视角 (Isometric-like)
    ax.view_init(elev=30, azim=-45)
    
    # 移除坐标轴
    ax.set_axis_off()
    
    # 保存
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=dpi, transparent=True)
    plt.close(fig)

# ==========================================
# 2. CLIP 评分核心
# ==========================================
class CLIPScorer:
    def __init__(self):
        print(f"🔹 Loading CLIP model (openai/clip-vit-base-patch32) on {DEVICE}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP Loaded.")

    def calculate_score(self, image_path, text):
        try:
            image = Image.open(image_path)
            inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # 计算图像-文本相似度 (Logits_per_image)
            logits_per_image = outputs.logits_per_image # this is the cosine similarity * 100
            score = logits_per_image.item()
            return score
        except Exception as e:
            # print(f"CLIP Error: {e}")
            return 0.0

# ==========================================
# 3. 工具函数
# ==========================================
def get_pure_id(filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    for suffix in ["_bricks", "_naive", "_greedy", "_legolization", "_ours"]:
        name = name.replace(suffix, "")
    return name

def extract_class_name(pure_id):
    # filename: "airplane_0001" -> "airplane"
    # filename: "adult_beds_0002" -> "adult beds"
    # 假设格式是 {class}_{id}
    parts = pure_id.split('_')
    # 移除最后的数字部分
    if parts[-1].isdigit():
        class_parts = parts[:-1]
    else:
        # 有些可能是 airplane_0001_s00
        # 简单的启发式：取第一个词，或者除了数字以外的所有词
        class_parts = [p for p in parts if not p.isdigit() and len(p) > 1]
    
    return " ".join(class_parts)

def get_consistent_sample_ids(folders):
    id_sets = []
    for name, path in folders.items():
        if not os.path.exists(path): continue
        files = glob.glob(os.path.join(path, "*.json"))
        id_sets.append(set(get_pure_id(f) for f in files))
    
    if not id_sets: return set()
    common_ids = set.intersection(*id_sets)
    
    sorted_ids = sorted(list(common_ids))
    random.seed(RANDOM_SEED)
    if len(sorted_ids) > SAMPLE_SIZE:
        return set(random.sample(sorted_ids, SAMPLE_SIZE))
    return set(sorted_ids)

# ==========================================
# 主流程
# ==========================================
def run_clip_evaluation(dataset_root):
    folders = {
        "0_init": os.path.join(dataset_root, "0_init"),
        "1_greedy": os.path.join(dataset_root, "1_greedy"),
        "2_legolization": os.path.join(dataset_root, "2_legolization"),
        "3_ours": os.path.join(dataset_root, "3_ours")
    }
    
    # 1. 确定抽样 ID
    target_ids = get_consistent_sample_ids(folders)
    if not target_ids:
        print("❌ Error: No common files found.")
        return
        
    # 2. 初始化 CLIP
    scorer = CLIPScorer()
    
    # 3. 创建临时渲染目录
    temp_render_dir = "temp_renders"
    os.makedirs(temp_render_dir, exist_ok=True)
    
    print(f"\n🚀 Starting CLIP Score Evaluation (Sample Size: {len(target_ids)})")
    print(f"   Prompt Template: '{PROMPT_TEMPLATE}'")
    print("=" * 70)
    print(f"{'Method':<18} | {'Avg CLIP Score':<15}")
    print("-" * 70)
    
    for method_name, folder_path in folders.items():
        if not os.path.exists(folder_path): continue
        
        all_files = glob.glob(os.path.join(folder_path, "*.json"))
        target_files = [f for f in all_files if get_pure_id(f) in target_ids]
        
        clip_scores = []
        
        # 进度条
        for fpath in tqdm(target_files, desc=f"Eval {method_name}", leave=False, unit="img"):
            pure_id = get_pure_id(fpath)
            class_name = extract_class_name(pure_id)
            text_prompt = PROMPT_TEMPLATE.format(class_name=class_name)
            
            # 临时图片路径
            img_path = os.path.join(temp_render_dir, f"{method_name}_{pure_id}.png")
            
            try:
                # A. 读取
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    bricks = data if isinstance(data, list) else data.get("bricks", [])
                
                if not bricks:
                    clip_scores.append(0.0)
                    continue

                # B. 渲染 (如果图片不存在才渲染，加速调试)
                if not os.path.exists(img_path):
                    render_bricks_to_image(bricks, img_path)
                
                # C. 打分
                score = scorer.calculate_score(img_path, text_prompt)
                clip_scores.append(score)
                
            except Exception as e:
                # print(f"Error: {e}")
                pass
                
        # 统计
        if clip_scores:
            avg_score = np.mean(clip_scores)
            print(f"{method_name:<18} | {avg_score:<15.4f}")
        else:
            print(f"{method_name:<18} | [No Data]")

    # 清理临时文件 (可选)
    # import shutil
    # shutil.rmtree(temp_render_dir)
    print("=" * 70)
    print(f"Note: Scores represent semantic similarity to the text description.")
    print(f"      Rendered images are saved in '{temp_render_dir}' for inspection.")

if __name__ == "__main__":
    DATASET_ROOT = "benchmark_data"
    run_clip_evaluation(DATASET_ROOT)