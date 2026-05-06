import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==========================================
#               用户配置区域
#   (这里修改的所有参数都会真正生效)
# ==========================================

# 1. 输入输出
INPUT_CSV = "scalability_results.csv"
OUTPUT_IMG = "scalability_final_plot_2025.png"
DPI = 300

# 2. 画布与字体
FIGURE_SIZE = (10, 6)
FONT_FAMILY = 'Arial' 
GRID_STYLE = '--'    
GRID_ALPHA = 0.5     

# 3. 曲线样式
STYLE_OURS = {
    'label': 'Ours (Graph Decomposition)',
    'color': '#1f77b4', 'linestyle': '-', 'linewidth': 2.5,
    'marker': 'o', 'markersize': 8, 'markeredgecolor': 'white', 'markeredgewidth': 1.5
}

STYLE_BASE = {
    'label': 'Baseline (RL + Action Mask)',
    'color': '#d62728', 'linestyle': '--', 'linewidth': 2.5,
    'marker': 's', 'markersize': 8, 'markeredgecolor': 'white', 'markeredgewidth': 1.5
}

# 4. 坐标轴与标题
# TITLE_TEXT = 'Scalability Analysis'
X_LABEL = 'Number of Bricks'
Y_LABEL = 'Computation Time (s)'
FONT_SIZES = {
    'title': 16, 'label': 14, 'tick': 12, 'legend': 12, 'annotation': 11
}

# 5. 标注策略 (哪些点需要标数值?)
# 例如：标注后 1/4 段均匀分布的 5 个点
ANNOTATE_INDICES = lambda n: [n-1]

# 6. 【关键修复】标注位置配置 (Offset Configuration)
# Ours (下方曲线) 的偏移量 (x, y) - 单位是 points
OFFSET_OURS = (0, 10)     

# Baseline (上方曲线) 的默认偏移量 (x, y)
OFFSET_BASE = (0, 1)    

# 特殊处理：Baseline 最后一个点是否要向左放？(防止出界)
BASE_LAST_POINT_ADJUST = True       # 开关
BASE_LAST_POINT_ALIGN = 'right'     # 对齐方式 'center' / 'right' / 'left'
BASE_LAST_POINT_OFFSET_X = -10      # 特殊的X偏移量

# ==========================================
#             核心绘图逻辑 
# ==========================================

def plot_chart():
    # 1. 读取数据
    try:
        df = pd.read_csv(INPUT_CSV)
        df = df.sort_values("Bricks").reset_index(drop=True)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_CSV}。")
        return

    # 全局字体
    plt.rcParams['font.family'] = FONT_FAMILY
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # 2. 绘制曲线
    ax.plot(df["Bricks"], df["Ours"], **STYLE_OURS)
    ax.plot(df["Bricks"], df["Baseline"], **STYLE_BASE)

    # 3. 绘制标注
    indices = ANNOTATE_INDICES(len(df))
    # 去重、排序、防越界
    indices = sorted(list(set([i for i in indices if 0 <= i < len(df)])))

    for i in indices:
        row = df.iloc[i]
        x_val = row["Bricks"]
        
        # --- 标注 Ours (蓝色) ---
        y_ours = row["Ours"]
        ax.annotate(f"{y_ours:.2f}s", 
                    xy=(x_val, y_ours), 
                    xytext=OFFSET_OURS, # 【修复】现在使用配置变量
                    textcoords="offset points", 
                    ha='center', va='bottom',
                    fontsize=FONT_SIZES['annotation'], 
                    fontweight='bold', 
                    color=STYLE_OURS['color'])
        
        # --- 标注 Baseline (红色) ---
        y_base = row["Baseline"]
        
        # 判断是否是最后一个点，且是否开启了特殊处理
        is_last = (i == len(df) - 1) or (i == indices[-1])
        
        if is_last and BASE_LAST_POINT_ADJUST:
            # 使用特殊配置
            current_ha = BASE_LAST_POINT_ALIGN
            current_offset = (BASE_LAST_POINT_OFFSET_X, OFFSET_BASE[1])
        else:
            # 使用默认配置
            current_ha = 'center'
            current_offset = OFFSET_BASE
            
        ax.annotate(f"{y_base:.1f}s", 
                    xy=(x_val, y_base), 
                    xytext=current_offset, # 【修复】现在使用配置变量
                    textcoords="offset points", 
                    ha=current_ha, va='bottom',
                    fontsize=FONT_SIZES['annotation'], 
                    fontweight='bold', 
                    color=STYLE_BASE['color'])

    # 4. 装饰图表
    # ax.set_title(TITLE_TEXT, fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.set_xlabel(X_LABEL, fontsize=FONT_SIZES['label'])
    ax.set_ylabel(Y_LABEL, fontsize=FONT_SIZES['label'])
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    ax.grid(True, linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=True, fancybox=True, shadow=True, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=DPI)
    print(f"✅ 修正版图表已生成: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    plot_chart()