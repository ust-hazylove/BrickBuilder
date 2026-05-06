import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, generate_binary_structure

class PrimitiveMapper:
    """
    将体素网格转换为乐高图元列表。
    实现了 'Boundary-Aware Vertical Splitting' 策略。
    """
    def __init__(self, voxel_grid):
        self.voxels = voxel_grid
        self.shape = voxel_grid.shape
        
    def get_boundary_mask(self):
        """
        计算边界掩码：如果是实体且周围有空隙，则为边界。
        """
        # 定义 6-邻域结构
        struct = generate_binary_structure(3, 1)
        # 腐蚀操作：只有完全被包裹的内部体素会被保留
        eroded = binary_erosion(self.voxels, structure=struct)
        # 边界 = 原图 - 内部 (即原图中存在，但不是内部的点)
        boundary = (self.voxels > 0) & (~eroded)
        return boundary

    def map_primitives(self, split_boundary=False):
        """
        生成渲染用的图元列表。
        :param split_boundary: 是否启用边界拆解策略
        """
        primitives = []
        boundary_mask = self.get_boundary_mask()
        
        # 遍历所有体素
        # 注意: 这里的坐标系假设是 (x, y, z)，y 是高度轴
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    if self.voxels[x, y, z] == 0:
                        continue
                    
                    is_boundary = boundary_mask[x, y, z]
                    
                    if split_boundary and is_boundary:
                        # --- 策略：拆解为 3 块板 (Plate) ---
                        # 乐高单位：1 Brick Height = 1.0, 1 Plate Height = 0.333
                        # 为了可视化方便，我们在 y 轴上归一化
                        base_y = y
                        # 添加 3 块板，每块高度 1/3
                        for i in range(3):
                            primitives.append({
                                'type': 'Plate 1x1',
                                'pos': (x, base_y + i * 0.333, z),
                                'size': (1, 0.3, 1), # 稍微留一点缝隙(0.3 vs 0.333)以显示边缘
                                'color': '#D32F2F' if i % 2 == 0 else '#B71C1C' # 稍微变色以突显层级
                            })
                    else:
                        # --- 策略：保持为 1 块砖 (Brick) ---
                        primitives.append({
                            'type': 'Brick 1x1',
                            'pos': (x, y, z),
                            'size': (1, 1, 1),
                            'color': '#1976D2' # 内部用蓝色表示坚固
                        })
                        
        return primitives

# ==============================================================================
# 可视化工具 (用于生成论文对比图)
# ==============================================================================
def generate_mock_data(grid_size=16):
    """生成一个球体数据用于演示"""
    x, y, z = np.indices((grid_size, grid_size, grid_size))
    center = grid_size // 2
    radius = grid_size // 2 - 2
    sphere = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
    return sphere.astype(int)

def render_scene(ax, primitives, title):
    """使用 Matplotlib 绘制 3D 箱体"""
    ax.set_title(title, fontsize=15)
    ax.set_axis_off()
    
    # 简单的遮挡排序 (Painter's Algorithm)，防止渲染错误
    # 根据离摄像机的距离排序 (这里简单按 x+y+z 排序)
    primitives.sort(key=lambda p: p['pos'][0] + p['pos'][1] + p['pos'][2], reverse=True)

    for p in primitives:
        x, y, z = p['pos']
        dx, dy, dz = p['size']
        c = p['color']
        
        # 绘制立方体
        ax.bar3d(x, z, y, dx, dz, dy, color=c, edgecolor='k', linewidth=0.1, shade=True, alpha=0.9)

    # 设置比例
    ax.set_box_aspect((1, 1, 1))

def main():
    print("Generating Mock Data (Sphere)...")
    grid_size = 12 # 稍微小一点以便渲染清晰
    voxels = generate_mock_data(grid_size)
    
    mapper = PrimitiveMapper(voxels)
    
    print("Mapping: Naive (Pure Bricks)...")
    prims_naive = mapper.map_primitives(split_boundary=False)
    
    print("Mapping: Refined (Split Plates)...")
    prims_refined = mapper.map_primitives(split_boundary=True)
    
    print(f"Stats: Naive Bricks: {len(prims_naive)} | Refined Parts: {len(prims_refined)}")
    
    # --- Plotting ---
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Baseline
    ax1 = fig.add_subplot(121, projection='3d')
    # 调整视角以看清边缘锯齿
    ax1.view_init(elev=20, azim=45)
    render_scene(ax1, prims_naive, "Baseline: Pure Voxelization (Minecraft-like)")
    
    # Right: Ours
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(elev=20, azim=45)
    render_scene(ax2, prims_refined, "Ours: Boundary-Aware Plate Splitting")
    
    plt.tight_layout()
    save_path = "comparison_plate_split.png"
    plt.savefig(save_path, dpi=300)
    print(f"Comparison image saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()