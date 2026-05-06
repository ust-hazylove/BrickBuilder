import numpy as np
from scipy.spatial import cKDTree

class LegoColorPalette:
    def __init__(self):
        # LDraw 标准色卡 (ID, R, G, B, Name)
        # 选取了最常用的基础色，你可以根据需要扩展
        self.colors = [
            (0,   5,  19,  29, "Black"),
            (1,   0,  85, 191, "Blue"),
            (2,  37, 122,  62, "Green"),
            (4, 191,   0,   0, "Red"),
            (14, 242, 205,  55, "Yellow"),
            (15, 255, 255, 255, "White"),
            (19, 160, 165, 169, "Tan"),
            (71, 160, 160, 160, "Light Bluish Gray"),
            (72, 100, 100, 100, "Dark Bluish Gray"),
            (320, 170,  61,  54, "Dark Red"),
            (321,  64,  84, 142, "Dark Azure"),
            (322,  96, 116, 161, "Medium Blue"),
            (323, 170, 224, 208, "Light Aqua"),
            (326, 226, 249, 154, "Yellowish Green"),
            (330, 255, 209, 143, "Olive Green"),
            (28, 204, 142, 104, "Nougat"), # 皮肤色近似
            (25, 214, 121, 35,  "Orange")
        ]
        
        # 构建 KDTree 用于快速查找最近颜色
        self.rgb_data = np.array([[c[1], c[2], c[3]] for c in self.colors])
        self.ids = [c[0] for c in self.colors]
        self.tree = cKDTree(self.rgb_data)

    def get_nearest_color_id(self, rgb):
        """
        输入: (R, G, B) 范围 0-255
        输出: LDraw Color ID
        """
        # 查询最近邻
        dist, idx = self.tree.query(rgb)
        return self.ids[idx]

    def get_nearest_color_id_batch(self, rgb_array):
        """
        批量查询
        输入: (N, 3) array
        输出: (N,) IDs
        """
        dist, idx = self.tree.query(rgb_array)
        return np.array(self.ids)[idx]

# 全局单例
PALETTE = LegoColorPalette()