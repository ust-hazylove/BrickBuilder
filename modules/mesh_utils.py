# modules/mesh_utils.py
import numpy as np
import open3d as o3d
import trimesh
from collections import deque
import os

class MeshUtils:
    @staticmethod
    def normalize_to_unit_cube(mesh_o3d):
        """
        将 Open3D 网格归一化到单位立方体，并移动到原点。
        """
        v = np.asarray(mesh_o3d.vertices)
        if len(v) == 0:
            raise ValueError("Mesh has no vertices.")
            
        vmin, vmax = v.min(0), v.max(0)
        extent = vmax - vmin
        # 计算缩放比例，保留 1e-8 防止除零
        scale = 1.0 / max(float(extent.max()), 1e-8)
        
        # 归一化顶点坐标
        v2 = (v - vmin) * scale
        # 移动到中心
        center = (vmax - vmin) * scale / 2.0
        v2 = v2 - center
        
        # 重新创建 Mesh
        m2 = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v2), 
            mesh_o3d.triangles
        )
        m2.compute_vertex_normals()
        return m2

    @staticmethod
    def voxelize_surface(mesh_o3d, res=64):
        """
        生成表面体素（壳）。
        """
        # 均匀采样点云
        pts = mesh_o3d.sample_points_uniformly(number_of_points=max(50000, res * res * 20))
        # +0.5 将中心对齐到 [0.5, 0.5, 0.5] 以适应体素网格
        P = np.clip(np.asarray(pts.points) + 0.5, 0.0, 1.0)
        
        # 离散化坐标
        vox = np.zeros((res, res, res), dtype=np.uint8)
        idx = np.clip(np.floor(P * res).astype(int), 0, res - 1)
        
        # 标记占据
        vox[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
        return vox

    @staticmethod
    def flood_fill_solid(surface_voxels):
        """
        使用洪水填充算法将闭合的表面体素转为实心体素。
        """
        assert surface_voxels.ndim == 3
        X, Y, Z = surface_voxels.shape
        visited = np.zeros_like(surface_voxels, dtype=np.uint8)
        
        q = deque()

        def try_push(x, y, z):
            if 0 <= x < X and 0 <= y < Y and 0 <= z < Z:
                if surface_voxels[x, y, z] == 0 and visited[x, y, z] == 0:
                    visited[x, y, z] = 1
                    q.append((x, y, z))

        # 1. 将六个面的边界空气入队
        for x in range(X):
            try_push(x, 0, 0);           try_push(x, 0, Z - 1)
            try_push(x, Y - 1, 0);       try_push(x, Y - 1, Z - 1)
        for y in range(Y):
            try_push(0, y, 0);           try_push(0, y, Z - 1)
            try_push(X - 1, y, 0);       try_push(X - 1, y, Z - 1)
        for z in range(Z):
            try_push(0, 0, z);           try_push(0, Y - 1, z)
            try_push(X - 1, 0, z);       try_push(X - 1, Y - 1, z)

        # 2. BFS 扩散（标记外部空气）
        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < X and 0 <= ny < Y and 0 <= nz < Z:
                    if surface_voxels[nx, ny, nz] == 0 and visited[nx, ny, nz] == 0:
                        visited[nx, ny, nz] = 1
                        q.append((nx, ny, nz))

        # 3. 反转：未访问的即为内部实体
        solid = (1 - visited).astype(np.uint8)
        return solid


    @staticmethod
    def rotate_voxels(voxel_grid: np.ndarray, axis='x', k=1):
        """
        旋转体素网格，用于修正“躺倒”的模型。
        axis: 'x', 'y', 'z'
        k: 旋转次数 (1 = 90度)
        """
        if axis == 'x':
            # 绕 X 轴旋转 (在 Y-Z 平面旋转)
            return np.rot90(voxel_grid, k=k, axes=(1, 2))
        elif axis == 'y':
            # 绕 Y 轴旋转 (在 X-Z 平面旋转)
            return np.rot90(voxel_grid, k=k, axes=(0, 2))
        elif axis == 'z':
            # 绕 Z 轴旋转 (在 X-Y 平面旋转)
            return np.rot90(voxel_grid, k=k, axes=(0, 1))
        else:
            return voxel_grid

    @staticmethod
    def glb_to_voxels(glb_path, resolution=64, fill=True):
        """
        读取 GLB -> 归一化 -> 体素化 -> (可选) 填充实心
        fill=True: 返回实心体素 (用于 RL 修复)
        fill=False: 返回空心壳体素 (用于快速预览)
        """
        # 读取 Mesh
        mesh = o3d.io.read_triangle_mesh(str(glb_path))
        mesh.compute_vertex_normals()
        
        # 归一化
        mesh_norm = MeshUtils.normalize_to_unit_cube(mesh)
        
        # 1. 获取表面体素 (Shell)
        surface_vox = MeshUtils.voxelize_surface(mesh_norm, res=resolution)
        
        if fill:
            # 2. 如果开启填充，执行洪水填充算法
            solid_vox = MeshUtils.flood_fill_solid(surface_vox)
            return solid_vox.astype(bool)
        else:
            # 3. 如果关闭填充，直接返回表面壳
            return surface_vox.astype(bool)

    @staticmethod
    def save_voxels_as_mesh(voxel_grid: np.ndarray, save_path: str):
        """
        将体素网格导出为 GLB 模型，用于前端预览。
        """
        if voxel_grid.sum() == 0:
            return None

        try:
            # 使用 trimesh 将体素转换为立方体网格
            v = trimesh.voxel.VoxelGrid(trimesh.voxel.encoding.DenseEncoding(voxel_grid))
            # 转换为网格 (每个体素一个 cube)
            mesh = v.as_boxes()
            
            # 导出
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            mesh.export(save_path)
            return save_path
        except Exception as e:
            print(f"[MeshUtils] Voxel export failed: {e}")
            return None