import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

class VoxelStabilityAnalyzer:
    def __init__(self, config=None):
        if config is None: config = {}
        self.g = config.get("g", 9.8)
        self.T = config.get("T", 20.0)
        self.friction_coeff = config.get("mu", 0.3)
        self.voxel_mass = config.get("voxel_mass", 0.002)
        self.print_log = config.get("print_log", False)
        self.time_limit = config.get("time_limit", 20.0)
        
        # [开关] 刚体模式：忽略乐高咬合力限制，只要平衡就行 (用于检测几何是否合理)
        self.rigid_body_mode = config.get("rigid_body_mode", False)

    def analyze(self, voxel_grid):
        t_start = time.time()
        active_indices = np.argwhere(voxel_grid > 0)
        n_voxels = len(active_indices)
        
        if n_voxels == 0:
            return {"stable": True, "stability_score": 0.0, "max_force": 0.0}

        voxel_map = {tuple(idx): i for i, idx in enumerate(active_indices)}
        
        # 初始化 Gurobi
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            model = gp.Model("voxel_stability", env=env)
        except:
            model = gp.Model("voxel_stability")

        model.setParam("OutputFlag", 1 if self.print_log else 0)
        model.setParam("TimeLimit", self.time_limit)
        
        # 力的大小限制
        force_limit = 1e9 if self.rigid_body_mode else (self.T * 100.0)
        
        # 存储力变量
        forces = {} 
        # 存储地面反作用力 (Ground Reaction Forces)
        ground_forces = {}

        # 1. 创建内部连接力变量 (Internal Forces)
        for i, (x, y, z) in enumerate(active_indices):
            # 检查 6 个方向的邻居
            check_dirs = [
                ([x+1, y, z], 'x_pos'), ([x-1, y, z], 'x_neg'),
                ([x, y+1, z], 'y_pos'), ([x, y-1, z], 'y_neg'),
                ([x, y, z+1], 'z_up'),  ([x, y, z-1], 'z_down') 
            ]
            
            for coord, direction in check_dirs:
                coord_tuple = tuple(coord)
                
                # 情况 A: 邻居是另一个体素 -> 创建相互作用力
                if coord_tuple in voxel_map:
                    neighbor_idx = voxel_map[coord_tuple]
                    # 为了避免重复创建，只在 i < neighbor_idx 时创建变量
                    if i < neighbor_idx:
                        f_x = model.addVar(lb=-force_limit, ub=force_limit, name=f"F_{i}_{neighbor_idx}_x")
                        f_y = model.addVar(lb=-force_limit, ub=force_limit, name=f"F_{i}_{neighbor_idx}_y")
                        f_z = model.addVar(lb=-force_limit, ub=force_limit, name=f"F_{i}_{neighbor_idx}_z")
                        forces[(i, neighbor_idx)] = (f_x, f_y, f_z)

            # 情况 B: [关键修复] 体素在地面 (Z=0) -> 创建地面支撑力
            if z == 0:
                # 地面能提供无限的支撑力 (Fz > 0) 和 摩擦力 (Fx, Fy)
                # Fz_ground: 地面对体素向上的力
                g_x = model.addVar(lb=-force_limit, ub=force_limit, name=f"G_{i}_x")
                g_y = model.addVar(lb=-force_limit, ub=force_limit, name=f"G_{i}_y")
                g_z = model.addVar(lb=0, ub=force_limit, name=f"G_{i}_z") # 只能向上支撑，不能把地面拉起来
                ground_forces[i] = (g_x, g_y, g_z)

        # 2. 建立平衡方程 (Equilibrium Constraints)
        max_load_z = model.addVar(lb=0, name="max_load_z")
        
        for i, (x, y, z) in enumerate(active_indices):
            fx_sum, fy_sum, fz_sum = [], [], []
            tx_sum, ty_sum = [], [] 
            
            # 重力 (向下)
            gravity = -self.voxel_mass * self.g
            
            # A. 收集邻居的力
            check_dirs = [
                ([x+1, y, z], 'x_pos'), ([x-1, y, z], 'x_neg'),
                ([x, y+1, z], 'y_pos'), ([x, y-1, z], 'y_neg'),
                ([x, y, z+1], 'z_up'),  ([x, y, z-1], 'z_down') 
            ]
            for coord, direction in check_dirs:
                coord_tuple = tuple(coord)
                if coord_tuple in voxel_map:
                    n_idx = voxel_map[coord_tuple]
                    
                    # 获取之前定义的变量
                    if i < n_idx:
                        fx, fy, fz = forces[(i, n_idx)]
                        sign = 1.0 # n_idx 施加给 i 的力
                    else:
                        fx, fy, fz = forces[(n_idx, i)]
                        sign = -1.0 # i 施加给 n_idx 的力 => n_idx 给 i 的是反作用力
                    
                    # 作用在当前体素 i 上的力
                    fx_curr, fy_curr, fz_curr = sign*fx, sign*fy, sign*fz
                    
                    fx_sum.append(fx_curr)
                    fy_sum.append(fy_curr)
                    fz_sum.append(fz_curr)
                    
                    # 力矩计算 (简化: 力臂 0.5)
                    L = 0.5
                    if direction == 'z_up':    # 上面的砖给的力，作用点在 (0,0,0.5)
                        tx_sum.append(-L * fy_curr); ty_sum.append(L * fx_curr)
                    elif direction == 'z_down': # 下面的砖给的力，作用点在 (0,0,-0.5)
                        tx_sum.append(L * fy_curr); ty_sum.append(-L * fx_curr)
                    elif direction == 'x_pos':
                        ty_sum.append(L * fz_curr)
                    elif direction == 'x_neg':
                        ty_sum.append(-L * fz_curr)
                    elif direction == 'y_pos':
                        tx_sum.append(-L * fz_curr)
                    elif direction == 'y_neg':
                        tx_sum.append(L * fz_curr)

            # B. [关键修复] 收集地面的力
            if i in ground_forces:
                gx, gy, gz = ground_forces[i]
                fx_sum.append(gx)
                fy_sum.append(gy)
                fz_sum.append(gz)
                
                # 地面力作用在底部 (0,0,-0.5)
                L = 0.5
                tx_sum.append(L * gy)
                ty_sum.append(-L * gx)

            # C. 平衡方程 Sigma F = 0
            model.addConstr(gp.quicksum(fx_sum) == 0, name=f"Eq_Fx_{i}")
            model.addConstr(gp.quicksum(fy_sum) == 0, name=f"Eq_Fy_{i}")
            model.addConstr(gp.quicksum(fz_sum) + gravity == 0, name=f"Eq_Fz_{i}")
            
            # D. 力矩平衡 Sigma T = 0
            # 刚体模式下 (glue)，如果觉得太难收敛，可以注释掉力矩约束
            # 但为了准确性，建议保留。如果依然无解，可尝试暂时注释掉。
            if not self.rigid_body_mode:
                model.addConstr(gp.quicksum(tx_sum) == 0, name=f"Eq_Tx_{i}")
                model.addConstr(gp.quicksum(ty_sum) == 0, name=f"Eq_Ty_{i}")

            # 记录最大受力 (只看内部 Z 向力)
            # 地面支撑力不计入"积木断裂"风险
            # 只有积木间的拉力才算
            pass 

        # 3. 目标函数: 最小化能量 (Sum of Forces)
        obj_terms = []
        for (idx_a, idx_b), (fx, fy, fz) in forces.items():
            abs_fz = model.addVar(lb=0); model.addConstr(abs_fz >= fz); model.addConstr(abs_fz >= -fz)
            obj_terms.append(abs_fz)
            model.addConstr(max_load_z >= fz) # 记录最大拉/压力
        
        # 也惩罚地面力，避免数值飘逸
        for i in ground_forces:
            gx, gy, gz = ground_forces[i]
            obj_terms.append(gz)

        model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)
        
        # 4. 求解
        model.optimize()
        
        solve_time = time.time() - t_start
        
        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
            max_force = max_load_z.X
            stability_score = 1.0 # 只要能平衡就是 1.0 (Rigid Mode)
            
            if not self.rigid_body_mode:
                # 乐高模式：如果受力太大，分数降低
                stability_score = max(0.0, 1.0 - max_force / (self.T * 10))
                
            return {
                "stable": True, 
                "stability_score": stability_score, 
                "max_force": max_force, 
                "status": "OPTIMAL"
            }
        elif model.status == GRB.INFEASIBLE:
            return {"stable": False, "stability_score": 0.0, "max_force": float('inf'), "status": "INFEASIBLE"}
        else:
            return {"stable": False, "stability_score": 0.0, "max_force": float('inf'), "status": f"CODE_{model.status}"}

if __name__ == "__main__":
    # 测试一下
    grid = np.zeros((5, 5, 5), dtype=int)
    grid[2, 2, 0] = 1 # 地面基座
    grid[2, 2, 1] = 1 # 上面的砖
    analyzer = VoxelStabilityAnalyzer(config={"rigid_body_mode": True, "print_log": True})
    res = analyzer.analyze(grid)
    print("Test Result:", res)