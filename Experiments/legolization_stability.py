import cvxpy as cp
import numpy as np

class LegoSolverCVX:
    """
    基于 Legolization (Luo et al. 2015) 物理模型的稳定性求解器。
    修复了 num_edges=0 时的崩溃问题。
    """
    def __init__(self, friction=71.6, gravity=10.0):
        self.T = friction 
        self.G = gravity

    def solve(self, brick_list):
        if not brick_list: 
            return 0.0
        
        # 1. 预处理：强制落地
        min_z = min([b['z'] for b in brick_list])
        bricks = [{**b, 'z': b['z'] - min_z} for b in brick_list]
        
        # 2. 性能优化：大模型降级
        if len(bricks) > 800:
            return self._calculate_geometric_score(bricks)

        # 3. 构建连接图
        grid = {}
        for idx, b in enumerate(bricks):
            for dx in range(b.get('sx', 1)):
                for dy in range(b.get('sy', 1)):
                    grid[(b['x']+dx, b['y']+dy, b['z'])] = idx
        
        fixed_indices = {idx for idx, b in enumerate(bricks) if b['z'] == 0}
        
        # 如果完全没接地
        if not fixed_indices:
            total_mass = sum([b.get('sx',1)*b.get('sy',1) for b in bricks])
            return -1.0 * total_mass * self.G

        edges = []
        processed_pairs = set()
        
        for idx, b in enumerate(bricks):
            x, y, z = b['x'], b['y'], b['z']
            for dx in range(b.get('sx', 1)):
                for dy in range(b.get('sy', 1)):
                    curr_x, curr_y = x + dx, y + dy
                    # 检查上方连接
                    if (curr_x, curr_y, z + 1) in grid:
                        neighbor_idx = grid[(curr_x, curr_y, z + 1)]
                        if idx != neighbor_idx:
                            pair = tuple(sorted((idx, neighbor_idx)))
                            if pair not in processed_pairs:
                                edges.append((idx, neighbor_idx))
                                processed_pairs.add(pair)

        # =======================================================
        # [CRITICAL FIX] 修复 edges 为空导致的崩溃
        # =======================================================
        if not edges:
            # 如果没有边，检查是否所有积木都在地上
            # 如果有积木不在地上且没有连接，那就是悬空 -> 不稳定
            unstable_mass = 0
            for idx, b in enumerate(bricks):
                if idx not in fixed_indices:
                    unstable_mass += b.get('sx',1) * b.get('sy',1)
            
            if unstable_mass > 0:
                return -1.0 * unstable_mass * self.G  # 返回重力惩罚
            else:
                return 0.0  # 所有积木都在地上，且互不相连 -> 稳定

        # 4. 优化问题建模
        num_bricks = len(bricks)
        num_edges = len(edges)
        
        f_glue = cp.Variable(num_edges) # 这里现在安全了
        slacks = cp.Variable(num_bricks)
        
        constrs = []
        constrs.append(f_glue <= self.T)
        constrs.append(f_glue >= -self.T)
        
        brick_forces = [[] for _ in range(num_bricks)]
        for e_i, (u, v) in enumerate(edges):
            brick_forces[u].append((e_i, -1.0)) 
            brick_forces[v].append((e_i, 1.0))

        for i in range(num_bricks):
            if i in fixed_indices:
                constrs.append(slacks[i] == 0)
            else:
                b = bricks[i]
                mass = b.get('sx', 1) * b.get('sy', 1)
                gravity_force = -1.0 * mass * self.G
                
                force_expr = 0
                for edge_idx, sign in brick_forces[i]:
                    force_expr += sign * f_glue[edge_idx]
                
                constrs.append(force_expr + slacks[i] + gravity_force == 0)

        prob = cp.Problem(cp.Minimize(cp.norm(slacks, 1)), constrs)
        
        try:
            prob.solve(solver=cp.ECOS)
        except:
            try: prob.solve(solver=cp.SCS)
            except: return -999.0

        if prob.status in ['optimal', 'optimal_inaccurate']:
            raw_score = prob.value
            if raw_score < 1e-3: return 0.0
            return -raw_score
        else:
            return -999.0

    def _calculate_geometric_score(self, bricks):
        grid = {}
        for idx, b in enumerate(bricks):
            for dx in range(b.get('sx',1)):
                for dy in range(b.get('sy',1)):
                    grid[(b['x']+dx, b['y']+dy, b['z'])] = idx
        
        roots = {idx for idx, b in enumerate(bricks) if b['z'] == 0}
        visited = set(roots)
        queue = list(roots)
        
        head = 0
        while head < len(queue):
            curr_idx = queue[head]; head += 1
            b = bricks[curr_idx]
            x,y,z = b['x'], b['y'], b['z']
            neighbors = [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]
            for np_pos in neighbors:
                if np_pos in grid:
                    n_idx = grid[np_pos]
                    if n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)
        
        unstable_count = len(bricks) - len(visited)
        return -10.0 * unstable_count