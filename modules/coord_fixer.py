# modules/coord_fixer.py
# 基于你提供的 fix_ldr_orientation.py 改编，适配内存中的 dict 列表
import math

class CoordinateFixer:
    @staticmethod
    def scan_bbox(bricks):
        if not bricks: return 0,0,0,0,0,0
        xs = [b['pos'][0] for b in bricks]
        ys = [b['pos'][1] for b in bricks]
        zs = [b['pos'][2] for b in bricks]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    @staticmethod
    def process(bricks, mode='rotate', axis='x', plane='z', pivot_mode='center'):
        """
        处理内存中的 brick_list
        """
        xmn, xmx, ymn, ymx, zmn, zmx = CoordinateFixer.scan_bbox(bricks)
        
        if pivot_mode == 'center':
            px, py, pz = 0.5*(xmn+xmx), 0.5*(ymn+ymx), 0.5*(zmn+zmx)
        elif pivot_mode == 'zero':
            px, py, pz = 0.0, 0.0, 0.0
        else:
            px, py, pz = 0.0, 0.0, 0.0

        new_bricks = []
        for b in bricks:
            d = b.copy()
            # 解构 pos 和 rot
            d['x'], d['y'], d['z'] = d['pos']
            # rot 是列表 [a,b,c, d,e,f, g,h,i]
            d['a'], d['b'], d['c'] = d['rot'][0:3]
            d['d'], d['e'], d['f'] = d['rot'][3:6]
            d['g'], d['h'], d['i'] = d['rot'][6:9]

            # --- 核心变换逻辑 (来自你的脚本) ---
            if mode == 'rotate':
                # 1. 平移
                d['x'] -= px; d['y'] -= py; d['z'] -= pz
                
                # 2. 位置旋转
                if axis == "x":   d['y'], d['z'] = -d['y'], -d['z']
                elif axis == "y": d['x'], d['z'] = -d['x'], -d['z']
                elif axis == "z": d['x'], d['y'] = -d['x'], -d['y']
                
                # 3. 还原
                d['x'] += px; d['y'] += py; d['z'] += pz

                # 4. 矩阵旋转
                # 临时变量保存原始值
                oa,ob,oc = d['a'],d['b'],d['c']
                od,oe,of = d['d'],d['e'],d['f']
                og,oh,oi = d['g'],d['h'],d['i']

                if axis == "x": # Rx(pi) -> 取反第2,3行
                    d['d'], d['e'], d['f'] = -od, -oe, -of
                    d['g'], d['h'], d['i'] = -og, -oh, -oi
                elif axis == "y": # Ry(pi) -> 取反第1,3行
                    d['a'], d['b'], d['c'] = -oa, -ob, -oc
                    d['g'], d['h'], d['i'] = -og, -oh, -oi
                elif axis == "z": # Rz(pi) -> 取反第1,2行
                    d['a'], d['b'], d['c'] = -oa, -ob, -oc
                    d['d'], d['e'], d['f'] = -od, -oe, -of

            elif mode == 'mirror':
                z0 = {"x": px, "y": py, "z": pz}[plane]
                if plane == "z": # XY平面
                    d['z'] = 2*z0 - d['z']
                    d['g'], d['h'], d['i'] = -d['g'], -d['h'], -d['i']
                elif plane == "x": # YZ平面
                    d['x'] = 2*z0 - d['x']
                    d['a'], d['b'], d['c'] = -d['a'], -d['b'], -d['c']
                elif plane == "y": # XZ平面
                    d['y'] = 2*z0 - d['y']
                    d['d'], d['e'], d['f'] = -d['d'], -d['e'], -d['f']

            # 重组回 dict
            new_b = b.copy()
            new_b['pos'] = (d['x'], d['y'], d['z'])
            new_b['rot'] = [d['a'], d['b'], d['c'], d['d'], d['e'], d['f'], d['g'], d['h'], d['i']]
            new_bricks.append(new_b)
            
        return new_bricks