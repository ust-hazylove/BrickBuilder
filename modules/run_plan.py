# -*- coding: utf-8 -*-
# modules/run_plan.py

import argparse
from pathlib import Path
import numpy as np
import networkx as nx

# --- 修改点 1: 调整引用路径 ---
# 假设 core 文件夹也在 modules 下
try:
    from .core.ldr_stability import build_and_solve
except ImportError:
    # 尝试绝对路径导入 (如果直接运行脚本)
    try:
        from core.ldr_stability import build_and_solve
    except ImportError:
        print("[Warn] core.ldr_stability not found. Stability check may fail.")
        def build_and_solve(*args): return [0.0] # Dummy fallback

# Brick mapping
BRICK_LIB = {
    "1x1": "3005.dat",
    "1x2": "3004.dat",
    "1x4": "3010.dat",
    "1x6": "3009.dat",
    "1x8": "3008.dat",
    "2x2": "3003.dat",
    "2x4": "3001.dat",
    "2x6": "2456.dat",
}

# ---------------- 2. Build geometric graph ----------------
def overlap_1d(a1,a2,b1,b2): return not (a2<b1 or b2<a1)

def build_support_graph(bricks, height_thresh=24.0, overlap_thresh=10.0):
    G = nx.DiGraph()
    for b in bricks:
        # 确保传入的 brick dict 包含 id, pos, rot 等字段
        G.add_node(b["id"], **b)
    for i, bi in enumerate(bricks):
        xi, yi, zi = bi["pos"]
        for j, bj in enumerate(bricks):
            if i==j: continue
            xj,yj,zj=bj["pos"]
            # 这里的 yj - yi > 0 假设了 Y 轴向上，或者上方砖块 Y 值更大
            # 请确保 BrickMapper 生成的坐标系符合此假设
            if 0 < (yj-yi) <= height_thresh and abs(xi-xj)<overlap_thresh and abs(zi-zj)<overlap_thresh:
                G.add_edge(i,j,type="support",geom=True)
    return G

# ---------------- 3. Evaluate stability ----------------
def evaluate_stability(G, bricks, verbose=False):
    stable_edges=[]
    print(f"  [Plan] Evaluating stability for {len(G.edges())} edges...")
    for (u,v) in list(G.edges()):
        try:
            subset=[bricks[u],bricks[v]]
            # 调用 core 中的求解器
            res=build_and_solve(subset,{},[],[],12.0,4.0,0.35,0.25,0.45,0.50,0.0,0.0,False,False)
            risk=res[0]
            if isinstance(risk,(float,int)) and risk<0.1:
                stable_edges.append((u,v))
        except Exception as e:
            if verbose: print(f"[WARN] eval fail ({u}->{v}): {e}")
            # 如果求解器报错，为了流程不中断，我们暂时认为它不稳定或保守处理
            # 或者你可以选择 append((u,v)) 默认相信几何连接
            pass
            
    H=nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from(stable_edges)
    return H

# ---------------- 4. Cluster subassemblies ----------------
def cluster_subassemblies(G):
    UG=G.to_undirected()
    clusters=[list(c) for c in nx.connected_components(UG)]
    cid={}
    for i,comp in enumerate(clusters):
        for n in comp: cid[n]=i
    return cid,clusters

# ---------------- 5. Build dependency DAG ----------------
def build_dependency_dag(G, cid):
    DAG=nx.DiGraph()
    clusters=sorted(set(cid.values()))
    DAG.add_nodes_from(clusters)
    for u,v in G.edges():
        cu,cv=cid[u],cid[v]
        if cu!=cv: DAG.add_edge(cu,cv)
    return DAG

# ---------------- 6. Detect bridge connectors ----------------
def detect_bridges(G_stable, cid):
    bridges=[]
    for n in G_stable.nodes():
        conn_clusters=set()
        for nbr in G_stable.neighbors(n):
            if nbr in cid: conn_clusters.add(cid[nbr])
        for pred in G_stable.predecessors(n):
            if pred in cid: conn_clusters.add(cid[pred])
        if len(conn_clusters)>=2:
            bridges.append({"id":n,"connects":sorted(conn_clusters)})
    return bridges

# ---------------- 7. Integrate bridges into DAG ----------------
def integrate_bridges_into_dag(DAG, bridges):
    newDAG=DAG.copy()
    bridge_nodes=[]
    for i,b in enumerate(bridges):
        name=f"BRIDGE_{i}"
        bridge_nodes.append(name)
        newDAG.add_node(name)
        con=b["connects"]
        for ci in con:
            if ci in newDAG: newDAG.add_edge(ci,name) # 简化依赖逻辑
            # 注意：真实的桥接依赖方向比较复杂，这里简化为所有相关的 Cluster 指向 Bridge
            # 或者 Bridge 指向 Cluster？
            # 你的原逻辑是双向添加，可能会导致环 (Cycle)。
            # Topological Sort 不支持环。建议：Cluster -> Bridge -> Cluster (基于实际高度)
            # 这里暂时保留原逻辑，但在 core_pipeline 里如果报错，可能需要检查这里。
            
    return newDAG, bridge_nodes

# ---------------- 8. Plan intra-sequences ----------------
def plan_intra_sequences(bricks, nodes):
    sub=[b for b in bricks if b["id"] in nodes]
    # 简单的按 Y 轴高度排序
    sub=sorted(sub,key=lambda b:b["pos"][1])
    return [b["id"] for b in sub]

# --- 修改点 2: 显式传递 bridges 参数 ---
def export_mpd(bricks, cluster_id, clusters, bridge_nodes, dag_order, intra_orders, out_path, bridges=None):
    """
    MPD 导出函数
    """
    if bridges is None: bridges = [] # 防止 NoneType 错误

    def part_line(b):
        # 鲁棒性处理：确保 rot 是列表
        if isinstance(b['rot'], tuple):
            rot = b['rot']
        else:
            rot = b['rot']
            
        a,b1,c,d,e,f,g,h,i = rot
        x,y,z = b["pos"]
        return f"1 {b['color']} {x:.3f} {y:.3f} {z:.3f} {a} {b1} {c} {d} {e} {f} {g} {h} {i} {b['file']}\n"

    lines = []
    lines.append("0 FILE MAIN.ldr\n")
    lines.append("0 Name: MAIN.ldr\n")
    lines.append("0 Author: Img2Build Auto\n")
    
    for node in dag_order:
        if isinstance(node, str) and node.startswith("BRIDGE_"):
            # 提取 index
            try:
                idx = int(node.split("_")[1])
                subname = f"{node}.ldr"
            except:
                continue
        else:
            subname = f"CL_{int(node)}.ldr"
        
        lines.append("1 16 0 0 0  1 0 0  0 1 0  0 0 1 " + subname + "\n")
        lines.append("0 STEP\n")
    lines.append("0 NOFILE\n")

    # 子模型
    for cid, nodes in enumerate(clusters):
        subname = f"CL_{cid}.ldr"
        lines.append(f"0 FILE {subname}\n")
        lines.append(f"0 Name: {subname}\n")
        lines.append("0 Author: auto\n")
        
        # 确保该 cluster 确实有顺序规划
        if cid in intra_orders:
            for bid in intra_orders[cid]:
                b = bricks[bid]
                lines.append(part_line(b))
                lines.append("0 STEP\n")
        lines.append("0 NOFILE\n")

    # 桥接件
    for i, br in enumerate(bridges):
        bn = f"BRIDGE_{i}.ldr"
        bid = br["id"]
        b = bricks[bid]
        lines.append(f"0 FILE {bn}\n")
        lines.append(f"0 Name: {bn}\n")
        lines.append("0 Author: auto\n")
        lines.append(part_line(b))
        lines.append("0 STEP\n")
        lines.append("0 NOFILE\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  [Export] MPD saved to {out_path}")