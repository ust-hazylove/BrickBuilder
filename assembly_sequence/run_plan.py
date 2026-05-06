# -*- coding: utf-8 -*-
"""
Physics-aware Subassembly Planning with Bridge Detection
--------------------------------------------------------
Implements:
  "Physics-Aware Combinatorial Assembly Sequence Planning
   using Data-Free Action Masking"

Adds bridge-aware connection refinement.
"""

import argparse
from pathlib import Path
import numpy as np
import networkx as nx
from core.ldr_stability import build_and_solve

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

# ---------------- 1. Parse LDraw ----------------
def parse_ldr(ldr_path):
    bricks = []
    with open(ldr_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.startswith("1 "): continue
            p = ln.split()
            if len(p) < 15: continue
            color = int(float(p[1]))
            x, y, z = map(float, p[2:5])
            rot = list(map(float, p[5:14]))
            file = p[14].lower()
            bricks.append({
                "id": len(bricks),
                "file": file,
                "color": color,
                "pos": (x, y, z),
                "rot": rot
            })
    return bricks

# ---------------- 2. Build geometric graph ----------------
def overlap_1d(a1,a2,b1,b2): return not (a2<b1 or b2<a1)
def build_support_graph(bricks, height_thresh=24.0, overlap_thresh=10.0):
    G = nx.DiGraph()
    for b in bricks:
        G.add_node(b["id"], **b)
    for i, bi in enumerate(bricks):
        xi, yi, zi = bi["pos"]
        for j, bj in enumerate(bricks):
            if i==j: continue
            xj,yj,zj=bj["pos"]
            if 0 < (yj-yi) <= height_thresh and abs(xi-xj)<overlap_thresh and abs(zi-zj)<overlap_thresh:
                G.add_edge(i,j,type="support",geom=True)
    return G

# ---------------- 3. Evaluate stability ----------------
def evaluate_stability(G, bricks, verbose=True):
    stable_edges=[]
    for (u,v) in list(G.edges()):
        try:
            subset=[bricks[u],bricks[v]]
            res=build_and_solve(subset,{},[],[],12.0,4.0,0.35,0.25,0.45,0.50,0.0,0.0,False,False)
            risk=res[0]
            if isinstance(risk,(float,int)) and risk<0.1:
                stable_edges.append((u,v))
        except Exception as e:
            if verbose: print(f"[WARN] eval fail ({u}->{v}): {e}")
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
    """
    A brick is a bridge if it connects >1 clusters via stable edges.
    Return: list of {id, connects:[c1,c2,...]}
    """
    bridges=[]
    for n in G_stable.nodes():
        conn_clusters=set()
        for nbr in G_stable.neighbors(n):
            conn_clusters.add(cid[nbr])
        for pred in G_stable.predecessors(n):
            conn_clusters.add(cid[pred])
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
        # connect A→bridge→B pattern (for all combinations)
        for ci in con:
            newDAG.add_edge(ci,name)
        for cj in con:
            newDAG.add_edge(name,cj)
    return newDAG, bridge_nodes

# ---------------- 8. Plan intra-sequences ----------------
def plan_intra_sequences(bricks, nodes):
    sub=[b for b in bricks if b["id"] in nodes]
    sub=sorted(sub,key=lambda b:b["pos"][1])
    return [b["id"] for b in sub]

def export_mpd(bricks, cluster_id, clusters, bridge_nodes, dag_order, intra_orders, out_path):
    """
    修正版 MPD 导出：
      - MAIN 放在第一个 0 FILE 段
      - 主模型引用子模型名需带扩展名（CL_x.ldr / BRIDGE_x.ldr）
      - 每次引用子模型后插入 0 STEP
      - 每个 0 FILE 段以 0 NOFILE 结束
    """
    def part_line(b):
        a,b1,c,d,e,f,g,h,i = b["rot"]
        x,y,z = b["pos"]
        return f"1 {b['color']} {int(x)} {int(y)} {int(z)} {a} {b1} {c} {d} {e} {f} {g} {h} {i} {b['file']}\n"

    lines = []

    # ---------- 1) MAIN 必须放在第一个 0 FILE 段 ----------
    lines.append("0 FILE MAIN.ldr\n")
    lines.append("0 Name: MAIN.ldr\n")
    lines.append("0 Author: auto\n")
    # 拓扑序列里既有 cluster id（int），也可能有 BRIDGE_x（str）
    for node in dag_order:
        if isinstance(node, str) and node.startswith("BRIDGE_"):
            subname = f"{node}.ldr"
        else:
            subname = f"CL_{int(node)}.ldr"
        # 引用子模型（单位矩阵）
        lines.append("1 16 0 0 0  1 0 0  0 1 0  0 0 1 " + subname + "\n")
        lines.append("0 STEP\n")
    lines.append("0 NOFILE\n")

    # ---------- 2) 子模型：每个子块一个文件 ----------
    for cid, nodes in enumerate(clusters):
        subname = f"CL_{cid}.ldr"
        lines.append(f"0 FILE {subname}\n")
        lines.append(f"0 Name: {subname}\n")
        lines.append("0 Author: auto\n")
        for bid in intra_orders[cid]:
            b = bricks[bid]
            lines.append(part_line(b))
            lines.append("0 STEP\n")      # 一砖一 STEP
        lines.append("0 NOFILE\n")

    # ---------- 3) 桥接件：每个桥节点一个文件 ----------
    # 注：这里假设上层逻辑中有全局变量 bridges，或你把 bridges 列表也传进来
    if 'bridges' in globals():
        br_list = globals()['bridges']
    else:
        br_list = []
    for i, br in enumerate(br_list):
        bn = f"BRIDGE_{i}.ldr"
        bid = br["id"]
        b   = bricks[bid]
        lines.append(f"0 FILE {bn}\n")
        lines.append(f"0 Name: {bn}\n")
        lines.append("0 Author: auto\n")
        lines.append(part_line(b))
        lines.append("0 STEP\n")
        lines.append("0 NOFILE\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[OK] MPD exported -> {out_path}")

# ---------------- 10. Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ldr",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--verbose",action="store_true")
    args=ap.parse_args()

    bricks=parse_ldr(args.ldr)
    print(f"[INFO] bricks={len(bricks)}")
    G=build_support_graph(bricks)
    G_stable=evaluate_stability(G,bricks,verbose=args.verbose)
    cid,clusters=cluster_subassemblies(G_stable)
    DAG=build_dependency_dag(G_stable,cid)
    print(f"[INFO] clusters={len(clusters)} edges={len(DAG.edges())}")

    bridges=detect_bridges(G_stable,cid)
    print(f"[INFO] bridges={len(bridges)}")
    DAG2,bridge_nodes=integrate_bridges_into_dag(DAG,bridges)
    dag_order=list(nx.topological_sort(DAG2))
    intra_orders={cid:plan_intra_sequences(bricks,nodes) for cid,nodes in enumerate(clusters)}
    export_mpd(bricks,cid,clusters,bridge_nodes,dag_order,intra_orders,args.out)
    print("[DONE]")

if __name__=="__main__":
    main()