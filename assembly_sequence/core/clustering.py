# clustering.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Set, Iterable, Any
import networkx as nx
from collections import defaultdict

########## 1) 桥与关节点检测 ##########

def find_bridges_and_articulations(G: nx.Graph) -> Tuple[Set[Tuple[int,int]], Set[int]]:
    """
    Returns:
      bridges: set of (u,v) with u < v
      arts: set of articulation point node ids
    """
    bridges = set()
    if G.number_of_edges() > 0:
        for u, v in nx.bridges(G):
            bridges.add((min(u, v), max(u, v)))
    arts = set(nx.articulation_points(G))
    return bridges, arts


########## 2) 层感知社区发现（轻量 Louvain） ##########
# 思想：对跨层边施加惩罚，对强接触/竖向支撑加分。
# 采用 Louvain 两阶段：
#   (A) Local Moving: 单节点尝试移入邻居社区，最大化 “层感知收益”
#   (B) Coarsening: 社区收缩为超节点，重复 (A)
#
# 我们不实现完整模块度公式，而是使用“加权凝聚 gain”：
# Δgain = Δ(社区内边权总和) - alpha * (社区内跨层边惩罚)
# 其中边权来自 G[u][v]["weight"]，跨层边指 layer_gap > 0。
#
# 可将 alpha 设大提升“层内聚合”、抑制层间混合；beta 提升竖向边加成已在权重里。

def _community_of(node:int, part:Dict[int,int])->int:
    return part[node]

def _compute_gain_for_move(
    G: nx.Graph, node: int, target_comm: int, part: Dict[int,int], *,
    alpha_cross: float, layer_attr: str = "layer"
) -> float:
    """
    Approximate local gain if `node` moves into `target_comm`.
    Sum weights to nodes already in target_comm minus alpha * cross-layer penalty on those edges.
    """
    gain = 0.0
    node_layer = G.nodes[node].get(layer_attr, 0)
    for nbr, edata in G[node].items():
        if part.get(nbr, -1) != target_comm:
            continue
        w = edata.get("weight", 1.0)
        # penalize cross-layer connections inside community
        nbr_layer = G.nodes[nbr].get(layer_attr, 0)
        layer_gap = abs(node_layer - nbr_layer)
        penalty = alpha_cross * (1.0 if layer_gap > 0 else 0.0)
        gain += (w - penalty)
    return gain

def _neighbors_communities(G: nx.Graph, node: int, part: Dict[int,int]) -> Set[int]:
    return {part[nbr] for nbr in G.neighbors(node)}

def _local_moving(
    G: nx.Graph, part: Dict[int,int], *,
    alpha_cross: float, max_passes: int = 5
) -> Tuple[Dict[int,int], bool]:
    """
    One-level local moving. Returns updated partition and a flag indicating any move happened.
    """
    moved_any = False
    nodes = list(G.nodes())
    # simple deterministic order; optionally shuffle for randomness
    for _ in range(max_passes):
        moved = False
        for node in nodes:
            current_comm = part[node]
            best_comm = current_comm
            best_gain = 0.0

            neigh_comms = _neighbors_communities(G, node, part)
            # consider staying put plus all neighbor communities
            for comm in neigh_comms | {current_comm}:
                gain = _compute_gain_for_move(G, node, comm, part, alpha_cross=alpha_cross)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = comm

            if best_comm != current_comm:
                part[node] = best_comm
                moved = True
        moved_any |= moved
        if not moved:
            break
    return part, moved_any

def _coarsen_graph(G: nx.Graph, part: Dict[int,int]) -> Tuple[nx.Graph, Dict[int,int]]:
    """
    Coarsen graph by contracting nodes in the same community into super-nodes.
    Returns (G_coarse, node_to_super) with fresh community ids {0..K-1}.
    """
    # remap communities to 0..K-1
    comm_ids = sorted(set(part.values()))
    comm_id_map = {c:i for i,c in enumerate(comm_ids)}

    H = nx.Graph()
    # super-node attributes as aggregated stats
    for c in comm_ids:
        H.add_node(comm_id_map[c], size=0, weight_sum=0.0)

    # aggregate node counts
    for n, c in part.items():
        sc = comm_id_map[c]
        H.nodes[sc]["size"] += 1

    # aggregate edges between communities
    for u, v, d in G.edges(data=True):
        cu = comm_id_map[part[u]]
        cv = comm_id_map[part[v]]
        w = d.get("weight", 1.0)
        if cu == cv:
            # intra-comm self-loop weight (store as node attr)
            H.nodes[cu]["weight_sum"] = H.nodes[cu].get("weight_sum", 0.0) + w
        else:
            if H.has_edge(cu, cv):
                H[cu][cv]["weight"] += w
            else:
                H.add_edge(cu, cv, weight=w)

    # mapping from original node -> super node id
    node_to_super = {n: comm_id_map[part[n]] for n in G.nodes()}
    return H, node_to_super

def layer_aware_communities(
    G: nx.Graph,
    *,
    alpha_cross: float = 0.8,   # penalty for cross-layer edges inside a community
    max_outer_levels: int = 3,
    max_local_passes: int = 6
) -> Dict[int,int]:
    """
    Lightweight, layer-aware Louvain-style clustering.
    Returns partition dict: node -> community_id (0..K-1).
    """
    # initial: each node = its own community
    part = {n:i for i,n in enumerate(G.nodes())}
    current_G = G.copy()
    node_maps: List[Dict[int,int]] = []  # history of coarsening maps

    for _ in range(max_outer_levels):
        # local moving on current level
        part, moved = _local_moving(
            current_G, part, alpha_cross=alpha_cross, max_passes=max_local_passes
        )
        # build coarse graph
        coarse_G, node_to_super = _coarsen_graph(current_G, part)

        # if no meaningful coarsening (i.e., same number of nodes), stop
        if coarse_G.number_of_nodes() == current_G.number_of_nodes():
            break

        # prepare next level
        node_maps.append(node_to_super)
        current_G = coarse_G
        # reset communities on coarse level
        part = {n:i for i,n in enumerate(current_G.nodes())}

    # unfold partition back to original nodes
    # current level communities: node -> comm
    final_comm_map = part.copy()
    # replay node_maps backwards (if any)
    for node_to_super in reversed(node_maps):
        # expand: original_node -> previous_super -> current_super_comm
        expanded = {}
        for orig_node, prev_super in node_to_super.items():
            expanded[orig_node] = final_comm_map[prev_super]
        final_comm_map = expanded
    # normalize community ids to 0..K-1
    unique = {c:i for i,c in enumerate(sorted(set(final_comm_map.values())))}
    final_part = {n: unique[c] for n, c in final_comm_map.items()}
    return final_part


########## 3) 后处理：小团合并 + 桥/接口标注 + 子块DAG ##########

def compute_cluster_cohesion(G: nx.Graph, nodes: Iterable[int]) -> float:
    """Average internal edge weight per node (simple cohesion proxy)."""
    sub = G.subgraph(nodes)
    if sub.number_of_nodes() <= 1:
        return 0.0
    w = sum(d.get("weight", 1.0) for _, _, d in sub.edges(data=True))
    return w / float(sub.number_of_nodes())

def merge_small_communities(
    G: nx.Graph,
    part: Dict[int,int],
    *,
    min_size: int = 4,
    min_cohesion: float = 0.5
) -> Dict[int,int]:
    """
    Merge tiny/weak communities into the best neighboring community
    (highest inter-edge weight).
    """
    comm_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for n,c in part.items():
        comm_to_nodes[c].append(n)

    # compute candidates
    for c, nodes in list(comm_to_nodes.items()):
        if len(nodes) >= min_size and compute_cluster_cohesion(G, nodes) >= min_cohesion:
            continue  # keep
        # find best neighbor community by total cut weight
        cut_weights = defaultdict(float)
        for u in nodes:
            for v, d in G[u].items():
                cv = part[v]
                if cv == c: continue
                cut_weights[cv] += d.get("weight", 1.0)
        if not cut_weights:
            continue
        best_c = max(cut_weights, key=cut_weights.get)
        for u in nodes:
            part[u] = best_c

    # reindex ids
    uniq = {c:i for i,c in enumerate(sorted(set(part.values())))}
    return {n: uniq[c] for n,c in part.items()}

def label_connectors(
    G: nx.Graph,
    part: Dict[int,int],
    bridges: Set[Tuple[int,int]],
    arts: Set[int],
) -> Dict[str, Set[Any]]:
    """
    Mark cluster-bridging edges/nodes:
      - bridge_edges_between_clusters
      - articulation_nodes
      - interface_edges (edges that cross clusters)
    """
    interface_edges = set()
    for u, v in G.edges():
        if part[u] != part[v]:
            interface_edges.add((min(u,v), max(u,v)))

    bridge_edges_between_clusters = {
        (u, v) for (u, v) in bridges if part[u] != part[v]
    }
    return dict(
        interface_edges=interface_edges,
        bridge_edges_between_clusters=bridge_edges_between_clusters,
        articulation_nodes=set(arts),
    )

def build_cluster_DAG(
    G: nx.Graph,
    part: Dict[int,int],
    *,
    layer_attr: str = "layer"
) -> nx.DiGraph:
    """
    Collapse to cluster DAG.
    Edge direction heuristics:
      - lower layer -> higher layer (bottom-up)
      - ties broken by interface edge weight (optional)
    """
    H = nx.DiGraph()
    clusters = sorted(set(part.values()))
    for c in clusters:
        H.add_node(c, size=0, avg_layer=0.0)

    # size / avg layer
    for n, c in part.items():
        H.nodes[c]["size"] += 1
        H.nodes[c]["avg_layer"] += G.nodes[n].get(layer_attr, 0)
    for c in clusters:
        size = max(1, H.nodes[c]["size"])
        H.nodes[c]["avg_layer"] /= float(size)

    # edges between clusters
    cut = defaultdict(float)
    for u, v, d in G.edges(data=True):
        cu, cv = part[u], part[v]
        if cu == cv: continue
        w = d.get("weight", 1.0)
        cut[(cu, cv)] += w
        cut[(cv, cu)] += w

    # direct bottom-up orientation
    for (cu, cv), wsum in cut.items():
        lu = H.nodes[cu]["avg_layer"]
        lv = H.nodes[cv]["avg_layer"]
        if lu < lv:
            H.add_edge(cu, cv, weight=wsum)
        elif lv < lu:
            H.add_edge(cv, cu, weight=wsum)
        else:
            # same average layer: keep bi-directed or pick heavier
            if cu < cv:
                H.add_edge(cu, cv, weight=wsum)
            else:
                H.add_edge(cv, cu, weight=wsum)

    # remove potential 2-cycles by keeping heavier direction
    to_remove = []
    for u, v in list(H.edges()):
        if H.has_edge(v, u):
            if H[u][v]["weight"] >= H[v][u]["weight"]:
                to_remove.append((v, u))
            else:
                to_remove.append((u, v))
    H.remove_edges_from(to_remove)

    return H


########## 4) 高层封装 ##########

def run_layer_aware_clustering(
    G: nx.Graph,
    *,
    alpha_cross: float = 0.8,
    max_outer_levels: int = 3,
    max_local_passes: int = 6,
    min_size: int = 4,
    min_cohesion: float = 0.5
) -> Dict[str, Any]:
    """
    Master routine:
      1) find bridges & articulations
      2) layer-aware louvain
      3) merge small/weak communities
      4) label connectors
      5) build cluster DAG
    Returns:
      {
        "partition": {node: cluster_id},
        "bridges": set((u,v)),
        "articulations": set(node),
        "interfaces": set((u,v)),
        "cluster_dag": nx.DiGraph
      }
    """
    bridges, arts = find_bridges_and_articulations(G)
    part = layer_aware_communities(
        G, alpha_cross=alpha_cross,
        max_outer_levels=max_outer_levels,
        max_local_passes=max_local_passes
    )
    part = merge_small_communities(G, part, min_size=min_size, min_cohesion=min_cohesion)
    labels = label_connectors(G, part, bridges, arts)
    dag = build_cluster_DAG(G, part)
    return dict(
        partition=part,
        bridges=bridges,
        articulations=arts,
        interfaces=labels["interface_edges"],
        bridge_edges_between_clusters=labels["bridge_edges_between_clusters"],
        cluster_dag=dag,
    )