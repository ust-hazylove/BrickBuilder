import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial import KDTree
from torch import Tensor, nn


class LocalEdgeConvLite(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_attr_dim: int = 0):
        super().__init__()
        mlp_in = in_channels * 2 + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index
        pieces = [x[dst], x[src] - x[dst]]
        if edge_attr is not None:
            pieces.append(edge_attr)
        messages = self.mlp(torch.cat(pieces, dim=-1))

        out = x.new_full((x.shape[0], messages.shape[-1]), -torch.inf)
        scatter_index = dst.unsqueeze(-1).expand(-1, messages.shape[-1])
        out = out.scatter_reduce(0, scatter_index, messages, reduce="amax", include_self=True)

        empty_rows = torch.isinf(out).all(dim=-1)
        if empty_rows.any():
            out = torch.where(empty_rows.unsqueeze(-1), torch.zeros_like(out), out)
        return out


def kd_tree_pooling(pos: Tensor, x: Tensor, leaf_size: int = 8) -> Tuple[Tensor, Tensor, Tensor]:
    if pos.shape[0] == 0:
        return (
            pos.new_empty((0, 3)),
            x.new_empty((0, x.shape[-1])),
            torch.empty((0,), dtype=torch.long, device=pos.device),
        )

    pos_np = pos.detach().cpu().numpy()
    tree = KDTree(pos_np, leafsize=leaf_size)
    num_nodes = pos_np.shape[0]
    cluster_ids = [-1] * num_nodes
    clusters: List[List[int]] = []

    k = min(max(int(leaf_size), 1), num_nodes)
    for node_idx in range(num_nodes):
        if cluster_ids[node_idx] >= 0:
            continue
        _, neighbor_idx = tree.query(pos_np[node_idx], k=k)
        neighbor_idx = [int(idx) for idx in torch.as_tensor(neighbor_idx).view(-1).tolist()]
        members = [idx for idx in neighbor_idx if cluster_ids[idx] < 0]
        if not members:
            members = [node_idx]
        cluster_index = len(clusters)
        clusters.append(members)
        for member in members:
            cluster_ids[member] = cluster_index

    cluster = torch.as_tensor(cluster_ids, dtype=torch.long, device=pos.device)
    pooled_pos = pos.new_zeros((len(clusters), pos.shape[-1]))
    pooled_x = x.new_zeros((len(clusters), x.shape[-1]))
    counts = pos.new_zeros((len(clusters), 1))
    pooled_pos.index_add_(0, cluster, pos)
    pooled_x.index_add_(0, cluster, x)
    counts.index_add_(0, cluster, pos.new_ones((pos.shape[0], 1)))
    pooled_pos = pooled_pos / counts.clamp_min(1.0)
    pooled_x = pooled_x / counts.clamp_min(1.0)
    return pooled_pos, pooled_x, cluster


class TAGUNetRiskPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        depth: int = 3,
        pool_leaf_size: int = 8,
        edge_attr_dim: int = 7,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.pool_leaf_size = pool_leaf_size
        self.edge_attr_dim = edge_attr_dim

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.encoder = nn.ModuleList(
            [LocalEdgeConvLite(hidden_channels, hidden_channels, edge_attr_dim=edge_attr_dim) for _ in range(depth)]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                for _ in range(depth - 1)
            ]
        )
        self.refine = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.stability_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),
        )
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = self.input_proj(x)
        pool_states = []
        current_pos = pos
        current_edge_index = edge_index
        current_edge_attr = edge_attr

        for level, conv in enumerate(self.encoder):
            x = conv(x, current_edge_index, current_edge_attr)
            if level == self.depth - 1 or x.shape[0] <= 1:
                continue

            pooled_pos, pooled_x, cluster = kd_tree_pooling(current_pos, x, leaf_size=self.pool_leaf_size)
            pool_states.append((cluster, x))
            current_pos = pooled_pos
            x = pooled_x
            current_edge_index, current_edge_attr = self._pool_graph(
                current_edge_index, current_edge_attr, cluster, current_pos
            )

        for decoder, (cluster, skip) in zip(reversed(self.decoder), reversed(pool_states)):
            x = x[cluster]
            x = decoder(torch.cat([x, skip], dim=-1))

        x = self.refine(x)
        return self.stability_head(x), self.stress_head(x)

    def _pool_graph(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        cluster: Tensor,
        pooled_pos: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if edge_index.numel() == 0:
            empty_edges = edge_index.new_empty((2, 0))
            return empty_edges, self._empty_edge_attr(edge_attr, pooled_pos.device)

        pooled_src = cluster[edge_index[0]]
        pooled_dst = cluster[edge_index[1]]
        keep = pooled_src != pooled_dst
        if not torch.any(keep):
            empty_edges = edge_index.new_empty((2, 0))
            return empty_edges, self._empty_edge_attr(edge_attr, pooled_pos.device)

        pooled_src = pooled_src[keep]
        pooled_dst = pooled_dst[keep]
        kept_attr = edge_attr[keep] if edge_attr is not None else None

        edge_map = {}
        for idx in range(pooled_src.shape[0]):
            key = (int(pooled_src[idx]), int(pooled_dst[idx]))
            edge_map.setdefault(key, [])
            if kept_attr is not None:
                edge_map[key].append(kept_attr[idx])

        pooled_edges = torch.as_tensor(sorted(edge_map.keys()), dtype=torch.long, device=pooled_pos.device)
        pooled_edge_index = pooled_edges.t().contiguous()
        if kept_attr is None:
            return self._ensure_connectivity(pooled_pos, pooled_edge_index, None)

        pooled_edge_attr = torch.stack(
            [torch.stack(values, dim=0).mean(dim=0) for values in edge_map.values()],
            dim=0,
        )
        return self._ensure_connectivity(pooled_pos, pooled_edge_index, pooled_edge_attr)

    def _empty_edge_attr(self, edge_attr: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
        if edge_attr is None:
            return None
        return edge_attr.new_empty((0, edge_attr.shape[-1]), device=device)

    def _ensure_connectivity(
        self,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if pos.shape[0] <= 1 or edge_index.numel() > 0:
            return edge_index, edge_attr

        pos_np = pos.detach().cpu().numpy()
        tree = KDTree(pos_np)
        k = min(2, pos.shape[0] - 1)
        if k <= 0:
            return edge_index.new_empty((2, 0)), self._empty_edge_attr(edge_attr, pos.device)

        edges = set()
        for idx in range(pos.shape[0]):
            _, neighbors = tree.query(pos_np[idx], k=k + 1)
            neighbors = neighbors[1:] if hasattr(neighbors, "__len__") else [neighbors]
            for neighbor in neighbors:
                src = int(idx)
                dst = int(neighbor)
                if src == dst:
                    continue
                edges.add((src, dst))
                edges.add((dst, src))

        new_edge_index = torch.as_tensor(sorted(edges), dtype=torch.long, device=pos.device).t().contiguous()
        if edge_attr is None:
            return new_edge_index, None

        new_edge_attr = edge_attr.new_zeros((new_edge_index.shape[1], edge_attr.shape[-1]), device=pos.device)
        return new_edge_index, new_edge_attr


class HighRiskPredictor:
    TYPE_TO_ID = {
        "1x1": 2,
        "1x2": 5,
        "1x3": 5,
        "1x4": 5,
        "1x6": 5,
        "1x8": 5,
        "2x1": 5,
        "2x2": 10,
        "2x3": 12,
        "2x4": 9,
        "2x6": 9,
        "2x8": 9,
        "tile_1x1": 13,
        "round_brick_1x1": 14,
        "tile_1x2": 15,
        "grille_tile_1x2": 16,
        "slope_1x2": 17,
        "tile_1x4": 18,
        "tile_2x2": 19,
        "slope_2x2": 20,
        "corner_slope_2x2": 21,
    }

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"High-risk checkpoint not found: {checkpoint_path}")

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = TAGUNetRiskPredictor(in_channels=128, hidden_channels=128, depth=3, edge_attr_dim=14).to(self.device)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        if "physics_proxy" not in payload:
            raise KeyError("Checkpoint does not contain 'physics_proxy'.")
        self.model.load_state_dict(payload["physics_proxy"], strict=True)
        self.model.eval()

    def predict(self, brick_list: Sequence[dict], risk_threshold: float = 0.45) -> List[dict]:
        if not brick_list:
            return []

        node_features, pos, rot, edge_index, edge_attr = self._build_graph(brick_list)
        with torch.no_grad():
            stability, stress = self.model(node_features, pos, edge_index, edge_attr)

        stability = stability.squeeze(-1).detach().cpu()
        stress = stress.detach().cpu()

        risky = []
        for idx, brick in enumerate(brick_list):
            risk_score = float(1.0 - stability[idx].item())
            if risk_score < risk_threshold:
                continue
            risky.append(
                {
                    "id": brick["id"],
                    "risk_score": risk_score,
                    "grid_pos": brick.get("grid_pos"),
                    "size": brick.get("size"),
                    "stress": stress[idx].tolist(),
                }
            )
        risky.sort(key=lambda item: item["risk_score"], reverse=True)
        return risky

    def _build_graph(self, brick_list: Sequence[dict]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        centered_x = [brick["grid_pos"][0] + brick["size"][0] / 2.0 for brick in brick_list]
        centered_y = [brick["grid_pos"][1] + brick["size"][1] / 2.0 for brick in brick_list]
        mean_x = sum(centered_x) / max(len(centered_x), 1)
        mean_y = sum(centered_y) / max(len(centered_y), 1)

        pos_rows = []
        rot_rows = []
        feature_rows = []
        for brick in brick_list:
            x, y, z = brick["grid_pos"]
            dx, dy = brick["size"]
            pos_row = [
                (x + dx / 2.0) - mean_x,
                (y + dy / 2.0) - mean_y,
                z * 0.96,
            ]
            rot_z = float(brick.get("ori_quarter", 0)) * (math.pi / 2.0)
            type_name = brick.get("type_name", brick.get("struct_type", "1x1"))
            type_idx = self.TYPE_TO_ID.get(type_name, 2)
            semantic_label = int(brick.get("semantic_label", 0))

            pos_rows.append(pos_row)
            rot_rows.append([0.0, 0.0, rot_z])
            raw_feature = pos_row + [float(type_idx), float(dx), float(dy), rot_z, float(semantic_label)]
            expanded = (raw_feature * 16)[:128]
            feature_rows.append(expanded)

        pos = torch.tensor(pos_rows, dtype=torch.float32, device=self.device)
        rot = torch.tensor(rot_rows, dtype=torch.float32, device=self.device)
        node_features = torch.tensor(feature_rows, dtype=torch.float32, device=self.device)
        edge_index, extra_edge_attr = self._infer_contact_pairs(pos, brick_list, mean_x, mean_y)

        if edge_index.numel() == 0:
            edge_attr = torch.empty((0, 14), dtype=torch.float32, device=self.device)
        else:
            undirected = edge_index.t().contiguous()
            rel_pos = pos[undirected[:, 1]] - pos[undirected[:, 0]]
            rot_delta = rot[undirected[:, 1]] - rot[undirected[:, 0]]
            edge_attr = torch.cat([rel_pos, rot_delta], dim=-1)
            edge_attr = torch.cat([edge_attr, rel_pos.norm(dim=-1, keepdim=True)], dim=-1)
            edge_attr = torch.cat([edge_attr, extra_edge_attr], dim=-1)
        return node_features, pos, rot, edge_index, edge_attr

    def _infer_contact_pairs(self, pos: Tensor, brick_list: Sequence[dict], mean_x: float, mean_y: float) -> Tuple[Tensor, Tensor]:
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), torch.empty((0, 7), dtype=torch.float32, device=self.device)

        pos_np = pos.detach().cpu().numpy()
        tree = KDTree(pos_np[:, :2])
        pairs = []
        extras = []
        for source in range(num_nodes):
            xy_candidates = tree.query_ball_point(pos_np[source, :2], r=0.25)
            for target in xy_candidates:
                if source == target:
                    continue
                delta = pos_np[target] - pos_np[source]
                xy_distance = float((delta[0] ** 2 + delta[1] ** 2) ** 0.5)
                z_gap = abs(float(delta[2]))
                if xy_distance > 0.25:
                    continue
                if abs(z_gap - 0.96) > 0.2:
                    continue
                if delta[2] > 0:
                    pairs.append((source, target))
                    pairs.append((target, source))
                    extra = self._build_contact_features(brick_list[source], brick_list[target], mean_x, mean_y)
                    extras.append(extra)
                    extras.append(extra)

        if not pairs:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), torch.empty((0, 7), dtype=torch.float32, device=self.device)
        edge_index = torch.as_tensor(pairs, dtype=torch.long, device=self.device).t().contiguous()
        extra_attr = torch.tensor(extras, dtype=torch.float32, device=self.device)
        return edge_index, extra_attr

    def _build_contact_features(self, lower_brick: dict, upper_brick: dict, mean_x: float, mean_y: float):
        lx, ly, lz = lower_brick["grid_pos"]
        ldx, ldy = lower_brick["size"]
        ux, uy, uz = upper_brick["grid_pos"]
        udx, udy = upper_brick["size"]

        overlap_x0 = max(lx, ux)
        overlap_x1 = min(lx + ldx, ux + udx)
        overlap_y0 = max(ly, uy)
        overlap_y1 = min(ly + ldy, uy + udy)

        contact_x = ((overlap_x0 + overlap_x1) / 2.0) - mean_x
        contact_y = ((overlap_y0 + overlap_y1) / 2.0) - mean_y
        contact_z = max(lz, uz) * 0.96
        contact_count = max(1.0, float(max(0, overlap_x1 - overlap_x0) * max(0, overlap_y1 - overlap_y0)))

        lower_center = np.array([lx + ldx / 2.0 - mean_x, ly + ldy / 2.0 - mean_y, lz * 0.96], dtype=np.float32)
        upper_center = np.array([ux + udx / 2.0 - mean_x, uy + udy / 2.0 - mean_y, uz * 0.96], dtype=np.float32)
        contact_center = np.array([contact_x, contact_y, contact_z], dtype=np.float32)
        lower_dist = float(np.linalg.norm(contact_center - lower_center))
        upper_dist = float(np.linalg.norm(contact_center - upper_center))

        return [
            float(contact_x),
            float(contact_y),
            float(contact_z),
            0.0,
            float(contact_count) / 4.0,
            lower_dist,
            upper_dist,
        ]
