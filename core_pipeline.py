import os
import traceback
import uuid
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from modules.brick_mapper import BrickMapper
from modules.coord_fixer import CoordinateFixer
from modules.high_risk_predictor import HighRiskPredictor
from modules.hunyuan_infer import Hunyuan3DInferencer
from modules.mesh_utils import MeshUtils
from modules.risk_analysis import detect_risky_bricks, summarize_rule_risk
from modules.rl_repair import RLRepairModule
import modules.run_plan as run_plan


class Img2BuildPipeline:
    def __init__(self, device="cuda"):
        self.output_base = Path("output")
        self.output_base.mkdir(exist_ok=True)
        self.max_resolution = 16
        self.risk_threshold = 0.9

        print(">>> [System] Initializing Img2Build Pipeline (Geometry Mode)...")
        print(">>> [Init] Loading Hunyuan3D-2.1 (DiT Only)...")
        self.generator = Hunyuan3DInferencer(device=device)

        rl_weights_path = "weights/ppo_lego_repair_final.zip"
        print(f">>> [Init] Loading RL Repair Agent from {rl_weights_path}...")
        self.repair_agent = None
        if os.path.exists(rl_weights_path):
            self.repair_agent = RLRepairModule(checkpoint_path=rl_weights_path, device=device)
        else:
            print("[Warning] RL weights not found. PPO repair will be disabled.")

        risk_checkpoint = Path("weights/high_risk_predictor_styled_best.pt")
        self.risk_predictor = None
        if risk_checkpoint.exists():
            try:
                print(f">>> [Init] Loading high-risk predictor from {risk_checkpoint}...")
                self.risk_predictor = HighRiskPredictor(str(risk_checkpoint), device=device)
            except Exception as exc:
                print(f"[Warning] Failed to load high-risk predictor: {exc}")
        else:
            print("[Warning] High-risk checkpoint not found. Risk-guided repair will be skipped.")

        self.mapper = BrickMapper()
        print(">>> [System] Initialization Complete.")

    def run(self, input_image_path: str, task_id: str = None, use_repair: bool = False, resolution: int = 16):
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        job_dir = self.output_base / task_id
        job_dir.mkdir(parents=True, exist_ok=True)
        resolution = self._clamp_resolution(resolution)

        logs = []

        def log(message):
            print(f"[{task_id}] {message}")
            logs.append(message)

        try:
            log("1. Generating 3D Geometry (Hunyuan3D)...")
            mesh_data = self.generator.predict(input_image_path, seed=42)

            raw_mesh_path = job_dir / "raw_mesh.glb"
            self.generator.save_mesh(mesh_data, str(raw_mesh_path))
            log(f"   -> Mesh saved to {raw_mesh_path}")

            do_fill = True
            log(f"2. Voxelizing Mesh (Resolution={resolution}, Solid_Fill={do_fill})...")
            voxel_grid = MeshUtils.glb_to_voxels(raw_mesh_path, resolution=resolution, fill=do_fill)
            log("2.5 Rotating Voxels (Re-orienting Upward)...")
            voxel_grid = MeshUtils.rotate_voxels(voxel_grid, axis="x", k=1)

            log("   -> Generating dummy color grid (White)...")
            color_grid = np.full(voxel_grid.shape + (3,), 255, dtype=np.uint8)
            log(f"   -> Solid Voxel Grid: {voxel_grid.shape}, Occupied: {int(np.sum(voxel_grid))}")

            log("3. Structural Stability Repair...")
            repaired_voxels = voxel_grid
            repaired_color_grid = color_grid
            risk_hints = []

            if use_repair:
                structural_bricks_seed = self.mapper.map_voxels_to_bricks(voxel_grid, color_grid)
                rule_stats = summarize_rule_risk(voxel_grid)
                log(
                    "   -> Rule risk:"
                    f" floating={rule_stats['floating_voxels']},"
                    f" unsupported={rule_stats['unsupported_voxels']},"
                    f" isolated={rule_stats['isolated_voxels']}"
                )
                if self.risk_predictor is not None:
                    try:
                        risk_hints, _, risk_source = detect_risky_bricks(
                            voxel_grid,
                            structural_bricks_seed,
                            risk_predictor=self.risk_predictor,
                            risk_threshold=self.risk_threshold,
                        )
                        log(f"   -> Risk detector ({risk_source}) flagged {len(risk_hints)} risky bricks.")
                    except Exception as exc:
                        log(f"   [Warn] High-risk prediction failed: {exc}")

                if self.repair_agent is not None:
                    repaired_voxels = self.repair_agent.inference(voxel_grid.astype(np.uint8), risk_hints=risk_hints)
                    log("   -> PPO repair finished with risk-guided hints.")
                elif risk_hints:
                    repaired_voxels = self._apply_fallback_repairs(voxel_grid.astype(np.uint8), risk_hints)
                    log("   -> Applied heuristic repairs from high-risk predictor.")
                else:
                    log("   -> No repair backend available. Using raw voxels.")
            else:
                log("   -> Repair disabled for this run.")

            log("4. Mapping Voxels to LEGO Bricks (Expanded Library)...")
            structural_bricks = self.mapper.map_voxels_to_bricks(repaired_voxels, repaired_color_grid)
            styled_bricks = self.mapper.apply_surface_finishing(structural_bricks, repaired_voxels)

            brick_df = self.mapper.generate_bom(styled_bricks)
            brick_count = len(styled_bricks)

            log("4.5 Fixing LDraw Orientation...")
            structural_bricks = CoordinateFixer.process(structural_bricks, mode="rotate", axis="x", pivot_mode="center")
            styled_bricks = CoordinateFixer.process(styled_bricks, mode="rotate", axis="x", pivot_mode="center")

            log("4.6 Generating Assembly Preview Mesh...")
            assembly_mesh_path = job_dir / "assembly_preview.glb"
            MeshUtils.save_voxels_as_mesh(repaired_voxels, str(assembly_mesh_path))
            log(f"   -> Generated {brick_count} bricks with slopes/smooth pieces in the export set.")

            log("5. Generating Assembly Sequence...")
            support_graph = run_plan.build_support_graph(structural_bricks)
            try:
                stable_graph = run_plan.evaluate_stability(support_graph, structural_bricks, verbose=False)
            except Exception:
                log("   [Warn] Solver failed/missing. Fallback to geometric graph.")
                stable_graph = support_graph

            cluster_id, clusters = run_plan.cluster_subassemblies(stable_graph)
            dag = run_plan.build_dependency_dag(stable_graph, cluster_id)
            bridges = run_plan.detect_bridges(stable_graph, cluster_id)

            try:
                dag2, bridge_nodes = run_plan.integrate_bridges_into_dag(dag, bridges)
                dag_order = list(nx.topological_sort(dag2))
            except nx.NetworkXUnfeasible:
                log("   [Warn] Cycle detected! Using fallback sort.")
                cluster_heights = []
                for cluster_index in range(len(clusters)):
                    avg_y = np.mean([structural_bricks[bid]["pos"][1] for bid in clusters[cluster_index]])
                    cluster_heights.append((cluster_index, avg_y))
                cluster_heights.sort(key=lambda item: item[1])
                dag_order = [item[0] for item in cluster_heights]
                bridge_nodes = []

            intra_orders = {
                cluster_index: run_plan.plan_intra_sequences(structural_bricks, nodes)
                for cluster_index, nodes in enumerate(clusters)
            }

            log("6. Exporting Final Model...")
            output_mpd = job_dir / "final_model.mpd"
            run_plan.export_mpd(
                bricks=styled_bricks,
                cluster_id=cluster_id,
                clusters=clusters,
                bridge_nodes=bridge_nodes,
                dag_order=dag_order,
                intra_orders=intra_orders,
                out_path=str(output_mpd),
                bridges=bridges,
            )
            log(f"Success! Job finished. Result: {output_mpd}")

            return str(output_mpd), str(assembly_mesh_path), brick_count, brick_df, "\n".join(logs)
        except Exception as exc:
            err_msg = f"Pipeline Error: {exc}\n{traceback.format_exc()}"
            print(err_msg)
            return None, None, 0, pd.DataFrame(), err_msg

    def _clamp_resolution(self, resolution: int) -> int:
        return max(8, min(int(resolution), self.max_resolution))

    def _apply_fallback_repairs(self, voxel_grid: np.ndarray, risk_hints):
        repaired = voxel_grid.copy()
        for hint in risk_hints[:24]:
            grid_pos = hint.get("grid_pos")
            size = hint.get("size")
            if grid_pos is None or size is None:
                continue
            x, y, z = [int(v) for v in grid_pos]
            dx, dy = [int(v) for v in size]
            cx = min(max(x + dx // 2, 0), repaired.shape[0] - 1)
            cy = min(max(y + dy // 2, 0), repaired.shape[1] - 1)
            repaired[cx, cy, : min(z + 1, repaired.shape[2])] = 1
        return repaired
