# modules/hunyuan_infer.py
# -*- coding: utf-8 -*-
import torch
import os
import gc
from pathlib import Path

# 只引入形状生成管道
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

class Hunyuan3DInferencer:
    def __init__(self, device='cuda', pretrained_model_name_or_path="tencent/Hunyuan3D-2.1"):
        """
        [Revert] 低显存纯几何模式初始化
        """
        self.device = device
        self.model_path = pretrained_model_name_or_path
        print(f"--- Hunyuan3D Inferencer (Geometry Only) initialized ---")

    def _clear_gpu_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    def predict(self, image_path: str, seed: int = 0, steps: int = 50):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # ==========================================
        # Step 1: 加载形状模型 & 生成几何
        # ==========================================
        print(f"1. [Low VRAM] Loading DiT Shape Model...")
        try:
            shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_path,
                subfolder="hunyuan3d-dit-v2-1",
                use_safetensors=False, 
                device_map="auto"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Shape Model: {e}")

        print(f"   -> Generating Geometry from {image_path}...")
        # output 是一个 list，取第一个 mesh
        mesh = shape_pipe(
            image=image_path, 
            num_inference_steps=steps,
            guidance_scale=4.0,
            seed=seed
        )[0]
        
        # 卸载模型
        print("   -> Unloading Shape Model...")
        del shape_pipe
        self._clear_gpu_memory()

        return mesh

    def save_mesh(self, mesh, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mesh.export(save_path, include_normals=True)
        return save_path