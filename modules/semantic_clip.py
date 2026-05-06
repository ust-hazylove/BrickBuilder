from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _render_triptych(voxel_grid: np.ndarray, canvas_size: int = 224) -> Image.Image:
    vox = (voxel_grid > 0).astype(np.uint8)
    if not vox.any():
        return Image.fromarray(np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8))

    proj_xy = vox.max(axis=2) * 255
    proj_xz = vox.max(axis=1) * 255
    proj_yz = vox.max(axis=0) * 255

    def normalize_panel(panel: np.ndarray, target_h: int, target_w: int):
        img = Image.fromarray(panel.astype(np.uint8), mode="L")
        img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
        return np.array(img, dtype=np.uint8)

    row_h = canvas_size // 2
    half_w = canvas_size // 2
    left = normalize_panel(proj_xy.T, row_h, half_w)
    top_right = normalize_panel(proj_xz.T, row_h, half_w)
    bottom_right = normalize_panel(proj_yz.T, row_h, half_w)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    canvas[:row_h, :half_w] = left
    canvas[:row_h, half_w:] = top_right
    canvas[row_h:, half_w:] = bottom_right
    canvas[row_h:, :half_w] = left[::-1]
    rgb = np.stack([canvas, canvas, canvas], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


@lru_cache(maxsize=2)
def _load_clip(model_name: str, device_str: str):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    device = torch.device(device_str if device_str == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device


class SemanticCLIPScorer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.model_name = model_name
        self.model, self.processor, self.device = _load_clip(model_name, device)

    def score_voxel_similarity(self, voxel_grid: np.ndarray, reference_voxels: np.ndarray, text_prompt: Optional[str] = None) -> float:
        image = _render_triptych(voxel_grid)
        reference = _render_triptych(reference_voxels)
        image_score = self._image_similarity(image, reference)
        if text_prompt:
            text_score = self._text_similarity(image, text_prompt)
            return float(0.7 * image_score + 0.3 * text_score)
        return float(image_score)

    def _image_similarity(self, image: Image.Image, reference: Image.Image) -> float:
        with torch.no_grad():
            inputs = self.processor(images=[image, reference], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return float((features[0] * features[1]).sum().item())

    def _text_similarity(self, image: Image.Image, text_prompt: str) -> float:
        with torch.no_grad():
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            image_feature = self.model.get_image_features(pixel_values=image_inputs["pixel_values"])
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            text_inputs = self.processor(text=[text_prompt], return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_feature = self.model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return float((image_feature[0] * text_feature[0]).sum().item())
