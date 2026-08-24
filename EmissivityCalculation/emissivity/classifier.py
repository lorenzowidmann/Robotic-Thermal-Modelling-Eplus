"""Zero-shot material classification with CLIP.

Uses openai/clip-vit-base-patch32 through Hugging Face transformers.
Class prompts come from the EmissivityTable, so every class the model can
predict has a corresponding tabulated emissivity value.
"""

import numpy as np
from PIL import Image

from .table import EmissivityTable

MODEL_NAME = "openai/clip-vit-base-patch32"


def _is_cached(model_name: str) -> bool:
    """Best-effort check for a local snapshot, just to print an honest message
    -- from_pretrained() below still handles the actual cache lookup/download."""
    from huggingface_hub import scan_cache_dir

    try:
        repos = scan_cache_dir().repos
    except Exception:
        return False
    return any(r.repo_id == model_name and r.size_on_disk > 0 for r in repos)


class MaterialClassifier:
    def __init__(self, table: EmissivityTable, model_name: str = MODEL_NAME, device: str | None = None):
        # Heavy imports kept local so the rest of the package loads fast.
        import torch
        from transformers import CLIPModel, CLIPProcessor

        print(
            "Loading cached CLIP model..." if _is_cached(model_name)
            else "Downloading CLIP model (~600 MB, first run only)..."
        )

        self._torch = torch
        self.table = table

        # Auto-detect CUDA (e.g. a lab GPU); explicit `device=` overrides.
        # fp16 on CUDA roughly halves both memory and time, which matters on
        # a small vGPU frame buffer (e.g. a GRID V100S-2Q's 2GB partition --
        # ViT-L/14 is ~1.7GB in fp32, tight; comfortably fits in fp16).
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.device.type == "cpu":
            # Leave one core free so a long grid scan can't starve the rest
            # of the system (display loop, camera server, desktop).
            import os
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

        self.model = CLIPModel.from_pretrained(model_name).to(self.device, dtype=self.dtype)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print(f"MaterialClassifier: {self.device} ({self.dtype})")

        # The prompt list never changes at runtime, so encode it once here.
        # classify() then only runs the image encoder -- grid mode was
        # re-encoding every material prompt through the text encoder for
        # every cell of every scan, which pegged the CPU with ViT-L/14.
        text_inputs = self.processor(text=table.prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_feats = self.model.get_text_features(**text_inputs)
        # transformers >= 5 returns BaseModelOutputWithPooling with the
        # projected embeddings in pooler_output; < 5 returns the tensor.
        if not torch.is_tensor(text_feats):
            text_feats = text_feats.pooler_output
        self._text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    def _image_inputs(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.dtype == self._torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        return inputs

    def classify(
        self, image: np.ndarray | Image.Image, top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Classify the material in the image.

        Args:
            image: RGB image as HxWx3 numpy array or PIL Image.
            top_k: number of (material, confidence) pairs to return.

        Returns:
            List of (material_name, confidence) sorted by confidence.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self._image_inputs(image)
        with self._torch.no_grad():
            image_feats = self.model.get_image_features(**inputs)
        if not self._torch.is_tensor(image_feats):
            image_feats = image_feats.pooler_output
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        # Same computation CLIPModel.forward does for logits_per_image,
        # just against the cached text features.
        logits = self.model.logit_scale.exp() * image_feats @ self._text_feats.T
        probs = logits.softmax(dim=1)[0]

        materials = self.table.materials
        top = probs.argsort(descending=True)[:top_k]
        return [(materials[i], float(probs[i])) for i in top]

    def classify_batch(
        self, images: list[np.ndarray | Image.Image], top_k: int = 1
    ) -> list[list[tuple[str, float]]]:
        """Same as classify(), but encodes every image in one batched forward
        pass through the CLIP image encoder instead of one call per image.

        A single frame's worth of superpixels (dozens to ~100) run through
        classify() sequentially is the main cost of classify_session.py on
        this CPU-only setup -- batching the encoder call is the practical
        difference between a session finishing in minutes vs. hours.

        Args:
            images: list of RGB HxWx3 numpy arrays or PIL Images.
            top_k: number of (material, confidence) pairs to return per image.

        Returns:
            One list of (material_name, confidence) per input image, same
            order as `images`.
        """
        pil_images = [Image.fromarray(im) if isinstance(im, np.ndarray) else im for im in images]

        inputs = self._image_inputs(pil_images)
        with self._torch.no_grad():
            image_feats = self.model.get_image_features(**inputs)
        if not self._torch.is_tensor(image_feats):
            image_feats = image_feats.pooler_output
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        logits = self.model.logit_scale.exp() * image_feats @ self._text_feats.T
        probs = logits.softmax(dim=1)

        materials = self.table.materials
        results = []
        for row in probs:
            top = row.argsort(descending=True)[:top_k]
            results.append([(materials[i], float(row[i])) for i in top])
        return results
