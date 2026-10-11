# -*- coding: utf-8 -*-
"""Frozen pretrained feature extractors / classifiers with on-disk output cache.

Every extractor maps a list of items (PIL images OR path strings) to a float32
array [N, D] (embedding or logits).  Path items are cached (results/emb/<name>.npz)
unless CACHE_ENABLED is False (used for latency measurement).
NO training happens anywhere: weights are used exactly as downloaded.
"""
import os, time, gc
import numpy as np
import torch
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.environ.get("VEHICLE_RESULTS", os.path.join(_REPO, "results"))
EMB_DIR = os.path.join(RESULTS, "emb")
CACHE_ENABLED = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def load_image(item):
    if isinstance(item, str):
        im = Image.open(item)
        im.load()
    else:
        im = item
    return im.convert("RGB")


class Extractor:
    name = "base"
    kind = "embedding"          # or "logits"
    fp16 = True

    def __init__(self):
        self._model = None
        self._cache = None
        self._dirty = False
        self.embed_time = 0.0   # seconds: preprocessing + forward (GPU-synchronised)
        self.n_embedded = 0

    # ---- subclass hooks ----
    def _load(self):            # must set self._model, self._pre
        raise NotImplementedError

    def _forward(self, batch):  # tensor on device -> tensor [B,D]
        raise NotImplementedError

    # ---- cache ----
    @property
    def cache_file(self):
        return os.path.join(EMB_DIR, self.name + ".npz")

    def _load_cache(self):
        if self._cache is not None:
            return
        self._cache = {}
        if os.path.exists(self.cache_file):
            try:
                z = np.load(self.cache_file, allow_pickle=False)
                for p, v in zip(z["paths"], z["vecs"]):
                    self._cache[str(p)] = v
            except Exception:
                self._cache = {}

    def save(self):
        if self._cache and self._dirty:
            os.makedirs(EMB_DIR, exist_ok=True)
            keys = sorted(self._cache)
            np.savez(self.cache_file, paths=np.array(keys),
                     vecs=np.stack([self._cache[k] for k in keys]).astype(np.float32))
            self._dirty = False

    def ensure_model(self):
        if self._model is None:
            self._load()
            self._model.eval()
            if DEVICE == "cuda" and self.fp16:
                self._model.half()
            self._model.to(DEVICE)
        return self._model

    @torch.no_grad()
    def _run(self, items):
        self.ensure_model()
        outs = []
        for i in range(0, len(items), BATCH):
            chunk = items[i:i + BATCH]
            sync()
            t0 = time.perf_counter()
            x = torch.stack([self._pre(load_image(it)) for it in chunk]).to(DEVICE)
            if DEVICE == "cuda" and self.fp16:
                x = x.half()
            y = self._forward(x).float().cpu().numpy()
            sync()
            self.embed_time += time.perf_counter() - t0
            self.n_embedded += len(chunk)
            outs.append(y)
        return np.concatenate(outs, 0)

    def __call__(self, items):
        items = list(items)
        use = CACHE_ENABLED
        if use:
            self._load_cache()
        res = [None] * len(items)
        todo = []
        for i, it in enumerate(items):
            if use and isinstance(it, str) and it in self._cache:
                res[i] = self._cache[it]
            else:
                todo.append(i)
        if todo:
            ys = self._run([items[i] for i in todo])
            for j, i in enumerate(todo):
                res[i] = ys[j]
                if use and isinstance(items[i], str):
                    self._cache[items[i]] = ys[j]
                    self._dirty = True
        return np.stack(res).astype(np.float32)

    def release(self):
        self.save()
        self._model = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


class OpenClipExtractor(Extractor):
    """open_clip image tower (+ text tower for zero-shot).  Output: L2-normalised embedding."""

    def __init__(self, name, arch, pretrained):
        super().__init__()
        self.name, self.arch, self.pretrained = name, arch, pretrained
        self._tok = None

    def _load(self):
        import open_clip
        model, _, pre = open_clip.create_model_and_transforms(self.arch, pretrained=self.pretrained)
        self._model, self._pre = model, pre
        self._tok = open_clip.get_tokenizer(self.arch)

    def _forward(self, x):
        f = self._model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_features(self, prompts_per_class):
        """list (per class) of lists of prompts -> [C,D] normalised (mean of normalised)."""
        self.ensure_model()
        feats = []
        for ps in prompts_per_class:
            tok = self._tok(ps).to(DEVICE)
            t = self._model.encode_text(tok).float()
            t = t / t.norm(dim=-1, keepdim=True)
            t = t.mean(0)
            feats.append(t / t.norm())
        return torch.stack(feats).cpu().numpy()

    @property
    def logit_scale(self):
        self.ensure_model()
        return float(self._model.logit_scale.exp().float().item())


class DinoExtractor(Extractor):
    """HF DINOv2 CLS embedding (L2-normalised)."""

    def __init__(self, name="dinov2_base", hf_id="facebook/dinov2-base"):
        super().__init__()
        self.name, self.hf_id = name, hf_id

    def _load(self):
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(self.hf_id)
        self._pre = _imagenet_pre(224, 0.875)

    def _forward(self, x):
        h = self._model(pixel_values=x).last_hidden_state[:, 0]
        return h / h.norm(dim=-1, keepdim=True)


class TimmLogitsExtractor(Extractor):
    """A classifier; output = raw logits."""
    kind = "logits"

    def __init__(self, name, builder, fp16=True):
        super().__init__()
        self.name, self._builder, self.fp16 = name, builder, fp16

    def _load(self):
        self._model, self._pre = self._builder()

    def _forward(self, x):
        return self._model(x)


def _imagenet_pre(size=224, crop_pct=0.875, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    import torchvision.transforms as T
    return T.Compose([T.Resize(int(round(size / crop_pct)), interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


def _build_convnext():
    import timm
    m = timm.create_model("convnext_small.fb_in1k", pretrained=True)
    cfg = m.pretrained_cfg
    return m, _imagenet_pre(224, cfg.get("crop_pct", 0.875), cfg["mean"], cfg["std"])


def _build_cbc():
    """yigechundong/car-body-classifier: its config.json is not timm-native
    (timm.create_model('hf-hub:...') raises KeyError 'architecture'), so build
    tf_efficientnetv2_s with 8 classes and load model.safetensors (strict, all keys match)."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    m = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=8)
    sd = load_file(hf_hub_download("yigechundong/car-body-classifier", "model.safetensors"))
    m.load_state_dict(sd, strict=True)
    return m, _imagenet_pre(224, 0.875)


EXTRACTORS = {
    "clip_b32": lambda: OpenClipExtractor("clip_b32", "ViT-B-32", "openai"),
    "clip_l14": lambda: OpenClipExtractor("clip_l14", "ViT-L-14", "openai"),
    "siglip_b16": lambda: OpenClipExtractor("siglip_b16", "ViT-B-16-SigLIP", "webli"),
    "dinov2_base": lambda: DinoExtractor("dinov2_base", "facebook/dinov2-base"),
    "convnext_small_in1k": lambda: TimmLogitsExtractor("convnext_small_in1k", _build_convnext),
    "car_body_classifier": lambda: TimmLogitsExtractor("car_body_classifier", _build_cbc, fp16=False),
}
_INSTANCES = {}


def get_extractor(name):
    if name not in _INSTANCES:
        _INSTANCES[name] = EXTRACTORS[name]()
    return _INSTANCES[name]


def release_all():
    for e in list(_INSTANCES.values()):
        e.release()
    _INSTANCES.clear()
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def set_cache(enabled):
    global CACHE_ENABLED
    CACHE_ENABLED = bool(enabled)


def total_embed_time():
    return sum(e.embed_time for e in _INSTANCES.values())
