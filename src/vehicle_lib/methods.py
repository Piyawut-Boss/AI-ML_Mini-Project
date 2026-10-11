# -*- coding: utf-8 -*-
"""Named, swappable techniques.  All expose predict_proba(items) -> np.ndarray [N,16]
(CLASSES order).  `items` = list of PIL images or of image path strings.
No training/fine-tuning: only frozen pretrained models + prompts / k-NN / hand-written maps."""
import os
import numpy as np
import pandas as pd

from .classes import CLASSES, CLASS_IDX, prompt_sets, PROMPT_KINDS, COARSE_NAMES, COARSE_OF_IDX, COARSE_MATRIX, CBSC_TO_COARSE
from . import models

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_CSV = os.environ.get("VEHICLE_DATASET", os.path.join(_REPO, "data", "dataset.csv"))
SEED = 42
_SUPPORT_CACHE = {}


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _renorm(p):
    p = np.clip(p, 0, None) + 1e-9
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


def get_support(shots=10):
    """dict coarse group -> list of `shots` support image paths (fixed seed, nested subsets 3 < 5 < 10).
    Support images carry only coarse labels (CBSC classes -> 8 groups)."""
    key = (DATASET_CSV, shots)
    if key not in _SUPPORT_CACHE:
        df = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
        df = df[df["split_role"] == "support"]
        g = df["class"].map(CBSC_TO_COARSE)
        if "coarse_group" in df:
            g = df["coarse_group"].where(df["coarse_group"].isin(COARSE_NAMES), g)
        df = df.assign(_g=g)
        out = {}
        for gi, c in enumerate(COARSE_NAMES):
            paths = sorted(df[df["_g"] == c]["path"].tolist())
            rng = np.random.RandomState(SEED + gi)
            perm = rng.permutation(len(paths))
            out[c] = [paths[i] for i in perm[:shots]]
        _SUPPORT_CACHE[key] = out
    return _SUPPORT_CACHE[key]


class Method:
    name = "?"
    backbone = ""
    family = ""
    description = ""

    level = "16"          # '16' = predicts the 16 chart classes; 'coarse' = only the 8 groups
    caveat = ""

    def predict_proba(self, items):
        raise NotImplementedError

    def predict_proba_coarse(self, items):
        """[N,8] probabilities over COARSE_NAMES (16-class probs summed within groups)."""
        return _renorm(self.predict_proba(items) @ COARSE_MATRIX)

    def extractors(self):
        return []

    def embed_time(self):
        return sum(e.embed_time for e in {id(x): x for x in self.extractors()}.values())


# ---------------------------------------------------------------- (a) CLIP zero-shot
class ClipZeroShot(Method):
    family = "zeroshot"

    def __init__(self, backbone, prompt):
        self.backbone, self.prompt = backbone, prompt
        self.name = "clip_zeroshot_%s_%s" % (prompt, backbone)
        self.description = "zero-shot, %s prompts, %s" % (prompt, backbone)
        self._T = None

    def extractors(self):
        return [models.get_extractor(self.backbone)]

    def _text(self):
        if self._T is None:
            ex = models.get_extractor(self.backbone)
            self._T = ex.text_features(prompt_sets(self.prompt))
            self._scale = ex.logit_scale
        return self._T

    def predict_proba(self, items):
        T = self._text()
        F = models.get_extractor(self.backbone)(items)
        return _renorm(_softmax(self._scale * F @ T.T))


# ---------------------------------------------------------------- (b) k-NN few-shot
class KnnFewShot(Method):
    family = "knn"
    level = "coarse"
    TAU = 0.05

    def __init__(self, backbone, shots=10, k=5):
        self.backbone, self.shots, self.k = backbone, shots, k
        self.name = "knn_fewshot_%s_s%d_k%d" % (backbone, shots, k)
        self.description = "cosine k-NN k=%d, %d support/group, %s (coarse labels only)" % (k, shots, backbone)
        self._S = None

    def extractors(self):
        return [models.get_extractor(self.backbone)]

    def _support(self):
        if self._S is None:
            sup = get_support(self.shots)
            paths, labels = [], []
            for gi, c in enumerate(COARSE_NAMES):
                for p in sup[c]:
                    paths.append(p); labels.append(gi)
            self._S = models.get_extractor(self.backbone)(paths)
            self._y = np.array(labels)
        return self._S, self._y

    def predict_proba(self, items):
        raise NotImplementedError("kNN support images carry only coarse labels: use predict_proba_coarse")

    def predict_proba_coarse(self, items):
        S, y = self._support()
        Q = models.get_extractor(self.backbone)(items)
        sims = Q @ S.T
        k = min(self.k, S.shape[0])
        idx = np.argsort(-sims, axis=1)[:, :k]
        top = np.take_along_axis(sims, idx, 1)
        w = _softmax(top / self.TAU, axis=1)             # softmax-weighted vote
        out = np.zeros((len(Q), 8), np.float64)
        for j in range(k):
            np.add.at(out, (np.arange(len(Q)), y[idx[:, j]]), w[:, j])
        return _renorm(out + 1e-3 / 8)


# ---------------------------------------------------------------- (c) ImageNet map
# ImageNet-1k class id -> {our class: weight}.  Weights per id sum to 1 (mass split when ambiguous).
IMAGENET_MAP = {
    468: {"sedan": 1.0},                                  # cab, taxi
    627: {"sedan": 1.0},                                  # limousine
    511: {"cabriolet": 0.5, "roadster": 0.5},             # convertible (ambiguous -> split)
    817: {"coupe": 0.5, "supercar": 0.5},                 # sports car (split)
    751: {"supercar": 1.0},                               # racer / race car
    609: {"suv": 1.0},                                    # jeep, landrover
    436: {"cuv": 0.5, "hatchback": 0.5},                  # beach wagon / station wagon (approximate)
    717: {"pickup": 1.0},                                 # pickup truck
    675: {"van": 1.0},                                    # moving van
    734: {"van": 1.0},                                    # police van
    407: {"van": 1.0},                                    # ambulance
    654: {"van": 0.5, "minivan": 0.5},                    # minibus
    656: {"minivan": 1.0},                                # minivan
    757: {"campervan": 1.0},                              # recreational vehicle
    660: {"campervan": 1.0},                              # mobile home
    569: {"truck": 1.0},                                  # garbage truck
    864: {"truck": 1.0},                                  # tow truck
    555: {"truck": 1.0},                                  # fire engine
    867: {"big_truck": 1.0},                              # trailer truck / semi
}
IMAGENET_NAMES = {468: "cab", 627: "limousine", 511: "convertible", 817: "sports car", 751: "racer",
                  609: "jeep", 436: "beach wagon", 717: "pickup", 675: "moving van", 734: "police van",
                  407: "ambulance", 654: "minibus", 656: "minivan", 757: "recreational vehicle",
                  660: "mobile home", 569: "garbage truck", 864: "tow truck", 555: "fire engine",
                  867: "trailer truck"}
IMAGENET_NO_COUNTERPART = ["micro", "mini_truck"]   # no ImageNet class at all; hatchback/cuv/coupe/roadster only via split mass


def imagenet_matrix():
    M = np.zeros((1000, 16), np.float32)
    for i, d in IMAGENET_MAP.items():
        for c, w in d.items():
            M[i, CLASS_IDX[c]] = w
    return M


class ImageNetMap(Method):
    family = "imagenet_map"
    backbone = "convnext_small_in1k"

    def __init__(self):
        self.name = "imagenet_map_convnext_small"
        self.description = "ConvNeXt-S ImageNet-1k probs summed through hand-written map to 16 classes"
        self.M = imagenet_matrix()

    def extractors(self):
        return [models.get_extractor(self.backbone)]

    def predict_proba(self, items):
        logits = models.get_extractor(self.backbone)(items)
        p = _softmax(logits) @ self.M          # mass on unmapped ImageNet classes is dropped, then renormalised
        return _renorm(p)


# ---------------------------------------------------------------- (d) car_body_classifier
CBC_LABELS = ["SUV", "VAN", "STATION_WAGON", "MICRO", "ACIK_TEKERLEKLI_F1_ARACLARI",
              "SEDAN", "HATCHBACK", "PICK_UP"]            # from the model's config.json id2label
CBC_TEMP = 0.565484
CBC_TO_OURS = {"SUV": "suv", "VAN": "van", "MICRO": "micro", "SEDAN": "sedan",
               "HATCHBACK": "hatchback", "PICK_UP": "pickup"}   # STATION_WAGON, F1: no counterpart


class CarBodyClassifier(Method):
    family = "car_body_classifier"
    backbone = "car_body_classifier"
    caveat = "trained_on_cbsc_likely_inflated"
    known = [CLASS_IDX[c] for c in CBC_TO_OURS.values()]

    def __init__(self):
        self.name = "car_body_classifier"
        self.description = "yigechundong/car-body-classifier (EfficientNetV2-S, 8 classes -> 6 mapped), T=0.565"

    def extractors(self):
        return [models.get_extractor(self.backbone)]

    def predict_proba(self, items):
        logits = models.get_extractor(self.backbone)(items)
        p8 = _softmax(logits / CBC_TEMP)
        out = np.zeros((len(p8), 16), np.float32)
        for j, lab in enumerate(CBC_LABELS):
            if lab in CBC_TO_OURS:
                out[:, CLASS_IDX[CBC_TO_OURS[lab]]] = p8[:, j]
        s = out.sum(1, keepdims=True)
        out = out / np.clip(s, 1e-9, None)            # renormalise over the 6 known classes
        return (out + 1e-9).astype(np.float32)        # the other 10 classes stay ~0


# ---------------------------------------------------------------- fusion / hybrid
class Fusion(Method):
    """Mean of component probabilities at coarse level (kNN is coarse-only)."""
    family = "fusion"
    level = "coarse"

    def __init__(self, name, parts, weights=None, backbone=""):
        self.name, self.parts_names, self.backbone = name, parts, backbone
        self.weights = weights or [1.0] * len(parts)
        self.description = "coarse-level mean of probabilities of: " + " + ".join(parts)
        self._parts = None

    def _p(self):
        if self._parts is None:
            self._parts = [build(n) for n in self.parts_names]
        return self._parts

    @property
    def caveat(self):
        return "trained_on_cbsc_likely_inflated" if any(m.caveat for m in self._p()) else ""

    def extractors(self):
        return [e for m in self._p() for e in m.extractors()]

    def predict_proba(self, items):
        raise NotImplementedError("fusion with kNN is coarse-only")

    def predict_proba_coarse(self, items):
        ps = [m.predict_proba_coarse(items) for m in self._p()]
        w = np.array(self.weights, np.float32); w = w / w.sum()
        return _renorm(sum(wi * p for wi, p in zip(w, ps)))


# car_body_classifier labels collapsed to the 8 groups (STATION_WAGON / F1 unknown)
CBC_KNOWN_GROUPS = sorted({int(COARSE_OF_IDX[CLASS_IDX[c]]) for c in CBC_TO_OURS.values()})


class CbcHybrid(Method):
    """Restricted-mass fusion at coarse level: the general model keeps its mass on the groups the specialist
    does not know (sporty_closed, open_top, heavy_truck); inside the 5 groups it knows, the mass m is
    redistributed with p = (1-w)*general + w*m*specialist."""
    family = "hybrid"
    level = "coarse"
    caveat = "trained_on_cbsc_likely_inflated"

    def __init__(self, name, general, w=0.5, backbone=""):
        self.name, self.general_name, self.w, self.backbone = name, general, w, backbone
        self.description = "%s + car_body_classifier restricted-mass fusion (w=%.1f)" % (general, w)
        self._g = self._c = None

    def _parts(self):
        if self._g is None:
            self._g, self._c = build(self.general_name), build("car_body_classifier")
        return self._g, self._c

    def extractors(self):
        g, c = self._parts()
        return g.extractors() + c.extractors()

    def predict_proba(self, items):
        raise NotImplementedError("hybrid is coarse-only")

    def predict_proba_coarse(self, items):
        g, c = self._parts()
        pg, pc = g.predict_proba_coarse(items), c.predict_proba_coarse(items)
        k = CBC_KNOWN_GROUPS
        pc = pc[:, k] / np.clip(pc[:, k].sum(1, keepdims=True), 1e-9, None)
        out = pg.copy()
        m = pg[:, k].sum(1, keepdims=True)
        out[:, k] = (1 - self.w) * pg[:, k] + self.w * m * pc
        return _renorm(out)


# ---------------------------------------------------------------- registry
CLIP_BACKBONES = ["clip_b32", "clip_l14", "siglip_b16"]
KNN_BACKBONES = ["clip_b32", "clip_l14", "siglip_b16", "dinov2_base"]
BEST_PROMPT = "ensemble"

_FACTORIES = {}


def _reg(name, factory, backbone, family):
    _FACTORIES[name] = (factory, backbone, family)


for _bb in CLIP_BACKBONES:
    for _pk in PROMPT_KINDS:
        _n = "clip_zeroshot_%s_%s" % (_pk, _bb)
        _reg(_n, (lambda b=_bb, p=_pk: ClipZeroShot(b, p)), _bb, "zeroshot")
for _bb in KNN_BACKBONES:
    for _s in (3, 5, 10):
        for _k in (1, 3, 5):
            _n = "knn_fewshot_%s_s%d_k%d" % (_bb, _s, _k)
            _reg(_n, (lambda b=_bb, s=_s, k=_k: KnnFewShot(b, s, k)), _bb, "knn")
_reg("imagenet_map_convnext_small", ImageNetMap, "convnext_small_in1k", "imagenet_map")
_reg("car_body_classifier", CarBodyClassifier, "car_body_classifier", "car_body_classifier")
for _bb in CLIP_BACKBONES:
    _zs = "clip_zeroshot_%s_%s" % (BEST_PROMPT, _bb)
    _kn = "knn_fewshot_%s_s10_k5" % _bb
    _n = "fusion_zs_knn_%s" % _bb
    _reg(_n, (lambda n=_n, a=_zs, b=_kn, bb=_bb: Fusion(n, [a, b], backbone=bb)), _bb, "fusion")
_n = "fusion_zs_clip_l14_knn_dinov2"
_reg(_n, (lambda n=_n: Fusion(n, ["clip_zeroshot_ensemble_clip_l14", "knn_fewshot_dinov2_base_s10_k5"],
                              backbone="clip_l14+dinov2_base")), "clip_l14+dinov2_base", "fusion")
_n = "fusion_zs_knn_clip_l14_dinov2"
_reg(_n, (lambda n=_n: Fusion(n, ["clip_zeroshot_ensemble_clip_l14", "knn_fewshot_clip_l14_s10_k5",
                                  "knn_fewshot_dinov2_base_s10_k5"], backbone="clip_l14+dinov2_base")),
     "clip_l14+dinov2_base", "fusion")
for _bb in CLIP_BACKBONES:
    _n = "hybrid_zs_cbc_%s" % _bb
    _reg(_n, (lambda n=_n, g="clip_zeroshot_%s_%s" % (BEST_PROMPT, _bb), bb=_bb: CbcHybrid(n, g, backbone=bb + "+cbc")),
         _bb + "+cbc", "hybrid")
_n = "hybrid_fusion_cbc_clip_l14"
_reg(_n, (lambda n=_n: CbcHybrid(n, "fusion_zs_knn_clip_l14", backbone="clip_l14+cbc")), "clip_l14+cbc", "hybrid")
_n = "hybrid_fusion_cbc_clip_l14_dinov2"
_reg(_n, (lambda n=_n: CbcHybrid(n, "fusion_zs_knn_clip_l14_dinov2", backbone="clip_l14+dinov2_base+cbc")),
     "clip_l14+dinov2_base+cbc", "hybrid")

METHODS = {n: v[1:] for n, v in _FACTORIES.items()}   # name -> (backbone, family)
# methods that can output the 16 chart classes (all others are coarse-8 only)
LEVEL16 = {n for n in _FACTORIES if _FACTORIES[n][2] in ('zeroshot', 'imagenet_map', 'car_body_classifier')}


def build(name):
    if name not in _FACTORIES:
        raise KeyError("unknown method %s" % name)
    m = _FACTORIES[name][0]()
    m.backbone = m.backbone or _FACTORIES[name][1]
    m.name = name
    return m


def to_coarse(probs):
    from .classes import COARSE_MATRIX
    return probs @ COARSE_MATRIX
