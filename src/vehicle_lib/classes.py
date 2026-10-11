# -*- coding: utf-8 -*-
"""Fixed class list, coarse groups, prompt phrases and Thai names."""
import numpy as np

CLASSES = ["micro", "sedan", "cuv", "suv", "hatchback", "roadster", "pickup", "van",
           "coupe", "supercar", "campervan", "mini_truck", "cabriolet", "minivan",
           "truck", "big_truck"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

COARSE_GROUPS = {
    "small": ["micro", "hatchback"],
    "sedan": ["sedan"],
    "sporty_closed": ["coupe", "supercar"],
    "open_top": ["cabriolet", "roadster"],
    "suv_family": ["cuv", "suv"],
    "light_truck": ["pickup", "mini_truck"],
    "van_family": ["van", "minivan", "campervan"],
    "heavy_truck": ["truck", "big_truck"],
}
COARSE_NAMES = list(COARSE_GROUPS)
COARSE = {c: g for g, cs in COARSE_GROUPS.items() for c in cs}          # class -> group
COARSE_OF_IDX = np.array([COARSE_NAMES.index(COARSE[c]) for c in CLASSES])  # [16] -> 0..7
# 16x8 0/1 matrix: probs[N,16] @ COARSE_MATRIX -> [N,8]
COARSE_MATRIX = np.zeros((16, 8), dtype=np.float32)
COARSE_MATRIX[np.arange(16), COARSE_OF_IDX] = 1.0

# short noun phrase used in the 'plain' prompt
PLAIN_NAMES = {
    "micro": "microcar", "sedan": "sedan", "cuv": "crossover SUV", "suv": "SUV",
    "hatchback": "hatchback", "roadster": "roadster", "pickup": "pickup truck",
    "van": "van", "coupe": "coupe", "supercar": "supercar", "campervan": "campervan",
    "mini_truck": "mini truck", "cabriolet": "cabriolet", "minivan": "minivan",
    "truck": "truck", "big_truck": "big truck",
}

# longer descriptive phrases ('a photo of ' + phrase)
DESCRIPTIVE = {
    "micro": "a microcar, a tiny city car with two seats and a very short body",
    "sedan": "a sedan, a four-door passenger car with a separate trunk",
    "cuv": "a compact crossover SUV, a small car-based crossover with raised ride height",
    "suv": "a large SUV, a tall sport utility vehicle with a boxy body and off-road styling",
    "hatchback": "a hatchback, a small car with a rear tailgate door and a short tail",
    "roadster": "a roadster, an open two-seat sports car with no roof",
    "pickup": "a pickup truck with an open cargo bed",
    "van": "a van, a boxy cargo or passenger van with a sliding door",
    "coupe": "a coupe, a two-door car with a sloping roofline",
    "supercar": "a supercar, a low, wide exotic high-performance sports car",
    "campervan": "a campervan, a motorhome van converted for camping and sleeping",
    "mini_truck": "a mini truck, a compact light truck with a small flat cargo bed",
    "cabriolet": "a cabriolet, a convertible car with a folding soft-top roof",
    "minivan": "a minivan, a tall family people carrier with sliding side doors",
    "truck": "a medium-sized truck, a lorry with a cab and a cargo box",
    "big_truck": "a big heavy truck, a semi-trailer tractor-trailer rig",
}

# Thai display names (also used for the 'thai_names' prompt variant)
THAI = {
    "micro": "รถไมโครคาร์ขนาดเล็ก", "sedan": "รถเก๋งซีดาน", "cuv": "รถครอสโอเวอร์",
    "suv": "รถเอสยูวี", "hatchback": "รถแฮทช์แบ็ก", "roadster": "รถโรดสเตอร์เปิดประทุนสองที่นั่ง",
    "pickup": "รถกระบะ", "van": "รถตู้", "coupe": "รถคูเป้", "supercar": "รถซูเปอร์คาร์",
    "campervan": "รถบ้านแคมเปอร์แวน", "mini_truck": "รถบรรทุกเล็ก", "cabriolet": "รถเปิดประทุน",
    "minivan": "รถมินิแวน", "truck": "รถบรรทุก", "big_truck": "รถบรรทุกใหญ่ (รถพ่วง)",
}

# prompt template sets; {} is replaced by the class phrase
TEMPLATE_PLAIN = ["a photo of a {}."]
TEMPLATES_ENSEMBLE = [
    "a photo of a {}.", "a photo of the {}.", "a street photo of a {}.",
    "a photo of a {} parked on the road.", "a side view of a {}.",
    "a close-up photo of a {}.", "a low-resolution photo of a {}.",
    "a photo of a {}, a type of car.",
]
TEMPLATES_DESC_ENSEMBLE = ["a photo of {}.", "a street photo of {}.", "an image of {}."]
TEMPLATE_THAI = ["รูปถ่ายของ{}"]

def prompt_sets(kind):
    """Return list (per class, CLASSES order) of lists of prompt strings."""
    out = []
    for c in CLASSES:
        if kind == "plain":
            out.append([t.format(PLAIN_NAMES[c]) for t in TEMPLATE_PLAIN])
        elif kind == "descriptive":
            out.append(["a photo of %s." % DESCRIPTIVE[c]])
        elif kind == "ensemble":
            out.append([t.format(PLAIN_NAMES[c]) for t in TEMPLATES_ENSEMBLE])
        elif kind == "ensemble_desc":
            out.append([t.format(DESCRIPTIVE[c]) for t in TEMPLATES_DESC_ENSEMBLE])
        elif kind == "thai_names":
            out.append([t.format(THAI[c]) for t in TEMPLATE_THAI])
        else:
            raise ValueError(kind)
    return out
PROMPT_KINDS = ["plain", "descriptive", "ensemble", "ensemble_desc", "thai_names"]

# CBSC dataset class (lower-case) -> our coarse group
CBSC_TO_COARSE = {"hatchback": "small", "sedan": "sedan", "sports": "sporty_closed", "cabriolet": "open_top",
                  "suv": "suv_family", "pickup": "light_truck", "van": "van_family", "truck": "heavy_truck"}
