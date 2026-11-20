# explain_confusion.py
# ------------------------------------------------------------------
# Diagnoses WHY 2217010 <-> 2317046 appear confused in the matrices.
#  - BEFORE : original enrollment folders, original label, real-photo LOO
#  - AFTER  : corrected galleries (production DB), corrected label, real-photo LOO
#  - AUG AFTER: which augmentation variants drive the remaining cross-confusion
# ------------------------------------------------------------------
import os
import sys
import pickle
import json
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))

from evaluate_confusion import knn_match, FACE_THRESHOLD_DEFAULT

CACHE = os.path.join(BASE, "data", "eval_face_aug_embeddings.pkl")
RELABEL = os.path.join(BASE, "data", "face_relabel_proposal.json")
DB = os.path.join(BASE, "data", "face_embeddings", "student_embeddings.pkl")
ENROLLED = ["2217010", "2317009", "2317046"]

cache = pickle.load(open(CACHE, "rb"))
relabel = json.load(open(RELABEL, "rb"))["relabel"]
db = pickle.load(open(DB, "rb"))

# ---------- BEFORE: original folders, original label ----------
def loo(label_mode, galleries, self_exclude):
    """LOO over real identity embeddings. label_mode: 'orig' | 'corrected'."""
    counts = {p: Counter() for p in ENROLLED}
    rows = {"correct": 0, "fa": 0, "fr": 0}
    for lab in ENROLLED:
        for n, a, e in cache[lab]:
            if a != "identity":
                continue
            true = lab if label_mode == "orig" else relabel.get(f"{lab}/{n}", "imposter")
            if true not in ENROLLED:
                continue
            gal = []
            for q, embs in galleries.items():
                for g in embs:
                    if q == true and self_exclude(q, g, e):
                        continue
                    gal.append((q, g))
            pred, best_dist = knn_match(np.asarray(e), gal, threshold=FACE_THRESHOLD_DEFAULT)
            counts[true][pred if pred else "rejected"] += 1
            if pred == true:
                rows["correct"] += 1
            elif pred is None:
                rows["fr"] += 1
            else:
                rows["fa"] += 1
    return rows, counts

def self_is_dup(q, g, e):
    return np.linalg.norm(np.asarray(g) - np.asarray(e)) < 1e-6

# gallery BEFORE = original folders (identity embeddings)
gal_before = {p: [np.asarray(e) for n, a, e in cache[p] if a == "identity"] for p in ENROLLED}
rows_b, counts_b = loo("orig", gal_before, self_is_dup)

# gallery AFTER = corrected production DB
gal_after = {p: [np.asarray(e) for e in db[p]] for p in ENROLLED}
rows_a, counts_a = loo("corrected", gal_after, self_is_dup)

def show(tag, rows, counts):
    n = sum(rows.values())
    print(f"\n== REAL-photo LOO  [{tag}]  thr={FACE_THRESHOLD_DEFAULT}  n={n}")
    print(f"   correct={rows['correct']}  FA={rows['fa']}  FR={rows['fr']}  "
          f"acc={rows['correct']/n:.4f}" if n else "")
    for p in ENROLLED:
        c = counts[p]
        t = sum(c.values())
        print(f"   {p}: " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())) +
              (f"  acc={c[p] / t:.3f}" if t else ""))

show("BEFORE (original folders/labels)", rows_b, counts_b)
show("AFTER  (corrected galleries/labels)", rows_a, counts_a)

# ---------- Cross-pair distances after (real photos) ----------
print("\n== Distance stats AFTER (real identity embeddings) ==")
A = [(n, np.asarray(e)) for n, a, e in cache["2217010"] if a == "identity"
     and relabel.get("2217010/" + n) in ENROLLED]
B = [(n, np.asarray(e)) for n, a, e in cache["2317046"] if a == "identity"
     and relabel.get("2317046/" + n) in ENROLLED]
# note: includes 2217010/2317046 folders only; collect from all folders below
all_items = []
for lab in ENROLLED:
    for n, a, e in cache[lab]:
        if a == "identity":
            new = relabel.get(f"{lab}/{n}", "imposter")
            if new in ENROLLED:
                all_items.append((new, n, np.asarray(e)))
A2 = [(n, e) for p, n, e in all_items if p == "2217010"]
B2 = [(n, e) for p, n, e in all_items if p == "2317046"]
pairs = sorted((float(np.linalg.norm(ea - eb)), na, nb)
               for na, ea in A2 for nb, eb in B2)
under = [(d, na, nb) for d, na, nb in pairs if d <= FACE_THRESHOLD_DEFAULT]
print(f"min cross distance = {pairs[0][0]:.3f}  ({pairs[0][1]} <-> {pairs[0][2]})")
print(f"cross pairs under 4.10 = {len(under)} / {len(pairs)}")
for d, na, nb in under[:15]:
    print(f"   {d:6.3f}  {na} <-> {nb}")

# ---------- Augmented AFTER: which variants cause 2217010<->2317046 ----------
print("\n== Augmented probes AFTER: cross-confusion by variant ==")
variant = defaultdict(Counter)          # (true, variant) -> predicted counts
for lab in ENROLLED:
    for n, a, e in cache[lab]:
        true = relabel.get(f"{lab}/{n}", "imposter")
        if true not in ENROLLED:
            continue
        gal = [(q, np.asarray(g)) for q in ENROLLED for g in db[q]]
        pred, _ = knn_match(np.asarray(e), gal, threshold=FACE_THRESHOLD_DEFAULT)
        if pred and pred != true:
            variant[(true, a)][pred] += 1
for k in sorted(variant):
    true, aug = k
    if (true, aug) in [("2217010", x) for x in ["identity", "flip", "rot+8", "rot-8", "brightx125", "zoom108"]] or \
       (true, aug) in [("2317046", x) for x in ["identity", "flip", "rot+8", "rot-8", "brightx125", "zoom108"]]:
        if true in ("2217010", "2317046"):
            print(f"   {true} [{aug}]: " + "  ".join(f"{p}={c}" for p, c in variant[k].items()))