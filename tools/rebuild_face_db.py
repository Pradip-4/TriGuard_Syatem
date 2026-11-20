# rebuild_face_db.py
# ------------------------------------------------------------------
# Rebuilds the production enrollment DB  data/face_embeddings/student_embeddings.pkl
# from the auto-relabeled dataset (data/face_relabel_proposal.json).
#
#  - Uses REAL-photo embeddings only in the gallery (same as the evaluation).
#  - Backs up the existing DB before overwriting.
#  - Original tools/image_data folders are NOT modified.
# ------------------------------------------------------------------
import os
import sys
import pickle
import json
import shutil
import time
from collections import Counter

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

CACHE = os.path.join(BASE, "data", "eval_face_aug_embeddings.pkl")
RELABEL = os.path.join(BASE, "data", "face_relabel_proposal.json")
DB_PATH = os.path.join(BASE, "data", "face_embeddings", "student_embeddings.pkl")

ENROLLED = ["2217010", "2317009", "2317046"]


def main():
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    with open(RELABEL, "r") as f:
        proposal = json.load(f)
    relabel = proposal["relabel"]  # "folder/photo" -> target label

    # Backup existing DB
    if os.path.exists(DB_PATH):
        bak = DB_PATH + f".backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(DB_PATH, bak)
        print(f"backup -> {bak}")

    # Collect REAL identity embeddings per corrected identity
    db = {person: [] for person in ENROLLED}
    seen = {person: [] for person in ENROLLED}
    for lab in ENROLLED:
        for name, aug, emb in cache[lab]:
            if aug != "identity":
                continue
            key = f"{lab}/{name}"
            new_label = relabel.get(key, "imposter")
            if new_label in ENROLLED:
                db[new_label].append(np.asarray(emb, dtype=np.float32))
                seen[new_label].append(key)

    # Sanity: every embedding must be 512-dim ArcFace
    for person, embs in db.items():
        dims = {int(np.asarray(e).size) for e in embs}
        assert dims == {512}, f"{person}: unexpected dims {dims}"
        print(f"{person}: {len(embs)} embeddings (512-d)")

    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

    total = sum(len(v) for v in db.values())
    print(f"\nWrote corrected DB -> {DB_PATH}  ({total} embeddings, {len(db)} identities)")

    # Preview of per-identity photo counts in the corrected enrollment
    for person, keys in seen.items():
        print(f"  {person}: {len(keys)} photos")
        print("    " + ", ".join(sorted(keys)))


if __name__ == "__main__":
    main()