# analyze_distances.py - caches ArcFace embeddings + reports within/cross identity
# distance distributions to guide accuracy improvements.
import os
import sys
import pickle
import time
from itertools import combinations

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

ENROLLED = ["2217010", "2317009", "2317046"]
IMG_ROOT = os.path.join(BASE, "tools", "image_data")
CACHE = os.path.join(BASE, "data", "eval_face_embeddings.pkl")


def preload():
    from tools.face_embedding import preload_face_runtime as _p
    _p()


def get_embeddings():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    from deepface import DeepFace
    t0 = time.time()
    out = {}
    for person in ENROLLED:
        d = os.path.join(IMG_ROOT, person)
        out[person] = []
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            try:
                res = DeepFace.represent(img_path=p, model_name="ArcFace", enforce_detection=True)
                out[person].append((name, np.asarray(res[0]["embedding"], dtype=np.float32)))
            except Exception as e:
                print(f"  failed {person}/{name}: {type(e).__name__}")
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    print(f"Embedded+cached in {time.time() - t0:.1f}s")
    return out


def main():
    preload()
    embs = get_embeddings()
    for person, v in embs.items():
        print(f"{person}: {len(v)} photos")

    # Flat lookup by (person, photo)
    flat = [(p, n, e) for p in ENROLLED for n, e in embs[p]]

    print("\n=== genuine (same identity, LOO) min-distance stats ===")
    gen_dist = []
    for p, n, e in flat:
        same = [np.linalg.norm(e - e2) for q, n2, e2 in flat if q == p and n2 != n]
        if same:
            gen_dist.append(min(same))
    print(f"n={len(gen_dist)}  min={min(gen_dist):.3f}  median={np.median(gen_dist):.3f}  "
          f"max={max(gen_dist):.3f}")

    print("\n=== impostor (different identity) min-distance stats ===")
    imp_dist = []
    for p, n, e in flat:
        dif = [np.linalg.norm(e - e2) for q, n2, e2 in flat if q != p]
        if dif:
            imp_dist.append(min(dif))
    print(f"n={len(imp_dist)}  min={min(imp_dist):.3f}  median={np.median(imp_dist):.3f}  "
          f"max={max(imp_dist):.3f}")

    print("\n=== cross-identity confusion hotspots (photo vs nearest other identity) ===")
    print(f"{'photo':<34}{'id':>10}{'other':>10}{'dist':>8}")
    for p, n, e in flat:
        best_other, best_d = None, float("inf")
        for q, n2, e2 in flat:
            if q == p:
                continue
            d = np.linalg.norm(e - e2)
            if d < best_d:
                best_d, best_other = d, q
        if best_d < 5.0:
            print(f"{n:<34}{p:>10}{best_other:>10}{best_d:>8.3f}")

    # Pairwise identity centroid distances
    print("\n=== identity centroid distance matrix ===")
    centroids = {}
    for p, v in embs.items():
        centroids[p] = np.mean([e for _, e in v], axis=0)
    for a, b in combinations(ENROLLED, 2):
        d = np.linalg.norm(centroids[a] - centroids[b])
        print(f"  {a} <-> {b}: {d:.3f}")


if __name__ == "__main__":
    main()