# screen_enrollment.py
# ------------------------------------------------------------------
# Preventive QA for face enrollment: makes sure one student ID folder
# really contains ONE dominant face, so mixed-enrollment data (the
# root cause of the 2217010 <-> 2317046 confusion) can never silently
# re-enter the production database.
#
# For every identity folder it:
#   1. takes 512-d ArcFace embeddings (cached set is reused if present),
#   2. builds the within-folder face graph (single-linkage, L2 <= 4.10),
#   3. keeps the LARGEST connected component = the dominant face,
#   4. flags every other photo as "needs review / park in Unknown Data",
#   5. also flags photos with no detectable face.
#
# Output:  data/enrollment_screen_report.json  (read-only; original
#          image folders are never modified).
# ------------------------------------------------------------------
import os
import sys
import json
import pickle
import time
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))

ENROLLED = ["2217010", "2317009", "2317046"]
IMG_ROOT = os.path.join(BASE, "tools", "image_data")
CACHE = os.path.join(BASE, "data", "eval_face_aug_embeddings.pkl")
OUT = os.path.join(BASE, "data", "enrollment_screen_report.json")

DEFAULT_THRESHOLD = 4.10  # same as production distance_threshold


def load_embeddings(use_cache=True):
    """{(folder, photo): np.ndarray} — cached identity embeddings if usable."""
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
        out = {}
        for folder, items in cache.items():
            for name, aug, emb in items:
                if aug == "identity":
                    out[(folder, name)] = np.asarray(emb, dtype=np.float64)
        return out, True
    # fallback: embed fresh with the exact production pipeline
    from tools.evaluate_confusion import face_embedding
    out = {}
    for person in ENROLLED:
        d = os.path.join(IMG_ROOT, person)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not os.path.isfile(os.path.join(d, name)):
                continue
            try:
                out[(person, name)] = face_embedding(os.path.join(d, name))
            except Exception:
                pass  # no detectable face
    return out, False


def connected_components(emb_dict, threshold):
    """Single-linkage partitioning (L2 <= threshold) over embedding dict keys."""
    keys = list(emb_dict)
    n = len(keys)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for i in range(n):
        ei = emb_dict[keys[i]]
        for j in range(i + 1, n):
            if np.linalg.norm(ei - emb_dict[keys[j]]) <= threshold:
                union(i, j)

    comps = defaultdict(list)
    for i, k in enumerate(keys):
        comps[find(i)].append(k)
    return list(comps.values())


def main():
    threshold = float(os.environ.get("FACESCREEN_THRESHOLD", DEFAULT_THRESHOLD))
    embs, from_cache = load_embeddings(use_cache=True)
    print(f"embeddings: {len(embs)} (cache={'yes' if from_cache else 'no'})")

    report = {
        "threshold": threshold,
        "note": "Preventive enrollment QA. Source folders are NOT modified.",
        "identities": {},
        "flagged_photos": [],
    }

    # dominant centroids across all IDs (to report likely owners)
    centroids = {}

    for person in ENROLLED:
        d = os.path.join(IMG_ROOT, person)
        if not os.path.isdir(d):
            continue
        folder_embs = {k: e for k, e in embs.items() if k[0] == person}
        names = [(n, os.path.join(d, n)) for n in sorted(os.listdir(d))
                 if os.path.isfile(os.path.join(d, n))]
        missing = [n for n, _ in names if (person, n) not in folder_embs]

        comps = connected_components(folder_embs, threshold)
        comps = sorted(comps, key=len, reverse=True)
        dominant = comps[0] if comps else []
        total = len(names)

        keep = sorted(n for n, _ in names if (person, n) in dominant)
        flagged = []
        for n, path in names:
            if (person, n) in dominant:
                continue
            flagged.append({
                "photo": n,
                "reason": "no detectable face" if (person, n) not in folder_embs
                          else "off-cluster face (not the dominant face of this ID)",
            })

        if dominant:
            centroids[person] = np.mean(
                [folder_embs[k] for k in dominant], axis=0
            )

        comp_sizes = [len(c) for c in comps]
        report["identities"][person] = {
            "photos_in_folder": total,
            "embedded": total - len(missing),
            "no_face_detected": missing,
            "components": comp_sizes,
            "dominant_face_photos": keep,
            "flagged_photos": [f["photo"] for f in flagged],
        }
        print(f"\n{person}: {total} photos, {len(comps)} face cluster(s) "
              f"sizes={comp_sizes}  -> dominant={len(keep)}, flagged={len(flagged)}")
        for f in flagged:
            print(f"    FLAG  {f['photo']}  ({f['reason']})")

    # likely owner of each flagged face = nearest dominant centroid
    for person, info in report["identities"].items():
        d = os.path.join(IMG_ROOT, person)
        for n in info["flagged_photos"]:
            if (person, n) not in embs:
                continue
            e = embs[(person, n)]
            owner, best = None, float("inf")
            for q, c in centroids.items():
                dd = float(np.linalg.norm(e - c))
                if dd < best:
                    owner, best = q, dd
            report["flagged_photos"].append({
                "photo": f"{person}/{n}",
                "nearest_dominant_identity": owner,
                "distance_to_that_centroid": round(best, 3),
                "action": "park in Unknown Data; do not enroll",
            })

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {OUT}  ({len(report['flagged_photos'])} flagged entries)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.1f}s (cached embeddings -> fast)")