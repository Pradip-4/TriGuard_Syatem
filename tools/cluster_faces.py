# cluster_faces.py - cluster all face-photo embeddings to detect mislabeled/mixed
# enrollment photos. Reports cluster->folder-label contingency, so we can see how
# many DISTINCT people actually exist in the dataset and which photos are misplaced.
import os
import sys
import pickle
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

ENROLLED = ["2217010", "2317009", "2317046"]
UNKNOWN_DIR = "Unknown Data"
IMG_ROOT = os.path.join(BASE, "tools", "image_data")
AUG_CACHE = os.path.join(BASE, "data", "eval_face_aug_embeddings.pkl")
OUT = os.path.join(BASE, "data", "face_cluster_proposal.json")


def preload():
    from tools.face_embedding import preload_face_runtime as _p
    _p()


def main():
    preload()

    # real-photo embeddings already cached from the augmented run (identity aug)
    with open(AUG_CACHE, "rb") as f:
        aug = pickle.load(f)

    items = []  # (folder_label, photo_name, emb)
    for person in ENROLLED:
        for name, augname, emb in aug[person]:
            if augname == "identity":
                items.append((person, name, emb))

    # embed Unknown Data photos (identity only)
    from deepface import DeepFace
    ud = os.path.join(IMG_ROOT, UNKNOWN_DIR)
    t0 = time.time()
    for name in sorted(os.listdir(ud)):
        if not os.path.isfile(os.path.join(ud, name)):
            continue
        try:
            res = DeepFace.represent(img_path=os.path.join(ud, name),
                                     model_name="ArcFace", enforce_detection=True)
            items.append((UNKNOWN_DIR, name, np.asarray(res[0]["embedding"], dtype=np.float32)))
        except Exception as e:
            print(f"  failed {name}: {type(e).__name__}")
    print(f"embedded unknown data in {time.time() - t0:.1f}s; total items={len(items)}")

    # ---- agglomerative clustering by pairwise L2 distance threshold ----
    # Simple single-linkage: union photos within THRESH of an existing member.
    THRESH = 4.1  # same as production accept threshold
    clusters = []  # list of list of item indices
    for idx, (lab, nm, emb) in enumerate(items):
        placed = False
        for cl in clusters:
            if any(np.linalg.norm(emb - items[j][2]) <= THRESH for j in cl):
                cl.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    # merge chains (single-linkage transitive closure)
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if any(np.linalg.norm(items[a][2] - items[b][2]) <= THRESH
                       for a in clusters[i] for b in clusters[j]):
                    clusters[i].extend(clusters[j])
                    clusters[j] = []
                    merged = True
                    break
            if merged:
                break
    clusters = [c for c in clusters if c]

    print(f"\nDistinct person-clusters at threshold {THRESH}: {len(clusters)}\n")
    for ci, cl in enumerate(sorted(clusters, key=len, reverse=True)):
        counts = {}
        for idx in cl:
            lab = items[idx][0]
            counts[lab] = counts.get(lab, 0) + 1
        print(f"cluster {ci}: size={len(cl)}  folders={counts}")
        for idx in cl:
            lab, nm, _ = items[idx]
            print(f"    {lab:>14} / {nm}")

    # proposal: for each cluster, majority folder label
    import json
    proposal = {"threshold": THRESH, "n_clusters": len(clusters), "clusters": [], "relabel": {}}
    for ci, cl in enumerate(sorted(clusters, key=len, reverse=True)):
        counts = {}
        for idx in cl:
            lab = items[idx][0]
            counts[lab] = counts.get(lab, 0) + 1
        majority = max(counts, key=counts.get)
        proposal["clusters"].append({
            "size": len(cl),
            "labels": counts,
            "majority": majority,
            "photos": [f"{items[idx][0]}/{items[idx][1]}" for idx in cl],
        })
        for idx in cl:
            proposal["relabel"][f"{items[idx][0]}/{items[idx][1]}"] = majority
    with open(OUT, "w") as f:
        json.dump(proposal, f, indent=2)
    print(f"\nproposal saved -> {OUT}")


if __name__ == "__main__":
    main()