# evaluate_relabeled.py
# ------------------------------------------------------------------
# Auto-proposes a corrected face dataset from cluster analysis, then
# re-runs the face evaluation on it (identical matching logic).
#
#  - Clusters photos by single-linkage ArcFace L2 distance <= 4.1
#    (same threshold the system uses to accept a match).
#  - Each identity is assigned the cluster holding the plurality of
#    its own folder photos.
#  - A cluster whose majority label is X but which was NOT assigned to
#    X (a conflicting face, e.g. 2317046's second person) is parked in
#    "Unknown Data" (imposters) instead of polluting an identity.
#  - Nothing in tools/image_data is modified; the corrected mapping is
#    saved to data/face_relabel_proposal.json for verification.
# ------------------------------------------------------------------
import os
import sys
import pickle
import json
import time
from collections import Counter

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "tools"))

from evaluate_confusion import (  # noqa: E402
    ENROLLED, AUGS, knn_match, report,
)

CACHE = os.path.join(BASE, "data", "eval_face_aug_embeddings.pkl")
OUT_RELABEL = os.path.join(BASE, "data", "face_relabel_proposal.json")
OUT_RESULTS = os.path.join(BASE, "data", "evaluation_results_relabeled.json")
CLUSTER_THRESH = 4.1


def single_linkage(embs, thresh):
    """embs: list of embedding vectors -> list of index-clusters (<= thresh)."""
    clusters = []
    for idx in range(len(embs)):
        placed = False
        for cl in clusters:
            if any(np.linalg.norm(embs[idx] - embs[j]) <= thresh for j in cl):
                cl.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if any(np.linalg.norm(embs[a] - embs[b]) <= thresh
                       for a in clusters[i] for b in clusters[j]):
                    clusters[i].extend(clusters[j])
                    clusters[j] = []
                    merged = True
                    break
            if merged:
                break
    return [c for c in clusters if c]


def make_relabel(cache):
    # (label, photo, emb) for real (identity) embeddings
    items = [(lab, name, emb)
             for lab in cache
             for name, aug, emb in cache[lab] if aug == "identity"]
    embs = [it[2] for it in items]
    clusters = single_linkage(embs, CLUSTER_THRESH)
    cl_counts = [Counter(items[i][0] for i in cl) for cl in clusters]

    # per-identity plurality assignment
    assigned = {}
    for identity in ENROLLED:
        best_cl, best_n = None, 0
        for ci, counts in enumerate(cl_counts):
            n = counts.get(identity, 0)
            if n > best_n:
                best_n, best_cl = n, ci
        assigned[identity] = best_cl

    # relabel each photo: identity if its cluster is assigned to its majority,
    # else imposter (parked conflicting face).
    relabel = {}
    for ci, cl in enumerate(clusters):
        counts = Counter(items[i][0] for i in cl)
        majority = counts.most_common(1)[0][0]
        label = majority if assigned.get(majority) == ci else "imposter"
        for i in cl:
            relabel[(items[i][0], items[i][1])] = label

    # summary for the proposal file
    proposal = {"threshold": CLUSTER_THRESH, "assigned": {k: int(v) for k, v in assigned.items()},
                "clusters": [], "relabel": {f"{a}/{b}": c for (a, b), c in relabel.items()}}
    for ci, cl in enumerate(sorted(clusters, key=len, reverse=True)):
        counts = Counter(items[i][0] for i in cl)
        majority = counts.most_common(1)[0][0]
        target = majority if assigned.get(majority) == ci else "imposter (parked)"
        proposal["clusters"].append({
            "cluster": ci, "size": len(cl), "labels": dict(counts),
            "assigned_to": target,
            "photos": [f"{items[i][0]}/{items[i][1]}" for i in cl],
        })
    return relabel, proposal


def build_corrected_probes(cache, relabel):
    out = {person: [] for person in ENROLLED + ["imposter"]}
    for lab in cache:
        for name, aug, emb in cache[lab]:
            new_label = relabel.get((lab, name), "imposter")
            out[new_label].append((name, aug, emb))
    return out


def evaluate_face_on(probes, clean_gallery, tag):
    """Mirror of evaluate_confusion.evaluate_face but with a clean toggle."""
    print("=" * 70)
    print(f"FACE MODULE [{tag}]")
    print("=" * 70)

    enrolled_embs = {}
    for person in ENROLLED:
        enrolled_embs[person] = [(n, e) for n, aug, e in probes[person] if aug == "identity"]

    if clean_gallery:
        # reuse cluster-based cleaning from evaluate_confusion
        from evaluate_confusion import clean_gallery as _cg
        gallery, drop_log = _cg(enrolled_embs, {})
        for person, d in drop_log.items():
            print(f"    {person}: cleaned {len(d)}")
    else:
        gallery = enrolled_embs

    ppl = [(p, n, a, e) for p in ENROLLED for n, a, e in probes[p]]
    imp_ppl = [(p, n, a, e) for p in ["imposter"] if p in probes for n, a, e in probes[p]]
    full_gal = [(q, g) for q in ENROLLED for gn, g in gallery[q]]

    def eval_at(thr):
        ok = fa = fr = 0
        for person, name, aug, e in ppl:
            gal = [(q, g) for q in ENROLLED for gn, g in gallery[q]
                   if not (q == person and gn == name)]
            pred, _ = knn_match(e, gal, threshold=thr)
            if pred == person:
                ok += 1
            elif pred is None:
                fr += 1
            else:
                fa += 1
        im_rej = sum(1 for _, _, _, e in imp_ppl
                     if knn_match(e, full_gal, threshold=thr)[0] is None)
        total = ok + fa + fr + len(imp_ppl)
        return (ok + im_rej) / total if total else 0.0, ok, fa, fr, im_rej

    print(f"  probes: enrolled={len(ppl)}, imposters={len(imp_ppl)}")
    best = None
    for thr in sorted(set([x / 100 for x in range(230, 521, 10)] + [4.10])):
        acc, ok, fa, fr, im_rej = eval_at(thr)
        if best is None or acc > best[0]:
            best = (acc, thr)
    use_thr = best[1]
    print(f"  best threshold {use_thr:.2f} -> accuracy {best[0]:.4f}")

    labels = ENROLLED + ["imposter"]
    preds = ENROLLED + ["rejected"]
    cm = {a: {p: 0 for p in preds} for a in labels}
    row_totals = {a: 0 for a in labels}
    for person, name, aug, e in ppl:
        gal = [(q, g) for q in ENROLLED for gn, g in gallery[q]
               if not (q == person and gn == name)]
        pred, _ = knn_match(e, gal, threshold=use_thr)
        row_totals[person] += 1
        cm[person][pred if pred is not None else "rejected"] += 1
    for p, n, a, e in imp_ppl:
        pred, _ = knn_match(e, full_gal, threshold=use_thr)
        row_totals["imposter"] += 1
        cm["imposter"][pred if pred is not None else "rejected"] += 1
    return cm, row_totals, {"threshold": round(use_thr, 2)}


def main():
    t0 = time.time()
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)

    relabel, proposal = make_relabel(cache)
    with open(OUT_RELABEL, "w") as f:
        json.dump(proposal, f, indent=2)
    print(f"relabel proposal saved -> {OUT_RELABEL}")

    corrected = build_corrected_probes(cache, relabel)
    for person in ENROLLED + ["imposter"]:
        n_photos = len({n for n, a, e in corrected[person] if a == "identity"})
        print(f"  corrected {person}: {n_photos} photos")

    # BEFORE: original folder labels, full gallery, real+augmented probes
    cm_b, rt_b, ex_b = evaluate_face_on(cache, clean_gallery=False,
                                        tag="BEFORE - original folders (augmented probes)")
    res_before = report(cm_b, rt_b, "BEFORE CONFUSION MATRIX (original labels)", "face_before", ex_b)

    # AFTER: corrected labels, real+augmented probes
    cm_a, rt_a, ex_a = evaluate_face_on(corrected, clean_gallery=False,
                                        tag="AFTER - auto-relabeled dataset (augmented probes)")
    res_after = report(cm_a, rt_a, "AFTER CONFUSION MATRIX (auto-relabeled)", "face_after", ex_a)

    out = {"before": res_before, "after": res_after,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(OUT_RESULTS, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT_RESULTS}")


if __name__ == "__main__":
    main()