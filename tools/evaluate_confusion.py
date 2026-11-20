# evaluate_confusion.py
# ------------------------------------------------------------------
# Accuracy + confusion-matrix evaluation for TriGuard (face + voice).
#
# Face: ArcFace embeddings via DeepFace (same as production FaceModule).
#   - Gallery  = real enrolled photos (optionally cleaned of outliers).
#   - Probes   = real photos + deterministic augmentations (flip/rot/bright/zoom)
#                to enlarge the confusion-matrix sample count.
#   - Matching = production KNN top-5 majority vote, ArcFace L2 threshold
#                (swept; production default 4.10 shown as baseline).
# Voice: ECAPA-TDNN + mean cosine per speaker, threshold 0.35 (unchanged).
# ------------------------------------------------------------------
import os
import sys
import json
import pickle
import time
from collections import Counter

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

ENROLLED = ["2217010", "2317009", "2317046"]
UNKNOWN_DIR = "Unknown Data"
IMG_ROOT = os.path.join(BASE, "tools", "image_data")
VOICE_ROOT = os.path.join(BASE, "tools", "raw_voice_dataset")
VOICE_SAVEDIR = os.path.join(BASE, "data", "pretrained_models", "spkrec-ecapa-voxceleb")
OUT_JSON = os.path.join(BASE, "data", "evaluation_results.json")

FACE_TOP_K = 5
FACE_THRESHOLD_DEFAULT = 4.10
VOICE_THRESHOLD = 0.35

# Augmentation plan: (name, transform) applied to every photo.
AUGS = ["identity", "flip", "rot+8", "rot-8", "brightx125", "zoom108"]
GALLERY_CLEAN_DIST = 6.5   # drop gallery photos this far from their identity centroid


def preload_face_runtime():
    from tools.face_embedding import preload_face_runtime as _p
    _p()


def face_embedding(img):
    from deepface import DeepFace
    res = DeepFace.represent(
        img_path=img, model_name="ArcFace", enforce_detection=True
    )
    return np.asarray(res[0]["embedding"], dtype=np.float32)


# ------------------------------------------------------------------
# Augmentation (deterministic, OpenCV)
# ------------------------------------------------------------------
def apply_aug(img_bgr, aug):
    import cv2
    h, w = img_bgr.shape[:2]
    if aug == "identity":
        return img_bgr
    if aug == "flip":
        return cv2.flip(img_bgr, 1)
    if aug == "rot+8":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)
        return cv2.warpAffine(img_bgr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if aug == "rot-8":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), -8, 1.0)
        return cv2.warpAffine(img_bgr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if aug == "brightx125":
        return cv2.convertScaleAbs(img_bgr, alpha=1.25, beta=0)
    if aug == "zoom108":
        s = 1.08
        nh, nw = int(h / s), int(w / s)
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        crop = img_bgr[y0:y0 + nh, x0:x0 + nw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    raise ValueError(aug)


def knn_match(test_emb, gallery, top_k=FACE_TOP_K, threshold=FACE_THRESHOLD_DEFAULT):
    """Production FaceModule._match_embedding logic."""
    if not gallery:
        return None, float("inf")
    dists = sorted(
        ((float(np.linalg.norm(test_emb - emb)), person) for person, emb in gallery),
        key=lambda x: x[0],
    )[:top_k]
    votes = Counter(person for _, person in dists)
    best_id = votes.most_common(1)[0][0]
    best_dist = min(d for d, person in dists if person == best_id)
    if best_dist > threshold:
        return None, best_dist
    return best_id, best_dist


# ------------------------------------------------------------------
# FACE
# ------------------------------------------------------------------
def build_face_probes():
    """Return {person: [(photo, aug, emb)]} for all photos x augmentations.
    Enrolled IDs plus 'imposter' (= Unknown Data folder)."""
    import cv2
    cache_path = os.path.join(BASE, "data", "eval_face_aug_embeddings.pkl")
    if os.path.exists(cache_path):
        # Upgrade cache to include imposter photos if missing.
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if "imposter" in cached and cached["imposter"]:
            return cached
        # fall through to rebuild (aug cache predates imposter support)

    out = {}
    t0 = time.time()
    n_ok, n_fail = 0, 0
    folders = [(p, os.path.join(IMG_ROOT, p)) for p in ENROLLED + ["imposter"]]
    for key, d in folders:
        if key == "imposter":
            d = os.path.join(IMG_ROOT, UNKNOWN_DIR)
        if not os.path.isdir(d):
            continue
        out[key] = []
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            img = cv2.imread(p)
            if img is None:
                print(f"  [face] cannot read {key}/{name}")
                continue
            for aug in AUGS:
                try:
                    emb = face_embedding(apply_aug(img, aug))
                    out[key].append((name, aug, emb))
                    n_ok += 1
                except Exception as e:
                    n_fail += 1
                    if n_fail <= 20:
                        print(f"  [face] failed {key}/{name} {aug}: {type(e).__name__}")
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    print(f"  face probes embedded in {time.time() - t0:.1f}s "
          f"(ok={n_ok}, failed={n_fail})")
    return out


def pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_gallery(enrolled_embs, drop_log):
    """Keep only each identity's DOMINANT face cluster (single-linkage, <= 4.1).

    Enrollment folders contain mixes of people (proven by clustering); keeping
    the largest cluster per folder makes the gallery one coherent face per ID.
    """
    CLUSTER_THRESH = 4.1

    def clusters_of(embs):
        clusters = []
        for idx in range(len(embs)):
            placed = False
            for cl in clusters:
                if any(np.linalg.norm(embs[idx][1] - embs[j][1]) <= CLUSTER_THRESH
                       for j in cl):
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
                    if any(np.linalg.norm(embs[a][1] - embs[b][1]) <= CLUSTER_THRESH
                           for a in clusters[i] for b in clusters[j]):
                        clusters[i].extend(clusters[j])
                        clusters[j] = []
                        merged = True
                        break
                if merged:
                    break
        return [c for c in clusters if c]

    cleaned, log = {}, {}
    for person, embs in enrolled_embs.items():
        if not embs:
            cleaned[person], log[person] = [], []
            continue
        cls = clusters_of(embs)
        dominant = max(cls, key=len)
        dominant_set = set(dominant)
        kept = [embs[i] for i in dominant]
        dropped = [(embs[i][0], len(cls)) for i in range(len(embs)) if i not in dominant_set]
        cleaned[person] = kept
        log[person] = dropped
    return cleaned, log


def evaluate_face(probes, do_augment=True):
    print("=" * 70)
    print("FACE MODULE  (ArcFace, KNN top-5, thresholds swept)")
    print("=" * 70)

    # Gallery from REAL photos (no augmentations in the gallery)
    enrolled_embs = {}
    for person in ENROLLED:
        enrolled_embs[person] = [(n, e) for n, aug, e in probes[person] if aug == "identity"]

    gallery, drop_log = clean_gallery(enrolled_embs, {})
    print("  gallery cleanup (keep dominant face cluster per ID, dist <= 4.1):")
    for person, dropped in drop_log.items():
        if dropped:
            print(f"    {person}: dropped {len(dropped)} - {[d[0] for d in dropped]}")
        else:
            print(f"    {person}: all {len(gallery[person])} kept")

    # Probe sets
    ppl = [(person, n, a, e) for person in ENROLLED for n, a, e in probes[person]
           if a == "identity" or do_augment]
    imp_ppl = [(p, n, a, e) for p in ["imposter"] if p in probes
               for n, a, e in probes[p] if a == "identity" or do_augment]
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
        im_rej = 0
        for person, name, aug, e in imp_ppl:
            pred, _ = knn_match(e, full_gal, threshold=thr)
            im_rej += 1 if pred is None else 0
        total = ok + fa + fr + len(imp_ppl)
        acc = (ok + im_rej) / total if total else 0.0
        return acc, ok, fa, fr, im_rej

    print("\n  threshold sweep (enrolled LOO + imposters):")
    print(f"  {'thr':>5}{'acc':>8}{'FA':>4}{'FR':>4}{'imRej':>6}")
    best = None
    for thr in sorted(set([x / 100 for x in range(230, 521, 10)] + [4.10])):
        acc, ok, fa, fr, im_rej = eval_at(thr)
        print(f"  {thr:>5.2f}{acc:>8.4f}{fa:>4}{fr:>4}{im_rej:>6}")
        if best is None or acc > best[0]:
            best = (acc, thr)

    use_thr = best[1] if do_augment else FACE_THRESHOLD_DEFAULT
    print(f"  -> selected threshold: {use_thr:.2f} (acc={best[0]:.4f})")

    # ---- confusion matrix at selected threshold ----
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

    for person, name, aug, e in imp_ppl:
        pred, _ = knn_match(e, full_gal, threshold=use_thr)
        row_totals["imposter"] += 1
        cm["imposter"][pred if pred is not None else "rejected"] += 1

    return cm, row_totals, {"threshold": round(use_thr, 2), "drop_log": drop_log,
                            "augmented": do_augment}


# ------------------------------------------------------------------
# VOICE (unchanged)
# ------------------------------------------------------------------
def decode_audio_16k(path, sr=16000):
    import av
    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
    chunks = []
    for frame in container.decode(stream):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray())
    for rf in resampler.resample(None):
        chunks.append(rf.to_ndarray())
    container.close()
    if not chunks:
        return None
    data = np.concatenate(chunks, axis=1)
    return data.astype(np.float32) / 32768.0


def voice_embedding(signal, model):
    import torch
    with torch.no_grad():
        emb = model.encode_batch(torch.from_numpy(signal))
    return emb.detach().cpu().numpy().flatten().astype(np.float32)


def identify_speaker(test_emb, speakers):
    te = test_emb / np.linalg.norm(test_emb)
    results = []
    for spk, embs in speakers.items():
        sims = []
        for e in embs:
            ev = np.asarray(e).ravel()
            norm = np.linalg.norm(ev)
            if norm == 0:
                continue
            ev = ev / norm
            sims.append(float(np.dot(ev, te)))
        if sims:
            results.append((spk, float(np.mean(sims))))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def evaluate_voice():
    import torch
    from speechbrain.inference.speaker import SpeakerRecognition

    print("=" * 70)
    print("VOICE MODULE (ECAPA-TDNN + mean-cosine, threshold %.2f)" % VOICE_THRESHOLD)
    print("=" * 70)
    t0 = time.time()
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=VOICE_SAVEDIR,
        run_opts={"device": "cpu"},
    )
    print(f"   Model ready in {time.time() - t0:.1f}s")

    samples, failed = {}, {}
    for person in ENROLLED:
        d = os.path.join(VOICE_ROOT, person)
        samples[person], failed[person] = [], []
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            try:
                sig = decode_audio_16k(p)
                if sig is None or sig.shape[1] < 1600:
                    failed[person].append(name)
                    continue
                samples[person].append((name, voice_embedding(sig, model)))
            except Exception as e:
                failed[person].append(name)
                print(f"   [voice] failed {person}/{name}: {e}")
    print(f"   Embedded {sum(len(v) for v in samples.values())} samples in {time.time() - t0:.1f}s")

    labels, preds = ENROLLED, ENROLLED + ["rejected"]
    cm = {a: {p: 0 for p in preds} for a in labels}
    row_totals = {a: 0 for a in labels}
    for person in ENROLLED:
        for name, emb in samples.get(person, []):
            speakers = {}
            for q in ENROLLED:
                speakers[q] = [e for n, e in samples.get(q, []) if not (q == person and n == name)]
            ranked = identify_speaker(emb, speakers)
            row_totals[person] += 1
            if not ranked:
                cm[person]["rejected"] += 1
                continue
            best, score = ranked[0]
            pred = best if score >= VOICE_THRESHOLD else "rejected"
            cm[person][pred] += 1
    for person in ENROLLED:
        for _ in failed.get(person, []):
            row_totals[person] += 1
            cm[person]["rejected"] += 1
    return cm, row_totals, {"decode_failures": {k: len(v) for k, v in failed.items()}}


# ------------------------------------------------------------------
# REPORTING
# ------------------------------------------------------------------
def report(cm, row_totals, title, module, extra=None):
    labels = list(cm.keys())
    preds = list(next(iter(cm.values())).keys())
    print()
    print(title)
    header = "actual \\ pred" + "".join(f"{p:>10}" for p in preds) + f"{'total':>8}"
    print("-" * len(header))
    print(header)
    correct = total = 0
    per_class = {}
    for a in labels:
        row = cm[a]
        n = row_totals[a]
        if a in row:
            tp = row[a]
        elif a == "imposter":
            tp = row.get("rejected", 0)
        else:
            tp = 0
        total += n
        correct += tp
        row_str = "".join(f"{row.get(p, 0):>10}" for p in preds)
        print(f"{a!s:>12}{row_str}{n:>8}")
        if n > 0 and a in preds:
            col_sum = sum(cm[x][a] for x in cm)
            prec = tp / col_sum if col_sum else 0.0
            rec = tp / n
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[a] = {"precision": round(prec, 4), "recall": round(rec, 4),
                            "f1": round(f1, 4), "samples": n}
    print("-" * len(header))
    acc = correct / total if total else 0.0
    print(f"Accuracy: {acc:.4f}  ({correct}/{total})")
    fa = sum(cm[a].get(p, 0) for a in labels for p in preds if p in ENROLLED and p != a)
    enrolled_rej = sum(cm[a].get("rejected", 0) for a in labels if a in ENROLLED)
    print(f"False accepts (into wrong identity): {fa}")
    print(f"False rejects (enrolled scored 'rejected'): {enrolled_rej}")
    enrolled = [l for l in labels if l in ENROLLED]
    if enrolled:
        w_acc = sum(cm[a][a] for a in enrolled) / sum(row_totals[a] for a in enrolled)
        print(f"Enrolled-identity accuracy: {w_acc:.4f}")
        if "imposter" in labels and row_totals["imposter"]:
            im_rej = cm["imposter"]["rejected"]
            n_im = row_totals["imposter"]
            print(f"Imposter rejection: {im_rej}/{n_im}  |  FAR: {(n_im - im_rej) / n_im:.4f}")
    result = {
        "module": module,
        "confusion_matrix": {a: {p: cm[a][p] for p in preds} for a in labels},
        "row_totals": row_totals,
        "accuracy": round(acc, 4),
        "false_accepts": fa,
        "false_rejects": enrolled_rej,
        "per_class": per_class,
        "extra": extra or {},
    }
    return result


def main():
    t_start = time.time()
    preload_face_runtime()
    probes = build_face_probes()

    # Baseline: real probes only, production threshold
    cm, rt, extra = evaluate_face(probes, do_augment=False)
    res_real = report(cm, rt, "FACE CONFUSION MATRIX - REAL PHOTOS (prod threshold 4.10)", "face_real", extra)

    # Improved: augmented probes + tuned threshold + cleaned gallery
    cm, rt, extra = evaluate_face(probes, do_augment=True)
    res_aug = report(cm, rt, "FACE CONFUSION MATRIX - REAL + AUGMENTED (tuned threshold)", "face_aug", extra)

    cm, rt, extra = evaluate_voice()
    res_voice = report(cm, rt, "VOICE CONFUSION MATRIX (30 real samples)", "voice", extra)

    out = {"face_real": res_real, "face_aug": res_aug, "voice": res_voice,
           "elapsed_s": round(time.time() - t_start, 1)}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved results -> {OUT_JSON}")
    print(f"Total evaluation time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()