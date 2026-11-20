# plot_confusion.py - renders confusion-matrix heatmaps from data/evaluation_results.json
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, "data", "evaluation_results.json")) as f:
    data = json.load(f)

ORDER = {"2217010": 0, "2317009": 1, "2317046": 2, "rejected": 3, "imposter": 4}


def plot(module_key, title, fname):
    mod = data[module_key]
    cm = mod["confusion_matrix"]
    rows = list(cm.keys())
    cols = list(cm[rows[0]].keys())
    rc = [r for r in rows if r in ORDER]
    cc = [c for c in cols if c in ORDER]
    M = np.zeros((len(rc), len(cc)), dtype=int)
    for i, r in enumerate(rc):
        for j, c in enumerate(cc):
            M[i, j] = cm[r].get(c, 0)

    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(cc)))
    ax.set_yticks(range(len(rc)))
    ax.set_xticklabels(cc, rotation=30, ha="right")
    ax.set_yticklabels(rc)
    for i in range(len(rc)):
        for j in range(len(cc)):
            v = M[i, j]
            color = "white" if v > 0.5 * M.max(initial=1) else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title + f"  |  accuracy {mod['accuracy']:.4f}  |  n = {sum(mod['row_totals'].values())}")
    fig.colorbar(im)
    fig.tight_layout()
    out = os.path.join(BASE, "data", fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if "face_aug" in data:
    plot("face_real", "Face - real photos (LOO, prod thr 4.10)", "face_confusion_real.png")
    plot("face_aug", "Face - real + augmented (LOO)", "face_confusion_augmented.png")
else:
    plot("face", "Face confusion matrix (LOO)", "face_confusion_matrix.png")
plot("voice", "Voice confusion matrix (LOO, thr 0.35)", "voice_confusion_matrix.png")

# relabeled dataset comparison (from tools/evaluate_relabeled.py)
rel = os.path.join(BASE, "data", "evaluation_results_relabeled.json")
if os.path.exists(rel):
    with open(rel) as f:
        rd = json.load(f)
    for key, fname, label in [
        ("before", "face_before_relabel.png", "Face BEFORE - original folders"),
        ("after", "face_after_relabel.png", "Face AFTER - auto-relabeled"),
    ]:
        mod = rd[key]
        cm = mod["confusion_matrix"]
        rows = list(cm.keys())
        cols = list(cm[rows[0]].keys())
        rc = [r for r in rows if r in ORDER]
        cc = [c for c in cols if c in ORDER]
        M = np.zeros((len(rc), len(cc)), dtype=int)
        for i, r in enumerate(rc):
            for j, c in enumerate(cc):
                M[i, j] = cm[r].get(c, 0)
        fig, ax = plt.subplots(figsize=(7, 5.6))
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(len(cc)))
        ax.set_yticks(range(len(rc)))
        ax.set_xticklabels(cc, rotation=30, ha="right")
        ax.set_yticklabels(rc)
        for i in range(len(rc)):
            for j in range(len(cc)):
                v = M[i, j]
                color = "white" if v > 0.5 * M.max(initial=1) else "black"
                ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(label + f"  |  accuracy {mod['accuracy']:.4f}  |  n = {sum(mod['row_totals'].values())}")
        fig.colorbar(im)
        fig.tight_layout()
        out = os.path.join(BASE, "data", fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")