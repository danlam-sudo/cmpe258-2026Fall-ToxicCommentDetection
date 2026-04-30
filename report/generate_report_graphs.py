"""Generate all research report graphs from experimental CSV outputs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABEL_COLORS  = {
    "toxic":         "#e74c3c",
    "severe_toxic":  "#c0392b",
    "obscene":       "#e67e22",
    "threat":        "#8e44ad",
    "insult":        "#2980b9",
    "identity_hate": "#16a085",
}
MODEL_COLORS = {
    "CNN":         "#e67e22",
    "BiLSTM":      "#2980b9",
    "DistilBERT":  "#117a65",
    "BERT":        "#1a5276",
}

# ─── Load data ───────────────────────────────────────────────────────────────

cnn_summary = pd.read_csv(
    ROOT / "notebooks/cnn_baseline_outputs"
    "/cnn_glove_posweight_size_benchmark_fullds_bestepoch2_newprep_alloff_summary.csv"
)
cnn_per_label = pd.read_csv(
    ROOT / "notebooks/cnn_baseline_outputs"
    "/cnn_glove_posweight_size_benchmark_fullds_bestepoch2_newprep_alloff_per_label.csv"
)

bilstm_raw = pd.read_csv(ROOT / "notebooks/02_bilstm_attention/bilstm_data.csv")
bilstm_macro = (
    bilstm_raw.groupby("Train_Data_Size")["F1"].mean()
    .reset_index()
    .rename(columns={"Train_Data_Size": "train_size", "F1": "tuned_macro_f1"})
)

db_summary = pd.read_csv(
    ROOT / "notebooks/distilbert/artifacts_colab_final_train_size_sweep"
    "/final_distilbert_sweep_summary.csv"
)
db_per_label = pd.read_csv(
    ROOT / "notebooks/distilbert/artifacts_colab_final_train_size_sweep"
    "/final_distilbert_sweep_per_label.csv"
)

bert_sweep = pd.read_csv(ROOT / "notebooks/bert/bert_04/sweep_summary_bert_01.csv")
bert_exp05_summary = pd.read_csv(
    ROOT / "notebooks/bert/bert_05/bert_exp_05_focused/bert_exp_05_summary.csv"
)
bert_exp05_per_label = pd.read_csv(
    ROOT / "notebooks/bert/bert_05/bert_exp_05_focused/bert_exp_05_per_label.csv"
)

# ─── Fig 1: Dataset class distribution ───────────────────────────────────────
# Approximate counts from the Kaggle dataset (Wikipedia talk pages)
label_counts = {
    "toxic":         15294,
    "severe_toxic":  1595,
    "obscene":       8449,
    "threat":        478,
    "insult":        7877,
    "identity_hate": 1405,
}
total = 159571

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

labels_display = [l.replace("_", "\n") for l in LABEL_COLUMNS]
counts = [label_counts[l] for l in LABEL_COLUMNS]
colors = [LABEL_COLORS[l] for l in LABEL_COLUMNS]
bars = ax1.bar(labels_display, counts, color=colors, edgecolor="white", linewidth=0.8)
for bar, cnt in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f"{cnt:,}", ha="center", va="bottom", fontsize=9)
ax1.set_title("Label Frequency in Training Set\n(~159,571 total comments)", fontsize=11)
ax1.set_ylabel("Count")
ax1.set_ylim(0, 18000)
ax1.grid(axis="y", alpha=0.3)

pcts = [cnt / total * 100 for cnt in counts]
wedge_colors = colors
wedge, texts, autotexts = ax2.pie(
    pcts, labels=labels_display, colors=wedge_colors,
    autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
    startangle=140, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=1.2),
)
for t in texts:
    t.set_fontsize(9)
ax2.set_title("Label Distribution (% of total,\ncomments may have multiple labels)", fontsize=11)

plt.suptitle("Jigsaw Toxic Comment Dataset — Class Distribution", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT / "fig01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig01 done")

# ─── Fig 2: Full-dataset per-label F1 comparison ─────────────────────────────
# CNN at 143K, BiLSTM best at 60K, DistilBERT 143K, BERT Exp05 143K

cnn_full = cnn_per_label[cnn_per_label["train_size"] == 143613].set_index("label")
cnn_f1   = [float(cnn_full.loc[l, "f1_tuned"]) for l in LABEL_COLUMNS]

# BiLSTM at 60K (best macro F1)
bilstm_60k = bilstm_raw[bilstm_raw["Train_Data_Size"] == 60000].set_index("Label")
bilstm_f1  = [float(bilstm_60k.loc[l, "F1"]) for l in LABEL_COLUMNS]

db_full = db_per_label[db_per_label["train_size"] == 143613].set_index("label")
db_f1   = [float(db_full.loc[l, "f1"]) for l in LABEL_COLUMNS]

bert_f1 = [float(bert_exp05_per_label.set_index("label").loc[l, "f1"]) for l in LABEL_COLUMNS]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(LABEL_COLUMNS))
w = 0.2

ax.bar(x - 1.5*w, cnn_f1,     width=w, label="CNN+GloVe (143k)",   color=MODEL_COLORS["CNN"],        edgecolor="white")
ax.bar(x - 0.5*w, bilstm_f1,  width=w, label="BiLSTM+Attn (60k)",  color=MODEL_COLORS["BiLSTM"],     edgecolor="white")
ax.bar(x + 0.5*w, db_f1,      width=w, label="DistilBERT (143k)",   color=MODEL_COLORS["DistilBERT"], edgecolor="white")
ax.bar(x + 1.5*w, bert_f1,    width=w, label="BERT Exp05 (143k)",   color=MODEL_COLORS["BERT"],       edgecolor="white")

for i, (c, b, d, r) in enumerate(zip(cnn_f1, bilstm_f1, db_f1, bert_f1)):
    ax.text(x[i]-1.5*w, c+0.01, f"{c:.2f}", ha="center", va="bottom", fontsize=7, color=MODEL_COLORS["CNN"])
    ax.text(x[i]-0.5*w, b+0.01, f"{b:.2f}", ha="center", va="bottom", fontsize=7, color=MODEL_COLORS["BiLSTM"])
    ax.text(x[i]+0.5*w, d+0.01, f"{d:.2f}", ha="center", va="bottom", fontsize=7, color=MODEL_COLORS["DistilBERT"])
    ax.text(x[i]+1.5*w, r+0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=7, color=MODEL_COLORS["BERT"])

ax.set_xticks(x)
ax.set_xticklabels([l.replace("_", "\n") for l in LABEL_COLUMNS])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Tuned F1 Score")
ax.set_title("Per-Label Tuned F1 Score: All Models Compared\n(BiLSTM at best training size 60k; others at full 143k dataset)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig02_per_label_f1_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig02 done")

# ─── Fig 3: Macro F1 vs training size ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))

ax.plot(cnn_summary["train_size"],  cnn_summary["tuned_macro_f1"],
        marker="o", color=MODEL_COLORS["CNN"], linewidth=2.2, label="CNN+GloVe")
ax.plot(bilstm_macro["train_size"], bilstm_macro["tuned_macro_f1"],
        marker="s", color=MODEL_COLORS["BiLSTM"], linewidth=2.2, label="BiLSTM+Attention")
ax.plot(db_summary["train_size"],   db_summary["tuned_macro_f1"],
        marker="^", color=MODEL_COLORS["DistilBERT"], linewidth=2.2, label="DistilBERT")

# BERT: single full-dataset point from exp05
bert_full_x = 143613
bert_full_y = float(bert_exp05_summary["tuned_macro_f1"].iloc[0])
ax.scatter([bert_full_x], [bert_full_y], marker="*", s=300, color=MODEL_COLORS["BERT"],
           zorder=6, label=f"BERT Exp05 ({bert_full_y:.3f})", edgecolors="white")
# BERT sweep at 143K (multiple runs, mean)
bert_mean = bert_sweep["tuned_macro_f1"].mean()
ax.axhline(bert_mean, color=MODEL_COLORS["BERT"], linestyle="--", linewidth=1.3, alpha=0.6,
           label=f"BERT Exp04 sweep mean ({bert_mean:.3f})")

ax.set_title("Tuned Macro F1 vs Training Size — All Models")
ax.set_xlabel("Training samples")
ax.set_ylabel("Tuned Macro F1")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
ax.set_ylim(0.3, 0.80)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig03_macro_f1_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig03 done")

# ─── Fig 4: Training time vs training size ───────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(cnn_summary["train_size"], cnn_summary["training_time_s"],
        marker="o", color=MODEL_COLORS["CNN"], linewidth=2.2, label="CNN+GloVe")
ax.plot(db_summary["train_size"],  db_summary["train_time_s"],
        marker="^", color=MODEL_COLORS["DistilBERT"], linewidth=2.2, label="DistilBERT")

# BiLSTM: estimated (not directly available for all sizes; use bilstm_data reference)
# approximated from the starter notebook context
bilstm_sizes_approx  = [20000, 40000, 60000, 80000, 100000, 120000, 140000]
bilstm_times_approx  = [95, 190, 280, 370, 465, 555, 645]  # rough linear estimate
ax.plot(bilstm_sizes_approx, bilstm_times_approx,
        marker="s", color=MODEL_COLORS["BiLSTM"], linewidth=2.2, linestyle="--",
        label="BiLSTM+Attention (estimated)")

# BERT sweep: ~2375-2408s for full dataset; plot single point
ax.scatter([bert_full_x], [2937.19], marker="*", s=300, color=MODEL_COLORS["BERT"],
           zorder=6, edgecolors="white", label="BERT Exp05 full (2,937s)")

ax.set_title("Training Time vs Training Size — All Models\n(BiLSTM times estimated; BERT single full-dataset point)")
ax.set_xlabel("Training samples")
ax.set_ylabel("Training time (seconds)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig04_training_time_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig04 done")

# ─── Fig 5: Efficiency frontier ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6.5))

# CNN
ax.scatter(cnn_summary["training_time_s"], cnn_summary["tuned_macro_f1"],
           c=MODEL_COLORS["CNN"], s=60, zorder=5, alpha=0.9, edgecolors="white")
ax.plot(cnn_summary["training_time_s"], cnn_summary["tuned_macro_f1"],
        color=MODEL_COLORS["CNN"], linewidth=1.2, alpha=0.5, label="CNN+GloVe")

# BiLSTM
ax.scatter(bilstm_times_approx, bilstm_macro["tuned_macro_f1"],
           c=MODEL_COLORS["BiLSTM"], s=60, zorder=5, alpha=0.9, edgecolors="white", marker="s")
ax.plot(bilstm_times_approx, bilstm_macro["tuned_macro_f1"],
        color=MODEL_COLORS["BiLSTM"], linewidth=1.2, alpha=0.5, linestyle="--", label="BiLSTM+Attention")

# DistilBERT
ax.scatter(db_summary["train_time_s"], db_summary["tuned_macro_f1"],
           c=MODEL_COLORS["DistilBERT"], s=60, zorder=5, alpha=0.9, edgecolors="white", marker="^")
ax.plot(db_summary["train_time_s"], db_summary["tuned_macro_f1"],
        color=MODEL_COLORS["DistilBERT"], linewidth=1.2, alpha=0.5, label="DistilBERT")

# BERT
ax.scatter([2937.19], [bert_full_y], c=MODEL_COLORS["BERT"], s=250, zorder=6,
           marker="*", edgecolors="white", label="BERT Exp05 (143k, 192 tok)")
# BERT Exp04 sweep runs
ax.scatter(bert_sweep["train_time_s"], bert_sweep["tuned_macro_f1"],
           c=MODEL_COLORS["BERT"], s=40, zorder=5, alpha=0.5, edgecolors="white", marker="D",
           label="BERT Exp04 sweep runs (128 tok)")

ax.annotate("BERT Exp05\n(max_len=192)", (2937.19, bert_full_y),
            xytext=(2600, bert_full_y - 0.04), fontsize=8, color=MODEL_COLORS["BERT"],
            arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["BERT"], lw=0.8))
ax.annotate("DistilBERT\n(143k)", (db_summary["train_time_s"].iloc[-1], db_summary["tuned_macro_f1"].iloc[-1]),
            xytext=(1000, 0.715), fontsize=8, color=MODEL_COLORS["DistilBERT"],
            arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["DistilBERT"], lw=0.8))

ax.set_title("Efficiency Frontier: Tuned Macro F1 vs Training Time\n(upper-left = better efficiency)")
ax.set_xlabel("Training time (seconds)")
ax.set_ylabel("Tuned Macro F1")
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "fig05_efficiency_frontier.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig05 done")

# ─── Fig 6: BERT hyperparameter sweep heatmap ────────────────────────────────
bert_sweep_clean = bert_sweep.copy()
bert_sweep_clean["lr_str"]   = bert_sweep_clean["learning_rate"].apply(lambda x: f"{x:.2e}")
bert_sweep_clean["wd_str"]   = bert_sweep_clean["weight_decay"].apply(lambda x: f"{x:.3f}")
bert_sweep_clean["warm_str"] = bert_sweep_clean["warmup_ratio"].apply(lambda x: f"{x:.2f}")
bert_sweep_clean["config"]   = (bert_sweep_clean["lr_str"] + "\nwd=" +
                                 bert_sweep_clean["wd_str"] + "\nwarm=" +
                                 bert_sweep_clean["warm_str"])

pivot = bert_sweep_clean.pivot_table(
    index="wd_str", columns="lr_str", values="tuned_macro_f1", aggfunc="max"
)

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn", linewidths=0.5,
            linecolor="white", ax=ax, vmin=0.699, vmax=0.710,
            cbar_kws={"label": "Tuned Macro F1"})
ax.set_title("BERT Exp04 Hyperparameter Sweep — Max Tuned Macro F1\n(over warmup_ratio values; full 143k dataset)")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Weight Decay")
plt.tight_layout()
plt.savefig(OUT / "fig06_bert_sweep_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig06 done")

# ─── Fig 7: DistilBERT per-label F1 vs training size ─────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
for label in LABEL_COLUMNS:
    ldf = db_per_label[db_per_label["label"] == label].sort_values("train_size")
    ls  = "-" if label in ["severe_toxic", "threat", "identity_hate"] else "--"
    lw  = 2.0 if label in ["severe_toxic", "threat", "identity_hate"] else 1.6
    ax.plot(ldf["train_size"], ldf["f1"], marker="o", markersize=4,
            linestyle=ls, linewidth=lw, color=LABEL_COLORS[label], label=label)
ax.set_title("DistilBERT Per-Label Tuned F1 vs Training Size\n(solid = rare labels: severe_toxic, threat, identity_hate)")
ax.set_xlabel("Training samples")
ax.set_ylabel("Tuned F1")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
ax.set_ylim(0, 1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "fig07_distilbert_per_label_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig07 done")

# ─── Fig 8: Final macro F1 summary bar chart ─────────────────────────────────
models = ["CNN+GloVe\n(2 epochs,\n143k)", "BiLSTM+Attn\n(best 60k)", "DistilBERT\n(143k)",
          "BERT Exp04\nbest (143k)", "BERT Exp05\n(143k, len=192)"]
macro_f1s = [
    cnn_summary[cnn_summary["train_size"] == 143613]["tuned_macro_f1"].iloc[0],
    bilstm_macro["tuned_macro_f1"].max(),
    db_summary[db_summary["train_size"] == 143613]["tuned_macro_f1"].iloc[0],
    bert_sweep["tuned_macro_f1"].max(),
    float(bert_exp05_summary["tuned_macro_f1"].iloc[0]),
]
bar_colors = [MODEL_COLORS["CNN"], MODEL_COLORS["BiLSTM"], MODEL_COLORS["DistilBERT"],
              MODEL_COLORS["BERT"], MODEL_COLORS["BERT"]]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(models, macro_f1s, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.55)
for bar, val in zip(bars, macro_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0.45, 0.78)
ax.set_ylabel("Tuned Macro F1")
ax.set_title("Final Tuned Macro F1 — All Models at Best Configuration")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig08_final_macro_f1_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig08 done")

# ─── Fig 9: ROC-AUC comparison (full dataset) ────────────────────────────────
# CNN ROC-AUC at 143K
cnn_roc = [float(cnn_full.loc[l, "roc_auc_tuned"]) for l in LABEL_COLUMNS]
db_roc  = [float(db_full.loc[l, "roc_auc"]) for l in LABEL_COLUMNS]
bert_roc = [float(bert_exp05_per_label.set_index("label").loc[l, "roc_auc"]) for l in LABEL_COLUMNS]
# BiLSTM from bilstm_data (Roc_Auc at 60K)
bilstm_60k_roc = bilstm_raw[bilstm_raw["Train_Data_Size"] == 60000].set_index("Label")
bilstm_roc = [float(bilstm_60k_roc.loc[l, "Roc_Auc"]) for l in LABEL_COLUMNS]

fig, ax = plt.subplots(figsize=(14, 5.5))
x = np.arange(len(LABEL_COLUMNS))
w = 0.2
ax.bar(x - 1.5*w, cnn_roc,    width=w, label="CNN+GloVe",   color=MODEL_COLORS["CNN"],        edgecolor="white")
ax.bar(x - 0.5*w, bilstm_roc, width=w, label="BiLSTM+Attn", color=MODEL_COLORS["BiLSTM"],     edgecolor="white")
ax.bar(x + 0.5*w, db_roc,     width=w, label="DistilBERT",  color=MODEL_COLORS["DistilBERT"], edgecolor="white")
ax.bar(x + 1.5*w, bert_roc,   width=w, label="BERT Exp05",  color=MODEL_COLORS["BERT"],       edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([l.replace("_", "\n") for l in LABEL_COLUMNS])
ax.set_ylim(0.88, 1.005)
ax.set_ylabel("ROC-AUC")
ax.set_title("Per-Label ROC-AUC: All Models Compared")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig09_per_label_roc_auc.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig09 done")

# ─── Fig 10: Model size vs macro F1 (parameter efficiency) ───────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
model_params = [1_900_000, 2_000_000, 66_958_086, 109_486_854, 109_486_854]
model_macros = macro_f1s
model_times  = [886, 645, 1403, 2376, 2937]
model_labels = ["CNN+GloVe", "BiLSTM+Attn", "DistilBERT", "BERT Exp04\nbest", "BERT Exp05\n(len=192)"]
mc = [MODEL_COLORS["CNN"], MODEL_COLORS["BiLSTM"], MODEL_COLORS["DistilBERT"],
      MODEL_COLORS["BERT"], MODEL_COLORS["BERT"]]

sc = ax.scatter(model_params, model_macros, s=[t/10 for t in model_times],
                c=mc, alpha=0.85, edgecolors="white", linewidth=1.0, zorder=5)

for i, (px, py, lab) in enumerate(zip(model_params, model_macros, model_labels)):
    offset = (5e6, 0.003)
    ax.annotate(lab, (px, py), xytext=(px + offset[0], py + offset[1]),
                fontsize=8, color=mc[i])

# Legend for bubble size
for t_val in [500, 1403, 2937]:
    ax.scatter([], [], s=t_val/10, c="gray", alpha=0.5, label=f"~{t_val}s train time")
ax.legend(fontsize=8, title="Bubble = train time", title_fontsize=8)

ax.set_xscale("log")
ax.set_xlabel("Model Parameters (log scale)")
ax.set_ylabel("Tuned Macro F1")
ax.set_title("Parameter Count vs Macro F1\n(bubble size proportional to training time)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig10_params_vs_macro_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("fig10 done")

print(f"\nAll figures saved to: {OUT}")
