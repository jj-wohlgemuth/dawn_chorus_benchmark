import argparse
from pathlib import Path

import jiwer
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from datasets import Audio, load_dataset
from matplotlib.ticker import FuncFormatter
from whisper_normalizer.english import EnglishTextNormalizer

parser = argparse.ArgumentParser()
parser.add_argument("--hush-dir", type=Path, default=None)
parser.add_argument("--aic-dir", type=Path, default=None)
parser.add_argument("--output", default="wer_comparison.png")
args = parser.parse_args()


def _find_dir(pattern: str, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    matches = sorted(Path(".").glob(pattern))
    return matches[-1] if matches else None


def _hush_label(d: Path) -> str:
    name = d.name
    if "_atten" in name:
        model_part, atten_part = name.split("_atten", 1)
        return f"Hush (atten={atten_part} dB)"
    return f"Hush ({name})"


def _aic_label(d: Path) -> str:
    name = d.name
    if "_el" in name:
        model_part, el_part = name.rsplit("_el", 1)
        model_part = model_part.removeprefix("aic_").replace("_", "-")
        return f"AIC {model_part} EL={el_part}%"
    return f"AIC ({name})"


hush_dir = _find_dir("hush_*/", args.hush_dir)
aic_dir = _find_dir("aic_*/", args.aic_dir)

# Conditions: label -> base dir
CONDITIONS: dict[str, Path] = {"Mix": Path("mix")}
CONDITION_COLORS: dict[str, str] = {"Mix": "#888888"}
if hush_dir:
    label = _hush_label(hush_dir)
    CONDITIONS[label] = hush_dir
    CONDITION_COLORS[label] = "#4E9AF1"
if aic_dir:
    label = _aic_label(aic_dir)
    CONDITIONS[label] = aic_dir
    CONDITION_COLORS[label] = "#2F6DF6"

# STT systems: label -> transcripts subdir
STT_SYSTEMS = [
    ("Whisper distil-medium.en", "transcripts"),
    ("Kyutai STT-2.6B-EN", "transcripts_kyutai"),
]

BRAND = {"bg": "#1A1A1A", "fg": "#F2F2F0"}
HATCH = {"Deletion": "//", "Insertion": "", "Substitution": "\\\\"}

# Load ground truth
print("Loading dataset ground truth...")
dataset = load_dataset("ai-coustics/dawn_chorus_en", split="eval")
for col, feat in dataset.features.items():
    if isinstance(feat, Audio):
        dataset = dataset.cast_column(col, Audio(decode=False))
normalizer = EnglishTextNormalizer()
ground_truth = {row["id"]: normalizer(row["transcript"]) for row in dataset}
print(f"Loaded {len(ground_truth)} ground truth entries.")

# Compute WER: results[condition][stt_label] = {Del, Ins, Sub, WER}
results: dict[str, dict[str, dict]] = {}
for cond_name, base_dir in CONDITIONS.items():
    results[cond_name] = {}
    for stt_label, subdir in STT_SYSTEMS:
        transcript_dir = base_dir / subdir
        refs, hyps = [], []
        for txt_file in sorted(transcript_dir.glob("*.txt")):
            ref = ground_truth.get(txt_file.stem)
            if not ref:
                continue
            hyp = normalizer(txt_file.read_text().strip())
            refs.append(ref)
            hyps.append(hyp)
        print(f"{cond_name} / {stt_label}: computing WER over {len(refs)} files...")
        if not refs:
            print(f"  [WARN] No transcripts found in {transcript_dir} — skipping.")
            continue
        m = jiwer.process_words(refs, hyps)
        n = m.hits + m.substitutions + m.deletions
        results[cond_name][stt_label] = {
            "Deletion":     m.deletions   / n * 100,
            "Insertion":    m.insertions  / n * 100,
            "Substitution": m.substitutions / n * 100,
            "WER": (m.substitutions + m.deletions + m.insertions) / n * 100,
        }
        r = results[cond_name][stt_label]
        print(f"  WER: {r['WER']:.1f}%  Del: {r['Deletion']:.1f}%  "
              f"Ins: {r['Insertion']:.1f}%  Sub: {r['Substitution']:.1f}%")

# --- Plot: grouped bars (condition groups, STT systems side by side) ---
bg, fg = BRAND["bg"], BRAND["fg"]
cond_names = [c for c in CONDITIONS if any(results[c].values())]
stt_labels = [s for s, _ in STT_SYSTEMS]
n_cond = len(cond_names)
n_stt = len(stt_labels)

bar_w = 0.35
group_gap = 0.2
group_w = n_stt * bar_w + group_gap
x_centers = np.arange(n_cond) * group_w

# STT style: Whisper = solid, Kyutai = lighter (alpha overlay)
STT_ALPHA = [1.0, 0.65]

fig, ax = plt.subplots(figsize=(max(9, n_cond * 3.2), 7))
fig.set_facecolor(bg)
ax.set_facecolor(bg)

for stt_idx, (stt_label, stt_alpha) in enumerate(zip(stt_labels, STT_ALPHA)):
    offset = (stt_idx - (n_stt - 1) / 2) * bar_w
    bottoms = np.zeros(n_cond)
    for error_type in ("Deletion", "Insertion", "Substitution"):
        vals = np.array([
            results[c].get(stt_label, {}).get(error_type, 0.0)
            for c in cond_names
        ])
        colors = [CONDITION_COLORS[c] for c in cond_names]
        ax.bar(
            x_centers + offset, vals, bottom=bottoms,
            width=bar_w, color=colors, alpha=stt_alpha,
            edgecolor=bg, hatch=HATCH[error_type], linewidth=0.6,
        )
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 1.5:
                ax.text(x_centers[i] + offset, b + v / 2, f"{v:.1f}",
                        ha="center", va="center", fontsize=9,
                        color="white", zorder=5)
        bottoms += vals

    # Total WER labels above each bar
    for i, c in enumerate(cond_names):
        wer = results[c].get(stt_label, {}).get("WER", 0.0)
        if wer:
            ax.text(x_centers[i] + offset, wer + 0.4, f"{wer:.1f}",
                    ha="center", va="bottom", fontsize=11,
                    color=fg, fontweight="bold", zorder=6)

# X-axis labels: condition name
ax.set_xticks(x_centers)
ax.set_xticklabels(cond_names, fontsize=12, color=fg, fontweight="bold")

# Styling
for spine in ax.spines.values():
    spine.set_visible(False)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("Corpus-level Word Error Rate (%)", fontsize=13, color=fg, labelpad=12)
ax.tick_params(colors=fg)
ax.tick_params(axis="x", length=0, pad=10)
plt.yticks(color=fg, fontsize=12)
ax.grid(axis="y", linestyle="dotted", alpha=0.5, color=fg, linewidth=1.2)
ax.axhline(y=0, linestyle="dotted", alpha=0.5, color=fg, linewidth=1.2)
ax.set_title("Dawn Chorus — WER by Condition & STT System",
             fontsize=18, color=fg, pad=16)

# Legend: error types + STT systems
error_handles = [
    mpatches.Patch(facecolor="#555", edgecolor=fg, hatch=HATCH[t], label=f"{t}s")
    for t in ("Deletion", "Insertion", "Substitution")
]
stt_handles = [
    mpatches.Patch(facecolor="#888888", alpha=a, edgecolor=fg, label=s)
    for s, a in zip(stt_labels, STT_ALPHA)
]
leg = ax.legend(
    handles=error_handles + stt_handles,
    loc="upper center", bbox_to_anchor=(0.5, -0.10),
    ncol=len(error_handles) + len(stt_handles),
    fontsize=11, facecolor=bg, edgecolor="none", framealpha=1,
)
for text in leg.get_texts():
    text.set_color(fg)

plt.tight_layout()
plt.subplots_adjust(bottom=0.16)
out_path = Path(args.output)
plt.savefig(out_path, facecolor=bg, bbox_inches="tight")
print(f"Plot saved to {out_path}")
