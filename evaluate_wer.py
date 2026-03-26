import argparse
from pathlib import Path

import jiwer
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from datasets import Audio, load_dataset
from matplotlib.ticker import FuncFormatter
from whisper_normalizer.english import EnglishTextNormalizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tiny.en", help="Whisper model used for transcription")
parser.add_argument("--hush-dir", type=Path, default=None,
                    help="Hush output base dir (default: auto-detected by hush_* glob)")
parser.add_argument("--aic-dir", type=Path, default=None,
                    help="AIC output base dir (default: auto-detected by aic_* glob)")
args = parser.parse_args()


def _find_dir(pattern: str, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    matches = sorted(Path(".").glob(pattern))
    return matches[-1] if matches else None


def _hush_label(d: Path) -> str:
    """'advanced_dfnet16k_model_best_onnx · atten=35 dB' from dir name."""
    name = d.name  # e.g. hush_advanced_dfnet16k_model_best_onnx_atten35
    if "_atten" in name:
        model_part, atten_part = name.split("_atten", 1)
        model_part = model_part.removeprefix("hush_")
        return f"Hush\n{model_part}\natten={atten_part} dB"
    return f"Hush\n{name}"


def _aic_label(d: Path) -> str:
    """'quail-vf-2.0-l-16khz · EL=80%' from dir name."""
    name = d.name  # e.g. aic_quail_vf_2_0_l_16khz_el80
    if "_el" in name:
        model_part, el_part = name.rsplit("_el", 1)
        model_part = model_part.removeprefix("aic_").replace("_", "-")
        return f"AIC\n{model_part}\nEL={el_part}%"
    return f"AIC\n{name}"


hush_dir = _find_dir("hush_*/", args.hush_dir)
aic_dir = _find_dir("aic_*/", args.aic_dir)

# --- Experiments: label -> base directory ---
EXPERIMENTS: dict[str, Path] = {"Mix": Path("mix")}
EXPERIMENT_COLORS: dict[str, str] = {"Mix": "#929292"}

if hush_dir:
    label = _hush_label(hush_dir)
    EXPERIMENTS[label] = hush_dir
    EXPERIMENT_COLORS[label] = "#4E9AF1"

if aic_dir:
    label = _aic_label(aic_dir)
    EXPERIMENTS[label] = aic_dir
    EXPERIMENT_COLORS[label] = "#2F6DF6"

# --- Brand Styling (matching plot_aesthetic) ---
BRAND_COLORS = {"off_black": "#1A1A1A", "off_white": "#F2F2F0"}
HATCH_PATTERNS = {"Deletion": "//", "Insertion": "", "Substitution": "\\\\"}

# --- Load ground truth ---
print("Loading dataset ground truth...")
dataset = load_dataset("ai-coustics/dawn_chorus_en", split="eval")
for col, feat in dataset.features.items():
    if isinstance(feat, Audio):
        dataset = dataset.cast_column(col, Audio(decode=False))

normalizer = EnglishTextNormalizer()
ground_truth = {row["id"]: normalizer(row["transcript"]) for row in dataset}
print(f"Loaded {len(ground_truth)} ground truth entries.")

# --- Compute WER per experiment ---
results = {}
for name, base_dir in EXPERIMENTS.items():
    transcript_dir = base_dir / "transcripts"
    refs, hyps = [], []
    for txt_file in sorted(transcript_dir.glob("*.txt")):
        file_id = txt_file.stem
        ref = ground_truth.get(file_id)
        if not ref:
            continue
        hyp = normalizer(txt_file.read_text().strip())
        refs.append(ref)
        hyps.append(hyp)

    print(f"{name}: computing WER over {len(refs)} files...")
    if not refs:
        print(f"  [WARN] No transcripts found in {transcript_dir} — skipping.")
        continue
    measures = jiwer.process_words(refs, hyps)
    n = measures.hits + measures.substitutions + measures.deletions
    results[name] = {
        "Deletion": measures.deletions / n * 100,
        "Insertion": measures.insertions / n * 100,
        "Substitution": measures.substitutions / n * 100,
        "WER": (measures.substitutions + measures.deletions + measures.insertions)
        / n
        * 100,
    }
    print(
        f"  WER: {results[name]['WER']:.1f}%  "
        f"Del: {results[name]['Deletion']:.1f}%  "
        f"Ins: {results[name]['Insertion']:.1f}%  "
        f"Sub: {results[name]['Substitution']:.1f}%"
    )

# --- Plot ---
bg = BRAND_COLORS["off_black"]
fg = BRAND_COLORS["off_white"]

fig, ax = plt.subplots(figsize=(8, 7))
fig.set_facecolor(bg)
ax.set_facecolor(bg)

bar_width = 0.5
x_positions = list(range(len(results)))
experiment_names = list(results.keys())

bottoms = [0.0] * len(x_positions)
for error_type in ("Deletion", "Insertion", "Substitution"):
    vals = [results[name][error_type] for name in experiment_names]
    colors = [EXPERIMENT_COLORS[name] for name in experiment_names]
    ax.bar(
        x_positions,
        vals,
        bottom=bottoms,
        width=bar_width,
        color=colors,
        edgecolor=bg,
        hatch=HATCH_PATTERNS[error_type],
        linewidth=0.6,
    )
    # In-bar labels
    for i, (x, v, b) in enumerate(zip(x_positions, vals, bottoms)):
        if v > 1.0:
            ax.text(
                x,
                b + v / 2,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                zorder=5,
            )
    bottoms = [b + v for b, v in zip(bottoms, vals)]

# Total WER labels above bars
for i, (x, name) in enumerate(zip(x_positions, experiment_names)):
    total = results[name]["WER"]
    ax.text(
        x,
        total + 0.3,
        f"{total:.1f}",
        ha="center",
        va="bottom",
        fontsize=13,
        color=fg,
        fontweight="bold",
        zorder=6,
    )

# Styling
for spine in ax.spines.values():
    spine.set_visible(False)
ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}%"))
ax.set_ylabel("Corpus-level Word Error Rate (%)", fontsize=14, color=fg, labelpad=12)
ax.tick_params(colors=fg)
ax.tick_params(axis="x", length=0, pad=12)
ax.set_xticks(x_positions)
ax.set_xticklabels(experiment_names, fontsize=11, color=fg, fontweight="bold")
plt.yticks(color=fg, fontsize=13)
ax.grid(axis="y", linestyle="dotted", alpha=0.5, color=fg, linewidth=1.2)
ax.axhline(y=0, linestyle="dotted", alpha=0.5, color=fg, linewidth=1.2)
ax.set_title(
    f"Dawn Chorus — WER by Enhancement ({args.model})", fontsize=20, color=fg, pad=16
)

# Legend
legend_handles = [
    mpatches.Patch(
        facecolor="#555555", edgecolor=fg, hatch=HATCH_PATTERNS[t], label=f"{t}s"
    )
    for t in ("Deletion", "Insertion", "Substitution")
]
leg = ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    fontsize=12,
    facecolor=bg,
    edgecolor="none",
    framealpha=1,
)
for text in leg.get_texts():
    text.set_color(fg)

plt.tight_layout()
plt.subplots_adjust(bottom=0.14)
out_path = Path("wer_comparison.png")
plt.savefig(out_path, facecolor=bg, bbox_inches="tight")
print(f"Plot saved to {out_path}")
