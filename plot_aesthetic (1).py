import time

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
import re
from pathlib import Path

# --- Custom Font ---
_FONT_PATH = "/Users/theo/Library/Fonts/205TF-Milling-Duplex1mm.otf"
fm.fontManager.addfont(_FONT_PATH)
_FONT_PROP = fm.FontProperties(fname=_FONT_PATH)
_FONT_NAME = _FONT_PROP.get_name()
plt.rcParams["font.family"] = _FONT_NAME

# --- Brand Styling ---
BRAND_COLORS = {
    "off_black": "#1A1A1A",
    "off_white": "#F2F2F0",
}
ENHANCEMENT_PALETTE = [
    "#4E9AF1",
    "#F1C84E",
    "#4EF19A",
    "#F14E6E",
    "#B44EF1",
    "#F1944E",
    "#4EE8F1",
    "#F14EE0",
]
# Per-prefix color overrides (raw key -> hex). Falls back to ENHANCEMENT_PALETTE.
PREFIX_COLOR_OVERRIDES = {
    "mix": "#373737",  # dark gray
    "KRISP_BVCTelephony": "#575757",  # lighter gray
    "KRISP_BVC_Telephony": "#575757",  # lighter gray
    "KRISP_BVC": "#787878",  # lighter gray
    "KRISP_BVCTelephonyNew": "#787878",  # lighter gray
    "quail_vf_1_1_l_16khz_d00ghjzn_v15_el_100": "#787878",  # blue
    "quail_vf_2_0_l_16khz_0_15_0_rc_3_d42jls1e_v18_el_81": "#5F7CBC",  # blue
    "quail_vf_2_0_l_16khz_0_15_0_rc_3_d42jls1e_v18_el_100": "#2F6DF6",  # lighter blue
    "quail_vf_2_0_sdk_el_80": "#2F6DF6",  # blue
    "quail_vf_2_0_sdk_el_100": "#2F6DF6",  # Google Blue
    "speech": "#929292",  # light gray
}
# Hatch patterns: bottom→top stacking order is Deletions / Insertions / Substitutions
HATCH_PATTERNS = {"Deletion": "//", "Insertion": "", "Substitution": "\\\\"}

dark_mode = True

# --- Configuration: User Inputs ---
prefix_names_raw = [
    # "mix",
    # "quail-l-16khz_el_100",
    # "quail-vf-2.0-l-16khz_el_50",
    # "quail-vf-2.0-l-16khz_el_80",
    # "sparrow-l-16khz_el_100",
    # "KRISP_BVCTelephonyNew",
    # "quail_vf_stt_l16_v1",
    # "KRISP_BVC_Telephony",
    # "KRISP_BVC",
    # "quail_vf_1_1_l_16khz_d00ghjzn_v15_el_100",
    # "quail_vf_2_0_l_16khz_0_15_0_rc_3_d42jls1e_v18_el_81",
    # "quail_vf_2_0_sdk_el_80",
    # "quail_vf_2_0_sdk_el_100",
    # "quail-vf-2.0-l-16khz_el_100",
    # "speech",
    # "quail_vf_2_0_l_16khz_0_15_0_rc_3_d42jls1e_v18_el_100",
    "mix_pad2000ms_48kHz_livekit_first_100",
    "mix_pad2000ms_16kHz_livekit_first_100",
    "mix_bvc_pad2000ms_48kHz_livekit_first_100",
    "mix_bvc_pad2000ms_16kHz_livekit_first_100",
    "mix_aic_quail_vfl_el_80_pad2000ms_48kHz_livekit_first_100",
    "mix_aic_quail_vfl_el_80_pad2000ms_16kHz_livekit_first_100",
]

# Aliases for display (prefix_name -> label)
prefix_name_aliases = {
    "mix": "Raw",
    "quail-l-16khz_el_100": "Quail L 16kHz EL 100",
    "quail-vf-2.0-l-16khz_el_100": "Quail VF 2.0 L 16kHz EL 100",
    "quail-vf-2.0-l-16khz_el_80": "Quail VF 2.0 L 16kHz EL 80",
    "quail-vf-2.0-l-16khz_el_50": "Quail VF 2.0 L 16kHz EL 50",
    "sparrow-l-16khz_el_100": "Sparrow L 16kHz EL 100",
    "KRISP_BVCTelephonyNew": "Krisp BVC Telephony",
    "KRISP_BVCTelephony": "Krisp BVC Telephony",
    "KRISP_BVC_Telephony": "Krisp BVC Telephony",
    "KRISP_BVC": "Krisp BVC",
    "quail_vf_1_1_l_16khz_d00ghjzn_v15_el_100": "Quail VF 1.1",
    "quail_vf_2_0_l_16khz_0_15_0_rc_3_d42jls1e_v18_el_81": "Quail VF 2.0",
    "quail_vf_2_0_l_16khz_0_15_0_rc_3_d42jls1e_v18_el_100": "Quail VF 2.0",
    "quail_vf_2_0_sdk_el_100": "Quail VF 2.0",
    "quail_vf_2_0_sdk_el_80": "Quail VF 2.0",
    "mix_pad2000ms_48kHz_livekit_first_100": "Livekit Mix 48kHz",
    "mix_pad2000ms_16kHz_livekit_first_100": "Livekit Mix 16kHz",
    "mix_bvc_pad2000ms_48kHz_livekit_first_100": "Livekit Mix BVC 48kHz",
    "mix_bvc_pad2000ms_16kHz_livekit_first_100": "Livekit Mix BVC 16kHz",
    "mix_aic_quail_vfl_el_80_pad2000ms_48kHz_livekit_first_100": "Livekit AIC Quail VF L 48kHz",
    "mix_aic_quail_vfl_el_80_pad2000ms_16kHz_livekit_first_100": "Livekit AIC Quail VF L 16kHz",
    "speech": "Clean Speech",
    # Add more as needed
}

stt_models_aliases = {
    "deepgram": "Deepgram Nova 3",
    "soniox": "Soniox STT Async v4",
    "cartesia": "Cartesia Ink Whisper",
    "assemblyai": "AssemblyAI Universal-2",
    "gladia": "Gladia",
    "speechmatics": "Speechmatics",
    "mistral": "Mistral Voxtral Mini",
    "deepgram_rt": "Deepgram Nova 3 Live",
    "soniox_rt": "Soniox STT Async v4 Live",
    "cartesia_rt": "Cartesia Ink Whisper Live",
    "assemblyai_rt": "AssemblyAI Universal-2 Live",
    "gladia_rt": "Gladia Live",
    "speechmatics_rt": "Speechmatics Live",
    "mistral_rt": "Mistral Voxtral Mini Live",
}

allowed_asrs = [
    # "elevenlabs",
    # "assemblyai",
    # "cartesia",
    "deepgram",
    # "gladia",
    # "mistral",
    # "soniox",
    # "speechmatics",
    # "assemblyai_rt",
    # "cartesia_rt",
    # "deepgram_rt",
    # "gladia_rt",
    # "mistral_rt",
    # "soniox_rt",
    # "speechmatics_rt",
]

scores_csv_root = Path(
    "/Users/theo/ai-coustics/Data/evaluation/dawn_chorus_en_v1_16khz"
    # "/Users/theo/ai-coustics/Data/evaluation/switchboard_multispeaker_200_normalized"
    # "/Users/theo/ai-coustics/Data/evaluation/switchboard_multispeaker_200_normalized_new"
    # "/Users/theo/ai-coustics/Data/evaluation/leo_butch_voice_focus_aic_en_vad_selected_normalized_16khz"
)
plot_name = scores_csv_root.name


def _sanitize_column_key(key: str) -> str:
    key = str(key).strip()
    key = re.sub(r"[^\w]+", "_", key)
    key = re.sub(r"_+", "_", key)
    return key.strip("_")


def _aggregate_pct(rates: pd.Series | None, ref_len: pd.Series) -> float:
    """Weighted aggregate: sum(rate_i * ref_len_i) / sum(ref_len_i) * 100."""
    if rates is None:
        return 0.0
    r = pd.to_numeric(rates, errors="coerce")
    w = pd.to_numeric(ref_len, errors="coerce")
    mask = r.notna() & w.notna() & (w > 0)
    total = w[mask].sum()
    return float((r[mask] * w[mask]).sum() / total * 100.0) if total > 0 else 0.0


def _parse_percent_str(val) -> float:
    """Convert '5.3%' or a float to a plain float."""
    if pd.isna(val):
        return 0.0
    try:
        return float(str(val).strip().rstrip("%"))
    except ValueError:
        return 0.0


def _resolve_theme(dark: bool) -> dict:
    """Return bg/fg/edge colors for the chosen mode."""
    if dark:
        return {
            "bg": BRAND_COLORS["off_black"],
            "fg": BRAND_COLORS["off_white"],
            "edge": BRAND_COLORS["off_black"],
        }
    return {
        "bg": BRAND_COLORS["off_white"],
        "fg": BRAND_COLORS["off_black"],
        "edge": "#2E3440",
    }


def _apply_ax_style(fig, ax, theme: dict, title: str) -> None:
    """Apply brand styling to figure and axes."""
    bg, fg = theme["bg"], theme["fg"]
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}%"))
    ax.set_ylabel("Word Error Rate (%)", fontsize=16, color=fg, labelpad=12)
    ax.tick_params(colors=fg)
    ax.tick_params(axis="x", length=0, pad=12)
    plt.yticks(color=fg, fontsize=16)
    ax.grid(axis="y", linestyle="dotted", alpha=0.5, color=fg, linewidth=1.2)
    ax.axhline(y=0, linestyle="dotted", alpha=0.5, color=fg, linewidth=1.2)
    ax.set_title(title, fontsize=26, color=fg, pad=20)


def _draw_stacked_bars(
    ax, x_positions, bar_values_by_type, bar_colors, bar_width, edge_color
):
    """Draw stacked bars bottom→top in Deletion / Insertion / Substitution order."""
    bottoms = [0.0] * len(x_positions)
    for error_type in ("Deletion", "Insertion", "Substitution"):
        vals = bar_values_by_type[error_type]
        ax.bar(
            x_positions,
            vals,
            bottom=bottoms,
            width=bar_width,
            color=bar_colors,
            edgecolor=edge_color,
            hatch=HATCH_PATTERNS[error_type],
            linewidth=0.6,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]


def _add_inbar_labels(
    ax, x_positions, bar_values_by_type, label_color="white", min_pct=1.0
):
    """Draw percentage labels inside each error-type segment."""
    for i, x in enumerate(x_positions):
        bottom = 0.0
        for error_type in ("Deletion", "Insertion", "Substitution"):
            v = bar_values_by_type[error_type][i]
            if v > min_pct:
                ax.text(
                    x,
                    bottom + v / 2,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=label_color,
                    zorder=5,
                )
            bottom += v


def _add_total_labels(ax, x_positions, bar_values_by_type, label_color, fontsize=12):
    """Draw total WER percentage above each stacked bar."""
    for i, x in enumerate(x_positions):
        total = sum(
            bar_values_by_type[t][i] for t in ("Deletion", "Insertion", "Substitution")
        )
        ax.text(
            x,
            total + 0.3,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=label_color,
            fontweight="bold",
            zorder=6,
        )


# --- Main Processing ---
try:
    leaderboard_path = scores_csv_root / f"leaderboard_{plot_name}.csv"
    leaderboard_df = pd.read_csv(leaderboard_path, index_col=0)

    parsed_data = []
    for experiment in prefix_names_raw:
        if experiment not in leaderboard_df.index:
            print(f"Warning: {experiment} not found in leaderboard.")
            continue
        row = leaderboard_df.loc[experiment]
        for asr in allowed_asrs:
            col_key = _sanitize_column_key(asr)
            base = f"WER_{col_key}"
            if base not in row:
                continue
            parsed_data.append(
                {
                    "Experiment": experiment,
                    "ASR": asr,
                    "Score": float(row[base]) * 100.0,
                    "Insertions": float(row.get(f"{base}_insertions", 0)) * 100.0,
                    "Deletions": float(row.get(f"{base}_deletions", 0)) * 100.0,
                    "Substitutions": float(row.get(f"{base}_substitutions", 0)) * 100.0,
                }
            )

    df_filtered = pd.DataFrame(parsed_data)

    if not df_filtered.empty:
        experiment_means = (
            df_filtered.groupby("Experiment")[["Score"]].mean().reset_index()
        )
        experiment_means_sorted = experiment_means.sort_values("Score", ascending=True)
        ordered_experiments = experiment_means_sorted["Experiment"].tolist()

        theme = _resolve_theme(dark_mode)
        bg, fg = theme["bg"], theme["fg"]

        # Experiments in config order, filtered to those present in data
        experiments_present = [
            e for e in prefix_names_raw if e in df_filtered["Experiment"].values
        ]
        enhancement_colors = {
            exp: PREFIX_COLOR_OVERRIDES.get(
                exp, ENHANCEMENT_PALETTE[i % len(ENHANCEMENT_PALETTE)]
            )
            for i, exp in enumerate(experiments_present)
        }
        enhancement_labels = {
            exp: prefix_name_aliases.get(exp, exp) for exp in experiments_present
        }

        # One row per (ASR subgroup, enhancement bar)
        plot_rows = []
        for asr in allowed_asrs:
            asr_data = df_filtered[df_filtered["ASR"] == asr]
            for exp in experiments_present:
                row_data = asr_data[asr_data["Experiment"] == exp]
                if row_data.empty:
                    continue
                plot_rows.append(
                    {
                        "ASR": stt_models_aliases[asr],
                        "Experiment": exp,
                        "Deletion": row_data["Deletions"].values[0],
                        "Insertion": row_data["Insertions"].values[0],
                        "Substitution": row_data["Substitutions"].values[0],
                    }
                )

        # Assign x positions; group ASRs with gaps between subgroups
        bar_width, group_gap = 0.8, 1.2
        x_positions, bar_colors_list = [], []
        group_centers, group_labels = [], []
        current_x, group_start_x, last_asr = 0.0, 0.0, None

        for row_d in plot_rows:
            asr = row_d["ASR"]
            if last_asr is not None and asr != last_asr:
                group_centers.append((group_start_x + current_x - 1.0) / 2.0)
                group_labels.append(last_asr)
                current_x += group_gap
                group_start_x = current_x
            x_positions.append(current_x)
            bar_colors_list.append(enhancement_colors[row_d["Experiment"]])
            last_asr = asr
            current_x += 1.0

        if last_asr is not None:
            group_centers.append((group_start_x + current_x - 1.0) / 2.0)
            group_labels.append(last_asr)

        bar_values = {
            t: [r[t] for r in plot_rows]
            for t in ("Deletion", "Insertion", "Substitution")
        }

        # --- Draw ---
        fig, ax = plt.subplots(figsize=(12, 10))
        _draw_stacked_bars(
            ax, x_positions, bar_values, bar_colors_list, bar_width, edge_color=bg
        )
        _add_inbar_labels(ax, x_positions, bar_values)
        _add_total_labels(ax, x_positions, bar_values, label_color=fg)

        # X-axis: centered STT group labels
        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            group_labels,
            rotation=0,
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=fg,
        )

        # Legend: enhancement color patches + error-type hatch patches
        legend_handles = [
            mpatches.Patch(
                facecolor=enhancement_colors[exp],
                edgecolor="none",
                label=enhancement_labels.get(exp, exp),
            )
            for exp in experiments_present
        ] + [
            mpatches.Patch(
                facecolor="#555555",
                edgecolor=fg,
                hatch=HATCH_PATTERNS[t],
                label=f"{t}s",
            )
            for t in ("Deletion", "Insertion", "Substitution")
        ]
        leg = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=len(legend_handles),
            fontsize=14,
            facecolor=bg,
            edgecolor="none",
            framealpha=1,
        )
        for text in leg.get_texts():
            text.set_color(fg)

        _apply_ax_style(fig, ax, theme, title=f"{plot_name} (first 100)")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        plt.savefig(f"{plot_name}_{time.time()}.png", facecolor=bg, bbox_inches="tight")
        print("Plot saved successfully.")

        # --- Print Summary Tables ---
        print("\n" + "=" * 100)
        print("SUMMARY TABLES")
        print("=" * 100)

        print("\n--- WER Scores (%) by Experiment and ASR ---\n")
        pivot_wer = df_filtered.pivot_table(
            index="Experiment", columns="ASR", values="Score", aggfunc="mean"
        )
        pivot_wer = pivot_wer.reindex(ordered_experiments)
        pivot_wer["Mean"] = pivot_wer.mean(axis=1)
        print(pivot_wer.to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n\n--- Detailed Error Breakdown by Experiment and ASR ---\n")
        for experiment in ordered_experiments:
            exp_data = df_filtered[df_filtered["Experiment"] == experiment]
            print(f"\n{experiment}:")
            print("-" * 80)

            detail_rows = []
            for asr in allowed_asrs:
                row_data = exp_data[exp_data["ASR"] == asr]
                if not row_data.empty:
                    detail_rows.append(
                        {
                            "ASR": asr,
                            "WER (%)": row_data["Score"].values[0],
                            "Insertions (%)": row_data["Insertions"].values[0],
                            "Deletions (%)": row_data["Deletions"].values[0],
                            "Substitutions (%)": row_data["Substitutions"].values[0],
                        }
                    )

            if detail_rows:
                detail_df = pd.DataFrame(detail_rows)
                print(
                    detail_df.to_string(index=False, float_format=lambda x: f"{x:.2f}")
                )
                print(f"\nExperiment Mean WER: {exp_data['Score'].mean():.2f}%")

        print("\n\n--- Summary Statistics ---\n")
        summary_stats = (
            df_filtered.groupby("Experiment")
            .agg(
                {
                    "Score": "mean",
                    "Insertions": "mean",
                    "Deletions": "mean",
                    "Substitutions": "mean",
                }
            )
            .round(2)
        )
        summary_stats = summary_stats.reindex(ordered_experiments)
        print(summary_stats.to_string())

        print("\n" + "=" * 100 + "\n")

    else:
        print("No data found after filtering.")

except Exception as e:
    print(f"An error occurred: {e}")
