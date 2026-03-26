# Dawn Chorus Benchmark

End-to-end benchmark comparing speech enhancement approaches on the [ai-coustics/dawn_chorus_en](https://huggingface.co/datasets/ai-coustics/dawn_chorus_en) dataset. Each approach enhances noisy audio, transcribes the result with faster-whisper, and computes corpus-level Word Error Rate (WER) broken down by deletions, insertions, and substitutions.

**Approaches compared:**

- **Mix** — raw unprocessed audio (baseline)
- **Hush** — offline denoising via the prebuilt `libweya_nc` ONNX model
- **AIC** — enhancement via the ai-coustics Python SDK

---

## Structure

```
.
├── hush/                  # Hush enhancement (independent uv project)
├── aic/                   # AIC SDK enhancement (independent uv project)
├── extract_mix_audio.py   # Extracts raw mix audio from the HuggingFace dataset
├── generate_transcripts.py# Transcribes a folder of WAVs with faster-whisper
├── evaluate_wer.py        # Computes WER and plots the bar chart
├── sky/run.yaml           # SkyPilot job definition
└── .env                   # AIC SDK license key (see setup below)
```

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An **AIC SDK license key**. Self-service SDK Keys can be generated on the [developer portal](https://developers.ai-coustics.com/)

```bash
cp .env.sample .env
# edit .env and set AIC_SDK_LICENSE=your_key_here
```

---

## Run locally

### 1. Install dependencies

```bash
uv sync                  # root project (transcription + evaluation)
uv --project hush sync   # Hush sub-project
uv --project aic sync    # AIC sub-project
```

### 2. Extract raw mix audio

```bash
uv run python extract_mix_audio.py
# → mix/audio/*.wav
```

### 3. Enhance with Hush

```bash
uv --project hush run python hush/enhance_dawn_chorus_with_hush_onnx.py \
  --atten-lim-db 100
# → hush_advanced_dfnet16k_model_best_onnx_atten100/audio/*.wav
```

### 4. Enhance with AIC SDK

```bash
uv --project aic run python aic/enhance_dawn_chorus_with_aic.py \
  --model-id quail-vf-2.0-l-16khz --enhancement-level 0.8
# → aic_quail_vf_2_0_l_16khz_el80/audio/*.wav
```

### 5. Transcribe

`MODEL` choices: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `distil-small.en`, `medium`, `medium.en`, `distil-medium.en`, `large-v1`, `large-v2`, `large-v3`, `large`, `distil-large-v2`, `distil-large-v3`, `large-v3-turbo`, `turbo` (default: `tiny.en`)

```bash
uv run python generate_transcripts.py mix --model distil-medium.en
uv run python generate_transcripts.py hush_advanced_dfnet16k_model_best_onnx_atten100 --model distil-medium.en
uv run python generate_transcripts.py aic_quail_vf_2_0_l_16khz_el80 --model distil-medium.en
```

### 6. Evaluate and plot

```bash
uv run python evaluate_wer.py --model distil-medium.en
# → wer_comparison.png
```

Output dirs are auto-detected via glob (`hush_*/`, `aic_*/`). Pass `--hush-dir` and `--aic-dir` explicitly if you have multiple runs.

---

## Run on a GPU VM via SkyPilot

The `sky/run.yaml` job runs all steps end-to-end on an AWS `g5.xlarge` (NVIDIA A10G). Parameters are defined at the top of the `run:` block.

### Setup

Install [SkyPilot](https://docs.skypilot.co/en/latest/getting-started/installation.html) and configure AWS credentials, then:

```bash
cd sky
sky launch run.yaml --cluster hush-benchmark
```

To re-run on an existing cluster:

```bash
sky exec hush-benchmark run.yaml
```

To download results:

```bash
sky rsync-down hush-benchmark ~/sky_workdir/wer_comparison.png .
```

### Changing parameters

Edit the variables at the top of the `run:` block in `sky/run.yaml`:

```yaml
WHISPER_MODEL="distil-medium.en"
HUSH_MODEL="advanced_dfnet16k_model_best_onnx"
HUSH_ATTEN=100
AIC_MODEL="quail-vf-2.0-l-16khz"
AIC_EL=0.8
```
