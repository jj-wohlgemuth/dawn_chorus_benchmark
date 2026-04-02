# Dawn Chorus Benchmark

End-to-end benchmark comparing speech enhancement approaches on the [ai-coustics/dawn_chorus_en](https://huggingface.co/datasets/ai-coustics/dawn_chorus_en) dataset. Each approach enhances noisy audio, transcribes the result with two STT systems, and plots corpus-level Word Error Rate (WER) broken down by deletions, insertions, and substitutions.

**Enhancement conditions:**

- **Mix** — raw unprocessed audio (baseline)
- **Hush** — offline denoising via the prebuilt `libweya_nc` ONNX model (CPU, parallelised across all cores)
- **AIC** — enhancement via the ai-coustics Python SDK

**STT systems:**

- **Whisper** (`distil-medium.en`) via faster-whisper
- **Kyutai STT** (`stt-2.6b-en-candle`) via moshi-server WebSocket streaming

---

## Structure

```
.
├── kyutai/                        # Kyutai STT WebSocket client
│   └── kyutai_api.py
├── hush/                          # Hush enhancement (independent uv project)
├── aic/                           # AIC SDK enhancement (independent uv project)
├── extract_mix_audio.py           # Extracts raw mix audio from the HuggingFace dataset
├── generate_transcripts.py        # Transcribes a folder of WAVs with faster-whisper
├── generate_transcripts_kyutai.py # Transcribes a folder of WAVs via moshi-server
├── evaluate_wer.py                # Computes WER and plots the combined bar chart
├── sky/run.yaml                   # SkyPilot job definition (A10G end-to-end)
└── .env                           # AIC SDK license key (see setup below)
```

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An **AIC SDK license key** — generate one at the [developer portal](https://developers.ai-coustics.com/)

```bash
cp .env.sample .env
# edit .env and set AIC_SDK_LICENSE=your_key_here
```

---

## Run locally

### 1. Install dependencies

```bash
uv sync                  # root project (Whisper transcription + evaluation)
uv --project hush sync   # Hush sub-project
uv --project aic sync    # AIC sub-project
```

### 2. Download Hush native library

The Hush enhancement script requires the prebuilt `libweya_nc` C library:

```bash
mkdir -p hush/lib
curl -fL "https://github.com/pulp-vision/Hush/raw/main/deployment/lib/libweya_nc.so" \
  -o hush/lib/libweya_nc.so
```

### 3. Extract raw mix audio

```bash
uv run python extract_mix_audio.py
# → mix/audio/*.wav
```

### 4. Enhance with Hush

Processing is parallelised across all CPU cores (one worker per core).

```bash
uv --project hush run python hush/enhance_dawn_chorus_with_hush_onnx.py \
  --atten-lim-db 100
# → hush_advanced_dfnet16k_model_best_onnx_atten100/audio/*.wav
```

### 5. Enhance with AIC SDK

```bash
uv --project aic run python aic/enhance_dawn_chorus_with_aic.py \
  --model-id quail-vf-2.0-l-16khz --enhancement-level 0.8
# → aic_quail_vf_2_0_l_16khz_el80/audio/*.wav
```

### 6. Transcribe with Whisper

```bash
uv run python generate_transcripts.py mix --model distil-medium.en
uv run python generate_transcripts.py hush_advanced_dfnet16k_model_best_onnx_atten100 --model distil-medium.en
uv run python generate_transcripts.py aic_quail_vf_2_0_l_16khz_el80 --model distil-medium.en
```

### 7. Transcribe with Kyutai STT

Kyutai requires [moshi-server](https://github.com/kyutai-labs/moshi) running locally. See [moshi-server setup](#moshi-server-setup) below.

```bash
# Start moshi-server (from sky/ so static_dir resolves)
cd sky && moshi-server worker --config configs/config-stt-en-hf.toml &
cd ..

# Wait for "starting asr loop" in server output, then:
uv run python generate_transcripts_kyutai.py mix --concurrency 10
uv run python generate_transcripts_kyutai.py hush_advanced_dfnet16k_model_best_onnx_atten100 --concurrency 10
uv run python generate_transcripts_kyutai.py aic_quail_vf_2_0_l_16khz_el80 --concurrency 10
```

Conditions must be run **sequentially** — moshi-server has `batch_size=16`, and concurrent jobs from multiple conditions would saturate all slots.

### 8. Evaluate and plot

Produces a single chart showing Whisper and Kyutai side-by-side for each enhancement condition.

```bash
uv run python evaluate_wer.py
# → wer_comparison.png
```

Output dirs are auto-detected via glob (`hush_*/`, `aic_*/`). Pass `--hush-dir` and `--aic-dir` explicitly if you have multiple runs.

---

## moshi-server setup

moshi-server is a Rust binary that requires CUDA. Build it with:

```bash
# Requires CUDA 12.2 compiler (driver 535 is incompatible with PTX from CUDA 12.4+)
sudo apt-get install -y cuda-compiler-12-2 libopus-dev libssl-dev

uv python install 3.11   # moshi-server links against libpython via PyO3

export PATH=/usr/local/cuda-12.2/bin:$PATH
export CUDA_COMPUTE_CAP=86          # A10G; use 89 for L40S/H100
export PYO3_PYTHON=$(uv python find 3.11)

cargo install --features cuda --locked moshi-server

# At runtime, libpython must be discoverable:
MOSHI_LIBDIR=$($(uv python find 3.11) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH=$MOSHI_LIBDIR
```

The server downloads ~5 GB of model weights from HuggingFace on first start.

---

## Run on a GPU VM via SkyPilot

The `sky/run.yaml` job runs all steps end-to-end on an AWS `g5.xlarge` (NVIDIA A10G, CUDA driver 535). It builds moshi-server during setup and waits for the server to be fully ready before streaming audio.

### Launch

```bash
cd sky
sky launch run.yaml --cluster dawn-chorus
```

To re-run on an existing cluster:

```bash
sky exec dawn-chorus run.yaml
```

To download the result plot:

```bash
sky rsync-down dawn-chorus ~/sky_workdir/wer_comparison.png .
```

### Parameters

Edit the variables at the top of the `run:` block in `sky/run.yaml`:

```yaml
WHISPER_MODEL="distil-medium.en"
HUSH_MODEL="advanced_dfnet16k_model_best_onnx"
HUSH_ATTEN=100
AIC_MODEL="quail-vf-2.0-l-16khz"
AIC_EL=0.8
KYUTAI_CONCURRENCY=10
```
