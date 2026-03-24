## Structure

- `hush/` — hush baseline
- `aic/` — ai-coustics SDK inference

Both directories are independent uv projects.

Enhance with Hush
1. `uv --project hush run python hush/enhance_dawn_chorus_with_hush_onnx.py`


Enhance with AIC SDK
2. `uv --project aic run python aic/enhance_dawn_chorus_with_aic.py`


Generate transcriptions for Hush
3. `uv run python generate_transcripts.py advanced_dfnet16k_model_best_onnx [--model MODEL]`


Generate transcriptions for AIC-SDK
4. `uv run python generate_transcripts.py quail_vf_2_0_l_16khz_el_80 [--model MODEL]`

`MODEL` choices: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `distil-small.en`, `medium`, `medium.en`, `distil-medium.en`, `large-v1`, `large-v2`, `large-v3`, `large`, `distil-large-v2`, `distil-large-v3`, `large-v3-turbo`, `turbo` (default: `tiny.en`)


Run and plot evaluation
5. `uv run python evaluate_wer.py`