## Structure

- `hush/` — hush baseline
- `aic/` — ai-coustics SDK inference

Both directories are independent uv projects.

Enhance with Hush
1. `uv --project hush run python hush/enhance_dawn_chorus_with_hush_onnx.py`

Enhance with AIC SDK
2. `uv --project aic run python aic/enhance_dawn_chorus_with_aic.py`

Generate transcriptions for Hush
3. `uv run python generate_transcripts.py advanced_dfnet16k_model_best_onnx`

Generate transcriptions for AIC-SDK
4. `uv run python generate_transcripts.py quail_vf_2_0_l_16khz_el_80`

Run and plot evaluation
5. `uv run python evaluate_wer.py`