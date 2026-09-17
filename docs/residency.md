# Cantor model residency

Set `[engine] keep_loaded = true` in the node configuration and restart the
node to keep each lazily loaded LeLM, Flow and VAE model on its selected
backend between stages and generations. CPU weights remain in RAM; GPU
weights remain in the backend device buffers. Backend handles outlive weights.
Fresh KV sessions, request conditioning, samplers and renderers are created
for every stage invocation, including checkpoint replay.

With `keep_loaded = false` (the default), each stage releases its model and
backend on completion, pause or error. The context still holds paths and the
latest audio; caching that context alone does not cache model weights.
`vram_budget_bytes` is not implemented by this engine. Retention is all-or-none
and requires enough memory for all loaded weights plus active inference work.
Destroying the context releases every retained model.

`cantor_engine_resident_modules` and `cantor_engine_resident_bytes` report
retained models and their weight-buffer bytes; they exclude transient graphs,
KV and driver allocations. Test repeated generation, changed requests and
pause/resume on a GPU before treating a new backend build as GPU-validated.

Cantor input uses `caption`, optional `lyrics` and `duration`. Legacy
`description`/`duration_seconds` requests and checkpoints remain readable.
Flat `inference_steps`, `guidance_scale`, `cfg_scale`, `temperature`,
`top_k_mixed` and `top_k_detail` controls support catalog-driven clients.
