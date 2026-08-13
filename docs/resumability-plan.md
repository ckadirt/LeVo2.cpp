# Cross-process pause and resume plan

## Status

Proposed implementation contract. No resumability code has landed yet.

This plan was prepared from the current LeVo2.cpp `main` at commit `12a1253`
and the local ACE-Step reference at commit `79994ed` (whose staged-engine merge
is `7985032`). The reference implementation and its `docs/ABI.md` are the
behavioral baseline; this document resolves the model-specific differences for
LeVo before implementation starts.

## Current behavior

LeVo generation is **not resumable today**.

- LeLM polls cancellation at delayed autoregressive positions, then throws
  `operation_cancelled`. The in-memory delayed sequence, sampler state, EOS
  state, and both K/V sessions are destroyed.
- Flow polls before each 1000-frame window and Euler velocity evaluation, then
  throws the same exception. The current normalized latent and all completed
  windows are destroyed.
- VAE polls only between complete 1000-frame window graphs. A cancellation
  discards every decoded window.
- The CLIs exit with status 130 and intentionally write neither partial tokens
  nor a WAV. There is no state-blob API, resume sniffing, or stable C ABI.

The observability milestone supplied the necessary cancellation and progress
seams, but cancellation currently means clean abandonment rather than pause.

## Outcome

Add a staged engine in which a cancellation callback is a normal paused outcome:

```text
request JSON
    |
    v
 CODES (LeLM) -------- paused: request + delayed IDs + RNG cursor
    |
    | enriched request + canonical [3,T] codes
    v
 DIFFUSE (Flow) ------ paused: request/codes + Euler state + window state
    |
    | windowed denormalized latent envelope
    v
 DECODE (VAE) ------- paused: no new blob; retry from durable DIFFUSE result
    |
    v
 48 kHz stereo PCM
```

A paused blob must be sufficient to resume the same stage after a daemon
restart, with a newly created context in another process. The caller does not
re-supply the request. A same-backend resumed run must finish with exactly the
same token IDs and Flow latent bytes as an uninterrupted run.

## Scope and compatibility

The primary integration surface will be the same Cantor engine ABI v1 used by
ACE-Step. Import `include/cantor_engine.h` unchanged so the host can load either
engine without a model-specific FFI. The LeVo library will report:

- model family: `levo2`;
- supported stages: `CANTOR_STAGE_CODES`, `CANTOR_STAGE_DIFFUSE`, and
  `CANTOR_STAGE_DECODE`;
- no `CANTOR_STAGE_PLAN`, because LeVo consumes caller-provided lyrics and a
  style description rather than generating a planning response;
- component role `lm` for the LeLM GGUF, `dit` for the self-contained Flow GGUF,
  and `vae` for the Oobleck VAE GGUF. LeVo has no separate `embed` component.

`CANTOR_DONE` remains 0, `CANTOR_PAUSED` remains 1, and `CANTOR_ERR` remains
-1. A cancellation callback must never turn into `CANTOR_ERR`; the thread-local
error code may be `CANTOR_ERR_CANCEL`, but status 1 controls the outcome.

The existing `generate_tokens()` and `render_tokens_to_audio()` C++ functions
remain source compatible. They become blocking wrappers over the new internal
state machines and preserve their current exception behavior for callers that
do not opt into staged execution. Existing token NPY/JSON and WAV/JSON artifact
formats do not change.

Build a hidden-visibility shared engine target in addition to the existing
static `levo-core`. Only the Cantor symbols are exported. The executable tools
continue to link the ordinary C++ library.

## Request contract

Fresh `CODES` input is bounded UTF-8 JSON. It contains everything required to
reconstruct a `generation_config` and the later renderer request:

- lyrics, description, and duration;
- all sampling controls, LeLM CFG, and an optional LeLM seed;
- Flow seed, Euler steps, and Flow CFG;
- schema and model-family identifiers.

Model paths are not embedded: model content is supplied to `cantor_engine_load`
as content-addressed components. Paused state instead records the expected
component identities and rejects a context whose loaded components do not
match.

Vendor a pinned `yyjson` revision for strict parsing and canonical serialization
rather than growing a handwritten Unicode JSON parser. Record it in
`THIRD_PARTY_NOTICES.md`. Reject duplicate fields, unknown schema versions,
trailing input, invalid UTF-8, non-finite numbers, and values outside the
existing public configuration bounds.

Before the first cancellable unit, resolve an omitted LeLM seed exactly once and
write the resolved value into the canonical request. The final token metadata
must distinguish `seed_was_supplied` from `resolved_seed`; it must no longer be
possible for an unseeded paused run to resume with different randomness. Flow
already has an explicit seed, but its exact generated F32 noise is captured at
the first Flow pause as described below.

On `CANTOR_DONE`, `CODES` returns canonical request JSON enriched with the three
canonical code arrays and their tensor SHA-256. This remains opaque to the host
but is a self-contained, inspectable input to `DIFFUSE`.

## Blob envelope

Paused blobs and the completed Flow boundary use engine-private binary
envelopes. Use distinct eight-byte ASCII magics:

| Blob | Magic | Purpose |
|---|---|---|
| paused LeLM | `LEVOLM01` | resume `CODES` |
| paused Flow | `LEVOFL01` | resume `DIFFUSE` |
| completed Flow | `LEVOLT01` | durable input to `DECODE` |

No valid fresh JSON request can begin with these prefixes. Resume detection is
therefore a prefix sniff; there is no external `resuming` flag.

Do not serialize a native C++ struct. Encode every integer explicitly as
little-endian fixed-width bytes so padding, alignment, compiler, and host
architecture cannot reinterpret a checkpoint. A common envelope contains:

- magic, format version, producing stage, flags, header byte count, and exact
  total byte count;
- canonical-request byte count and a directory of typed payload sections;
- LeVo build ID and checkpoint-format revision;
- actual backend/device fingerprint and resolved numerical-mode fingerprint;
- required model names, content SHA-256 values, and runtime/tokenizer revisions;
- SHA-256 over the complete envelope with the digest field zeroed.

Every decoder performs checked addition and multiplication before allocating,
requires the declared total to equal the supplied byte length exactly, enforces
the node's 2 GiB hard ceiling, validates section uniqueness and non-overlap, and
rejects trailing bytes. Tensor sections validate element type, shape, finite F32
contents, ID ranges, and stage-specific invariants before any model work.

The backend fingerprint includes at least the GGML backend name, registered
device name and description, LeVo/GGML build revision, device index, and the
resolved CUDA accumulation/TF32 modes. The model fingerprint uses the provenance
SHA-256 embedded in each strict GGUF. A mismatch is a hard error explaining the
expected and actual values; it is never downgraded to a warning.

## LeLM checkpoint design

### Saved state

The paused `CODES` blob stores:

- the canonical resolved request JSON;
- the full delayed stream including the initial all-special position, as
  little-endian int32 `[3, sequence_steps]`;
- the next delayed position to generate;
- the resolved sampler seed and count of raw `mt19937_64` draws consumed;
- the per-stream EOS state and earliest EOS position as validation redundancy;
- a digest of the three current next-position logit arrays;
- backend, LeLM, tokenizer, runtime, sampling-policy, delayed-pattern, and
  numerical-mode stamps;
- accumulated stage timings needed for honest post-resume provenance.

Persist the delayed sequence, not only the canonical `[3,T]` result. With delays
`[0,250,250]`, a partial delayed position contains valid values from different
canonical frames; reverting it alone loses the exact continuation boundary.

Do not serialize the K/V caches. At the maximum 270-second request the delayed
payload is only about 84 KB (`3 * 7001 * sizeof(int32_t)`), while the paired
conditional/null K/V caches are hundreds of megabytes or more.

### Stable random continuation

The current sampler uses standard `std::mt19937_64` but does not expose its
state. Do not depend on an implementation-specific stream serialization. Add a
draw counter at the point where the engine is actually invoked. Restore with
the resolved seed plus `discard(draw_count)`. LeVo already avoids
`std::discrete_distribution` and derives each uniform variate directly from one
raw engine result, so this is a small, cross-process cursor over the standardized
Mersenne Twister sequence.

Reconstruct repetition history and EOS tracking by replaying the saved delayed
IDs through their host-side state machines. Validate the stored redundant EOS
fields rather than trusting them independently.

### K/V reconstruction

On resume:

1. parse and validate the entire blob before loading or running the model;
2. rebuild tokenizer conditioning from the embedded request;
3. create fresh conditional and null K/V sessions;
4. execute the same prefix-only prefill used by an uninterrupted run;
5. decode the all-special position and every saved generated delayed position
   sequentially through the same one-token graph shape;
6. compare the rebuilt final-logit digest with the checkpoint;
7. restore the sampler cursor and host-side history, then continue at the saved
   next position.

The existing combined condition+audio prefill is not an acceptable shortcut:
this repository already documents CUDA kernel-boundary changes and near-tied
logit drift between combined and split graph shapes. Sequential replay costs
compute, but it protects song identity. A future batched replay optimization may
land only after real CPU and CUDA tests prove that its continuation logits and
tokens are byte-identical.

Normalize the pause boundary to "ready to sample the next delayed position."
Refactor progress/cancel ordering so a checkpoint is never ambiguous about
whether the last saved ID has already been appended to K/V. A callback before
the first generated ID still returns a valid paused blob containing the resolved
request and initial special position.

During cross-process replay, poll cancellation between every one-token graph. A
second cancellation returns the same logical checkpoint rather than an error or
a half-rebuilt cache.

## Flow checkpoint design

LeVo currently implements one stateless, fixed-step Euler update:

```text
x(step + 1) = x(step) + dt * velocity(x(step), t(step))
```

There is no multistep velocity history and LeVo CFG is the direct stateless
`uncond + scale * (cond - uncond)` expression. Unlike ACE-Step's APG path,
LeVo Flow is therefore resumable with CFG greater than 1.0. The checkpoint still
stamps solver name/version, resolved CFG, sigma-min, step count, and schedule so
a changed request is refused.

### Saved state

The paused `DIFFUSE` blob stores:

- canonical resolved request JSON and canonical `[3,T]` LeVo codes;
- the complete exact window-major F32 initial-noise tensor;
- source/padded frame counts and the fixed 1000/750/250 window schedule;
- all completed denormalized latent windows needed for continuation and later
  VAE overlap processing;
- the current window index, number of completed Euler steps, and current
  normalized F32 Euler state;
- optional completed normalized windows only when diagnostic window capture is
  enabled;
- backend/numerical-mode, token digest, Flow-model digest/revisions, solver, and
  all resolved renderer settings;
- accumulated timing/progress counters.

Persist exact noise even when it was derived from the seed. The native
SplitMix64 sequence is stable, but Box-Muller uses platform libm; keeping the
already resolved F32 boundary avoids a quiet cross-libm change and also supports
the existing explicit-noise input without special cases.

For a 270-second request there are at most nine windows. A worst mid-final-window
checkpoint is approximately:

- 2.30 MB exact noise;
- 2.05 MB for eight completed denormalized windows;
- 0.26 MB current Euler state;
- 0.08 MB codes;
- request/header data.

The normal production blob is therefore about 4.7 MB. Diagnostic capture can
add roughly 2.05 MB of completed normalized windows. Both are intentionally far
below the 2 GiB sanity limit and small enough for one atomic pause write.

### Resume algorithm

On resume, validate all stamps before computation. Recreate the current
window's token/mask inputs and conditioning from the embedded request, codes,
completed prior raw window, and saved noise. Start the Euler loop at
`completed_steps` with the saved normalized state. At each later step, preserve
the existing in-context interpolation and final hard restore exactly.

When a window completes, denormalize it and move it to the completed-window
payload before observing another cancel boundary. Later windows recover their
250-frame raw continuation from the saved previous denormalized window. A pause
between windows therefore has one unambiguous representation and does not
require a separate state variant.

On `CANTOR_DONE`, emit `LEVOLT01`: source/schedule metadata plus every
denormalized `[1000,64]` window. A simple assembled `[T,64]` tensor is not a
sufficient LeVo stage boundary because the official VAE path decodes the
original overlapping windows and crossfades audio afterward.

## VAE and audio stage

The completed `LEVOLT01` Flow blob is the durable boundary for `DECODE`.
Initially, cancellation during VAE decode returns `CANTOR_PAUSED` with a null
blob. The host retains/reuses the prior completed Flow checkpoint and reruns the
whole decode stage. This mirrors ACE-Step and avoids checkpointing tens to more
than 100 MB of partially decoded PCM.

The null blob is safe only because `LEVOLT01` is complete, self-validating, and
persisted as `Done` before `DECODE` begins. Document this explicitly in the ABI
guide and exercise the host fallback path in an integration test.

VAE work currently consists of monolithic 1000-frame graphs. The measured
20-second CPU render spent roughly 40 seconds in load/decode/assembly after
Flow, so polling only between windows cannot satisfy the node's 20-second
shutdown checkpoint deadline on all CPUs. Before declaring CPU resume complete:

1. measure load, graph-build, graph-compute, and readback cancellation gaps;
2. install GGML's dynamically discovered CPU abort callback and translate
   `GGML_STATUS_ABORTED` caused by the user predicate into a paused outcome;
3. verify an aborted graph leaves immutable weights and the context reusable;
4. derive the Oobleck decoder's exact convolutional receptive-field halo and
   add chunked overlap/crop decode if a single CPU graph node still exceeds the
   latency budget;
5. freeze the chosen chunk/overlap values in the request/backend stamp and prove
   uninterrupted tiled output against the existing VAE parity thresholds.

CUDA lacks the same generic mid-graph guarantee, but current full-window CUDA
decode is only a few seconds. It still participates in the measured latency
gate rather than receiving an assumption-based exemption.

The PCM buffer owned by the Cantor context is valid only until the next
`run_stage` call, exactly as the shared ABI specifies. Clear stale PCM at the
start of every stage call and copy it immediately in the host.

## Cancellation latency

The host waits 20 seconds for a worker to return a checkpoint during shutdown.
Set a stricter implementation target: callback assertion to `CANTOR_PAUSED`
must be below 15 seconds on the supported reference CPU and CUDA runners,
leaving time for the host's atomic write, fsync, digest, and directory fsync.

Poll at these boundaries:

- LeLM: before every delayed sample and between every K/V replay/decode graph;
- Flow: before every window, before every Euler graph, and through a supported
  backend abort callback while a long CPU graph executes;
- VAE: before every decode chunk/window and through a supported CPU abort
  callback;
- GGUF loading: after each existing 8 MiB read/upload chunk and between tensors;
- blob parsing/serialization: between large tensor sections where useful.

Record the largest observed callback gap per stage/backend in the implementation
report. A heartbeat is observability, not cancellation; it does not satisfy this
gate.

When a backend cannot interrupt a graph and a measured unit exceeds the target,
that backend/stage combination is reported unsupported for resumable execution
until the unit is split. Do not claim resume support that can miss the host's
hard shutdown deadline.

## Context lifetime and errors

`cantor_engine_load` creates one context for one inference thread. The context
owns the selected backend, immutable loaded model components, residency policy,
callbacks for only the active call, and the most recent PCM buffer. It is not
concurrently callable.

Keep heavy component payload loading lazy and inside `run_stage`; the load ABI
has no cancel callback. A stage first reads the small GGUF header, resolves the
request and creates its initial checkpointable state, then polls between the
existing 8 MiB tensor uploads. This lets a cancellation during a multi-gigabyte
load return `CANTOR_PAUSED` instead of trapping the host in an uncancellable
`cantor_engine_load` call.

Refactor existing pipelines to accept a context-owned backend/model without
changing their GGML graph math. Default residency keeps at most one heavy module
resident so LeLM, Flow, and VAE retain the current low-peak handoff. Implement
the ABI's byte-budget/keep-loaded policy with an LRU module store, and include
the policy in re-entry tests.

All per-run mutable objects are scoped jobs: K/V sessions, sampler/history,
Euler state, temporary graphs, decoded windows, and live callback pointers.
Destroy or reset the job after done, pause, cancellation, and error. A paused or
failed call must not poison the backend or require `dlclose`; the same context
must immediately accept another stage call.

Every exported C entry point is wrapped in an exception guard. Map at least:

- `std::bad_alloc` and backend OOM to `CANTOR_ERR_OOM`;
- strict GGUF/schema/provenance failures to `CANTOR_ERR_MODEL`;
- unavailable or failed backend initialization to `CANTOR_ERR_BACKEND`;
- callback cancellation to status `CANTOR_PAUSED` and code
  `CANTOR_ERR_CANCEL`;
- all other exceptions to `CANTOR_ERR_OTHER`.

No exception crosses C. No `exit()`, `abort()`, assertion termination, or GGML
fatal callback is reachable below the ABI. Add the ACE-Step fatal-call scanner
pattern as a CI test adapted to LeVo. Store error code/message in thread-local
state; the host reads them immediately after an error.

## Progress

Continue reporting freely; host throttling owns persistence frequency.

- `CODES`: completed delayed positions / total delayed positions (`Tokens`);
- `DIFFUSE`: completed Euler updates over all windows, with a stable mapping
  from `(window, step)` to one monotonic unit count (`Steps`);
- `DECODE`: completed VAE chunks/tiles (`Tiles`).

On resume, emit the saved completed count before new computation. K/V and
conditioning reconstruction retain the saved stage count while normal LeVo
progress/logging identifies the replay substage. Counters never regress across
multiple pauses.

## Optional CLI checkpointing

After the engine contract works, expose the same machinery without requiring a
daemon:

- `levo-cli --checkpoint FILE` atomically writes a paused `LEVOLM01` blob on
  SIGINT/SIGTERM; `levo-cli --resume FILE` needs only model/backend selection and
  output location;
- `levo-render --checkpoint FILE` writes paused Flow state or the durable
  completed Flow boundary before VAE; `--resume FILE` sniffs its stage magic;
- successful stage completion removes only the checkpoint file explicitly
  owned by that invocation;
- temp-file write, file fsync, rename, and parent-directory fsync match the
  node's durability discipline;
- malformed or mismatched checkpoints are never overwritten automatically.

Keep exit status 130 for a successfully checkpointed signal pause and print the
checkpoint path. A checkpoint write failure is a real nonzero error and leaves
the previous valid checkpoint untouched.

## Implementation sequence and commit boundaries

Each numbered slice is independently reviewed, tested, committed, pushed, and
recorded in `docs/implementation_report.md` before proceeding.

1. **Freeze contracts and test fixtures.** Add the unchanged Cantor header,
   request schema, blob codec skeleton, exact status/error rules, size limits,
   and asset-free malformed-blob tests. Add the shared-library symbol map and
   C ABI smoke test.
2. **Make LeLM a resumable state machine.** Resolve seeds early, add sampler
   draw accounting, normalize cancel boundaries, encode delayed sequences, and
   implement exact sequential K/V replay with final-logit digest validation.
3. **Prove LeLM resume.** Pause at first/middle/final positions, pause twice,
   cross an EOS, resume in a child process, reject mismatches, and compare real
   CPU/CUDA final token payloads byte-for-byte with uninterrupted baselines.
4. **Make Euler and window rendering resumable.** Accept a starting
   `(window,step,state)`, retain completed windows and exact noise, encode
   `LEVOFL01`, and emit durable `LEVOLT01` on completion.
5. **Prove Flow resume.** Pause at every synthetic Euler boundary and selected
   real one-/multi-window boundaries, including repeated cross-process pauses.
   Require byte-identical final window latents on the same backend and loud
   refusal for model/backend/config changes.
6. **Add decode fallback and latency controls.** Implement null-blob VAE pause,
   CPU backend abort plumbing, loader polling, and—if measurement requires it—
   exact halo-based VAE chunking. Prove context re-entry after graph abort.
7. **Finish context residency and safety.** Exercise stage alternation, LRU
   eviction, errors after pauses, callback cleanup, thread-local errors, audio
   lifetime, and fatal-call scanning through the shared library.
8. **Add CLI checkpoint adapters.** Implement atomic checkpoint files, signal
   behavior, stage sniffing, progress continuity, and completed-checkpoint
   cleanup without changing the existing default commands.
9. **Run release gates and document.** Run the full CPU/CUDA suites, frozen
   LeLM fixtures, eight-case renderer parity matrix, cross-process resume
   matrix, maximum blob sizes, cancellation latency measurements, and a real
   interrupted end-to-end song through WAV.

## Verification matrix

The milestone is incomplete until all of the following pass.

### Blob and ABI tests

- pure-C load/run/free test plus ABI version, stage mask, hidden-symbol, and
  audio-lifetime checks;
- truncation at every header/section boundary, wrong magic/version/stage,
  integer overflow, duplicate/overlapping sections, impossible shapes, NaN/Inf,
  wrong checksum, and trailing-byte rejection;
- `PAUSED == 1` for cancellation at every stage; `ERR == -1` only for failures;
- same context successfully re-entered after done, pause, callback cancel,
  malformed input, model mismatch, backend abort, and OOM injection;
- separate-process producer/consumer for every resumable blob type.

### Determinism tests

- asset-free LeLM controller: greedy and sampling, every delayed-mask region,
  EOS on each stream, cancel before first token, every step, and repeated pause;
- real LeLM CPU and CUDA: uninterrupted versus pause/resume final token bytes,
  plus final-logit replay digest equality;
- asset-free Euler: pause after every step and at every window transition;
- real Flow CPU and CUDA: one and multiple windows, early/middle/last Euler
  steps, repeated resume, exact per-window latent bytes;
- refusal after changing backend/device fingerprint, any relevant model digest,
  resolved numerical mode, token digest, seed/noise, steps, CFG, solver, or
  window schedule;
- current frozen LeLM oracle fixtures and renderer parity thresholds unchanged.

### Operational tests

- worst normal LeLM blob remains below 128 KiB plus bounded request data;
- worst 270-second Flow blob and measured fsync time are recorded and remain far
  below 2 GiB;
- callback-to-paused latency remains below 15 seconds on supported reference
  CPU/CUDA stage/backend pairs;
- SIGINT during LeLM, Flow, Flow replay, GGUF load, and VAE decode writes or
  reuses the correct durable boundary and resumes after a process restart;
- a forced kill during checkpoint write preserves the previous valid file;
- complete resumed WAV has the exact requested length, finite stereo samples,
  and the expected provenance/sidecar hashes.

## Explicit differences from ACE-Step

These are model-driven decisions, not accidental deviations:

- LeVo advertises no `PLAN` stage.
- LeVo's direct CFG expression has no APG momentum, so Flow CFG greater than 1.0
  is resumable and is stamped rather than refused.
- LeVo supports only fixed-step Euler today; any future stateful solver is
  refused until its history is checkpointed.
- The completed Flow boundary contains overlapping 1000-frame latent windows,
  not one raw `[T,64]` tensor, because LeVo decodes then crossfades those windows.
- Paused LeLM blobs stamp backend/model/numerical identity and a replay-logit
  digest in addition to token IDs, because this repository has already observed
  graph-boundary changes that can flip near-tied logits.
- LeVo persists exact generated Flow noise rather than relying only on seed
  re-derivation, avoiding cross-libm Box-Muller drift.
- The bundled GGML CPU backend exposes a node-boundary abort callback. The
  implementation will use and verify it for long Flow/VAE graphs rather than
  assuming every graph is uninterruptible.

The governing rule remains the same: a resumed song that quietly drifts is
worse than a loudly refused resume.
