# SBaC-MoE

Communication-aware sequence regrouping with a distribution constraint.

## Asynchronous predictor integration

`analysis/train_olmoe_async.py` adds a CPU-only process that plans one window
ahead while normal forward/backward and optimizer work continues. This is
planner/training overlap; it does not enable Megatron A2A/compute overlap.

The predictor maintains prefix/rest token-to-expert inclusion marginals for
16 layers, 64 experts and native Top-8 routing. A 32-observation prior smooths
rare tokens. Training captures the first microbatch of each optimizer step.
Physical expert IDs are used consistently with the reordered Megatron router.
Pinned double buffers and a separate CUDA copy stream transfer observations to
a writer thread. A full pool skips an observation, never a training sample.
Activation-recomputation calls are excluded from duplicate capture.

Only fully published feedback from completed earlier windows is eligible for
updates. The next window is predicted without running its model forward. The
CPU process updates the predictor, constructs MMD features/kernel, repairs
initial infeasibility at fixed epsilon=0.02, searches, audits, and atomically
publishes the plan. Each search round proposes W swaps; search stops after
10 consecutive rounds with relative objective improvement below 0.1%.

Startup uses FIFO until feedback and a valid plan are available. Missing, late,
failed, or invalid plans also select FIFO; late plans cannot replace a window
already in use. FIFO fallback preserves samples but is **not** a guarantee that
the fallback satisfies the MMD constraint. Worker failures and fallback choices
are recorded. A single rank chooses the plan and broadcasts it to all ranks.

Using the same prepared `work/` layout described below:

```bash
export OLMOE_QUALITY_ROOT="$PWD/work"
export PYTHONPATH="$PWD"
torchrun --standalone --nproc_per_node=4 analysis/train_olmoe_async.py \
  --arm fifo --run-name async-control --updates 384
torchrun --standalone --nproc_per_node=4 analysis/train_olmoe_async.py \
  --arm regroup --run-name async-regroup --window 4096 --updates 384
```

This portable entry point uses constant learning rate (default 2e-5), evaluates
validation PPL every 200 updates and at the endpoints, and evaluates test PPL
at the end. It does not save training checkpoints or run downstream task scorers.
Set `--lr`, `--eval-every`, and optionally `--planner-cpu` explicitly for a study.
The two arms must use the same model, data, placement and optimization settings.
The token bank is consumed once; provide enough sequences for the chosen updates.

`async_planner/client_metrics.json` records readiness and local decision time;
feedback manifests record host enqueue/writer time and skipped observations;
responses record update, prediction and search timing; `train_loss.jsonl`
records step wall time and the actual planner choice. Initialization/JIT is not
silently removed from worker latency. Local decision time excludes the subsequent
distributed broadcast. These logs do not by themselves prove that resource
competition is absent or that overhead is fully hidden.

CPU replay and a synthetic CUDA router/backward check verified sample
conservation, past-only feedback, late-plan FIFO, saved route IDs, bounded-buffer
behavior and unchanged gradients. Full four-rank training overlap/performance
validation remains outstanding; no end-to-end speedup is claimed here.

The synchronous current-window profiling integration below remains available.

## Scheduler

A fixed expert placement supplies per-sequence, per-layer traffic counts. Within
each lookahead window, the scheduler swaps samples across dispatches and source
ranks while preserving the exact sample multiset and samples per rank. The cost is

`sum_dispatch sum_layer max(cross_0_to_1, cross_1_to_0)`.

Counts refer to token-expert assignments; this objective does not assume
destination-deduplicated activation transmission. It is a communication proxy,
not a measurement of execution time. Search is greedy and is not certified optimal.
The planner accepts a two-node rank mapping; the included training integration
uses four ranks with mapping `[0, 0, 0, 1]`, no expert replicas, and a fixed
compute-balanced placement. Adjust the integration for other topologies.

## MMD constraint

Each sequence is represented by a 256-dimensional token-ID modulo histogram,
normalized by its L2 norm. The RBF kernel uses sigma=1. The biased estimator is

`MMD2(batch, window) = mean K(batch,batch) + mean K(window,window) - 2 mean K(batch,window)`.

Diagonal entries are included. This matches the original SBaC feature and kernel
definitions. Every **dispatch** is constrained against its source window (not
against each GPU's subset or the accumulated optimizer batch). Default epsilon is
0.02. Cross-dispatch swaps must leave both affected dispatches within epsilon;
within-dispatch swaps leave MMD unchanged. The final audit recomputes every MMD.

The input FIFO schedule must already satisfy epsilon. An infeasible input raises
an error; the scheduler never silently increases epsilon or emits a violating
fallback. Choose epsilon explicitly using a separate calibration set and the
same dispatch size. Hashed token histograms are distribution proxies, not semantic
embeddings or a guarantee of preserved model quality.

## CPU tests and offline planning

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
PYTHONPATH=. python analysis/plan_window.py window.npz --epsilon 0.02 --microbatch 4
```

The NPZ contains aligned `tokens [W,S]`, `a [W,L]`, and `b [W,L]` arrays.
`a` counts remote assignments if a sequence originates on NUMA node 0; `b`
counts remote assignments if it originates on node 1. Output indices refer to
the input window. W must be divisible by ranks times microbatch size. W=4096
is the training default; the scheduler supports other complete windows.

## OLMoE full-parameter fine-tuning

The integration uses Megatron Core's native router and All-to-All dispatcher,
sequential experts, Torch SDPA, activation recomputation and a sharded FP32-master
AdamW implementation. Shared optimizer states are CPU-resident and GPU updates
are chunked. It does not enable GroupedGEMM or communication/compute overlap.

Prepare `work/model/` with the Hugging Face checkpoint
`allenai/OLMoE-1B-7B-0924-SFT` at revision
`215cc4f73147dd68bd11e7a7dcc56bac397f4221`. Put the parquet shards of
`allenai/tulu-v3.1-mix-preview-4096-OLMoE`, revision
`a16e10fdbec0b3430b4cc1e1716bec5ec39d08ce`, in `work/tulu_raw/`.
Model weights and datasets are not redistributed here.

Provide `work/data/native_placements.npz` with integer `compute_balanced [16,64]`
rank IDs: each layer must have exactly 16 experts on each of four ranks. The
optional `analysis/build_placement.py` constructs this mapping from calibration
route counts. Freeze the same placement for both training arms.

```bash
python -m pip install -r requirements-training.txt
export OLMOE_QUALITY_ROOT="$PWD/work"
export PYTHONPATH="$PWD"
export PYTORCH_ALLOC_CONF=expandable_segments:True
python analysis/prepare_data.py
torchrun --standalone --nproc_per_node=4 analysis/train_olmoe_quality.py --arm fifo --updates 384
torchrun --standalone --nproc_per_node=4 analysis/train_olmoe_quality.py --arm regroup --updates 384 --mmd-epsilon 0.02
```

Both arms start from identical weights and the same globally shuffled sequences.
Only regroup changes order inside each 4096-sequence window. Native Top-8 routing
continues to update during training. Each window is profiled with the current
model before planning; no learned predictor or forced route replay is used.
The MMD-constrained planner is called directly by the regroup training arm.

Each GPU receives four 2048-token sequences per microbatch, with four
microbatches per optimizer update. 384 updates consume 50,331,648 input tokens.
Loss is assistant-token weighted. Validation runs at step 0, every 32 updates
and at the end; test PPL is evaluated only at the end. Evaluation splits are
disjoint from this round's training data by normalized first-user prompt hash.
The initial SFT model may have seen this dataset, so these are not guaranteed
unseen-data evaluations.

Only final full states are saved, under `work/final_states/{arm}`. The trainer
waits for 110 GiB free on the workspace filesystem before saving. Keep sufficient
space for both arms. Final state includes parameters, optimizer states, RNGs,
data manifest, placement and scheduling configuration. `RESUME.txt` records how
to continue. No server credentials, private paths, experiment logs or Git history
are included.
