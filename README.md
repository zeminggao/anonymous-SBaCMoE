# SBaC-MoE

Communication-aware sequence regrouping with a distribution constraint.

## Base configuration: 16 GPUs (EP16)

The primary asynchronous entry point defaults to `configs/ep16.json`: 16 global
ranks, EP=16, TP=PP=1, 64 experts with **4 experts per rank**, four sequences per
rank per microbatch, and four microbatches per optimizer update. The global batch
is **256 sequences** and a dispatch contains 64 sequences. With length 2048,
96 updates consume 50,331,648 input tokens (the prepared 24,576-sequence bank).
A regroup window W is a sequence count; optimizer-step length is W/256, not W/64.
The older four-GPU synchronous example is retained below as a legacy reference.

### Calibrate the actual bottleneck before choosing the objective

**Do not reuse another server's NUMA grouping or bandwidth weights. Measure both
cross-NUMA and inter-node effective bandwidth on the target machines**, in both
directions and under concurrent traffic representative of the actual A2A sizes.
GPU rank numbering does not identify physical locality. Check GPU/PCIe/NVLink,
CPU NUMA and network/NIC affinity, then map global ranks to the congested link
or network cut. Low latency in a one-way isolated copy does not establish the
bandwidth available to bidirectional, concurrent All-to-All.

`configs/ep16.json` contains an **uncalibrated example**, two nodes of eight GPUs.
Replace `rank_to_bottleneck_side` with the actual two sides of the dominant cut.
Set `inverse_bandwidth_weights` to `[B_ref/B_0to1, B_ref/B_1to0]` using measured
effective bandwidth in consistent units; clear `example_only` after calibration.
The implemented proxy is:

`sum_dispatch sum_layer max(w_0to1 * tokens_0to1, w_1to0 * tokens_1to0)`.

A slower direction therefore receives a higher weight. Equal weights preserve
the original token-count proxy; they are not measured bandwidth values. A common
bytes-per-assignment factor cancels for a fixed activation representation. If
representations or deduplication differ, convert the corresponding traffic to
bytes explicitly. Confirm the proxy against actual A2A and full-step timings.
The directional maximum assumes the two directions can progress concurrently;
if they compete for a shared link, calibrate a shared-capacity/latency objective
instead of assuming full-duplex independence. Recheck MMD calibration at the
EP16 dispatch size of 64 sequences; the example epsilon remains 0.02.

The current solver models **one dominant two-sided bottleneck cut**. If cross-NUMA
and inter-node links both limit execution, use a per-link traffic/latency model
and extend the objective and incremental swap evaluation accordingly; this
release does not automatically model arbitrary multi-link overlap, startup
latency or contention. Expert placement and regrouping must use the actual
physical destinations consistently. The placement builder remains compute-only.

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
torchrun --standalone --nproc_per_node=16 analysis/train_olmoe_async.py \
  --arm fifo --run-name async-control --updates 96 --topology configs/ep16.json
torchrun --standalone --nproc_per_node=16 analysis/train_olmoe_async.py \
  --arm regroup --run-name async-regroup --window 4096 --updates 96 --topology configs/ep16.json
```

The commands above require a single host with 16 GPUs; its bottleneck mapping
must be recalibrated rather than assumed to match the two-node example. For two
8-GPU nodes, run the following on **each node** with its own `NODE_RANK` (0 or 1):

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29500 analysis/train_olmoe_async.py \
  --arm regroup --run-name async-regroup --updates 96 --topology configs/ep16.json
```

All ranks need the same code, model, placement and data. The run directory must
be a shared filesystem: the current feedback/plan transport uses atomic files,
not a cross-node RPC service. Use global ranks for data, feedback and ownership;
local rank is used only to select the CUDA device. Build the EP16 placement with
`python analysis/build_placement.py calibration_counts.npy --ranks 16`.

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
behavior and unchanged gradients. Full 16-rank training correctness and overlap/performance
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
The primary asynchronous planner uses the explicit EP16 topology configuration.
The legacy synchronous planner defaults to four ranks and `[0, 0, 0, 1]`. Both
use fixed compute-balanced placements without expert replicas; do not confuse
the legacy topology with the EP16 base configuration.

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

## Offline planning

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python analysis/plan_window.py window.npz --epsilon 0.02 --microbatch 4
```

The NPZ contains aligned `tokens [W,S]`, `a [W,L]`, and `b [W,L]` arrays.
`a` counts remote assignments if a sequence originates on NUMA node 0; `b`
counts remote assignments if it originates on node 1. Output indices refer to
the input window. W must be divisible by ranks times microbatch size. W=4096
is the training default; the scheduler supports other complete windows.

## Legacy four-GPU synchronous fine-tuning

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
optional `analysis/build_placement.py --ranks 4` constructs this legacy mapping
from calibration route counts (its default is now 16 ranks). Freeze the same
placement for both training arms.

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
