Questions (40)
1) Which AD mode most efficiently computes vector–Jacobian products for a scalar loss over millions of parameters?
☐ A. Forward‑mode AD
☐ B. Reverse‑mode AD
☐ C. Numerical finite differences
☐ D. Pure symbolic differentiation only
Show answer & explanation
2) For dual numbers, a unary function obeys f(a + ε·b) = ? (first‑order exact)
☐ A. f(a) + f''(a)·ε·b
☐ B. f(a) + f'(a)·b²
☐ C. f(a) + f'(a)·ε·b
☐ D. f(a) (no change)
Show answer & explanation
3) In TensorFlow Lite for Microcontrollers (TFLM), the core memory strategy for inference is:
☐ A. Rely on malloc/free for each inference
☐ B. Use the OS paging system to overcommit RAM
☐ C. Store all activations in external flash at runtime
☐ D. Provide a fixed tensor arena and avoid dynamic allocation in the hot path
Show answer & explanation
4) In XLA‑backed JIT compilation (e.g., JAX/TF), which is a primary benefit?
☐ A. Guaranteed zero memory usage
☐ B. Operation fusion, layout selection, and buffer reuse
☐ C. Eliminates the need for autograd
☐ D. Arbitrary dynamic shapes without retracing
Show answer & explanation
5) Which linker option (used with -ffunction-sections/-fdata-sections) removes unused code/data from the final binary?
☐ A. -Os
☐ B. -flto
☐ C. -fno-plt
☐ D. --gc-sections
Show answer & explanation
6) Which hierarchical all‑reduce sequence is correct?
☐ A. Inter‑node all‑gather → intra‑node reduce‑scatter → inter‑node reduce‑scatter → intra‑node all‑gather
☐ B. Intra‑node all‑gather → inter‑node all‑gather → intra‑node reduce‑scatter → inter‑node reduce‑scatter
☐ C. Intra‑node reduce‑scatter → inter‑node reduce‑scatter → inter‑node all‑gather → intra‑node all‑gather
☐ D. Inter‑node reduce‑scatter → intra‑node reduce‑scatter → inter‑node all‑gather → intra‑node all‑gather
Show answer & explanation
7) In secure aggregation for FL, handling client dropouts typically requires:
☐ A. Decrypting all updates at the server
☐ B. Replaying client updates from logs
☐ C. Homomorphic inference on raw audio
☐ D. Threshold mask‑recovery so pairwise masks still cancel in the sum
Show answer & explanation
8) In TensorFlow 2.x, the default execution style is:
☐ A. Eager (imperative) execution with optional tf.function
☐ B. Only static graphs with sessions
☐ C. Ahead‑of‑time compilation only
☐ D. TorchScript‑style tracing only
Show answer & explanation
9) TensorFlow 1.x’s “define‑then‑run” programming corresponds to:
☐ A. Immediate execution
☐ B. Pure functional transformations
☐ C. Symbolic computational graphs executed in a Session
☐ D. Device‑only kernels with no graphs
Show answer & explanation
10) In TFLM, the FlatBuffer model format primarily enables:
☐ A. Online backprop on MCUs
☐ B. Zero‑copy, schema‑checked access without heavy parsing
☐ C. On‑device code generation per model
☐ D. Dynamic operator linking at runtime
Show answer & explanation
11) In TinyML systems, the dominant energy cost is often:
☐ A. ALU adds
☐ B. Branch prediction
☐ C. Stack frame setup
☐ D. Radio uplink (e.g., BLE/Wi‑Fi)
Show answer & explanation
12) A canonical fusion in CNN blocks that cuts memory traffic is:
☐ A. BatchNorm → Softmax → Dropout
☐ B. Pool → Pool → Pool
☐ C. Conv2D → BatchNorm → ReLU
☐ D. ReLU → Argmax → Sort
Show answer & explanation
13) For NHWC tensors, a cache‑friendly inner loop typically iterates with:
☐ A. h → w → c (c innermost)
☐ B. c → h → w (c outermost)
☐ C. Randomized strides
☐ D. h → c → w (w outermost)
Show answer & explanation
14) Which statement is not characteristic of synchronous all‑reduce training?
☐ A. Per‑step global barrier
☐ B. Weight staleness by design across workers
☐ C. Equivalent to a single device with the global batch
☐ D. Straggler sensitivity
Show answer & explanation
15) A practical advantage of parameter‑server (async/stale‑sync) approaches is:
☐ A. No staleness
☐ B. Higher throughput and tolerance to slow clients
☐ C. Exact equivalence to single‑device SGD
☐ D. Perfect reproducibility across runs
Show answer & explanation
16) In ZeRO Stage‑2, the optimizer shards:
☐ A. Only parameter tensors
☐ B. Optimizer states and gradients
☐ C. Parameters and nothing else
☐ D. Nothing (just fuses kernels)
Show answer & explanation
17) In sharded training, the phase that computes partial reductions and partitions across ranks is:
☐ A. All‑gather
☐ B. Broadcast
☐ C. All‑to‑all
☐ D. Reduce‑scatter
Show answer & explanation
18) If the TFLM tensor arena is too small at initialization, the correct behavior is to:
☐ A. Silently reallocate on the heap
☐ B. Compress the model in place
☐ C. Return an immediate error at init
☐ D. Stream tensors from flash automatically
Show answer & explanation
19) MLIR is best described as:
☐ A. A multi‑level compiler framework with dialects enabling transformations from TF ops down to LLVM
☐ B. A runtime that replaces TF Serving
☐ C. A visualization tool for graphs only
☐ D. A GPU vendor library for GEMMs
Show answer & explanation
20) Which approach explicitly reduces cross‑switch traffic by doing reductions within nodes first?
☐ A. Ring all‑reduce
☐ B. Tree all‑reduce
☐ C. Parameter server
☐ D. Hierarchical all‑reduce
Show answer & explanation
21) A personalization strategy discussed for FL is:
☐ A. Keep only the server head and freeze all client layers
☐ B. Upload all personal parameters for global averaging
☐ C. Split model: shared backbone aggregated globally + private head kept on‑device
☐ D. Train only embeddings on device and discard the rest
Show answer & explanation
22) For non‑IID data, one aggregation tactic to improve convergence is:
☐ A. Cohort‑wise aggregation: average within clusters then combine
☐ B. Weight only by network speed
☐ C. Discard minority cohorts
☐ D. Aggregate only the largest clients
Show answer & explanation
23) In client‑level DP for FL, what bounds a single client’s influence prior to adding noise?
☐ A. Momentum correction
☐ B. Per‑client norm clipping
☐ C. Homomorphic encryption
☐ D. Secure enclaves
Show answer & explanation
24) In DDP, gradient bucketing is primarily used to:
☐ A. Increase peak memory
☐ B. Reduce flops by pruning tensors
☐ C. Start all‑reduce on ready buckets to overlap communication with compute
☐ D. Disable mixed precision safely
Show answer & explanation
25) A versatile CPU/GPU collective library often used as a fallback in DL frameworks is:
☐ A. NCCL
☐ B. MPI
☐ C. Gloo
☐ D. NVSHMEM
Show answer & explanation
26) On TPUs, a common practice to improve systolic array utilization is to:
☐ A. Use many tiny GEMMs
☐ B. Pad/align matrix dimensions to preferred multiples (e.g., 128)
☐ C. Disable XLA’s layout selection
☐ D. Replace all convs with FFTs by default
Show answer & explanation
27) For small messages, which reduce algorithm often has lower latency?
☐ A. Tree all‑reduce
☐ B. Ring all‑reduce
☐ C. All‑to‑all
☐ D. Broadcast
Show answer & explanation
28) A graph‑level transformation that avoids re‑computing identical subgraphs is:
☐ A. Kernel inlining
☐ B. Layer freezing
☐ C. Gradient checkpointing
☐ D. Common subexpression elimination
Show answer & explanation
29) (Select all that apply) Code‑size optimization tactics discussed include:
☐ A. Compile with -Os for size
☐ B. Link‑time optimization (LTO)
☐ C. Link with --gc-sections after -ffunction-sections/-fdata-sections
☐ D. Enable verbose logging strings in release builds
Show answers & explanation
30) (Select all that apply) Benefits of computational graphs include:
☐ A. Automatic differentiation via chain rule on the DAG
☐ B. Global scheduling and kernel fusion opportunities
☐ C. Memory planning and buffer reuse
☐ D. Guaranteed elimination of all communication
Show answers & explanation
31) (Select all that apply) Communication‑efficient FL updates may use:
☐ A. Quantization (e.g., 8/4/2‑bit with per‑tensor scales)
☐ B. Sparsification with error feedback
☐ C. Low‑rank adapters (e.g., LoRA)
☐ D. Larger deltas sent as raw samples
Show answers & explanation
32) (Select all that apply) Systolic array‑friendly GEMM tiling on TPUs involves:
☐ A. Partitioning A and B into fixed‑size tiles
☐ B. Padding boundaries so tiles align
☐ C. Streaming A horizontally and B vertically across cells
☐ D. Random tile sizes per step to avoid regularity
Show answers & explanation
33) (Select all that apply) Robust FL aggregation under non‑IID and noise may use:
☐ A. Trimmed means / coordinate‑wise medians
☐ B. Update norm clipping
☐ C. Outlier filtering
☐ D. Always weighting solely by client RTT
Show answers & explanation
34) (Select all that apply) Reproducibility at scale recommendations include:
☐ A. Seed Python/NumPy/framework/CUDA RNGs
☐ B. Save RNG states and dataloader epoch/offset in checkpoints
☐ C. Pin environment versions (CUDA/cuDNN/NCCL, drivers, framework)
☐ D. Prefer nondeterministic kernels to speed up runs
Show answers & explanation
35) (Select all that apply) In TFLM on MCUs, good memory practices are:
☐ A. Fixed tensor arena; fail fast if insufficient
☐ B. Avoid malloc/free in the hot path
☐ C. Preallocate/reuse buffers to avoid fragmentation
☐ D. Allocate per layer dynamically every inference
Show answers & explanation
36) (Select all that apply) FSDP typically:
☐ A. All‑gathers param shards just‑in‑time before compute
☐ B. Frees full params promptly after layer compute
☐ C. Reduce‑scatters gradients after backward
☐ D. Requires full params on all ranks at all times
Show answers & explanation
37) (Select all that apply) Pipeline parallelism knobs/trade‑offs include:
☐ A. Increase microbatches b to shrink bubbles ((m−1)/b)
☐ B. 1F1B cuts activation footprint but can add weight staleness
☐ C. Activation checkpointing reduces memory at extra compute
☐ D. GPipe always interleaves fwd/bwd per microbatch
Show answers & explanation
38) (Select all that apply) Strategies to reduce serial fraction s or hide communication include:
☐ A. Overlap comm/compute via bucketization and fused kernels
☐ B. Lower‑precision or compressed gradients with error feedback
☐ C. Increase local batch and consider partial/delayed sync (with care)
☐ D. Ignore stragglers and remove health checks
Show answers & explanation
39) (Select all that apply) Memory‑bandwidth‑aware kernel practices:
☐ A. Align loop order with layout (NHWC/NCHW) to read contiguous memory
☐ B. Use in‑place ops where safe to halve traffic
☐ C. Prefer many tiny kernel launches to improve “interleaving”
☐ D. Use memory pools/arenas to avoid runtime alloc/free overhead
Show answers & explanation
40) (Select all that apply) Elastic distributed training on world‑size change:
☐ A. Recompute effective global batch; adjust LR accordingly
☐ B. Rebuild process groups and reshard data/checkpoints
☐ C. Keep loss‑scaling consistent if precision unchanged
☐ D. Disable checkpoints to avoid mismatches
Show answers & explanation
Answers & Explanations
1) B — Reverse‑mode AD is efficient for a scalar loss with many parameters (classic NN training). Back
2) C — f(a+εb) = f(a) + f'(a)·ε·b because ε²=0 (first‑order Taylor). Back
3) D — TFLM uses a fixed tensor arena; no malloc/free in the hot path. Back
4) B — XLA fuses ops, picks layouts, and reuses buffers for speed/efficiency. Back
5) D — --gc-sections removes unused sections (with -ffunction/-fdata‑sections). Back
6) C — Intra reduce‑scatter → inter reduce‑scatter → inter all‑gather → intra all‑gather is canonical. Back
7) D — Pairwise masks cancel; threshold recovery fixes dropouts in secure agg. Back
8) A — TF 2.x defaults to eager; tf.function can trace/compile hot paths. Back
9) C — TF 1.x used symbolic graphs executed in a Session (define‑then‑run). Back
10) B — FlatBuffers allow zero‑copy deserialization with schema checks on MCUs. Back
11) D — Radios dominate energy; compute locally to avoid sending bytes. Back
12) C — Conv+BN+ReLU fusion is standard to cut memory traffic/launches. Back
13) A — For NHWC, reading c contiguously (h→w→c) improves cache/coalescing. Back
14) B — Sync all‑reduce does not have weight staleness by design; it has per‑step barriers and straggler sensitivity; it’s equivalent to a large batch on one device (with same optimizer/schedule). Back
15) B — Async PS can increase throughput and tolerate slow clients (but risks staleness). Back
16) B — ZeRO‑2 shards optimizer states and gradients to cut memory/traffic. Back
17) D — Reduce‑scatter partitions work and accumulates partial reductions across ranks. Back
18) C — TFLM must fail fast if the arena is insufficient (determinism > silent fallback). Back
19) A — MLIR is a multi‑level compiler framework (TF dialect → linalg → LLVM IR), not a runtime. Back
20) D — Hierarchical all‑reduce reduces within nodes before inter‑node, cutting cross‑switch traffic. Back
21) C — Shared backbone aggregated globally + private head on device is a common personalization split in FL. Back
22) A — Aggregating within cohorts (clusters) then combining mitigates non‑IID skew and speeds convergence. Back
23) B — Clip per‑client update norms; then add calibrated noise for client‑level DP guarantees (with an accountant). Back
24) C — Bucketing starts all‑reduce early to overlap comm with remaining backprop compute. Back
25) C — Gloo is a versatile CPU/GPU collective library often used as a fallback (NCCL is GPU‑centric). Back
26) B — Padding/alignment to TPU tile sizes improves systolic utilization. Back
27) A — Tree all‑reduce tends to have lower latency for small messages than ring (fewer hops). Back
28) D — CSE avoids recomputing repeated subgraphs; complements fusion and scheduling passes. Back
29) A, B, C — -Os, LTO, and --gc‑sections shrink binaries; verbose logs increase size. Back
30) A, B, C — Graphs enable autodiff, global optimization, and memory planning (not “no communication”). Back
31) A, B, C — Quantization, sparsification+error feedback, and low‑rank adapters cut bytes; sending raw data is not used. Back
32) A, B, C — Fixed tiles with padding stream A right/B down; “random tile sizes” break regularity and utilization. Back
33) A, B, C — Fixed arena + no dynamic allocs in hot path + reuse buffers; don’t malloc per inference on MCUs. Back
34) A, B, C — Seed/Save/Pin ensure reproducibility; nondeterministic kernels break it. Back
35) A, B, C — These TFLM patterns keep memory deterministic and fragmentation‑free. Back
36) A, B, C — FSDP all‑gathers JIT, frees promptly, reduce‑scatters grads; it doesn’t require full params on all ranks always. Back
37) A, B, C — b increases shrink bubbles; 1F1B reduces memory but risks staleness; checkpointing trades compute for memory; GPipe is all‑forward then all‑backward. Back
38) A, B, C — Overlap comm/compute, compress gradients, and adjust batching/sync variants to lower s; ignoring stragglers harms stability. Back
39) A, B, D — Contiguous access + safe in‑place ops + allocators/memory pools help bandwidth; many tiny launches hurt throughput. Back
40) A, B, C — Recompute effective batch & LR; rebuild groups/reshard; keep loss scaling consistent; never disable checkpoints. Back