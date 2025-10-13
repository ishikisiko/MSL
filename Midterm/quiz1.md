1) In which scenario is forward‑mode automatic differentiation the best fit?
☐ A. Computing sensitivities w.r.t. a few inputs for a function with many outputs
☐ B. Training a deep neural network with millions of parameters and a scalar loss
☐ C. Efficiently computing many VJPs (vector–Jacobian products)
☐ D. Finite differences for large parameter vectors
Show answer & explanation
2) In JAX, which transformation most directly vectorizes a function over a batch dimension without manual loops?
☐ A. pmap
☐ B. vmap
☐ C. jit
☐ D. grad
Show answer & explanation
3) On modern NVIDIA GPUs, which tensor layout often yields more coalesced memory access for many 2D conv workloads?
☐ A. NLC
☐ B. CHWN
☐ C. NHWC
☐ D. NCHW
Show answer & explanation
4) Regarding gradient checkpointing in reverse‑mode AD, which is most accurate?
☐ A. Cuts compute ~50% but increases memory
☐ B. Reduces memory O(log n) at ~10× compute
☐ C. Replaces backward with numerical differentiation
☐ D. Reduces peak memory (≈√n layers) with modest compute (~33%)
Show answer & explanation
5) For a ring all‑reduce with payload size P and N devices, the per‑device traffic per step (idealized) is:
☐ A. P·(N−1)/N
☐ B. 2·P/N
☐ C. 2·P·(N−1)/N
☐ D. P·(N−1)
Show answer & explanation
6) Which statement about Amdahl’s Law is correct?
☐ A. With serial fraction s=0.1, the maximum speedup is 10× as N→∞
☐ B. Speedup scales linearly with N for any s<1
☐ C. If s<1, the maximum speedup is unbounded
☐ D. With s=0.1, speedup at N=10 must be exactly 10×
Show answer & explanation
7) Which pipeline scheduler interleaves forward and backward and may introduce weight staleness?
☐ A. GPipe (all‑forward then all‑backward)
☐ B. 1F1B
☐ C. Synchronous data parallel
☐ D. Ring all‑reduce
Show answer & explanation
8) Which property defines dual numbers used for forward‑mode AD?
☐ A. ε = 0
☐ B. ε encodes second‑order terms (ε² ≠ 0)
☐ C. They implement finite differences
☐ D. ε² = 0 (nilpotency)
Show answer & explanation
9) Dual‑number multiplication obeys:
☐ A. (a,b)×(c,d)=(a+c, b+d)
☐ B. (a,b)×(c,d)=(ac, bd)
☐ C. (a,b)×(c,d)=(ac, ad+bc)
☐ D. (a,b)×(c,d)=(a+c, ad−bc)
Show answer & explanation
10) Under severe MCU memory limits for on‑device learning, which approach is most effective?
☐ A. Increase batch size to fill the arena
☐ B. Disable backprop in the last k layers
☐ C. Store all activations in flash and never recompute
☐ D. Aggressive checkpointing with full recomputation during backward
Show answer & explanation
11) Which collective is bandwidth‑optimal for large messages?
☐ A. Ring all‑reduce
☐ B. Tree all‑reduce
☐ C. Parameter server pull/push
☐ D. All‑to‑all
Show answer & explanation
12) Which framework centers on pure functions transformed by jit/grad/vmap/pmap with immutable arrays?
☐ A. TensorFlow 1.x
☐ B. PyTorch (eager + TorchScript)
☐ C. TensorFlow 2.x eager + tf.function
☐ D. JAX
Show answer & explanation
13) On TPUs, which practice generally improves systolic array utilization for dense layers?
☐ A. Pad/align matrix dims to preferred multiples (e.g., 128)
☐ B. Use many tiny GEMMs to vary launch patterns
☐ C. Disable XLA layout optimization
☐ D. Convert all convs to FFT‑based forms by default
Show answer & explanation
14) Which permutation converts NHWC → NCHW?
☐ A. [0,2,3,1]
☐ B. [0,1,3,2]
☐ C. [0,3,1,2]
☐ D. [3,0,1,2]
Show answer & explanation
15) Which framework popularized define‑by‑run dynamic graphs by default?
☐ A. TensorFlow 1.x
☐ B. PyTorch
☐ C. JAX
☐ D. Core ML
Show answer & explanation
16) In privacy‑preserving federated learning:
☐ A. DP alone ensures the server sees only encrypted sums
☐ B. Secure aggregation hides individuals; DP adds calibrated noise for a formal bound
☐ C. Secure aggregation is unnecessary with 8‑bit quantization
☐ D. DP requires sending raw data to the server
Show answer & explanation
17) In elastic distributed training, when world size changes you should:
☐ A. Recompute effective global batch and adjust LR accordingly
☐ B. Preserve the old process groups indefinitely
☐ C. Disable checkpoints to avoid mismatches
☐ D. Keep loss scaling fixed regardless of precision
Show answer & explanation
18) Which statement about FSDP (fully sharded data parallel) is correct?
☐ A. Keeps full parameters resident on every rank always
☐ B. Avoids all-gather in forward
☐ C. Only shards optimizer states
☐ D. Fully shards params/states; all‑gather before compute; reduce‑scatter gradients after
Show answer & explanation
19) In TinyML deployments, which is often the dominant energy cost?
☐ A. Radio communication (e.g., BLE/Wi‑Fi uplink)
☐ B. Scalar additions in ALU
☐ C. Instruction decoding
☐ D. Stack frame setup/teardown
Show answer & explanation
20) Which stack is hardware‑agnostic, imports models, auto‑schedules, and can emit standalone C/LLVM for deployment?
☐ A. XLA-only
☐ B. cuDNN
☐ C. TVM
☐ D. ONNX as a runtime
Show answer & explanation
21) Synchronous data‑parallel training with gradient averaging is equivalent to:
☐ A. Single device with a smaller batch
☐ B. Training only with parameter servers
☐ C. Single device only if FP16 is used
☐ D. Single device with batch equal to sum of local batches (same optimizer/LR schedule)
Show answer & explanation
22) In JAX, which transformation compiles a function for optimized execution while preserving semantics?
☐ A. grad
☐ B. vmap
☐ C. jit
☐ D. pmap
Show answer & explanation
23) For mixed‑precision training stability, which is generally preferred when available?
☐ A. FP64 only
☐ B. BF16 (wider dynamic range) vs FP16
☐ C. INT8 everywhere
☐ D. FP16 without loss scaling
Show answer & explanation
24) Amdahl’s Law upper bound on speedup is:
☐ A. 1 / (1−s·N)
☐ B. (1−s)/N
☐ C. 1 / (s + (1−s)/N)
☐ D. N / (1+s)
Show answer & explanation
25) Which approach reduces cross‑node traffic by keeping heavy reductions inside nodes before inter‑node?
☐ A. Ring all‑reduce
☐ B. Tree all‑reduce
☐ C. Parameter server
☐ D. Hierarchical all‑reduce
Show answer & explanation
26) In FedAvg, which weighting reduces bias when aggregating client updates?
☐ A. Weight by each client’s number of local examples
☐ B. Weight all clients equally regardless of data size
☐ C. Weight by client battery level
☐ D. Weight by client round‑trip time
Show answer & explanation
27) For small messages, which all‑reduce often has lower latency?
☐ A. Ring
☐ B. Tree
☐ C. All‑to‑all
☐ D. Broadcast
Show answer & explanation
28) In JAX, which transformation maps a function SPMD‑style across devices?
☐ A. vmap
☐ B. pmap
☐ C. grad
☐ D. jit
Show answer & explanation
29) (Select all that apply) Effective MCU optimizations for depthwise convolution:
☐ A. Precompute index/offset tables to avoid repeated address arithmetic
☐ B. Use wider, aligned loads (e.g., 8‑byte) when safe
☐ C. Loop unrolling to improve instruction‑level parallelism
☐ D. Frequent malloc/free per row to reduce peak use
Show answers & explanation
30) (Select all that apply) Federated learning client‑scheduling practices that reduce round time and bias:
☐ A. Over‑invite and accept the first N updates before deadline
☐ B. Stratified sampling across device or data strata
☐ C. Cap per‑client participation frequency
☐ D. Always wait for all stragglers beyond the deadline
Show answers & explanation
31) (Select all that apply) Differential privacy in federated learning:
☐ A. Client‑side norm clipping + adding Gaussian noise before secure aggregation
☐ B. Server decrypts individual updates and adds noise later
☐ C. Track ε, δ with a privacy accountant and stop/adjust at budget limit
☐ D. Transport encryption alone suffices for DP guarantees
Show answers & explanation
32) (Select all that apply) Energy‑efficiency strategies for TinyML deployments:
☐ A. Duty cycling (short active bursts, long sleeps)
☐ B. Cascaded detection (cheap stage gates a heavier classifier)
☐ C. Compute more locally to transmit fewer bytes
☐ D. Trigger training during active user interaction windows only
Show answers & explanation
33) (Select all that apply) Memory management best practices for TFLM on MCUs:
☐ A. Use a fixed tensor arena; avoid heap allocation in the hot path
☐ B. Immediate error if arena is too small
☐ C. Preallocate and reuse buffers; avoid fragmentation
☐ D. Prefer malloc/free per inference for flexibility
Show answers & explanation
34) (Select all that apply) Pipeline parallelism statements:
☐ A. Bubble fraction per step ≈ (m−1)/b
☐ B. 1F1B reduces bubbles/activation footprint but may introduce weight staleness
☐ C. GPipe interleaves forward/backward by design
☐ D. Increasing microbatches reduces bubbles but can increase memory/latency
Show answers & explanation
35) (Select all that apply) Phases typically present in hierarchical all‑reduce:
☐ A. Intra‑node reduce‑scatter
☐ B. Inter‑node reduce‑scatter
☐ C. Inter‑node all‑gather
☐ D. Intra‑node all‑gather
Show answers & explanation
36) (Select all that apply) ZeRO optimizer stages:
☐ A. Stage 1: shard optimizer states
☐ B. Stage 2: shard optimizer states + gradients
☐ C. Stage 3: shard parameters as well (optionally offload)
☐ D. ZeRO eliminates the need for collectives entirely
Show answers & explanation
37) (Select all that apply) FSDP behavior:
☐ A. All‑gather param shards before compute; free promptly; reduce‑scatter grads after
☐ B. Works with mixed precision to reduce memory
☐ C. Requires full params to be present on all ranks at all times
☐ D. Wrapping/granularity choices impact comm/compute balance
Show answers & explanation
38) (Select all that apply) Reproducibility at scale:
☐ A. Seed global and per‑device RNGs (Python/NumPy/framework/CUDA)
☐ B. Prefer nondeterministic kernels to improve speed
☐ C. Save RNG states and dataloader epoch/offsets in checkpoints
☐ D. Pin versions (CUDA/cuDNN/NCCL, drivers, framework) and log topology/world size
Show answers & explanation
39) (Select all that apply) Reducing serial fraction s / hiding comm for better scaling:
☐ A. Overlap communication with compute (bucketization, fused kernels)
☐ B. Lower‑precision or compressed gradients with error feedback
☐ C. Increase local batch judiciously; consider partial/delayed sync variants with care
☐ D. Ignore stragglers and disable health checks
Show answers & explanation
40) (Select all that apply) Memory bandwidth optimization in kernels:
☐ A. Align loop order with layout to read contiguous memory (NHWC vs NCHW)
☐ B. Prefer many tiny kernel launches for better “interleaving”
☐ C. Use in‑place ops where safe to halve traffic
☐ D. Use memory pools/arenas to avoid runtime malloc/free overhead
Show answers & explanation
Answers & Explanations
1) A — Forward mode scales with inputs; best for few inputs, many outputs. Back to top
2) B — vmap vectorizes across a batch dimension. Back
3) C — NHWC often coalesces better for many conv kernels. Back
4) D — Checkpointing reduces peak memory (~√n) with ~33% compute overhead. Back
5) C — Ring traffic per device ≈ 2·P·(N−1)/N. Back
6) A — Amdahl bound with s=0.1 → max ~10×. Back
7) B — 1F1B interleaves and can introduce weight staleness. Back
8) D — Dual numbers: ε² = 0 (nilpotent). Back
9) C — (ac, ad+bc) implements product rule. Back
10) D — Recompute activations during backward to save memory. Back
11) A — Ring is bandwidth‑optimal for large messages. Back
12) D — JAX functional transforms over immutable arrays. Back
13) A — Align dims to TPU tiles (e.g., 128) for utilization. Back
14) C — NHWC → NCHW is [0,3,1,2]. Back
15) B — PyTorch popularized define‑by‑run. Back
16) B — Secure aggregation hides individuals; DP adds noise & accounting. Back
17) A — Adjust LR to new effective global batch; rebuild groups. Back
18) D — FSDP shards params/states; all‑gather JIT; reduce‑scatter grads. Back
19) A — Radios often dominate energy cost. Back
20) C — TVM imports/auto‑schedules/emits C/LLVM. Back
21) D — Equivalent to single device with global batch = sum of local batches. Back
22) C — jit compiles a function for optimized execution. Back
23) B — BF16 preferred for stability vs FP16. Back
24) C — Amdahl bound: 1 / (s + (1−s)/N). Back
25) D — Hierarchical all‑reduce: intra‑node, then inter‑node, then local broadcast. Back
26) A — Weight by client sample counts. Back
27) B — Tree lower latency for small messages. Back
28) B — pmap distributes SPMD across devices. Back
29) A, B, C — Precompute indices; aligned wide loads; unroll loops. Back
30) A, B, C — Over‑invite/accept first N; stratify; cap frequency. Back
31) A, C — Client‑side clip+noise; privacy accountant for ε,δ. Back
32) A, B, C — Duty cycle; cascades; compute locally to save radio. Back
33) A, B, C — Fixed arena; fail fast if too small; reuse buffers. Back
34) A, B, D — Bubble ≈ (m−1)/b; 1F1B reduces bubbles but may add staleness; more microbatches reduce bubbles with memory/latency trade‑offs. Back
35) A, B, C, D — Hierarchical reduce‑scatter/all‑gather intra‑/inter‑node. Back
36) A, B, C — ZeRO stages 1/2/3 shard states, grads, then params (offload optional). Back
37) A, B, D — FSDP all‑gathers JIT, supports mixed precision, wrapping granularity matters. Back
38) A, C, D — Seed RNGs; save RNG/dataloader state; pin versions/topology. Back
39) A, B, C — Overlap comm/compute; compress gradients; adjust batch/sync variants carefully. Back
40) A, C, D — Contiguous access, in‑place where safe, memory pools; avoid many tiny launches. Back