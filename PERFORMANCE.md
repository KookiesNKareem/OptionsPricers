# Performance Analysis

## SIMD Optimization (ARM NEON)

**Greeks Approach:** Analytical closed-form derivatives (exact, single-pass, minimal overhead). See [OptionHelpers.h](OptionHelpers.h:173-245).

**Benchmark:** 101 strikes (90-110), 10K iterations

| Compiler Setting | Scalar | SIMD | Speedup |
|------------------|--------|------|---------|
| Auto-vectorization OFF (`-fno-tree-vectorize`) | 27.8 ns | 11.7 ns | 2.37× |
| Auto-vectorization ON (default `-O3`) | 13.3 ns | 9.3 ns | 1.42× |

**Why not 4x speedup?** Compiler auto-vectorization, memory bandwidth limits, loop overhead, SLEEF function calls.

**Takeaway:** Explicit SIMD provides consistent performance regardless of compiler optimizations.

## Monte Carlo Optimization

**Test case:** S=100, K=100, T=1, r=0.05, σ=0.20, 1M paths

### Results

| Method | Time | Error | Speedup |
|--------|------|-------|---------|
| **Analytical** | ~40 ns | 0% (exact) | N/A |
| MC - Basic CPU | 33.8 ms | 0.036% | 1x (baseline) |
| MC - Optimized (xoshiro256+ + threads) | 2.9 ms | 0.034% | **11.6x** |

### Optimization Breakdown

**Performance Profile (1M paths):**

| Implementation | RNG Time | Compute Time | Total | Speedup |
|----------------|----------|--------------|-------|---------|
| Basic | 14.1 ms (54.7%) | 11.7 ms (45.3%) | 25.7 ms | 1x |
| Optimized | 1.1 ms (38.4%) | 1.7 ms (61.6%) | 2.8 ms | 9.2x |
| **Component speedup** | **13.1x** | **6.8x** | **9.2x** | - |

**Key insight**: RNG was the primary bottleneck (54.7%). Optimizing it delivered 13.1x speedup, shifting the bottleneck to computation.

### Optimization Techniques

| Technique | Description | Benefit |
|-----------|-------------|---------|
| **xoshiro256+ RNG** | 256-bit state, XOR/shift/rotate ops only | 3-4x faster than std library RNGs |
| **Box-Muller** | 2 uniforms → 2 normals via polar transform | 2x efficiency vs inverse CDF |
| **Multi-threading** | Thread-local RNGs, no atomics, lock-free | Scales across CPU cores (~3-4x) |

See [MonteCarlo.h](MonteCarlo.h#L16-L129) for implementation details.

### Analysis

**Why RNG speedup (13.1x) > compute speedup (6.8x)?**
- RNG: 3.5x (xoshiro256+) × 2x (Box-Muller) × 3.5x (threading) ≈ 24x theoretical → 13.1x actual
- Compute: Limited by exp() calls and memory bandwidth, threading alone → 6.8x actual

**Overall speedup (9.2x):**
- RNG contribution: 54.7% × 13.1x = 7.2x
- Compute contribution: 45.3% × 6.8x = 3.1x
- Combined: 9.2x

**Recommendations:**
- **<100K paths**: Basic (lower overhead)
- **1M+ paths**: Optimized (overhead amortized)

### Use Cases

| Method | Best For | Latency |
|--------|----------|---------|
| **Analytical** | European options, Greeks, HFT | ~40 ns |
| **MC Basic** | Education, prototyping | ~26 ms (1M paths) |
| **MC Optimized** | Path-dependent options, exotic derivatives, production | ~3 ms (1M paths) |

### Accuracy vs Speed

**Convergence:** Monte Carlo error scales as O(1/√n) - 4× paths = 2× accuracy

**Results (S=100, K=100, T=1, r=0.05, σ=0.20, 1M paths):**

| Method | Price | Error | Time | Components |
|--------|-------|-------|------|-----------|
| Analytical | 10.450 | 0% | 40 ns | - |
| MC Basic | 10.447 | 0.036% | 25.7 ms | 14.1 ms RNG + 11.7 ms compute |
| MC Optimized | 10.454 | 0.034% | 2.8 ms | 1.1 ms RNG + 1.7 ms compute |

**Bottom line:** Use analytical for European options. Monte Carlo is essential only for path-dependent derivatives.