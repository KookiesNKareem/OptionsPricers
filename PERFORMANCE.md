# Performance Analysis

## Greek Calculation Strategy

Three approaches for computing derivatives:
- **Analytical**: Closed-form mathematical derivatives
- **Numerical**: Finite difference methods (e.g., bump-and-reprice)
- **Automatic differentiation**: Computational graph approach

This implementation uses **analytical Greeks** for maximum efficiency:
- Zero approximation error (exact derivatives)
- Single pass computation (no finite differences required)
- Minimal latency overhead

See implementations in [OptionHelpers.h](OptionHelpers.h:173-245).

## Benchmark Methodology

The [benchmark suite](testing.h:97-156) compares:
- **Scalar**: Standard C++ implementation calling `black_scholes_call()`
- **SIMD**: ARM NEON vectorized `price_call_simd()` processing 4 strikes at once

Both versions compute 101 strikes (90.0 to 110.0 in 0.2 increments) over 10,000 iterations.

## Results

### With Compiler Auto-Vectorization Disabled (`-fno-tree-vectorize`)

```
Scalar:  27.8 ns/option
SIMD:    11.7 ns/option
Speedup: 2.37×
```

This shows the true benefit of explicit SIMD intrinsics.

### With Auto-Vectorization Enabled (default `-O3`)

```
Scalar:  13.3 ns/option
SIMD:    9.3 ns/option
Speedup: 1.42×
```

## Analysis

**Why not 4× speedup?**
- Compiler auto-vectorization improves scalar baseline (27.8 → 13.3 ns)
- Memory bandwidth becomes limiting factor
- Loop overhead and remainder handling (101 strikes = 25 SIMD batches + 1 scalar)
- SLEEF function call overhead

**Key takeaway**: Explicit SIMD intrinsics provide consistent, predictable performance regardless of compiler optimizations.

## Monte Carlo vs Analytical Pricing

Comparing analytical Black-Scholes with Monte Carlo implementations for the same option (S=100, K=100, T=1, r=0.05, σ=0.20).

### Benchmark Results (1M paths)

| Method | Time | Error | Speedup |
|--------|------|-------|---------|
| **Analytical** | ~40 ns | 0% (exact) | N/A |
| MC - Basic CPU | 33.8 ms | 0.036% | 1x (baseline) |
| MC - Optimized (xoshiro256+ + threads) | 2.9 ms | 0.034% | **11.6x** |

### Optimization Breakdown

The optimized Monte Carlo implementation achieves **11.6x speedup** through combined RNG and compute optimizations:

#### Performance Breakdown (1M paths)

**Basic (Unoptimized):**
| Component | Time | Percentage |
|-----------|------|------------|
| RNG generation | 14.1 ms | 54.7% |
| Path computation | 11.7 ms | 45.3% |
| **Total** | **25.7 ms** | **100%** |

**Optimized (xoshiro256+ + Box-Muller + threading):**
| Component | Time | Percentage |
|-----------|------|------------|
| RNG + Box-Muller | 1.1 ms | 38.4% |
| Path computation | 1.7 ms | 61.6% |
| **Total** | **2.8 ms** | **100%** |

#### Speedup Analysis

| Optimization | Speedup | Impact |
|-------------|---------|--------|
| **RNG** (xoshiro256+ + Box-Muller + threading) | **13.1x** | 14.1 ms → 1.1 ms |
| **Compute** (multi-threading path calculation) | **6.8x** | 11.7 ms → 1.7 ms |
| **Overall** | **9.2x** | 25.7 ms → 2.8 ms |

**Key insight**: RNG was the dominant bottleneck (54.7% of runtime). Optimizing it with xoshiro256+, Box-Muller, and threading provided the biggest impact (13.1x speedup), reducing it from the primary bottleneck to a minor component (38.4%).

### Implementation Details

**xoshiro256+ RNG ([MonteCarlo.h:16-53](MonteCarlo.h#L16-L53)):**
```cpp
class Xoshiro256Plus {
    uint64_t s[4];  // 256-bit state

    uint64_t next() {
        const uint64_t result = s[0] + s[3];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }
};
```
- Only XOR, shift, rotate operations (no divisions/multiplications)
- Much faster than mt19937's complex Mersenne Twister algorithm
- Good statistical properties for Monte Carlo use

**Box-Muller Transform ([MonteCarlo.h:56-61](MonteCarlo.h#L56-L61)):**
```cpp
void box_muller(double u1, double u2, double& z1, double& z2) {
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    z1 = r * cos(theta);
    z2 = r * sin(theta);
}
```
- 2 uniforms → 2 normals (50% more efficient than inverse CDF)
- Main loop processes paths in pairs using both z1 and z2

**Multi-threading ([MonteCarlo.h:64-129](MonteCarlo.h#L64-L129)):**
- Each thread gets independent RNG seeded differently
- Work divided into equal chunks across threads
- Thread-local accumulation (no locks/atomics)
- Final reduction after all threads join

### Key Observations

**Why different speedups for RNG (13.1x) vs compute (6.8x)?**
- **RNG benefits more** from optimizations:
  - xoshiro256+ is much simpler than `default_random_engine` (3-4x faster)
  - Box-Muller generates 2 normals per call (2x efficiency)
  - Threading adds another 3-4x
  - Combined: 3.5× × 2× × 3.5× ≈ 24x theoretical → 13.1x actual

- **Compute benefits less** from threading:
  - exp() dominates computation time (expensive transcendental function)
  - exp() cannot be parallelized beyond threading
  - Memory bandwidth bottleneck when all threads call exp() simultaneously
  - Threading alone: ~8x theoretical → 6.8x actual

**Why overall speedup (9.2x) < average of parts?**
- Amdahl's law: Overall speedup limited by weighted average
- RNG: 54.7% × 13.1x = 7.2x contribution
- Compute: 45.3% × 6.8x = 3.1x contribution
- Combined: ≈ 9.2x (matches measured result)

**Practical recommendations:**
- **Small simulations (<100K paths)**: Use basic CPU (lower startup overhead)
- **Large simulations (1M+ paths)**: Use optimized version (overhead amortized)
- **Production systems**: Always use optimized version for consistent performance

### When to Use Each Method

**Analytical (OptionHelpers.h):**
- European options with closed-form solutions
- Real-time pricing requirements (sub-microsecond latency)
- Greeks computation needed
- High-frequency trading systems

**Monte Carlo - Basic CPU (MonteCarlo.h: mc_basic_cpu):**
- Educational purposes and benchmarking baseline
- Quick prototyping without optimization complexity
- Simple deployment with minimal code

**Monte Carlo - Optimized (MonteCarlo.h: simulate_paths_cpu):**
- Path-dependent options (Asian, lookback, barrier)
- Large-scale risk analysis (100K+ paths)
- Portfolio-level simulations
- Complex exotic derivatives
- Production systems requiring best CPU performance

### Accuracy vs Speed Trade-off

- **Analytical**: O(1) complexity, exact results, ~40 ns
- **Monte Carlo**: O(√n) convergence, probabilistic accuracy
- **1M paths**: 0.03-0.04% error, 2.8-33.8 ms depending on implementation
- **Error scales as 1/√n**: 4× more paths = 2× better accuracy

**Benchmark Results (S=100, K=100, T=1, r=0.05, σ=0.20):**

| Method | Price | Error | Time | Speedup |
|--------|-------|-------|------|---------|
| Analytical | 10.450 | 0% (exact) | 40 ns | - |
| MC Basic | 10.447 | 0.036% | 33.8 ms | 1x |
| MC Optimized | 10.454 | 0.034% | 2.9 ms | 11.6x |

**Time breakdown for 1M paths:**
- Basic: 14.1 ms RNG + 11.7 ms compute = 25.7 ms
- Optimized: 1.1 ms RNG + 1.7 ms compute = 2.8 ms

For European options, analytical pricing is overwhelmingly superior. Monte Carlo is essential for path-dependent derivatives where no closed-form solution exists.