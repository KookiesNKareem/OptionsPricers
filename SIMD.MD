# SIMD Optimization Details

## Challenges

ARM NEON lacks intrinsics for transcendental functions needed in Black-Scholes:
- No vectorized `log()` or `exp()`
- No vectorized `erf()` for computing N(x)

## Solution: SLEEF Integration

Using [SLEEF](https://sleef.org/) (SIMD Library for Evaluating Elementary Functions) to provide:
- `Sleef_logf4_u10()` - vectorized logarithm
- `Sleef_erff4_u10()` - vectorized error function
- `Sleef_expf4_u10()` - vectorized exponential (if needed)

This enables processing 4 options in parallel on ARM NEON.

## Vectorized CDF Implementation

```cpp
float32x4_t N_simd(float32x4_t x)
{
    const float32x4_t half = vdupq_n_f32(0.5);
    const float32x4_t inv_sqrt2 = vdupq_n_f32(0.707106781);

    float32x4_t x_scaled = vmulq_f32(x, inv_sqrt2);
    float32x4_t erf_result = Sleef_erff4_u10(x_scaled);

    float32x4_t sum = vaddq_f32(vdupq_n_f32(1.0), erf_result);
    return vmulq_f32(sum, half);
}
```

Computes N(x) = 0.5 × (1 + erf(x/√2)) for 4 values simultaneously.

## Memory Layout Strategy

```
Input strikes:  [K₀, K₁, K₂, K₃]  → vld1q_f32()
Broadcast S:    [S, S, S, S]      → vdupq_n_f32(S)
Compute d1:     [d1₀, d1₁, d1₂, d1₃]  (parallel)
Compute d2:     [d2₀, d2₁, d2₂, d2₃]  (parallel)
Result prices:  [P₀, P₁, P₂, P₃]  → vst1q_f32()
Result deltas:  [Δ₀, Δ₁, Δ₂, Δ₃]  → vst1q_f32()
```

**Key operations:**
- Load 4 strikes as a vector
- Broadcast common parameters (S, T, r, σ) to all lanes
- Compute all intermediate values vectorized
- Store results back to arrays

## Implementation Notes

The SIMD version in [OptionHelpers.h](OptionHelpers.h:78-117) computes:
- **Price** and **delta** only (not full Greeks)
- **Calls only** (puts would require separate implementation)
- Pre-computes scalar values (√T, e^(-rT), etc.) to avoid redundant work
- Uses NEON intrinsics explicitly for predictable performance