# Black-Scholes Options Pricer

C++ implementation of options pricing with both analytical and Monte Carlo methods.

## Pricing Approaches

### 1. Analytical Black-Scholes (Real-Time)
- **Latency**: ~40 ns per option
- **Accuracy**: Exact closed-form solution
- **SIMD**: ARM NEON intrinsics for 4x parallelization
- **Greeks**: Delta, gamma, vega, theta, rho computed analytically
- **Use case**: European options, high-frequency trading, real-time systems

### 2. Monte Carlo Simulation (Risk Analysis)
- **Latency**: 25.7 ms (basic) / 2.8 ms (optimized) for 1M paths
- **Accuracy**: ~0.03-0.04% error (converges as O(√n))
- **Optimizations**: xoshiro256+ RNG, Box-Muller transform, multi-threading (9.2x speedup)
- **Use case**: Path-dependent options, exotic derivatives, portfolio risk

## Build

```bash
# Install dependencies
brew install sleef eigen

# Compile
g++ -std=c++20 -O3 -march=native -ffast-math \
    -I/opt/homebrew/include -I/opt/homebrew/include/eigen3 \
    main.cpp \
    -L/opt/homebrew/lib -lsleef \
    -o pricer

# Run benchmarks
./pricer
```

Or use the Makefile:
```bash
make
make run
```

## Usage

### Single Option Pricing

```cpp
OptionParams call = {
    .S = 100.0,     // Spot price
    .T = 1.0,       // Time to expiry (years)
    .K = 100.0,     // Strike price
    .r = 0.05,      // Risk-free rate
    .sigma = 0.20,  // Volatility
    .type = OptionType::CALL
};

AnalyticalPricer pricer;
OptionPrice result = pricer.price(call);

// Access price and all Greeks
std::cout << "Price: " << result.price << "\n";
std::cout << "Delta: " << result.delta << "\n";
std::cout << "Gamma: " << result.gamma << "\n";
```

### SIMD Batch Pricing

```cpp
AnalyticalPricer pricer;
float strikes[4] = {95.0f, 100.0f, 105.0f, 110.0f};

PriceResult4 result = pricer.price_call_simd(
    100.0f,  // Spot
    1.0f,    // Time
    strikes, // 4 strikes
    0.05f,   // Rate
    0.20f    // Vol
);

// Access results (price and delta only)
for (int i = 0; i < 4; i++) {
    std::cout << "K=" << strikes[i]
              << " Price=" << result.price[i]
              << " Delta=" << result.delta[i] << "\n";
}
```

## Implementation

### Black-Scholes Formula

**Call price:**
```
C = S × N(d₁) - K × e^(-rT) × N(d₂)

d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

Where N(x) is the cumulative standard normal distribution.

### Greeks (Closed-Form)

- **Delta**: N(d₁) for calls, N(d₁) - 1 for puts
- **Gamma**: n(d₁) / (S × σ × √T)
- **Vega**: S × √T × n(d₁)
- **Theta**: -(S × n(d₁) × σ)/(2√T) - r × K × e^(-rT) × N(d₂)
- **Rho**: K × T × e^(-rT) × N(d₂)

Where n(x) is the standard normal PDF.

## Architecture

- **OptionHelpers.h**: Analytical Black-Scholes with ARM NEON SIMD optimization
- **MonteCarlo.h**: Monte Carlo implementations (basic CPU, multi-threaded CPU with xoshiro256+)
- **testing.h**: Benchmark suite comparing all pricing methods
- **main.cpp**: Runs all benchmarks

## Performance Summary

| Method | Latency (1M paths) | Speedup | Use Case |
|--------|---------|---------|----------|
| Analytical | ~40 ns | N/A | European options, real-time trading |
| SIMD (4 options) | ~9 ns/option | 2.4x | Batch pricing, option chains |
| MC - Basic CPU | 25.7 ms | 1x | Path-dependent, simple implementation |
| MC - Optimized (xoshiro256+ + threads) | 2.8 ms | **9.2x** | Large-scale risk, high path counts |

**Monte Carlo Optimizations (1M paths benchmark):**
- **RNG speedup**: 13.1x (xoshiro256+ + Box-Muller + threading: 1.1 ms vs 14.1 ms)
- **Compute speedup**: 6.8x (multi-threaded path calculation: 1.7 ms vs 11.7 ms)
- **Overall speedup**: 9.2x (2.8 ms vs 25.7 ms total)

**Bottleneck analysis:**
- Basic: 54.7% RNG, 45.3% computation
- Optimized: 38.4% RNG, 61.6% computation
- RNG optimization had biggest impact, reducing it from dominant bottleneck to minor component

See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks and [SIMD.MD](SIMD.MD) for implementation details.