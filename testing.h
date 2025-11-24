#ifndef TESTING_H
#define TESTING_H

#include "OptionHelpers.h"
#include "MonteCarlo.h"
#include <chrono>
#include <iostream>

using namespace std;


template<typename T>
inline void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

void benchmark_single(AnalyticalPricer& pricer) {
    OptionParams test = {100.0, 1.0, 100.0, 0.05, 0.20, OptionType::CALL};
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100000; i++) {
        OptionPrice result = pricer.price(test);
        DoNotOptimize(result);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    
    cout << "Single option + Greeks: " << duration.count() / 100000 << " ns\n";
}

void benchmark_chain(AnalyticalPricer& pricer) {
    // price 100 strikes from 90 to 110
    vector<double> strikes;
    for (int i = 90; i <= 110; i++) {
        strikes.push_back(i);
        DoNotOptimize(i);
    }
    
    auto start = chrono::high_resolution_clock::now();
    
    for (int run = 0; run < 1000; run++) {
        for (double K : strikes) {
            OptionParams test = {100.0, 1.0, K, 0.05, 0.20, OptionType::CALL};
            OptionPrice result = pricer.price(test);
            DoNotOptimize(result);
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    
    cout << "100 strikes: " << duration.count() / 1000.0 << " μs\n";
    cout << "Per strike: " << duration.count() / 100000.0 << " μs\n";
}

void profile_components(AnalyticalPricer& pricer) {
    OptionParams test = {100.0, 1.0, 100.0, 0.05, 0.20, OptionType::CALL};
    
    // just N(x), vary strike to avoid compiler optimization
    double sum1 = 0.0;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
        double x = 0.35 + (i % 100) * 0.0001;
        double result = pricer.N(x);
        sum1 += result;
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "N(x) call: " << duration.count() / 100000 << " ns (sum=" << sum1 << ")\n";
    
    // just pricing, vary strike to avoid compiler optimizations
    double sum2 = 0.0;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
        test.K = 100.0 + (i % 100) * 0.1;
        double price = pricer.black_scholes_call(test);
        sum2 += price;
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "Price only: " << duration.count() / 100000 << " ns (sum=" << sum2 << ")\n";
    
    // full greeks, varying strikes to avoid compiler optimization
    double sum3 = 0.0;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
        test.K = 100.0 + (i % 100) * 0.1;
        OptionPrice result = pricer.price(test);
        sum3 += result.price;
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::nanoseconds>(end - start);
    cout << "Price + Greeks: " << duration.count() / 100000 << " ns (sum=" << sum3 << ")\n";
}

void benchmark_simd_vs_scalar(AnalyticalPricer& pricer) {
    // generate 100 strikes
    vector<float> strikes;
    for (float K = 90.0f; K < 110.0f; K += 0.2f) {
        strikes.push_back(K);
    }
    int n_strikes = strikes.size();
    
    cout << "Pricing " << n_strikes << " strikes\n\n";
    
    // scalar
    double sum_scalar = 0.0;
    auto start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < 10000; iter++) {
        for (float K : strikes) {
            OptionParams p = {100.0f, 1.0f, K, 0.05f, 0.20f, OptionType::CALL};
            double price = pricer.black_scholes_call(p);
            sum_scalar += price;
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto scalar_time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    
    cout << "SCALAR:\n";
    cout << "  Total: " << scalar_time / 10000.0 << " ns\n";
    cout << "  Per option: " << scalar_time / (10000.0 * n_strikes) << " ns\n";
    cout << "  Sum: " << sum_scalar << "\n\n";
    
    // SIMD
    double sum_simd = 0.0;
    start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < 10000; iter++) {
        int i = 0;
        // groups of 4
        for (; i + 3 < n_strikes; i += 4) {
            float K4[4] = {strikes[i], strikes[i+1], strikes[i+2], strikes[i+3]};
            PriceResult4 result = pricer.price_call_simd(100.0f, 1.0f, K4, 0.05f, 0.20f);
            sum_simd += result.price[0] + result.price[1] + result.price[2] + result.price[3];
        }
        
        // remainder with scalar
        for (; i < n_strikes; i++) {
            OptionParams p = {100.0f, 1.0f, strikes[i], 0.05f, 0.20f, OptionType::CALL};
            sum_simd += pricer.black_scholes_call(p);
        }
    }
    
    end = chrono::high_resolution_clock::now();
    auto simd_time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    
    cout << "SIMD:\n";
    cout << "  Total: " << simd_time / 10000.0 << " ns\n";
    cout << "  Per option: " << simd_time / (10000.0 * n_strikes) << " ns\n";
    cout << "  Sum: " << sum_simd << "\n\n";
    
    cout << "SPEEDUP: " << (double)scalar_time / simd_time << "x\n";
}

void profile_monte_carlo_breakdown() {
    OptionParams test = {100.0, 1.0, 100.0, 0.05, 0.20, OptionType::CALL};
    int paths = 1000000;

    cout << "\n=== Monte Carlo Performance Breakdown (1M paths) ===\n\n";

    double drift = (test.r - 0.5*test.sigma*test.sigma) * test.T;
    double diffusion = test.sigma * sqrt(test.T);

    // ===== BASIC (UNOPTIMIZED) =====

    // Benchmark 1: RNG only for basic
    auto start = chrono::high_resolution_clock::now();
    {
        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> dist(0.0, 1.0);

        double dummy = 0.0;
        for (int i = 0; i < paths; i++) {
            double Z = dist(gen);
            dummy += Z;
        }
        DoNotOptimize(dummy);
    }
    auto end = chrono::high_resolution_clock::now();
    auto rng_basic = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Benchmark 2: Full basic computation
    start = chrono::high_resolution_clock::now();
    double price_basic = mc_basic(test.S, test.K, test.T, test.r, test.sigma, paths);
    end = chrono::high_resolution_clock::now();
    auto total_basic = chrono::duration_cast<chrono::microseconds>(end - start).count();

    auto compute_basic = total_basic - rng_basic;

    // ===== OPTIMIZED =====

    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8;

    // Benchmark 3: RNG only for optimized (xoshiro256+ + Box-Muller + threads)
    start = chrono::high_resolution_clock::now();
    {
        vector<thread> threads;
        for (unsigned int tid = 0; tid < num_threads; tid++) {
            threads.emplace_back([&](int thread_id) {
                uint64_t seed = 42 + thread_id * 123456789ULL;
                Xoshiro256Plus rng(seed);

                int paths_per_thread = paths / num_threads;
                int start_idx = thread_id * paths_per_thread;
                int end_idx = (thread_id == num_threads - 1) ? paths : start_idx + paths_per_thread;

                double dummy = 0.0;
                for (int i = start_idx; i < end_idx; i += 2) {
                    double u1 = rng.uniform();
                    double u2 = rng.uniform();
                    double z1, z2;
                    box_muller(u1, u2, z1, z2);
                    dummy += z1 + z2;
                }
                DoNotOptimize(dummy);
            }, tid);
        }
        for (auto& t : threads) t.join();
    }
    end = chrono::high_resolution_clock::now();
    auto rng_optimized = chrono::duration_cast<chrono::microseconds>(end - start).count();

    // Benchmark 4: Full optimized computation
    start = chrono::high_resolution_clock::now();
    double price_optimized = simulate_paths(test.S, test.K, test.T, test.r, test.sigma, 252, paths);
    end = chrono::high_resolution_clock::now();
    auto total_optimized = chrono::duration_cast<chrono::microseconds>(end - start).count();

    auto compute_optimized = total_optimized - rng_optimized;

    // ===== RESULTS =====

    cout << "BASIC (Unoptimized):\n";
    cout << "  RNG generation:    " << rng_basic << " μs (" << (rng_basic/1000.0) << " ms) - "
         << (100.0*rng_basic/total_basic) << "%\n";
    cout << "  Path computation:  " << compute_basic << " μs (" << (compute_basic/1000.0) << " ms) - "
         << (100.0*compute_basic/total_basic) << "%\n";
    cout << "  Total time:        " << total_basic << " μs (" << (total_basic/1000.0) << " ms)\n";
    cout << "  Price:             " << price_basic << "\n\n";

    cout << "OPTIMIZED (xoshiro256+ + Box-Muller + threads):\n";
    cout << "  RNG + Box-Muller:  " << rng_optimized << " μs (" << (rng_optimized/1000.0) << " ms) - "
         << (100.0*rng_optimized/total_optimized) << "%\n";
    cout << "  Path computation:  " << compute_optimized << " μs (" << (compute_optimized/1000.0) << " ms) - "
         << (100.0*compute_optimized/total_optimized) << "%\n";
    cout << "  Total time:        " << total_optimized << " μs (" << (total_optimized/1000.0) << " ms)\n";
    cout << "  Price:             " << price_optimized << "\n\n";

    cout << "SPEEDUP BREAKDOWN:\n";
    cout << "  RNG speedup:       " << (double)rng_basic / rng_optimized << "x\n";
    cout << "  Compute speedup:   " << (double)compute_basic / compute_optimized << "x\n";
    cout << "  Overall speedup:   " << (double)total_basic / total_optimized << "x\n\n";
}

void benchmark_monte_carlo_vs_analytical(AnalyticalPricer& pricer) {
    OptionParams test = {100.0, 1.0, 100.0, 0.05, 0.20, OptionType::CALL};

    cout << "\n=== Analytical vs Monte Carlo Implementations ===\n\n";

    // analytical pricing
    auto start = chrono::high_resolution_clock::now();
    double analytical = pricer.black_scholes_call(test);
    auto end = chrono::high_resolution_clock::now();
    auto analytical_time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

    cout << "ANALYTICAL (Real-Time Pricing):\n";
    cout << "  Price: " << analytical << "\n";
    cout << "  Time: " << analytical_time << " ns\n";

    // test with 1M paths to amortize thread overhead
    int paths = 1000000;

    cout << "--- Monte Carlo Comparison (" << paths << " paths) ---\n\n";

    // basic CPU
    start = chrono::high_resolution_clock::now();
    double price_basic = mc_basic(test.S, test.K, test.T, test.r, test.sigma, paths);
    end = chrono::high_resolution_clock::now();
    auto time_basic = chrono::duration_cast<chrono::microseconds>(end - start).count();

    cout << "MC - Basic CPU (Unoptimized):\n";
    cout << "  Price: " << price_basic << "\n";
    cout << "  Error: " << abs(price_basic - analytical) << " (" << (abs(price_basic - analytical)/analytical*100) << "%)\n";
    cout << "  Time: " << time_basic << " μs (" << (time_basic/1000.0) << " ms)\n\n";

    // Optimized CPU
    start = chrono::high_resolution_clock::now();
    double price_optimized = simulate_paths(test.S, test.K, test.T, test.r, test.sigma, 252, paths);
    end = chrono::high_resolution_clock::now();
    auto time_optimized = chrono::duration_cast<chrono::microseconds>(end - start).count();

    cout << "MC - Optimized (xoshiro256+ + threads):\n";
    cout << "  Price: " << price_optimized << "\n";
    cout << "  Error: " << abs(price_optimized - analytical) << " (" << (abs(price_optimized - analytical)/analytical*100) << "%)\n";
    cout << "  Time: " << time_optimized << " μs (" << (time_optimized/1000.0) << " ms)\n";
    cout << "  Speedup: " << (double)time_basic / time_optimized << "x\n";

    // Performance breakdown
    profile_monte_carlo_breakdown();
}

#endif