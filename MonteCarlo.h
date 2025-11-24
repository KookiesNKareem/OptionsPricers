#ifndef MONTECARLO_H
#define MONTECARLO_H

#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>
#include <cstdint>
#include <thread>
#include <vector>

using namespace std;

// xoshiro256+ RNG - much faster than other RNGs
class Xoshiro256Plus {
private:
    uint64_t s[4];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    Xoshiro256Plus(uint64_t seed) {
        s[0] = seed;
        s[1] = seed * 0x9e3779b97f4a7c15ULL;
        s[2] = seed * 0x94d049bb133111ebULL;
        s[3] = seed * 0xbf58476d1ce4e5b9ULL;

        // warm up
        for(int i = 0; i < 20; i++) next();
    }

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

    // generate uniform double in [0, 1)
    double uniform() {
        return (next() >> 11) * 0x1.0p-53;
    }
};

// Box-Muller transform, convert uniform to normal
inline void box_muller(double u1, double u2, double& z1, double& z2) {
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    z1 = r * cos(theta);
    z2 = r * sin(theta);
}

// parallel Monte Carlo with fast xoshiro256+ RNG
double simulate_paths(double S0, double K, double T,
    double r, double sigma, int N, int num_paths)
{
    double drift = (r - 0.5*sigma*sigma) * T;
    double diffusion = sigma * sqrt(T);
    double discount = exp(-r * T);

    // # of threads
    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 8; // fallback

    vector<double> thread_payoffs(num_threads, 0.0);
    vector<thread> threads;

    // lambda for each thread
    auto worker = [&](int thread_id) {
        // each thread has an RNG
        uint64_t seed = 42 + thread_id * 123456789ULL;
        Xoshiro256Plus rng(seed);

        // calculate this thread's chunk
        int paths_per_thread = num_paths / num_threads;
        int start = thread_id * paths_per_thread;
        int end = (thread_id == num_threads - 1) ? num_paths : start + paths_per_thread;

        double local_payoff = 0.0;

        for (int i = start; i < end; i += 2) {
            // generate normals using Box-Muller
            double u1 = rng.uniform();
            double u2 = rng.uniform();
            double z1, z2;
            box_muller(u1, u2, z1, z2);

            // Path 1
            double ST1 = S0 * exp(drift + diffusion * z1);
            local_payoff += max(ST1 - K, 0.0);

            // Path 2
            if (i + 1 < end) {
                double ST2 = S0 * exp(drift + diffusion * z2);
                local_payoff += max(ST2 - K, 0.0);
            }
        }

        thread_payoffs[thread_id] = local_payoff;
    };

    // Launch threads
    for (unsigned int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    // Sum up all thread results
    double total_payoff = 0.0;
    for (double payoff : thread_payoffs) {
        total_payoff += payoff;
    }

    return discount * (total_payoff / num_paths);
}

// unoptimized Monte Carlo
double mc_basic(double S, double K, double T, double r, double sigma, int num_paths) {
    random_device rd;
    default_random_engine gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    double drift = (r - 0.5 * sigma * sigma) * T;
    double diffusion = sigma * sqrt(T);
    double discount = exp(-r * T);

    double sum_payoff = 0.0;
    for (int i = 0; i < num_paths; i++) {
        double Z = dist(gen);
        double ST = S * exp(drift + diffusion * Z);
        double payoff = max(ST - K, 0.0);
        sum_payoff += payoff;
    }

    return discount * (sum_payoff / num_paths);
}

#endif
