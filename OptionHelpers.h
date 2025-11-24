#ifndef OPTIONHELPERS_H
#define OPTIONHELPERS_H

#include <cmath>
#include <arm_neon.h>
#include <sleef.h>

using namespace std;

enum OptionType{
    CALL,
    PUT
};

struct DValues
{
    double d1, d2;
};

struct OptionParams{
    double S;
    double T;
    double K;
    double r;
    double sigma;
    OptionType type;
};

struct OptionPrice {
    double price;
    double gamma;
    double theta;
    double delta;
    double rho;
    double vega;
};

struct PriceResult4 {
    float price[4];
    float delta[4];
};

class AnalyticalPricer {
    public:
        OptionPrice price(const OptionParams& params);

        double N(double x); // cumulative normal
        double n(double x); // normal probability density function
        DValues calculate_d(const OptionParams& p);
        double black_scholes_call(const OptionParams& p);
        double black_scholes_put(const OptionParams& p);
        double delta_call(const OptionParams& p);
        double delta_put(const OptionParams& p);
        double gamma(const OptionParams& p);
        double vega(const OptionParams& p);
        double theta_call(const OptionParams& p);
        double theta_put(const OptionParams& p);
        double rho_call(const OptionParams& p);
        double rho_put(const OptionParams& p);
        PriceResult4 price_call_simd(float S, float T, const float K[4], float r, float sigma);
};


float32x4_t N_simd(float32x4_t x)
{
    const float32x4_t half = vdupq_n_f32(0.5);
    const float32x4_t one = vdupq_n_f32(1.0);
    const float32x4_t inv_sqrt2 = vdupq_n_f32(0.707106781);

    float32x4_t x_scaled = vmulq_f32(x, inv_sqrt2);
    float32x4_t erf_result = Sleef_erff4_u10(x_scaled);

    float32x4_t sum = 1 + erf_result;
    return vmulq_f32(sum, half);
}


PriceResult4 AnalyticalPricer::price_call_simd(float S, float T, const float K[4],
                              float r, float sigma) {
    // scalars
    float sqrt_T = sqrtf(T);
    float sigma_sqrt_T = sigma * sqrt_T;
    float exp_neg_rT = expf(-r * T);
    float drift_term = (r + 0.5f * sigma * sigma) * T;
    
    // load and broadcast into float32x4_t
    float32x4_t K_vec = vld1q_f32(K);
    float32x4_t S_vec = vdupq_n_f32(S);
    float32x4_t sigma_sqrt_T_vec = vdupq_n_f32(sigma_sqrt_T);
    float32x4_t drift_vec = vdupq_n_f32(drift_term);
    float32x4_t exp_neg_rT_vec = vdupq_n_f32(exp_neg_rT);
    
    // calculate d1
    float32x4_t S_over_K = vdivq_f32(S_vec, K_vec);
    float32x4_t log_S_over_K = Sleef_logf4_u10(S_over_K);
    float32x4_t numerator = vaddq_f32(log_S_over_K, drift_vec);
    float32x4_t d1 = vdivq_f32(numerator, sigma_sqrt_T_vec);
    

    // calculate d2
    float32x4_t d2 = vsubq_f32(d1, sigma_sqrt_T_vec);
    
    float32x4_t Nd1 = N_simd(d1);
    float32x4_t Nd2 = N_simd(d2);

    // S * N(d1) - K * exp(-rT) * N(d2)
    float32x4_t term1 = vmulq_f32(S_vec, Nd1);
    float32x4_t K_discounted = vmulq_f32(K_vec, exp_neg_rT_vec);
    float32x4_t term2 = vmulq_f32(K_discounted, Nd2);
    float32x4_t prices = vsubq_f32(term1, term2);
    
    PriceResult4 result;
    vst1q_f32(result.price, prices);
    vst1q_f32(result.delta, Nd1);
    
    return result;
}

double AnalyticalPricer::N(double x)
{
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}

double AnalyticalPricer::n(double x)
{
    const double inv_sqrt2pi = 0.3989422804014327;

    return inv_sqrt2pi * exp(-0.5 * x * x);
}

DValues AnalyticalPricer::calculate_d(const OptionParams& p)
{
    // d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)

    double sqrt_T = sqrt(p.T);
    double sigma_sqrt_T = p.sigma * sqrt_T;

    double d1 = (log(p.S / p.K) + (p.r + 0.5 * p.sigma * p.sigma) * p.T) / sigma_sqrt_T;

    // d2 = d1 - σ√T

    double d2 = d1 - sigma_sqrt_T;

    return {d1, d2};
}

double AnalyticalPricer::black_scholes_call(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    
    double Nd1 = N(d1);
    double Nd2 = N(d2);
    
    double discount = exp(-p.r * p.T);
    
    // S × N(d1) - K × e^(-rT) × N(d2)
    double price = p.S * Nd1 - p.K * discount * Nd2;
    return price;
}

double AnalyticalPricer::black_scholes_put(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    
    double N_minus_d1 = N(-d1);
    double N_minus_d2 = N(-d2);
    
    double discount = exp(-p.r * p.T);
    
    // Put = K × e^(-rT) × N(-d2) - S × N(-d1)
    double price = p.K * discount * N_minus_d2 - p.S * N_minus_d1;
    return price;
}

double AnalyticalPricer::delta_call(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    return N(d1);
}

double AnalyticalPricer::delta_put(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    return N(d1) - 1.0;
}

double AnalyticalPricer::gamma(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    
    // n(d1) / (S × σ × √T)
    double nd1 = n(d1);
    double sqrt_T = sqrt(p.T);
    
    return nd1 / (p.S * p.sigma * sqrt_T);
}

double AnalyticalPricer::vega(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    
    // S × √T × n(d1)
    double nd1 = n(d1);
    double sqrt_T = sqrt(p.T);
    
    return p.S * sqrt_T * nd1;
}

double AnalyticalPricer::theta_call(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    
    double nd1 = n(d1);
    double Nd2 = N(d2);
    double sqrt_T = sqrt(p.T);
    
    //  -(S × n(d1) × σ) / (2√T) - r × K × e^(-rT) × N(d2)
    double term1 = -(p.S * nd1 * p.sigma) / (2.0 * sqrt_T);
    double term2 = -p.r * p.K * exp(-p.r * p.T) * Nd2;
    
    return term1 + term2;
}

double AnalyticalPricer::theta_put(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    
    double nd1 = n(d1);
    double N_minus_d2 = N(-d2);
    double sqrt_T = sqrt(p.T);
    
    // -(S × n(d1) × σ) / (2√T) + r × K × e^(-rT) × N(d2)
    double term1 = -(p.S * nd1 * p.sigma) / (2.0 * sqrt_T);
    double term2 = p.r * p.K * exp(-p.r * p.T) * N_minus_d2;
    
    return term1 + term2;
}

double AnalyticalPricer::rho_call(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    
    // K × T × e^(-rT) × N(d2)
    double Nd2 = N(d2);
    return p.K * p.T * exp(-p.r * p.T) * Nd2;
}

double AnalyticalPricer::rho_put(const OptionParams& p) {
    auto [d1, d2] = calculate_d(p);
    
    // -K × T × e^(-rT) × N(-d2)
    double N_minus_d2 = N(-d2);
    return -p.K * p.T * exp(-p.r * p.T) * N_minus_d2;
}

OptionPrice AnalyticalPricer::price(const OptionParams& p) {
    OptionPrice result;
    
    if (p.type == OptionType::CALL) {
        result.price = black_scholes_call(p);
        result.delta = delta_call(p);
        result.theta = theta_call(p);
        result.rho = rho_call(p);
    } else {
        result.price = black_scholes_put(p);
        result.delta = delta_put(p);
        result.theta = theta_put(p);
        result.rho = rho_put(p);
    }
    
    result.gamma = gamma(p);
    result.vega = vega(p);
    
    return result;
}

#endif