#include <iostream>
#include "OptionHelpers.h"
#include "testing.h"

using namespace std;


int main() {
    AnalyticalPricer pricer;

    benchmark_simd_vs_scalar(pricer);
    benchmark_monte_carlo_vs_analytical(pricer);
}