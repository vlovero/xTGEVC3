#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "tgevc3.h"

constexpr int sizes[] = { 500, 707, 1000, 1414, 2000, 2828, 4000, 5657, 8000 };


extern "C" {
    void dtgevc_(const char *side, const char *howmny, const int *select, const int *n, const double *s, const int *lds, const double *p, const int *ldp, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, int *info);
    void dtrevc_(const char *side, const char *howmny, const int *select, const int *n, const double *t, const int *ldt, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, int *info);
    void dtrevc3_(const char *side, const char *howmny, const int *select, const int *n, const double *t, const int *ldt, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, const int *lwork, int *info);
}

static std::vector<double> eye(const ptrdiff_t n)
{
    std::vector<double> I(n * n);
    for (ptrdiff_t i = 0; i < n; i++) {
        for (ptrdiff_t j = 0; j < n; j++) {
            I[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return I;
}

static void cool_down(const benchmark::State &state)
{
    // Only sleep if the current run used the last size in the array (8000)
    if (state.range(0) == sizes[std::size(sizes) - 1]) {
        std::this_thread::sleep_for(std::chrono::seconds(90));
    }
}

static void apply_args(benchmark::internal::Benchmark *b)
{
    for (const auto size : sizes) {
        b->Args({ size });
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
    b->Complexity();
    b->Teardown(cool_down);
}

// Data generator for Generalized Eigenvalue Problem
static void generate_generalized_real_triangular(int n, std::vector<double> &S, std::vector<double> &P, std::vector<double> &alphar, std::vector<double> &alphai, std::vector<double> &beta)
{
    S.assign(n * n, 0.0);
    P.assign(n * n, 0.0);
    alphar.assign(n, 0.0);
    alphai.assign(n, 0.0);
    beta.assign(n, 0.0);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            S[i + j * n] = dist(gen);
            P[i + j * n] = dist(gen);
        }
        double p_real = dist(gen);
        P[j + j * n] = p_real + (p_real >= 0 ? 1.0 : -1.0);

        alphar[j] = S[j + j * n];
        beta[j] = P[j + j * n];
    }
}

// Data generator for Standard Eigenvalue Problem
static void generate_standard_real_triangular(int n, std::vector<double> &T)
{
    T.assign(n * n, 0.0);
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            T[i + j * n] = dist(gen);
        }
    }
}

// --- Generalized Eigenvalue Problem Benchmarks ---
static void BM_dtgevc(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'B';
    double total_flops = (8.0 * n * n * n / 3.0) + (3 * n * n) - (11.0 * n / 3.0);

    std::vector<double> S(n * n), P(n * n);
    std::vector<double> alphar(n), alphai(n), beta(n);
    generate_generalized_real_triangular(n, S, P, alphar, alphai, beta);

    std::vector<double> VL{ eye(n) }, VR{ eye(n) };
    std::vector<double> work(std::max(1, 6 * n), 0.0);
    int m_out = 0, info = 0;

    for (auto _ : state) {
        dtgevc_(&side, &howmny, nullptr, &n, S.data(), &n, P.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

static void BM_dtgevc3(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'B';
    double total_flops = (8.0 * n * n * n / 3.0) + (3 * n * n) - (11.0 * n / 3.0);

    std::vector<double> S(n * n), P(n * n);
    std::vector<double> alphar(n), alphai(n), beta(n);
    generate_generalized_real_triangular(n, S, P, alphar, alphai, beta);

    std::vector<double> VL{ eye(n) }, VR{ eye(n) };
    int m_out = 0, info = 0;

    double dummy_work;
    dtgevc3(side, howmny, nullptr, n, S.data(), n, P.data(), n, alphar.data(), alphai.data(), beta.data(), VL.data(), n, VR.data(), n, n, &m_out, &dummy_work, -1, &info);

    int lwork = static_cast<int>(dummy_work);
    std::vector<double> work(lwork, 0.0);

    for (auto _ : state) {
        dtgevc3(side, howmny, nullptr, n, S.data(), n, P.data(), n, alphar.data(), alphai.data(), beta.data(), VL.data(), n, VR.data(), n, n, &m_out, work.data(), lwork, &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

// --- Standard Eigenvalue Problem Benchmarks ---
static void BM_dtrevc(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'B';
    double total_flops = (5.0 * n * n * n / 3.0) + (n * n) - (2.0 * n / 3.0);

    std::vector<double> T(n * n);
    generate_standard_real_triangular(n, T);

    std::vector<double> VL{ eye(n) }, VR{ eye(n) };
    std::vector<double> work(std::max(1, 3 * n), 0.0);
    int m_out = 0, info = 0;

    for (auto _ : state) {
        dtrevc_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

static void BM_dtrevc3(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'B';
    double total_flops = (5.0 * n * n * n / 3.0) + (n * n) - (2.0 * n / 3.0);

    std::vector<double> T(n * n);
    generate_standard_real_triangular(n, T);

    std::vector<double> VL{ eye(n) }, VR{ eye(n) };
    int m_out = 0, info = 0;

    int lwork_query = -1;
    double dummy_work;
    dtrevc3_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, &dummy_work, &lwork_query, &info);

    int lwork = static_cast<int>(dummy_work);
    lwork = n * 130;
    std::vector<double> work(lwork, 0.0);

    for (auto _ : state) {
        dtrevc3_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), &lwork, &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

BENCHMARK(BM_dtgevc)->Apply(apply_args);
BENCHMARK(BM_dtgevc3)->Apply(apply_args);
BENCHMARK(BM_dtrevc)->Apply(apply_args);
BENCHMARK(BM_dtrevc3)->Apply(apply_args);

BENCHMARK_MAIN();