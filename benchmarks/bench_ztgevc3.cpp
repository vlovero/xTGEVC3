#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <complex>
#include <random>
#include <thread>
#include <vector>

#include "tgevc3.h"

constexpr int sizes[] = { 500, 707, 1000, 1414, 2000, 2828, 4000, 5657, 8000 };

extern "C" {
    void ztgevc_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<double> *s, const int *lds, const std::complex<double> *p, const int *ldp, std::complex<double> *vl, const int *ldvl, std::complex<double> *vr, const int *ldvr, const int *mm, int *m, std::complex<double> *work, double *rwork, int *info);

    void ztrevc_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<double> *t, const int *ldt, std::complex<double> *vl, const int *ldvl, std::complex<double> *vr, const int *ldvr, const int *mm, int *m, std::complex<double> *work, double *rwork, int *info);

    // Note: ztrevc3 requires LRWORK as well
    void ztrevc3_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<double> *t, const int *ldt, std::complex<double> *vl, const int *ldvl, std::complex<double> *vr, const int *ldvr, const int *mm, int *m, std::complex<double> *work, const int *lwork, double *rwork, const int *lrwork, int *info);
}

static std::vector<std::complex<double>> eye(const ptrdiff_t n)
{
    std::vector<std::complex<double>> I(n * n);
    for (ptrdiff_t i = 0; i < n; i++) {
        for (ptrdiff_t j = 0; j < n; j++) {
            I[i * n + j] = (i == j) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
        }
    }
    return I;
}

static void cool_down(const benchmark::State &state)
{
    // Only sleep if the current run used the last size in the array (8000)
    if (state.range(0) == sizes[std::size(sizes) - 1]) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
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
static void generate_generalized_complex_triangular(int n, std::vector<std::complex<double>> &S, std::vector<std::complex<double>> &P, std::vector<std::complex<double>> &alpha, std::vector<std::complex<double>> &beta)
{
    S.assign(n * n, { 0.0, 0.0 });
    P.assign(n * n, { 0.0, 0.0 });
    alpha.assign(n, { 0.0, 0.0 });
    beta.assign(n, { 0.0, 0.0 });

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            S[i + j * n] = { dist(gen), dist(gen) };
            P[i + j * n] = { dist(gen), dist(gen) };
        }
        double p_real = dist(gen);
        P[j + j * n] = { p_real + (p_real >= 0 ? 1.0 : -1.0), 0.0 };

        alpha[j] = S[j + j * n];
        beta[j] = P[j + j * n];
    }
}

// Data generator for Standard Eigenvalue Problem
static void generate_standard_complex_triangular(int n, std::vector<std::complex<double>> &T)
{
    T.assign(n * n, { 0.0, 0.0 });
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            T[i + j * n] = { dist(gen), dist(gen) };
        }
    }
}

// --- Generalized Eigenvalue Problem Benchmarks ---

static void BM_ztgevc(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'B';
    double total_flops = 4.0 * ((8.0 * n * n * n / 3.0) + (3 * n * n) - (11.0 * n / 3.0));

    std::vector<std::complex<double>> S(n * n), P(n * n);
    std::vector<std::complex<double>> alpha(n), beta(n);
    generate_generalized_complex_triangular(n, S, P, alpha, beta);

    std::vector<std::complex<double>> VL{ eye(n) }, VR{ eye(n) };
    std::vector<std::complex<double>> work(std::max(1, 2 * n), { 0.0, 0.0 });
    std::vector<double> rwork(std::max(1, 2 * n), 0.0);
    int m_out = 0, info = 0;

    for (auto _ : state) {
        ztgevc_(&side, &howmny, nullptr, &n, S.data(), &n, P.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), rwork.data(), &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

static void BM_ztgevc3(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'B';
    double total_flops = 4.0 * ((8.0 * n * n * n / 3.0) + (3 * n * n) - (11.0 * n / 3.0));

    std::vector<std::complex<double>> S(n * n), P(n * n);
    std::vector<std::complex<double>> alpha(n), beta(n);
    generate_generalized_complex_triangular(n, S, P, alpha, beta);

    std::vector<std::complex<double>> VL{ eye(n) }, VR{ eye(n) };
    int m_out = 0, info = 0;

    std::complex<double> dummy_work;
    ztgevc3(side, howmny, nullptr, n, S.data(), n, P.data(), n, alpha.data(), beta.data(), VL.data(), n, VR.data(), n, n, &m_out, &dummy_work, -1, &info);

    int lwork = static_cast<int>(dummy_work.real());
    std::vector<std::complex<double>> work(lwork, { 0.0, 0.0 });

    for (auto _ : state) {
        ztgevc3(side, howmny, nullptr, n, S.data(), n, P.data(), n, alpha.data(), beta.data(), VL.data(), n, VR.data(), n, n, &m_out, work.data(), lwork, &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

// --- Standard Eigenvalue Problem Benchmarks ---

static void BM_ztrevc(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'B';
    double total_flops = 4.0 * ((5.0 * n * n * n / 3.0) + (n * n) - (2.0 * n / 3.0));

    std::vector<std::complex<double>> T(n * n);
    generate_standard_complex_triangular(n, T);

    std::vector<std::complex<double>> VL{ eye(n) }, VR{ eye(n) };
    std::vector<std::complex<double>> work(std::max(1, 2 * n), { 0.0, 0.0 });
    std::vector<double> rwork(std::max(1, n), 0.0);
    int m_out = 0, info = 0;

    for (auto _ : state) {
        ztrevc_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), rwork.data(), &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

static void BM_ztrevc3(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'B';
    double total_flops = 4.0 * ((5.0 * n * n * n / 3.0) + (n * n) - (2.0 * n / 3.0));

    std::vector<std::complex<double>> T(n * n);
    generate_standard_complex_triangular(n, T);

    std::vector<std::complex<double>> VL{ eye(n) }, VR{ eye(n) };
    int m_out = 0, info = 0;

    int lwork_query = -1;
    int lrwork_query = -1;
    std::complex<double> dummy_work;
    double dummy_rwork;

    ztrevc3_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, &dummy_work, &lwork_query, &dummy_rwork, &lrwork_query, &info);

    int lwork = static_cast<int>(dummy_work.real());
    int lrwork = static_cast<int>(dummy_rwork);
    std::vector<std::complex<double>> work(lwork, { 0.0, 0.0 });
    std::vector<double> rwork(lrwork, 0.0);

    for (auto _ : state) {
        ztrevc3_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), &lwork, rwork.data(), &lrwork, &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

BENCHMARK(BM_ztgevc)->Apply(apply_args);
BENCHMARK(BM_ztgevc3)->Apply(apply_args);
BENCHMARK(BM_ztrevc)->Apply(apply_args);
BENCHMARK(BM_ztrevc3)->Apply(apply_args);

BENCHMARK_MAIN();