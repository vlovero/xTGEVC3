#include <algorithm>
#include <benchmark/benchmark.h>
#include <complex>
#include <random>
#include <vector>

#include "tgevc3.h"

extern "C" {
    void ctgevc_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<float> *s, const int *lds, const std::complex<float> *p, const int *ldp, std::complex<float> *vl, const int *ldvl, std::complex<float> *vr, const int *ldvr, const int *mm, int *m, std::complex<float> *work, float *rwork, int *info);

    void ctrevc_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<float> *t, const int *ldt, std::complex<float> *vl, const int *ldvl, std::complex<float> *vr, const int *ldvr, const int *mm, int *m, std::complex<float> *work, float *rwork, int *info);

    void ctrevc3_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<float> *t, const int *ldt, std::complex<float> *vl, const int *ldvl, std::complex<float> *vr, const int *ldvr, const int *mm, int *m, std::complex<float> *work, const int *lwork, float *rwork, const int *lrwork, int *info);
}

static void apply_args(benchmark::internal::Benchmark *b)
{
    constexpr int sizes[] = { 500, 707, 1000, 1414, 2000, 2828, 4000, 5657, 8000 };
    for (const auto size : sizes) {
        b->Args({ size });
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
}

// Data generator for Generalized Eigenvalue Problem
static void generate_generalized_complex_triangular(int n, std::vector<std::complex<float>> &S, std::vector<std::complex<float>> &P, std::vector<std::complex<float>> &alpha, std::vector<std::complex<float>> &beta)
{
    S.assign(n * n, { 0.0f, 0.0f });
    P.assign(n * n, { 0.0f, 0.0f });
    alpha.assign(n, { 0.0f, 0.0f });
    beta.assign(n, { 0.0f, 0.0f });

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            S[i + j * n] = { dist(gen), dist(gen) };
            P[i + j * n] = { dist(gen), dist(gen) };
        }
        float p_real = dist(gen);
        P[j + j * n] = { p_real + (p_real >= 0 ? 1.0f : -1.0f), 0.0f };

        alpha[j] = S[j + j * n];
        beta[j] = P[j + j * n];
    }
}

// Data generator for Standard Eigenvalue Problem
static void generate_standard_complex_triangular(int n, std::vector<std::complex<float>> &T)
{
    T.assign(n * n, { 0.0f, 0.0f });
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            T[i + j * n] = { dist(gen), dist(gen) };
        }
    }
}

// --- Generalized Eigenvalue Problem Benchmarks ---

static void BM_ctgevc(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'A';

    std::vector<std::complex<float>> S(n * n), P(n * n);
    std::vector<std::complex<float>> alpha(n), beta(n);
    generate_generalized_complex_triangular(n, S, P, alpha, beta);

    std::vector<std::complex<float>> VL(n * n, { 0.0f, 0.0f }), VR(n * n, { 0.0f, 0.0f });
    std::vector<std::complex<float>> work(std::max(1, 2 * n), { 0.0f, 0.0f });
    std::vector<float> rwork(std::max(1, 2 * n), 0.0f);
    int m_out = 0, info = 0;

    for (auto _ : state) {
        ctgevc_(&side, &howmny, nullptr, &n, S.data(), &n, P.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), rwork.data(), &info);
    }
}

static void BM_ctgevc3(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'A';

    std::vector<std::complex<float>> S(n * n), P(n * n);
    std::vector<std::complex<float>> alpha(n), beta(n);
    generate_generalized_complex_triangular(n, S, P, alpha, beta);

    std::vector<std::complex<float>> VL(n * n, { 0.0f, 0.0f }), VR(n * n, { 0.0f, 0.0f });
    int m_out = 0, info = 0;

    std::complex<float> dummy_work;
    ctgevc3(side, howmny, nullptr, n, S.data(), n, P.data(), n, alpha.data(), beta.data(), VL.data(), n, VR.data(), n, n, &m_out, &dummy_work, -1, &info);

    int lwork = static_cast<int>(dummy_work.real());
    std::vector<std::complex<float>> work(lwork, { 0.0f, 0.0f });

    for (auto _ : state) {
        ctgevc3(side, howmny, nullptr, n, S.data(), n, P.data(), n, alpha.data(), beta.data(), VL.data(), n, VR.data(), n, n, &m_out, work.data(), lwork, &info);
    }
}

// --- Standard Eigenvalue Problem Benchmarks ---

static void BM_ctrevc(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'A';

    std::vector<std::complex<float>> T(n * n);
    generate_standard_complex_triangular(n, T);

    std::vector<std::complex<float>> VL(n * n, { 0.0f, 0.0f }), VR(n * n, { 0.0f, 0.0f });
    std::vector<std::complex<float>> work(std::max(1, 2 * n), { 0.0f, 0.0f });
    std::vector<float> rwork(std::max(1, n), 0.0f);
    int m_out = 0, info = 0;

    for (auto _ : state) {
        ctrevc_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), rwork.data(), &info);
    }
}

static void BM_ctrevc3(benchmark::State &state)
{
    int n = state.range(0);
    char side = 'B', howmny = 'A';

    std::vector<std::complex<float>> T(n * n);
    generate_standard_complex_triangular(n, T);

    std::vector<std::complex<float>> VL(n * n, { 0.0f, 0.0f }), VR(n * n, { 0.0f, 0.0f });
    int m_out = 0, info = 0;

    int lwork_query = -1;
    int lrwork_query = -1;
    std::complex<float> dummy_work;
    float dummy_rwork;

    ctrevc3_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, &dummy_work, &lwork_query, &dummy_rwork, &lrwork_query, &info);

    int lwork = static_cast<int>(dummy_work.real());
    int lrwork = static_cast<int>(dummy_rwork);
    std::vector<std::complex<float>> work(lwork, { 0.0f, 0.0f });
    std::vector<float> rwork(lrwork, 0.0f);

    for (auto _ : state) {
        ctrevc3_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), &lwork, rwork.data(), &lrwork, &info);
    }
}

BENCHMARK(BM_ctgevc)->Apply(apply_args);
BENCHMARK(BM_ctgevc3)->Apply(apply_args);
BENCHMARK(BM_ctrevc)->Apply(apply_args);
BENCHMARK(BM_ctrevc3)->Apply(apply_args);

BENCHMARK_MAIN();