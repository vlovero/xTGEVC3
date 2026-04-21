// test_lapack_c.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <random>

#include "fmt/core.h"
#include "tgevc3.h"

extern "C" {
    void ctgevc_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<float> *s, const int *lds, const std::complex<float> *p, const int *ldp, std::complex<float> *vl, const int *ldvl, std::complex<float> *vr, const int *ldvr, const int *mm, int *m, std::complex<float> *work, float *rwork, int *info);
}


static std::mt19937 gen(0);
static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

inline float c_abs1(const std::complex<float> &z)
{
    return std::abs(z.real()) + std::abs(z.imag());
}

float clange_1(int n, const std::complex<float> *A, int lda)
{
    float max_norm = 0.0f, col_sum;
    int c, r;

    for (c = 0; c < n; ++c) {
        col_sum = 0.0f;
        for (r = 0; r <= c; ++r) {
            col_sum += c_abs1(A[r + c * lda]);
        }
        max_norm = std::max(max_norm, col_sum);
    }

    return max_norm;
}

float cget52_residual(bool left, int n, const std::complex<float> *S, int lds, const std::complex<float> *P, int ldp, const std::complex<float> *alpha, const std::complex<float> *beta, const std::complex<float> *V, int ldv)
{
    float normS, normP, safemin, eps, max_err, normV, normR, denom, almax;
    int j, i, c, r;
    std::complex<float> *r_vec;
    std::complex<float> a, bVal;

    normS = clange_1(n, S, lds);
    normP = clange_1(n, P, ldp);
    safemin = std::numeric_limits<float>::min();
    eps = std::numeric_limits<float>::epsilon();

    if (normS == 0.0f) {
        normS = 1.0f;
    }
    if (normP == 0.0f) {
        normP = 1.0f;
    }

    max_err = 0.0f;
    r_vec = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));

    for (j = 0; j < n; ++j) {
        a = alpha[j];
        bVal = beta[j];

        almax = std::max({ 1.0f, c_abs1(a), c_abs1(bVal) });
        a /= almax;
        bVal /= almax;

        normV = 0.0f;
        for (i = 0; i < n; ++i) {
            normV += c_abs1(V[i + j * ldv]);
        }
        if (normV == 0.0f) {
            normV = 1.0f;
        }

        for (i = 0; i < n; i++) {
            r_vec[i] = std::complex<float>(0.0f, 0.0f);
        }

        if (!left) {
            for (c = 0; c < n; ++c) {
                for (r = 0; r <= c; ++r) {
                    r_vec[r] += bVal * S[r + c * lds] * V[c + j * ldv];
                    r_vec[r] -= a * P[r + c * ldp] * V[c + j * ldv];
                }
            }
        }
        else {
            for (c = 0; c < n; ++c) {
                for (r = 0; r <= c; ++r) {
                    r_vec[c] += std::conj(V[r + j * ldv]) * (bVal * S[r + c * lds] - a * P[r + c * ldp]);
                }
            }
        }

        normR = 0.0f;
        for (i = 0; i < n; ++i) {
            normR += c_abs1(r_vec[i]);
        }

        denom = (c_abs1(bVal) * normS + c_abs1(a) * normP) * normV;
        denom = std::max(safemin, denom);

        max_err = std::max(max_err, normR / denom);
    }

    free(r_vec);
    return max_err / eps;
}

void generate_generalized_upper_triangular(int n, std::complex<float> *S, int lds, std::complex<float> *P, int ldp, std::complex<float> *alpha, std::complex<float> *beta)
{
    int i, c, r;
    float p_real;

    for (i = 0; i < n * n; i++) {
        S[i] = { 0.0f, 0.0f };
        P[i] = { 0.0f, 0.0f };
    }

    for (c = 0; c < n; c++) {
        for (r = 0; r < c; r++) {
            S[r + c * lds] = std::complex<float>(dist(gen), dist(gen));
            P[r + c * ldp] = std::complex<float>(dist(gen), dist(gen));
        }

        S[c + c * lds] = std::complex<float>(dist(gen), dist(gen));

        p_real = dist(gen);
        p_real += (p_real >= 0 ? 1.0f : -1.0f);
        P[c + c * ldp] = std::complex<float>(p_real, 0.0f);

        alpha[c] = S[c + c * lds];
        beta[c] = P[c + c * ldp];
    }
}

void scale_matrices(int n, std::complex<float> *S, int lds, std::complex<float> *P, int ldp, std::complex<float> *alpha, std::complex<float> *beta, float scale_S, float scale_P)
{
    int j, i;
    for (j = 0; j < n; ++j) {
        for (i = 0; i <= j; ++i) {
            S[i + j * lds] *= scale_S;
            P[i + j * ldp] *= scale_P;
        }
        alpha[j] *= scale_S;
        beta[j] *= scale_P;
    }
}

void run_test_case(int n, const char *test_name, float scale_S = 1.0f, float scale_P = 1.0f)
{
    int lwork3, m_out, i;
    int info[1];
    std::complex<float> dummy[1];
    float res_right3, res_left3, res_right_lapack, res_left_lapack;
    char side = 'B', howmny = 'A';

    std::complex<float> *S, *P, *VR3, *VL3, *VR_lapack, *VL_lapack, *alpha, *beta, *work3, *work_lapack;
    int *select;
    float *rwork_lapack;

    std::chrono::high_resolution_clock::time_point start3, end3, start_lapack, end_lapack;
    std::chrono::duration<double> time3, time_lapack;

    S = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    P = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VR3 = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VL3 = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VR_lapack = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VL_lapack = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    alpha = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
    beta = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
    select = (int *)malloc(n * sizeof(int));
    work_lapack = (std::complex<float> *)malloc(2 * n * sizeof(std::complex<float>));
    rwork_lapack = (float *)malloc(2 * n * sizeof(float));

    for (i = 0; i < n; i++) {
        select[i] = 0;
    }
    for (i = 0; i < n * n; i++) {
        VR3[i] = { 0.0f, 0.0f };
        VL3[i] = { 0.0f, 0.0f };
        VR_lapack[i] = { 0.0f, 0.0f };
        VL_lapack[i] = { 0.0f, 0.0f };
    }

    fmt::println("--- Testing {}x{} [{}] ---", n, n, test_name);

    generate_generalized_upper_triangular(n, S, n, P, n, alpha, beta);

    if (scale_S != 1.0f || scale_P != 1.0f) {
        scale_matrices(n, S, n, P, n, alpha, beta, scale_S, scale_P);
    }

    ctgevc3(side, howmny, nullptr, n, S, n, P, n, alpha, beta, VL3, n, VR3, n, n, &m_out, dummy, -1, info);

    lwork3 = static_cast<int>(dummy[0].real());
    work3 = (std::complex<float> *)malloc(lwork3 * sizeof(std::complex<float>));

    start3 = std::chrono::high_resolution_clock::now();
    ctgevc3(side, howmny, nullptr, n, S, n, P, n, alpha, beta, VL3, n, VR3, n, n, &m_out, work3, lwork3, info);
    end3 = std::chrono::high_resolution_clock::now();
    time3 = end3 - start3;

    res_right3 = cget52_residual(false, n, S, n, P, n, alpha, beta, VR3, n);
    res_left3 = cget52_residual(true, n, S, n, P, n, alpha, beta, VL3, n);

    start_lapack = std::chrono::high_resolution_clock::now();
    ctgevc_(&side, &howmny, select, &n, S, &n, P, &n, VL_lapack, &n, VR_lapack, &n, &n, &m_out, work_lapack, rwork_lapack, info);
    end_lapack = std::chrono::high_resolution_clock::now();
    time_lapack = end_lapack - start_lapack;

    res_right_lapack = cget52_residual(false, n, S, n, P, n, alpha, beta, VR_lapack, n);
    res_left_lapack = cget52_residual(true, n, S, n, P, n, alpha, beta, VL_lapack, n);

    fmt::println("  [ctgevc3] Time: {:.4f}s | Right Res: {:.3e} | Left Res: {:.3e}", time3.count(), res_right3, res_left3);
    fmt::println("  [ctgevc]  Time: {:.4f}s | Right Res: {:.3e} | Left Res: {:.3e}\n", time_lapack.count(), res_right_lapack, res_left_lapack);

    if (res_right3 > 100.0f || res_left3 > 100.0f || std::isnan(res_right3)) {
        fmt::println("  => WARNING (ctgevc3): Residual is high or NaN! Numeric failure detected.");
    }

    free(S);
    free(P);
    free(VR3);
    free(VL3);
    free(VR_lapack);
    free(VL_lapack);
    free(alpha);
    free(beta);
    free(select);
    free(work_lapack);
    free(rwork_lapack);
    free(work3);
}


void test_dynamic_lapack(int n)
{
    run_test_case(n, "Standard Normal Scale");
}

void run_lapack_edge_cases(int n)
{
    float ulp, safemin, smlnum, bignum;

    ulp = std::numeric_limits<float>::epsilon();
    safemin = std::numeric_limits<float>::min();

    smlnum = safemin / ulp;
    bignum = 1.0f / smlnum;

    run_test_case(n, "Matrix S near Overflow", bignum, 1.0f);
    run_test_case(n, "Matrix S near Underflow", smlnum, 1.0f);
    run_test_case(n, "Matrix P near Overflow", 1.0f, bignum);
    run_test_case(n, "Matrix P near Underflow", 1.0f, smlnum);
    run_test_case(n, "Both S and P near Overflow", bignum, bignum);
    run_test_case(n, "Both S and P near Underflow", smlnum, smlnum);
}

int main()
{
    test_dynamic_lapack(100);
    run_lapack_edge_cases(100);

    return 0;
}