// test_lapack_s.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>

#include "fmt/core.h"
#include "tgevc3.h"

extern "C" {
    void stgevc_(const char *side, const char *howmny, const int *select, const int *n, const float *s, const int *lds, const float *p, const int *ldp, float *vl, const int *ldvl, float *vr, const int *ldvr, const int *mm, int *m, float *work, int *info);
}


static std::mt19937 gen(0);
static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
static std::uniform_int_distribution<int> dist_int(0, 99);

float slange_1(int n, const float *A, int lda)
{
    float max_norm, col_sum;
    int c, r;

    max_norm = 0.0f;
    for (c = 0; c < n; ++c) {
        col_sum = 0.0f;
        for (r = 0; r <= std::min(c + 1, n - 1); ++r) {
            col_sum += std::abs(A[r + c * lda]);
        }
        max_norm = std::max(max_norm, col_sum);
    }

    return max_norm;
}

float sget52_residual(bool left, int n, const float *S, int lds, const float *P, int ldp, const float *alphar, const float *alphai, const float *beta, const float *V, int ldv)
{
    float normS, normP, safemin, eps, max_err, aR, aI, bVal, normV, normR, denom, s_val, p_val, v_r, v_i;
    int j, i, c, r;
    float *r_r, *r_i;

    normS = slange_1(n, S, lds);
    normP = slange_1(n, P, ldp);
    safemin = std::numeric_limits<float>::min();
    eps = std::numeric_limits<float>::epsilon();

    if (normS == 0.0f) {
        normS = 1.0f;
    }
    if (normP == 0.0f) {
        normP = 1.0f;
    }

    max_err = 0.0f;
    j = 0;

    r_r = (float *)malloc(n * sizeof(float));
    r_i = (float *)malloc(n * sizeof(float));

    while (j < n) {
        if (alphai[j] == 0.0f) {
            aR = alphar[j];
            bVal = beta[j];
            normV = 0.0f;

            for (i = 0; i < n; ++i) {
                normV += std::abs(V[i + j * ldv]);
            }
            if (normV == 0.0f) {
                normV = 1.0f;
            }

            for (i = 0; i < n; i++) {
                r_r[i] = 0.0f;
            }

            if (!left) {
                for (c = 0; c < n; ++c) {
                    for (r = 0; r <= std::min(c + 1, n - 1); ++r) {
                        r_r[r] += bVal * S[r + c * lds] * V[c + j * ldv];
                        r_r[r] -= aR * P[r + c * ldp] * V[c + j * ldv];
                    }
                }
            }
            else {
                for (c = 0; c < n; ++c) {
                    for (r = 0; r <= std::min(c + 1, n - 1); ++r) {
                        r_r[c] += bVal * S[r + c * lds] * V[r + j * ldv];
                        r_r[c] -= aR * P[r + c * ldp] * V[r + j * ldv];
                    }
                }
            }

            normR = 0.0f;
            for (i = 0; i < n; ++i) {
                normR += std::abs(r_r[i]);
            }

            denom = (std::abs(bVal) * normS + std::abs(aR) * normP) * normV;
            denom = std::max(safemin, denom);

            max_err = std::max(max_err, normR / denom);
            j++;
        }
        else {
            aR = alphar[j];
            aI = alphai[j];
            bVal = beta[j];
            normV = 0.0f;

            for (i = 0; i < n; ++i) {
                normV += std::abs(V[i + j * ldv]) + std::abs(V[i + (j + 1) * ldv]);
            }
            if (normV == 0.0f) {
                normV = 1.0f;
            }

            for (i = 0; i < n; i++) {
                r_r[i] = 0.0f;
                r_i[i] = 0.0f;
            }

            if (!left) {
                for (c = 0; c < n; ++c) {
                    for (r = 0; r <= std::min(c + 1, n - 1); ++r) {
                        s_val = S[r + c * lds];
                        p_val = P[r + c * ldp];
                        v_r = V[c + j * ldv];
                        v_i = V[c + (j + 1) * ldv];

                        r_r[r] += bVal * s_val * v_r - aR * p_val * v_r + aI * p_val * v_i;
                        r_i[r] += bVal * s_val * v_i - aR * p_val * v_i - aI * p_val * v_r;
                    }
                }
            }
            else {
                for (c = 0; c < n; ++c) {
                    for (r = 0; r <= std::min(c + 1, n - 1); ++r) {
                        s_val = S[r + c * lds];
                        p_val = P[r + c * ldp];
                        v_r = V[r + j * ldv];
                        v_i = V[r + (j + 1) * ldv];

                        r_r[c] += bVal * s_val * v_r - aR * p_val * v_r - aI * p_val * v_i;
                        r_i[c] += bVal * s_val * v_i - aR * p_val * v_i + aI * p_val * v_r;
                    }
                }
            }

            normR = 0.0f;
            for (i = 0; i < n; ++i) {
                normR += std::abs(r_r[i]) + std::abs(r_i[i]);
            }

            denom = (std::abs(bVal) * normS + (std::abs(aR) + std::abs(aI)) * normP) * normV;
            denom = std::max(safemin, denom);

            max_err = std::max(max_err, normR / denom);
            j += 2;
        }
    }

    free(r_r);
    free(r_i);
    return max_err / eps;
}

void generate_generalized_quasi_triangular_s(int n, float *S, int lds, float *P, int ldp, float *alphar, float *alphai, float *beta)
{
    int i, c, r, k;
    float a, b;

    for (i = 0; i < n * n; i++) {
        S[i] = 0.0f;
        P[i] = 0.0f;
    }

    for (c = 0; c < n; c++) {
        for (r = 0; r <= c; r++) {
            S[r + c * lds] = dist(gen);
            P[r + c * ldp] = dist(gen);
        }
        P[c + c * ldp] += (P[c + c * ldp] >= 0 ? 1.0f : -1.0f);
    }

    k = 0;
    while (k < n) {
        if (k < n - 1 && dist_int(gen) < 40) {
            P[k + k * ldp] = 1.0f;
            P[k + (k + 1) * ldp] = 0.0f;
            P[k + 1 + k * ldp] = 0.0f;
            P[k + 1 + (k + 1) * ldp] = 1.0f;

            a = S[k + k * lds];
            S[(k + 1) + (k + 1) * lds] = a;

            b = S[k + (k + 1) * lds];
            if (b == 0.0f) {
                b = 1.0f;
            }
            S[(k + 1) + k * lds] = -b;

            alphar[k] = a;
            alphar[k + 1] = a;
            alphai[k] = std::abs(b);
            alphai[k + 1] = -std::abs(b);
            beta[k] = 1.0f;
            beta[k + 1] = 1.0f;
            k += 2;
        }
        else {
            alphar[k] = S[k + k * lds];
            alphai[k] = 0.0f;
            beta[k] = P[k + k * ldp];
            k += 1;
        }
    }
}

void scale_matrices_s(int n, float *S, int lds, float *P, int ldp, float *alphar, float *alphai, float *beta, float scale_S, float scale_P)
{
    int i, j;

    for (j = 0; j < n; ++j) {
        for (i = 0; i <= std::min(j + 1, n - 1); ++i) {
            S[i + j * lds] *= scale_S;
        }
        for (i = 0; i <= j; ++i) {
            P[i + j * ldp] *= scale_P;
        }
        alphar[j] *= scale_S;
        alphai[j] *= scale_S;
        beta[j] *= scale_P;
    }
}

void run_test_case_s(int n, const char *test_name, float scale_S = 1.0f, float scale_P = 1.0f)
{
    int i, j, lwork3, m_out;
    int info[1];
    float dummy[1];
    float res_right3, res_left3, res_right_lapack, res_left_lapack;
    char side, howmny;
    float *S, *P, *VR3, *VL3, *VR_lapack, *VL_lapack, *alphar, *alphai, *beta, *work3, *work_lapack;
    std::chrono::high_resolution_clock::time_point start3, end3, start_lapack, end_lapack;
    std::chrono::duration<double> time3, time_lapack;

    S = (float *)malloc(n * n * sizeof(float));
    P = (float *)malloc(n * n * sizeof(float));
    VR3 = (float *)malloc(n * n * sizeof(float));
    VL3 = (float *)malloc(n * n * sizeof(float));
    VR_lapack = (float *)malloc(n * n * sizeof(float));
    VL_lapack = (float *)malloc(n * n * sizeof(float));
    alphar = (float *)malloc(n * sizeof(float));
    alphai = (float *)malloc(n * sizeof(float));
    beta = (float *)malloc(n * sizeof(float));
    work_lapack = (float *)malloc(6 * n * sizeof(float));

    fmt::println("--- Testing {}x{} [{}] (Single Precision) ---", n, n, test_name);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR3[i * n + j] = (i == j) ? 1.0f : 0.0f;
            VL3[i * n + j] = (i == j) ? 1.0f : 0.0f;
            VR_lapack[i * n + j] = (i == j) ? 1.0f : 0.0f;
            VL_lapack[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    generate_generalized_quasi_triangular_s(n, S, n, P, n, alphar, alphai, beta);

    if (scale_S != 1.0f || scale_P != 1.0f) {
        scale_matrices_s(n, S, n, P, n, alphar, alphai, beta, scale_S, scale_P);
    }

    stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL3, n, VR3, n, n, &m_out, dummy, -1, info);

    lwork3 = static_cast<int>(dummy[0]);
    work3 = (float *)malloc(lwork3 * sizeof(float));

    start3 = std::chrono::high_resolution_clock::now();
    stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL3, n, VR3, n, n, &m_out, work3, lwork3, info);
    end3 = std::chrono::high_resolution_clock::now();
    time3 = end3 - start3;

    res_right3 = sget52_residual(false, n, S, n, P, n, alphar, alphai, beta, VR3, n);
    res_left3 = sget52_residual(true, n, S, n, P, n, alphar, alphai, beta, VL3, n);

    side = 'B';
    howmny = 'B';

    start_lapack = std::chrono::high_resolution_clock::now();
    stgevc_(&side, &howmny, nullptr, &n, S, &n, P, &n, VL_lapack, &n, VR_lapack, &n, &n, &m_out, work_lapack, info);
    end_lapack = std::chrono::high_resolution_clock::now();
    time_lapack = end_lapack - start_lapack;

    res_right_lapack = sget52_residual(false, n, S, n, P, n, alphar, alphai, beta, VR_lapack, n);
    res_left_lapack = sget52_residual(true, n, S, n, P, n, alphar, alphai, beta, VL_lapack, n);

    fmt::println("  [stgevc3] Time: {:.4f}s | Right Res: {:.3e} | Left Res: {:.3e}", time3.count(), res_right3, res_left3);
    fmt::println("  [stgevc]  Time: {:.4f}s | Right Res: {:.3e} | Left Res: {:.3e}\n", time_lapack.count(), res_right_lapack, res_left_lapack);

    if (res_right3 > 100.0f || res_left3 > 100.0f || std::isnan(res_right3)) {
        fmt::println("  => WARNING (stgevc3): Residual is high or NaN! Numeric failure detected.");
    }

    free(S);
    free(P);
    free(VR3);
    free(VL3);
    free(VR_lapack);
    free(VL_lapack);
    free(alphar);
    free(alphai);
    free(beta);
    free(work3);
    free(work_lapack);
}


void test_dynamic_lapack_s(int n)
{
    run_test_case_s(n, "Standard Normal Scale");
}

void run_lapack_edge_cases_s(int n)
{
    float ulp, safemin, smlnum, bignum;

    ulp = std::numeric_limits<float>::epsilon();
    safemin = std::numeric_limits<float>::min();

    smlnum = std::sqrt(safemin / ulp);
    bignum = std::sqrt(1.0f / (safemin / ulp));

    run_test_case_s(n, "Matrix S near Overflow", bignum, 1.0f);
    run_test_case_s(n, "Matrix S near Underflow", smlnum, 1.0f);
    run_test_case_s(n, "Matrix P near Overflow", 1.0f, bignum);
    run_test_case_s(n, "Matrix P near Underflow", 1.0f, smlnum);
    run_test_case_s(n, "Both S and P near Overflow", bignum, bignum);
    run_test_case_s(n, "Both S and P near Underflow", smlnum, smlnum);
}

int main()
{
    test_dynamic_lapack_s(100);
    run_lapack_edge_cases_s(100);

    return 0;
}