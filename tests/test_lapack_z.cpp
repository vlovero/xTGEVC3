// test_lapack_z.cpp
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
    void ztgevc_(const char *side, const char *howmny, const int *select, const int *n, const std::complex<double> *s, const int *lds, const std::complex<double> *p, const int *ldp, std::complex<double> *vl, const int *ldvl, std::complex<double> *vr, const int *ldvr, const int *mm, int *m, std::complex<double> *work, double *rwork, int *info);
}


static std::mt19937 gen(0);
static std::uniform_real_distribution<double> dist(-1.0, 1.0);

inline double c_abs1(const std::complex<double> &z)
{
    return std::abs(z.real()) + std::abs(z.imag());
}

double zlange_1(int n, const std::complex<double> *A, int lda)
{
    double max_norm = 0.0, col_sum;
    int c, r;

    for (c = 0; c < n; ++c) {
        col_sum = 0.0;
        for (r = 0; r <= c; ++r) {
            col_sum += c_abs1(A[r + c * lda]);
        }
        max_norm = std::max(max_norm, col_sum);
    }

    return max_norm;
}

double zget52_residual(bool left, int n, const std::complex<double> *S, int lds, const std::complex<double> *P, int ldp, const std::complex<double> *alpha, const std::complex<double> *beta, const std::complex<double> *V, int ldv)
{
    double normS, normP, safemin, eps, max_err, normV, normR, denom, almax;
    int j, i, c, r;
    std::complex<double> *r_vec;
    std::complex<double> a, bVal;

    normS = zlange_1(n, S, lds);
    normP = zlange_1(n, P, ldp);
    safemin = std::numeric_limits<double>::min();
    eps = std::numeric_limits<double>::epsilon();

    if (normS == 0.0) {
        normS = 1.0;
    }
    if (normP == 0.0) {
        normP = 1.0;
    }

    max_err = 0.0;
    r_vec = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));

    for (j = 0; j < n; ++j) {
        a = alpha[j];
        bVal = beta[j];

        almax = std::max({ 1.0, c_abs1(a), c_abs1(bVal) });
        a /= almax;
        bVal /= almax;

        normV = 0.0;
        for (i = 0; i < n; ++i) {
            normV += c_abs1(V[i + j * ldv]);
        }
        if (normV == 0.0) {
            normV = 1.0;
        }

        for (i = 0; i < n; i++) {
            r_vec[i] = std::complex<double>(0.0, 0.0);
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

        normR = 0.0;
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

void generate_generalized_upper_triangular(int n, std::complex<double> *S, int lds, std::complex<double> *P, int ldp, std::complex<double> *alpha, std::complex<double> *beta)
{
    int i, c, r;
    double p_real;

    for (i = 0; i < n * n; i++) {
        S[i] = { 0.0, 0.0 };
        P[i] = { 0.0, 0.0 };
    }

    for (c = 0; c < n; c++) {
        for (r = 0; r < c; r++) {
            S[r + c * lds] = std::complex<double>(dist(gen), dist(gen));
            P[r + c * ldp] = std::complex<double>(dist(gen), dist(gen));
        }

        S[c + c * lds] = std::complex<double>(dist(gen), dist(gen));

        p_real = dist(gen);
        p_real += (p_real >= 0 ? 1.0 : -1.0);
        P[c + c * ldp] = std::complex<double>(p_real, 0.0);

        alpha[c] = S[c + c * lds];
        beta[c] = P[c + c * ldp];
    }
}

void scale_matrices(int n, std::complex<double> *S, int lds, std::complex<double> *P, int ldp, std::complex<double> *alpha, std::complex<double> *beta, double scale_S, double scale_P)
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

void run_test_case(int n, const char *test_name, double scale_S = 1.0, double scale_P = 1.0)
{
    int lwork3, m_out, i;
    int info[1];
    std::complex<double> dummy[1];
    double res_right3, res_left3, res_right_lapack, res_left_lapack;
    char side = 'B', howmny = 'A';

    std::complex<double> *S, *P, *VR3, *VL3, *VR_lapack, *VL_lapack, *alpha, *beta, *work3, *work_lapack;
    int *select;
    double *rwork_lapack;

    std::chrono::high_resolution_clock::time_point start3, end3, start_lapack, end_lapack;
    std::chrono::duration<double> time3, time_lapack;

    S = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    P = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VR3 = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VL3 = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VR_lapack = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VL_lapack = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    alpha = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
    beta = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
    select = (int *)malloc(n * sizeof(int));
    work_lapack = (std::complex<double> *)malloc(2 * n * sizeof(std::complex<double>));
    rwork_lapack = (double *)malloc(2 * n * sizeof(double));

    for (i = 0; i < n; i++) {
        select[i] = 0;
    }
    for (i = 0; i < n * n; i++) {
        VR3[i] = { 0.0, 0.0 };
        VL3[i] = { 0.0, 0.0 };
        VR_lapack[i] = { 0.0, 0.0 };
        VL_lapack[i] = { 0.0, 0.0 };
    }

    fmt::println("--- Testing {}x{} [{}] ---", n, n, test_name);

    generate_generalized_upper_triangular(n, S, n, P, n, alpha, beta);

    if (scale_S != 1.0 || scale_P != 1.0) {
        scale_matrices(n, S, n, P, n, alpha, beta, scale_S, scale_P);
    }

    ztgevc3(side, howmny, nullptr, n, S, n, P, n, alpha, beta, VL3, n, VR3, n, n, &m_out, dummy, -1, info);

    lwork3 = static_cast<int>(dummy[0].real());
    work3 = (std::complex<double> *)malloc(lwork3 * sizeof(std::complex<double>));

    start3 = std::chrono::high_resolution_clock::now();
    ztgevc3(side, howmny, nullptr, n, S, n, P, n, alpha, beta, VL3, n, VR3, n, n, &m_out, work3, lwork3, info);
    end3 = std::chrono::high_resolution_clock::now();
    time3 = end3 - start3;

    res_right3 = zget52_residual(false, n, S, n, P, n, alpha, beta, VR3, n);
    res_left3 = zget52_residual(true, n, S, n, P, n, alpha, beta, VL3, n);

    start_lapack = std::chrono::high_resolution_clock::now();
    ztgevc_(&side, &howmny, select, &n, S, &n, P, &n, VL_lapack, &n, VR_lapack, &n, &n, &m_out, work_lapack, rwork_lapack, info);
    end_lapack = std::chrono::high_resolution_clock::now();
    time_lapack = end_lapack - start_lapack;

    res_right_lapack = zget52_residual(false, n, S, n, P, n, alpha, beta, VR_lapack, n);
    res_left_lapack = zget52_residual(true, n, S, n, P, n, alpha, beta, VL_lapack, n);

    fmt::println("  [ztgevc3] Time: {:.4f}s | Right Res: {:.3e} | Left Res: {:.3e}", time3.count(), res_right3, res_left3);
    fmt::println("  [ztgevc]  Time: {:.4f}s | Right Res: {:.3e} | Left Res: {:.3e}\n", time_lapack.count(), res_right_lapack, res_left_lapack);

    if (res_right3 > 100.0 || res_left3 > 100.0 || std::isnan(res_right3)) {
        fmt::println("  => WARNING (ztgevc3): Residual is high or NaN! Numeric failure detected.");
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
    double ulp, safemin, smlnum, bignum;

    ulp = std::numeric_limits<double>::epsilon();
    safemin = std::numeric_limits<double>::min();

    smlnum = safemin / ulp;
    bignum = 1.0 / smlnum;

    run_test_case(n, "Matrix S near Overflow", bignum, 1.0);
    run_test_case(n, "Matrix S near Underflow", smlnum, 1.0);
    run_test_case(n, "Matrix P near Overflow", 1.0, bignum);
    run_test_case(n, "Matrix P near Underflow", 1.0, smlnum);
    run_test_case(n, "Both S and P near Overflow", bignum, bignum);
    run_test_case(n, "Both S and P near Underflow", smlnum, smlnum);
}

int main()
{
    test_dynamic_lapack(100);
    run_lapack_edge_cases(100);

    return 0;
}