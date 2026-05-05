// test_ztgevc3.cpp
#include <algorithm>
#include <chrono>
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

template <typename T>
void time_code(const std::string_view what_it_is, T lambda_expression)
{
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;
    double total_time;

    start_time = std::chrono::high_resolution_clock::now();
    lambda_expression();
    stop_time = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double>(stop_time - start_time).count();

    fmt::println("{} took {:.3} seconds", what_it_is, total_time);
}

// Replicates LAPACK test routine zget52.
void zget52(bool compute_left, bool compute_right, int n, const std::complex<double> *S, int lds, const std::complex<double> *P, int ldp, const std::complex<double> *alpha, const std::complex<double> *beta, const std::complex<double> *VL, int ldvl, const std::complex<double> *VR, int ldvr, double *result)
{
    double normS, normP, csS, csP, norm_r, norm_v, den;
    double ulp, safmin, max_err_r, max_err_l;
    int c, r, i, j;

    ulp = std::numeric_limits<double>::epsilon();
    safmin = std::numeric_limits<double>::min();

    normS = 0.0;
    normP = 0.0;

    for (c = 0; c < n; c++) {
        csS = 0.0;
        csP = 0.0;
        for (r = 0; r < n; r++) {
            csS += std::abs(S[r + c * lds]);
            csP += std::abs(P[r + c * ldp]);
        }
        normS = std::max(normS, csS);
        normP = std::max(normP, csP);
    }

    if (normS == 0.0) {
        normS = 1.0;
    }
    if (normP == 0.0) {
        normP = 1.0;
    }

    max_err_r = 0.0;
    max_err_l = 0.0;

    if (compute_right) {
        for (c = 0; c < n; c++) {
            std::complex<double> a = alpha[c];
            std::complex<double> b = beta[c];
            norm_r = 0.0;
            norm_v = 0.0;

            for (i = 0; i < n; i++) {
                std::complex<double> val = { 0.0, 0.0 };
                for (j = 0; j < n; j++) {
                    val += b * S[i + j * lds] * VR[j + c * ldvr] - a * P[i + j * ldp] * VR[j + c * ldvr];
                }
                norm_r += std::abs(val);
                norm_v += std::abs(VR[i + c * ldvr]);
            }

            den = (std::abs(b) * normS + std::abs(a) * normP) * norm_v;
            den = std::max(den, safmin);
            max_err_r = std::max(max_err_r, norm_r / (den * ulp));
        }
    }

    if (compute_left) {
        for (c = 0; c < n; c++) {
            std::complex<double> a = alpha[c];
            std::complex<double> b = beta[c];
            norm_r = 0.0;
            norm_v = 0.0;

            for (j = 0; j < n; j++) {
                std::complex<double> val = { 0.0, 0.0 };
                for (i = 0; i < n; i++) {
                    val += std::conj(VL[i + c * ldvl]) * (b * S[i + j * lds] - a * P[i + j * ldp]);
                }
                norm_r += std::abs(val);
            }
            for (i = 0; i < n; i++) {
                norm_v += std::abs(VL[i + c * ldvl]);
            }

            den = (std::abs(b) * normS + std::abs(a) * normP) * norm_v;
            den = std::max(den, safmin);
            max_err_l = std::max(max_err_l, norm_r / (den * ulp));
        }
    }

    result[0] = max_err_r;
    result[1] = max_err_l;
}

void generate_generalized_upper_triangular(int n, std::complex<double> *S, int lds, std::complex<double> *P, int ldp, std::complex<double> *alpha, std::complex<double> *beta)
{
    int i, c, r;

    for (i = 0; i < n * n; i++) {
        S[i] = { 0.0, 0.0 };
        P[i] = { 0.0, 0.0 };
    }

    for (c = 0; c < n; c++) {
        for (r = 0; r < c; r++) { // Subdiagonal must be strictly 0
            S[r + c * lds] = std::complex<double>(dist(gen), dist(gen));
            P[r + c * ldp] = std::complex<double>(dist(gen), dist(gen));
        }

        // P diagonal elements must be real and non-negative
        S[c + c * lds] = std::complex<double>(dist(gen), dist(gen));
        P[c + c * ldp] = std::complex<double>(std::abs(dist(gen)) + 0.1, 0.0);

        alpha[c] = S[c + c * lds];
        beta[c] = P[c + c * ldp];
    }
}

void generate_lapack_matrix_type(int type, int n, std::complex<double> *S, int lds, std::complex<double> *P, int ldp, std::complex<double> *alpha, std::complex<double> *beta)
{
    int i, j, k;
    double big, small;

    big = 1e10;
    small = 1e-10;

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            S[i + j * lds] = { 0.0, 0.0 };
            P[i + j * ldp] = { 0.0, 0.0 };
        }
    }

    switch (type) {
    case 1:
        break;
    case 2:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { 1.0, 0.0 };
        }
        break;
    case 3:
        for (i = 0; i < n; i++) {
            P[i + i * ldp] = { 1.0, 0.0 };
        }
        break;
    case 4:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { 1.0, 0.0 };
            P[i + i * ldp] = { 1.0, 0.0 };
        }
        break;
    case 5:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { 1.0, 0.0 };
            P[i + i * ldp] = { 1.0, 0.0 };
            if (i < n - 1) {
                S[i + (i + 1) * lds] = { 1.0, 0.0 };
                P[i + (i + 1) * ldp] = { 1.0, 0.0 };
            }
        }
        break;
    case 6:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { (double)(i + 1) / n, 0.0 };
            P[i + i * ldp] = { 1.0, 0.0 };
        }
        break;
    case 7:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { 1.0, 0.0 };
            P[i + i * ldp] = { (double)(i + 1) / n, 0.0 };
        }
        break;
    case 8:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { (double)(i + 1) / n, 0.0 };
            P[i + i * ldp] = { (double)(n - i) / n, 0.0 };
        }
        break;
    case 9:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { big * (i + 1) / n, 0.0 };
            P[i + i * ldp] = { small, 0.0 };
        }
        break;
    case 10:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { small * (i + 1) / n, 0.0 };
            P[i + i * ldp] = { big, 0.0 };
        }
        break;
    case 11:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { big, 0.0 };
            P[i + i * ldp] = { small * (i + 1) / n, 0.0 };
        }
        break;
    case 12:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { small, 0.0 };
            P[i + i * ldp] = { big * (i + 1) / n, 0.0 };
        }
        break;
    case 13:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { big * (i + 1) / n, 0.0 };
            P[i + i * ldp] = { big, 0.0 };
        }
        break;
    case 14:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { small * (i + 1) / n, 0.0 };
            P[i + i * ldp] = { small, 0.0 };
        }
        break;
    case 15:
        for (i = 0; i < n; i++) {
            if (i == 0 || i == 1 || i == n - 1) {
                S[i + i * lds] = { 0.0, 0.0 };
            }
            else {
                S[i + i * lds] = { (double)(i - 1), 0.0 };
            }

            if (i == 0 || i == n - 2 || i == n - 1) {
                P[i + i * ldp] = { 0.0, 0.0 };
            }
            else {
                P[i + i * ldp] = { (double)(n - i - 1), 0.0 };
            }
        }
        break;
    default:
        for (j = 0; j < n; j++) {
            for (i = 0; i < j; i++) {
                S[i + j * lds] = { dist(gen), dist(gen) };
                P[i + j * ldp] = { dist(gen), dist(gen) };
            }
            // P diagonal elements must be real and non-negative
            S[j + j * lds] = { dist(gen), dist(gen) };
            P[j + j * ldp] = { std::abs(dist(gen)) + 0.1, 0.0 };
        }
        break;
    }

    for (k = 0; k < n; k++) {
        alpha[k] = S[k + k * lds];
        beta[k] = P[k + k * ldp];
    }
}

void test_ztgevc_static()
{
    int n = 4, info[1], max_bsize, work_size, i, j, m_out;
    double res3[2], resL[2], *rwork_lapack;
    std::complex<double> *work, *work_lapack;
    std::complex<double> S[16] = { { 0, 0 } }, P[16] = { { 0, 0 } }, alpha[4], beta[4], VR[16], VL[16];

    S[0 + 0 * 4] = { 1.0, 1.0 };
    P[0 + 0 * 4] = { 2.0, 0.0 };
    S[0 + 1 * 4] = { 2.0, 0.0 };
    P[0 + 1 * 4] = { 1.0, 1.0 };
    S[1 + 1 * 4] = { 2.0, -1.0 };
    P[1 + 1 * 4] = { 1.0, 0.0 }; // Real diagonal
    S[0 + 2 * 4] = { -5.0, 1.0 };
    P[0 + 2 * 4] = { 0.0, 0.0 };
    S[1 + 2 * 4] = { 0.0, 2.0 };
    P[1 + 2 * 4] = { -1.0, 0.0 };
    S[2 + 2 * 4] = { 3.0, 0.0 };
    P[2 + 2 * 4] = { 1.0, 0.0 }; // Real diagonal
    S[0 + 3 * 4] = { 5.0, 0.0 };
    P[0 + 3 * 4] = { 0.0, 0.0 };
    S[1 + 3 * 4] = { 2.0, 1.0 };
    P[1 + 3 * 4] = { 0.0, 2.0 };
    S[2 + 3 * 4] = { 4.0, -1.0 };
    P[2 + 3 * 4] = { 3.0, 0.0 };
    S[3 + 3 * 4] = { 3.0, 2.0 };
    P[3 + 3 * 4] = { 2.0, 0.0 }; // Real diagonal

    for (i = 0; i < n; i++) {
        alpha[i] = S[i + i * n];
        beta[i] = P[i + i * n];
    }

    fmt::print("--- Testing 4x4 Complex Static Matrix (Both Left/Right GEVP Evecs) ---\n");

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (std::complex<double> *)malloc(work_size * sizeof(std::complex<double>));
    work_lapack = (std::complex<double> *)malloc(2 * n * sizeof(std::complex<double>));
    rwork_lapack = (double *)malloc(2 * n * sizeof(double));

    // Test ztgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0, 0.0 };
            VL[i * n + j] = { 0.0, 0.0 };
        }
        VR[i * n + i] = { 1.0, 0.0 };
        VL[i * n + i] = { 1.0, 0.0 };
    }
    ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
    zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

    // Test LAPACK ztgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0, 0.0 };
            VL[i * n + j] = { 0.0, 0.0 };
        }
        VR[i * n + i] = { 1.0, 0.0 };
        VL[i * n + i] = { 1.0, 0.0 };
    }
    ztgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info);
    zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

    fmt::print("[ztgevc3] zget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[ztgevc3] zget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] zget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] zget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(work);
    free(work_lapack);
    free(rwork_lapack);
}

void test_ztgevc_infinite()
{
    int n = 4, info[1], max_bsize, work_size, i, j, m_out;
    double res3[2], resL[2], *rwork_lapack;
    std::complex<double> *work, *work_lapack;
    std::complex<double> S[16] = { { 0, 0 } }, P[16] = { { 0, 0 } }, alpha[4], beta[4], VR[16], VL[16];

    S[0 + 0 * 4] = { 1.0, 1.0 };
    P[0 + 0 * 4] = { 2.0, 0.0 };
    S[0 + 1 * 4] = { 2.0, 0.0 };
    P[0 + 1 * 4] = { 1.0, 1.0 };
    S[1 + 1 * 4] = { 2.0, -1.0 };
    P[1 + 1 * 4] = { 1.0, 0.0 }; // Real diagonal
    S[0 + 2 * 4] = { -5.0, 1.0 };
    P[0 + 2 * 4] = { 0.0, 0.0 };
    S[1 + 2 * 4] = { 0.0, 2.0 };
    P[1 + 2 * 4] = { -1.0, 0.0 };
    S[2 + 2 * 4] = { 3.0, 0.0 };
    P[2 + 2 * 4] = { 1.0, 0.0 }; // Real diagonal
    S[0 + 3 * 4] = { 5.0, 0.0 };
    P[0 + 3 * 4] = { 0.0, 0.0 };
    S[1 + 3 * 4] = { 2.0, 1.0 };
    P[1 + 3 * 4] = { 0.0, 2.0 };
    S[2 + 3 * 4] = { 4.0, -1.0 };
    P[2 + 3 * 4] = { 3.0, 0.0 };
    S[3 + 3 * 4] = { 3.0, 2.0 };
    P[3 + 3 * 4] = { 0.0, 0.0 };

    for (i = 0; i < n; i++) {
        alpha[i] = S[i + i * n];
        beta[i] = P[i + i * n];
    }

    fmt::print("--- Testing 4x4 Complex Static Matrix (Infinite Eigenvalue Case) ---\n");

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (std::complex<double> *)malloc(work_size * sizeof(std::complex<double>));
    work_lapack = (std::complex<double> *)malloc(2 * n * sizeof(std::complex<double>));
    rwork_lapack = (double *)malloc(2 * n * sizeof(double));

    // Test ztgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0, 0.0 };
            VL[i * n + j] = { 0.0, 0.0 };
        }
        VR[i * n + i] = { 1.0, 0.0 };
        VL[i * n + i] = { 1.0, 0.0 };
    }
    ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
    zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

    // Test LAPACK ztgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0, 0.0 };
            VL[i * n + j] = { 0.0, 0.0 };
        }
        VR[i * n + i] = { 1.0, 0.0 };
        VL[i * n + i] = { 1.0, 0.0 };
    }
    ztgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info);
    zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

    fmt::print("[ztgevc3] zget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[ztgevc3] zget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] zget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] zget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(work);
    free(work_lapack);
    free(rwork_lapack);
}

void test_ztgevc_scaling()
{
    int n = 4, info[1], max_bsize, work_size, i, j, test_idx, m_out;
    double res3[2], resL[2], *rwork_lapack, s;
    std::complex<double> *work, *work_lapack;
    std::complex<double> S_base[16] = { { 0, 0 } }, P_base[16] = { { 0, 0 } }, S[16], P[16], alpha[4], beta[4], VR[16], VL[16];
    double scales[2] = { 1e150, 1e-150 };
    const char *scale_names[2] = { "Overflow Risk (1e150)", "Underflow Risk (1e-150)" };

    S_base[0 + 0 * 4] = { 1.0, 1.0 };
    P_base[0 + 0 * 4] = { 2.0, 0.0 };
    S_base[0 + 1 * 4] = { 2.0, 0.0 };
    P_base[0 + 1 * 4] = { 1.0, 1.0 };
    S_base[1 + 1 * 4] = { 2.0, -1.0 };
    P_base[1 + 1 * 4] = { 1.0, 0.0 }; // Real diagonal
    S_base[0 + 2 * 4] = { -5.0, 1.0 };
    P_base[0 + 2 * 4] = { 0.0, 0.0 };
    S_base[1 + 2 * 4] = { 0.0, 2.0 };
    P_base[1 + 2 * 4] = { -1.0, 0.0 };
    S_base[2 + 2 * 4] = { 3.0, 0.0 };
    P_base[2 + 2 * 4] = { 1.0, 0.0 }; // Real diagonal
    S_base[0 + 3 * 4] = { 5.0, 0.0 };
    P_base[0 + 3 * 4] = { 0.0, 0.0 };
    S_base[1 + 3 * 4] = { 2.0, 1.0 };
    P_base[1 + 3 * 4] = { 0.0, 2.0 };
    S_base[2 + 3 * 4] = { 4.0, -1.0 };
    P_base[2 + 3 * 4] = { 3.0, 0.0 };
    S_base[3 + 3 * 4] = { 3.0, 2.0 };
    P_base[3 + 3 * 4] = { 2.0, 0.0 }; // Real diagonal

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (std::complex<double> *)malloc(work_size * sizeof(std::complex<double>));
    work_lapack = (std::complex<double> *)malloc(2 * n * sizeof(std::complex<double>));
    rwork_lapack = (double *)malloc(2 * n * sizeof(double));

    for (test_idx = 0; test_idx < 2; test_idx++) {
        fmt::print("--- Testing 4x4 Static Matrix (Scaling: {}) ---\n", scale_names[test_idx]);

        s = scales[test_idx];

        for (i = 0; i < 16; i++) {
            S[i] = S_base[i] * s;
            P[i] = P_base[i] * s;
        }
        for (i = 0; i < 4; i++) {
            alpha[i] = S[i + i * n];
            beta[i] = P[i + i * n];
        }

        // Test ztgevc3
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = { 0.0, 0.0 };
                VL[i * n + j] = { 0.0, 0.0 };
            }
            VR[i * n + i] = { 1.0, 0.0 };
            VL[i * n + i] = { 1.0, 0.0 };
        }
        ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
        zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

        // Test LAPACK ztgevc
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = { 0.0, 0.0 };
                VL[i * n + j] = { 0.0, 0.0 };
            }
            VR[i * n + i] = { 1.0, 0.0 };
            VL[i * n + i] = { 1.0, 0.0 };
        }
        ztgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info);
        zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

        fmt::print("[ztgevc3] zget52 Max Right Error Ratio: {:e}\n", res3[0]);
        fmt::print("[ztgevc3] zget52 Max Left Error Ratio:  {:e}\n", res3[1]);
        fmt::print("[lapack ] zget52 Max Right Error Ratio: {:e}\n", resL[0]);
        fmt::print("[lapack ] zget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);
    }

    free(work);
    free(work_lapack);
    free(rwork_lapack);
}

void test_dynamic_size(int n)
{
    std::complex<double> *S, *P, *VR, *VL, *alpha, *beta, *work, *work_lapack, dummy[1];
    double res3[2], resL[2], *rwork_lapack;
    int info[1], lwork, i, j, m_out;

    fmt::println("--- Testing {0}x{0} Dynamic Random Matrix ---", n);

    S = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    P = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VR = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VL = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    alpha = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
    beta = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));

    generate_generalized_upper_triangular(n, S, n, P, n, alpha, beta);
    ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);

    lwork = static_cast<int>(dummy[0].real());
    work = (std::complex<double> *)malloc(lwork * sizeof(std::complex<double>));
    work_lapack = (std::complex<double> *)malloc(2 * n * sizeof(std::complex<double>));
    rwork_lapack = (double *)malloc(2 * n * sizeof(double));

    // Test ztgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0, 0.0 };
            VL[i * n + j] = { 0.0, 0.0 };
        }
        VR[i * n + i] = { 1.0, 0.0 };
        VL[i * n + i] = { 1.0, 0.0 };
    }
    time_code("ztgevc3 ('B')", [&]() { ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, lwork, info); });
    zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

    // Test LAPACK ztgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0, 0.0 };
            VL[i * n + j] = { 0.0, 0.0 };
        }
        VR[i * n + i] = { 1.0, 0.0 };
        VL[i * n + i] = { 1.0, 0.0 };
    }
    time_code("LAPACK ztgevc ('B')", [&]() { ztgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info); });
    zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

    fmt::print("[ztgevc3] zget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[ztgevc3] zget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] zget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] zget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(S);
    free(P);
    free(VR);
    free(VL);
    free(alpha);
    free(beta);
    free(work);
    free(work_lapack);
    free(rwork_lapack);
}

void test_lapack_types(int n)
{
    std::complex<double> *S, *P, *VR, *VL, *alpha, *beta, *work, *work_lapack, dummy[1];
    double res3[2], resL[2], *rwork_lapack;
    int info[1], lwork, i, j, type, m_out;

    fmt::println("--- Testing {}x{} Matrices over 26 LAPACK Matrix Types ---", n, n);

    S = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    P = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VR = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VL = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    alpha = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
    beta = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
    work_lapack = (std::complex<double> *)malloc(2 * n * sizeof(std::complex<double>));
    rwork_lapack = (double *)malloc(2 * n * sizeof(double));

    for (type = 1; type <= 26; type++) {
        generate_lapack_matrix_type(type, n, S, n, P, n, alpha, beta);

        ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);
        lwork = static_cast<int>(dummy[0].real());
        if (lwork <= 0) {
            lwork = 2 * n * (32 + 1) + 4 * (32 + 1) * (32 + 1) + 2 * (32 + 1);
        }
        work = (std::complex<double> *)malloc(lwork * sizeof(std::complex<double>));

        // Test ztgevc3
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = { 0.0, 0.0 };
                VL[i * n + j] = { 0.0, 0.0 };
            }
            VR[i * n + i] = { 1.0, 0.0 };
            VL[i * n + i] = { 1.0, 0.0 };
        }
        ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, lwork, info);
        zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

        // Test LAPACK ztgevc
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = { 0.0, 0.0 };
                VL[i * n + j] = { 0.0, 0.0 };
            }
            VR[i * n + i] = { 1.0, 0.0 };
            VL[i * n + i] = { 1.0, 0.0 };
        }
        ztgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info);
        zget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

        fmt::println("Type {:2}:", type);
        fmt::println("  [ztgevc3] Max Right = {:e}, Max Left = {:e}", res3[0], res3[1]);
        fmt::println("  [lapack ] Max Right = {:e}, Max Left = {:e}", resL[0], resL[1]);

        free(work);
    }

    fmt::print("\n");

    free(S);
    free(P);
    free(VR);
    free(VL);
    free(alpha);
    free(beta);
    free(work_lapack);
    free(rwork_lapack);
}

int main()
{
    test_ztgevc_static();
    test_ztgevc_infinite();
    test_ztgevc_scaling();

    test_dynamic_size(10);
    test_dynamic_size(500);

    test_lapack_types(10);
    test_lapack_types(50);

    return 0;
}