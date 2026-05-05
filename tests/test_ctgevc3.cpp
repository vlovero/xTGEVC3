// test_ctgevc3.cpp
#include <algorithm>
#include <chrono>
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

// Replicates LAPACK test routine cget52.
void cget52(bool compute_left, bool compute_right, int n, const std::complex<float> *S, int lds, const std::complex<float> *P, int ldp, const std::complex<float> *alpha, const std::complex<float> *beta, const std::complex<float> *VL, int ldvl, const std::complex<float> *VR, int ldvr, float *result)
{
    float normS, normP, csS, csP, norm_r, norm_v, den;
    float ulp, safmin, max_err_r, max_err_l;
    int c, r, i, j;

    ulp = std::numeric_limits<float>::epsilon();
    safmin = std::numeric_limits<float>::min();

    normS = 0.0f;
    normP = 0.0f;

    for (c = 0; c < n; c++) {
        csS = 0.0f;
        csP = 0.0f;
        for (r = 0; r < n; r++) {
            csS += std::abs(S[r + c * lds]);
            csP += std::abs(P[r + c * ldp]);
        }
        normS = std::max(normS, csS);
        normP = std::max(normP, csP);
    }

    if (normS == 0.0f) {
        normS = 1.0f;
    }
    if (normP == 0.0f) {
        normP = 1.0f;
    }

    max_err_r = 0.0f;
    max_err_l = 0.0f;

    if (compute_right) {
        for (c = 0; c < n; c++) {
            std::complex<float> a = alpha[c];
            std::complex<float> b = beta[c];
            norm_r = 0.0f;
            norm_v = 0.0f;

            for (i = 0; i < n; i++) {
                std::complex<float> val = { 0.0f, 0.0f };
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
            std::complex<float> a = alpha[c];
            std::complex<float> b = beta[c];
            norm_r = 0.0f;
            norm_v = 0.0f;

            for (j = 0; j < n; j++) {
                std::complex<float> val = { 0.0f, 0.0f };
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

void generate_generalized_upper_triangular(int n, std::complex<float> *S, int lds, std::complex<float> *P, int ldp, std::complex<float> *alpha, std::complex<float> *beta)
{
    int i, c, r;

    for (i = 0; i < n * n; i++) {
        S[i] = { 0.0f, 0.0f };
        P[i] = { 0.0f, 0.0f };
    }

    for (c = 0; c < n; c++) {
        for (r = 0; r < c; r++) { // Subdiagonal must be strictly 0
            S[r + c * lds] = std::complex<float>(dist(gen), dist(gen));
            P[r + c * ldp] = std::complex<float>(dist(gen), dist(gen));
        }

        // P diagonal elements must be real and non-negative
        S[c + c * lds] = std::complex<float>(dist(gen), dist(gen));
        P[c + c * ldp] = std::complex<float>(std::abs(dist(gen)) + 0.1f, 0.0f);

        alpha[c] = S[c + c * lds];
        beta[c] = P[c + c * ldp];
    }
}

void generate_lapack_matrix_type(int type, int n, std::complex<float> *S, int lds, std::complex<float> *P, int ldp, std::complex<float> *alpha, std::complex<float> *beta)
{
    int i, j, k;
    float big, small;

    big = 1e10f;
    small = 1e-10f;

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            S[i + j * lds] = { 0.0f, 0.0f };
            P[i + j * ldp] = { 0.0f, 0.0f };
        }
    }

    switch (type) {
    case 1:
        break;
    case 2:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { 1.0f, 0.0f };
        }
        break;
    case 3:
        for (i = 0; i < n; i++) {
            P[i + i * ldp] = { 1.0f, 0.0f };
        }
        break;
    case 4:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { 1.0f, 0.0f };
            P[i + i * ldp] = { 1.0f, 0.0f };
        }
        break;
    case 5:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { 1.0f, 0.0f };
            P[i + i * ldp] = { 1.0f, 0.0f };
            if (i < n - 1) {
                S[i + (i + 1) * lds] = { 1.0f, 0.0f };
                P[i + (i + 1) * ldp] = { 1.0f, 0.0f };
            }
        }
        break;
    case 6:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { (float)(i + 1) / n, 0.0f };
            P[i + i * ldp] = { 1.0f, 0.0f };
        }
        break;
    case 7:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { 1.0f, 0.0f };
            P[i + i * ldp] = { (float)(i + 1) / n, 0.0f };
        }
        break;
    case 8:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { (float)(i + 1) / n, 0.0f };
            P[i + i * ldp] = { (float)(n - i) / n, 0.0f };
        }
        break;
    case 9:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { big * (i + 1) / n, 0.0f };
            P[i + i * ldp] = { small, 0.0f };
        }
        break;
    case 10:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { small * (i + 1) / n, 0.0f };
            P[i + i * ldp] = { big, 0.0f };
        }
        break;
    case 11:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { big, 0.0f };
            P[i + i * ldp] = { small * (i + 1) / n, 0.0f };
        }
        break;
    case 12:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { small, 0.0f };
            P[i + i * ldp] = { big * (i + 1) / n, 0.0f };
        }
        break;
    case 13:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { big * (i + 1) / n, 0.0f };
            P[i + i * ldp] = { big, 0.0f };
        }
        break;
    case 14:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = { small * (i + 1) / n, 0.0f };
            P[i + i * ldp] = { small, 0.0f };
        }
        break;
    case 15:
        for (i = 0; i < n; i++) {
            if (i == 0 || i == 1 || i == n - 1) {
                S[i + i * lds] = { 0.0f, 0.0f };
            }
            else {
                S[i + i * lds] = { (float)(i - 1), 0.0f };
            }

            if (i == 0 || i == n - 2 || i == n - 1) {
                P[i + i * ldp] = { 0.0f, 0.0f };
            }
            else {
                P[i + i * ldp] = { (float)(n - i - 1), 0.0f };
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
            P[j + j * ldp] = { std::abs(dist(gen)) + 0.1f, 0.0f };
        }
        break;
    }

    for (k = 0; k < n; k++) {
        alpha[k] = S[k + k * lds];
        beta[k] = P[k + k * ldp];
    }
}

void test_ctgevc_static()
{
    int n = 4, info[1], max_bsize, work_size, i, j, m_out;
    float res3[2], resL[2], *rwork_lapack;
    std::complex<float> *work, *work_lapack;
    std::complex<float> S[16] = { { 0, 0 } }, P[16] = { { 0, 0 } }, alpha[4], beta[4], VR[16], VL[16];

    S[0 + 0 * 4] = { 1.0f, 1.0f };
    P[0 + 0 * 4] = { 2.0f, 0.0f };
    S[0 + 1 * 4] = { 2.0f, 0.0f };
    P[0 + 1 * 4] = { 1.0f, 1.0f };
    S[1 + 1 * 4] = { 2.0f, -1.0f };
    P[1 + 1 * 4] = { 1.0f, 0.0f }; // Real diagonal
    S[0 + 2 * 4] = { -5.0f, 1.0f };
    P[0 + 2 * 4] = { 0.0f, 0.0f };
    S[1 + 2 * 4] = { 0.0f, 2.0f };
    P[1 + 2 * 4] = { -1.0f, 0.0f };
    S[2 + 2 * 4] = { 3.0f, 0.0f };
    P[2 + 2 * 4] = { 1.0f, 0.0f }; // Real diagonal
    S[0 + 3 * 4] = { 5.0f, 0.0f };
    P[0 + 3 * 4] = { 0.0f, 0.0f };
    S[1 + 3 * 4] = { 2.0f, 1.0f };
    P[1 + 3 * 4] = { 0.0f, 2.0f };
    S[2 + 3 * 4] = { 4.0f, -1.0f };
    P[2 + 3 * 4] = { 3.0f, 0.0f };
    S[3 + 3 * 4] = { 3.0f, 2.0f };
    P[3 + 3 * 4] = { 2.0f, 0.0f }; // Real diagonal

    for (i = 0; i < n; i++) {
        alpha[i] = S[i + i * n];
        beta[i] = P[i + i * n];
    }

    fmt::print("--- Testing 4x4 Complex Static Matrix (Both Left/Right GEVP Evecs) ---\n");

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (std::complex<float> *)malloc(work_size * sizeof(std::complex<float>));
    work_lapack = (std::complex<float> *)malloc(2 * n * sizeof(std::complex<float>));
    rwork_lapack = (float *)malloc(2 * n * sizeof(float));

    // Test ctgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0f, 0.0f };
            VL[i * n + j] = { 0.0f, 0.0f };
        }
        VR[i * n + i] = { 1.0f, 0.0f };
        VL[i * n + i] = { 1.0f, 0.0f };
    }
    ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
    cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

    // Test LAPACK ctgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0f, 0.0f };
            VL[i * n + j] = { 0.0f, 0.0f };
        }
        VR[i * n + i] = { 1.0f, 0.0f };
        VL[i * n + i] = { 1.0f, 0.0f };
    }
    ctgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info);
    cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

    fmt::print("[ctgevc3] cget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[ctgevc3] cget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] cget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] cget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(work);
    free(work_lapack);
    free(rwork_lapack);
}

void test_ctgevc_infinite()
{
    int n = 4, info[1], max_bsize, work_size, i, j, m_out;
    float res3[2], resL[2], *rwork_lapack;
    std::complex<float> *work, *work_lapack;
    std::complex<float> S[16] = { { 0, 0 } }, P[16] = { { 0, 0 } }, alpha[4], beta[4], VR[16], VL[16];

    S[0 + 0 * 4] = { 1.0f, 1.0f };
    P[0 + 0 * 4] = { 2.0f, 0.0f };
    S[0 + 1 * 4] = { 2.0f, 0.0f };
    P[0 + 1 * 4] = { 1.0f, 1.0f };
    S[1 + 1 * 4] = { 2.0f, -1.0f };
    P[1 + 1 * 4] = { 1.0f, 0.0f }; // Real diagonal
    S[0 + 2 * 4] = { -5.0f, 1.0f };
    P[0 + 2 * 4] = { 0.0f, 0.0f };
    S[1 + 2 * 4] = { 0.0f, 2.0f };
    P[1 + 2 * 4] = { -1.0f, 0.0f };
    S[2 + 2 * 4] = { 3.0f, 0.0f };
    P[2 + 2 * 4] = { 1.0f, 0.0f }; // Real diagonal
    S[0 + 3 * 4] = { 5.0f, 0.0f };
    P[0 + 3 * 4] = { 0.0f, 0.0f };
    S[1 + 3 * 4] = { 2.0f, 1.0f };
    P[1 + 3 * 4] = { 0.0f, 2.0f };
    S[2 + 3 * 4] = { 4.0f, -1.0f };
    P[2 + 3 * 4] = { 3.0f, 0.0f };
    S[3 + 3 * 4] = { 3.0f, 2.0f };
    P[3 + 3 * 4] = { 0.0f, 0.0f };

    for (i = 0; i < n; i++) {
        alpha[i] = S[i + i * n];
        beta[i] = P[i + i * n];
    }

    fmt::print("--- Testing 4x4 Complex Static Matrix (Infinite Eigenvalue Case) ---\n");

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (std::complex<float> *)malloc(work_size * sizeof(std::complex<float>));
    work_lapack = (std::complex<float> *)malloc(2 * n * sizeof(std::complex<float>));
    rwork_lapack = (float *)malloc(2 * n * sizeof(float));

    // Test ctgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0f, 0.0f };
            VL[i * n + j] = { 0.0f, 0.0f };
        }
        VR[i * n + i] = { 1.0f, 0.0f };
        VL[i * n + i] = { 1.0f, 0.0f };
    }
    ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
    cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

    // Test LAPACK ctgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0f, 0.0f };
            VL[i * n + j] = { 0.0f, 0.0f };
        }
        VR[i * n + i] = { 1.0f, 0.0f };
        VL[i * n + i] = { 1.0f, 0.0f };
    }
    ctgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info);
    cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

    fmt::print("[ctgevc3] cget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[ctgevc3] cget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] cget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] cget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(work);
    free(work_lapack);
    free(rwork_lapack);
}

void test_ctgevc_scaling()
{
    int n = 4, info[1], max_bsize, work_size, i, j, test_idx, m_out;
    float res3[2], resL[2], *rwork_lapack, s;
    std::complex<float> *work, *work_lapack;
    std::complex<float> S_base[16] = { { 0, 0 } }, P_base[16] = { { 0, 0 } }, S[16], P[16], alpha[4], beta[4], VR[16], VL[16];
    float scales[2] = { 1e30f, 1e-30f };
    const char *scale_names[2] = { "Overflow Risk (1e30)", "Underflow Risk (1e-30)" };

    S_base[0 + 0 * 4] = { 1.0f, 1.0f };
    P_base[0 + 0 * 4] = { 2.0f, 0.0f };
    S_base[0 + 1 * 4] = { 2.0f, 0.0f };
    P_base[0 + 1 * 4] = { 1.0f, 1.0f };
    S_base[1 + 1 * 4] = { 2.0f, -1.0f };
    P_base[1 + 1 * 4] = { 1.0f, 0.0f }; // Real diagonal
    S_base[0 + 2 * 4] = { -5.0f, 1.0f };
    P_base[0 + 2 * 4] = { 0.0f, 0.0f };
    S_base[1 + 2 * 4] = { 0.0f, 2.0f };
    P_base[1 + 2 * 4] = { -1.0f, 0.0f };
    S_base[2 + 2 * 4] = { 3.0f, 0.0f };
    P_base[2 + 2 * 4] = { 1.0f, 0.0f }; // Real diagonal
    S_base[0 + 3 * 4] = { 5.0f, 0.0f };
    P_base[0 + 3 * 4] = { 0.0f, 0.0f };
    S_base[1 + 3 * 4] = { 2.0f, 1.0f };
    P_base[1 + 3 * 4] = { 0.0f, 2.0f };
    S_base[2 + 3 * 4] = { 4.0f, -1.0f };
    P_base[2 + 3 * 4] = { 3.0f, 0.0f };
    S_base[3 + 3 * 4] = { 3.0f, 2.0f };
    P_base[3 + 3 * 4] = { 2.0f, 0.0f }; // Real diagonal

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (std::complex<float> *)malloc(work_size * sizeof(std::complex<float>));
    work_lapack = (std::complex<float> *)malloc(2 * n * sizeof(std::complex<float>));
    rwork_lapack = (float *)malloc(2 * n * sizeof(float));

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

        // Test ctgevc3
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = { 0.0f, 0.0f };
                VL[i * n + j] = { 0.0f, 0.0f };
            }
            VR[i * n + i] = { 1.0f, 0.0f };
            VL[i * n + i] = { 1.0f, 0.0f };
        }
        ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
        cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

        // Test LAPACK ctgevc
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = { 0.0f, 0.0f };
                VL[i * n + j] = { 0.0f, 0.0f };
            }
            VR[i * n + i] = { 1.0f, 0.0f };
            VL[i * n + i] = { 1.0f, 0.0f };
        }
        ctgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info);
        cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

        fmt::print("[ctgevc3] cget52 Max Right Error Ratio: {:e}\n", res3[0]);
        fmt::print("[ctgevc3] cget52 Max Left Error Ratio:  {:e}\n", res3[1]);
        fmt::print("[lapack ] cget52 Max Right Error Ratio: {:e}\n", resL[0]);
        fmt::print("[lapack ] cget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);
    }

    free(work);
    free(work_lapack);
    free(rwork_lapack);
}

void test_dynamic_size(int n)
{
    std::complex<float> *S, *P, *VR, *VL, *alpha, *beta, *work, *work_lapack, dummy[1];
    float res3[2], resL[2], *rwork_lapack;
    int info[1], lwork, i, j, m_out;

    fmt::println("--- Testing {0}x{0} Dynamic Random Matrix ---", n);

    S = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    P = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VR = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VL = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    alpha = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
    beta = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));

    generate_generalized_upper_triangular(n, S, n, P, n, alpha, beta);
    ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);

    lwork = static_cast<int>(dummy[0].real());
    work = (std::complex<float> *)malloc(lwork * sizeof(std::complex<float>));
    work_lapack = (std::complex<float> *)malloc(2 * n * sizeof(std::complex<float>));
    rwork_lapack = (float *)malloc(2 * n * sizeof(float));

    // Test ctgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0f, 0.0f };
            VL[i * n + j] = { 0.0f, 0.0f };
        }
        VR[i * n + i] = { 1.0f, 0.0f };
        VL[i * n + i] = { 1.0f, 0.0f };
    }
    time_code("ctgevc3 ('B')", [&]() { ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, lwork, info); });
    cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

    // Test LAPACK ctgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = { 0.0f, 0.0f };
            VL[i * n + j] = { 0.0f, 0.0f };
        }
        VR[i * n + i] = { 1.0f, 0.0f };
        VL[i * n + i] = { 1.0f, 0.0f };
    }
    time_code("LAPACK ctgevc ('B')", [&]() { ctgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info); });
    cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

    fmt::print("[ctgevc3] cget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[ctgevc3] cget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] cget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] cget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

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
    std::complex<float> *S, *P, *VR, *VL, *alpha, *beta, *work, *work_lapack, dummy[1];
    float res3[2], resL[2], *rwork_lapack;
    int info[1], lwork, i, j, type, m_out;

    fmt::println("--- Testing {}x{} Matrices over 26 LAPACK Matrix Types ---", n, n);

    S = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    P = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VR = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VL = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    alpha = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
    beta = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
    work_lapack = (std::complex<float> *)malloc(2 * n * sizeof(std::complex<float>));
    rwork_lapack = (float *)malloc(2 * n * sizeof(float));

    for (type = 1; type <= 26; type++) {
        generate_lapack_matrix_type(type, n, S, n, P, n, alpha, beta);

        ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);
        lwork = static_cast<int>(dummy[0].real());
        if (lwork <= 0) {
            lwork = 2 * n * (32 + 1) + 4 * (32 + 1) * (32 + 1) + 2 * (32 + 1);
        }
        work = (std::complex<float> *)malloc(lwork * sizeof(std::complex<float>));

        // Test ctgevc3
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = { 0.0f, 0.0f };
                VL[i * n + j] = { 0.0f, 0.0f };
            }
            VR[i * n + i] = { 1.0f, 0.0f };
            VL[i * n + i] = { 1.0f, 0.0f };
        }
        ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, lwork, info);
        cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, res3);

        // Test LAPACK ctgevc
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = { 0.0f, 0.0f };
                VL[i * n + j] = { 0.0f, 0.0f };
            }
            VR[i * n + i] = { 1.0f, 0.0f };
            VL[i * n + i] = { 1.0f, 0.0f };
        }
        ctgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, rwork_lapack, info);
        cget52(true, true, n, S, n, P, n, alpha, beta, VL, n, VR, n, resL);

        fmt::println("Type {:2}:", type);
        fmt::println("  [ctgevc3] Max Right = {:e}, Max Left = {:e}", res3[0], res3[1]);
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
    test_ctgevc_static();
    test_ctgevc_infinite();
    test_ctgevc_scaling();

    test_dynamic_size(10);
    test_dynamic_size(500);

    test_lapack_types(10);
    test_lapack_types(50);

    return 0;
}