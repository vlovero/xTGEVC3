// test_stgevc3.cpp
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <random>

#include "fmt/core.h"
#include "helpers.h"
#include "tgevc3.h"

extern "C" {
    void stgevc_(const char *side, const char *howmny, const int *select, const int *n, const float *s, const int *lds, const float *p, const int *ldp, float *vl, const int *ldvl, float *vr, const int *ldvr, const int *mm, int *m, float *work, int *info);
}

static std::mt19937 gen(0);
static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
static std::uniform_int_distribution<int> dist_int(0, 99);

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

void sget52(bool compute_left, bool compute_right, int n, const float *S, int lds, const float *P, int ldp, const float *alphar, const float *alphai, const float *beta, const float *VL, int ldvl, const float *VR, int ldvr, float *result)
{
    float normS, normP, csS, csP, a, b, aR, aI, norm_r, norm_v;
    float val, rr_val, ri_val, v_r, v_i, s_val, p_val, den, a_norm;
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
        c = 0;
        while (c < n) {
            if (alphai[c] == 0.0f) {
                a = alphar[c];
                b = beta[c];
                norm_r = 0.0f;
                norm_v = 0.0f;

                for (i = 0; i < n; i++) {
                    val = 0.0f;
                    for (j = 0; j < n; j++) {
                        val += b * S[i + j * lds] * VR[j + c * ldvr] - a * P[i + j * ldp] * VR[j + c * ldvr];
                    }
                    norm_r += std::abs(val);
                    norm_v += std::abs(VR[i + c * ldvr]);
                }

                den = (std::abs(b) * normS + std::abs(a) * normP) * norm_v;
                den = std::max(den, safmin);
                max_err_r = std::max(max_err_r, norm_r / (den * ulp));
                c++;
            }
            else {
                aR = alphar[c];
                aI = alphai[c];
                b = beta[c];
                norm_r = 0.0f;
                norm_v = 0.0f;

                for (i = 0; i < n; i++) {
                    rr_val = 0.0f;
                    ri_val = 0.0f;
                    for (j = 0; j < n; j++) {
                        v_r = VR[j + c * ldvr];
                        v_i = VR[j + (c + 1) * ldvr];
                        s_val = S[i + j * lds];
                        p_val = P[i + j * ldp];

                        rr_val += b * s_val * v_r - aR * p_val * v_r + aI * p_val * v_i;
                        ri_val += b * s_val * v_i - aR * p_val * v_i - aI * p_val * v_r;
                    }
                    norm_r += std::abs(rr_val) + std::abs(ri_val);
                    norm_v += std::abs(VR[i + c * ldvr]) + std::abs(VR[i + (c + 1) * ldvr]);
                }

                a_norm = std::abs(aR) + std::abs(aI);
                den = (std::abs(b) * normS + a_norm * normP) * norm_v;
                den = std::max(den, safmin);
                max_err_r = std::max(max_err_r, norm_r / (den * ulp));
                c += 2;
            }
        }
    }

    if (compute_left) {
        c = 0;
        while (c < n) {
            if (alphai[c] == 0.0f) {
                a = alphar[c];
                b = beta[c];
                norm_r = 0.0f;
                norm_v = 0.0f;

                for (j = 0; j < n; j++) {
                    val = 0.0f;
                    for (i = 0; i < n; i++) {
                        val += b * S[i + j * lds] * VL[i + c * ldvl] - a * P[i + j * ldp] * VL[i + c * ldvl];
                    }
                    norm_r += std::abs(val);
                }
                for (i = 0; i < n; i++) {
                    norm_v += std::abs(VL[i + c * ldvl]);
                }

                den = (std::abs(b) * normS + std::abs(a) * normP) * norm_v;
                den = std::max(den, safmin);
                max_err_l = std::max(max_err_l, norm_r / (den * ulp));
                c++;
            }
            else {
                aR = alphar[c];
                aI = alphai[c];
                b = beta[c];
                norm_r = 0.0f;
                norm_v = 0.0f;

                for (j = 0; j < n; j++) {
                    rr_val = 0.0f;
                    ri_val = 0.0f;
                    for (i = 0; i < n; i++) {
                        v_r = VL[i + c * ldvl];
                        v_i = VL[i + (c + 1) * ldvl];
                        s_val = S[i + j * lds];
                        p_val = P[i + j * ldp];

                        rr_val += b * s_val * v_r - aR * p_val * v_r - aI * p_val * v_i;
                        ri_val += b * s_val * v_i - aR * p_val * v_i + aI * p_val * v_r;
                    }
                    norm_r += std::abs(rr_val) + std::abs(ri_val);
                }
                for (i = 0; i < n; i++) {
                    norm_v += std::abs(VL[i + c * ldvl]) + std::abs(VL[i + (c + 1) * ldvl]);
                }

                a_norm = std::abs(aR) + std::abs(aI);
                den = (std::abs(b) * normS + a_norm * normP) * norm_v;
                den = std::max(den, safmin);
                max_err_l = std::max(max_err_l, norm_r / (den * ulp));
                c += 2;
            }
        }
    }

    result[0] = max_err_r;
    result[1] = max_err_l;
}

void generate_generalized_quasi_triangular(int n, float *S, int lds, float *P, int ldp, float *alphar, float *alphai, float *beta)
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
        P[c + c * ldp] += (P[c + c * ldp] >= 0.0f ? 1.0f : -1.0f);
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

void generate_lapack_matrix_type(int type, int n, float *S, int lds, float *P, int ldp, float *alphar, float *alphai, float *beta)
{
    int i, j, k;
    float big, small, a, b, p;

    big = 1e10f;
    small = 1e-10f;

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            S[i + j * lds] = 0.0f;
            P[i + j * ldp] = 0.0f;
        }
        alphai[j] = 0.0f;
    }

    switch (type) {
    case 1:
        break;
    case 2:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = 1.0f;
        }
        break;
    case 3:
        for (i = 0; i < n; i++) {
            P[i + i * ldp] = 1.0f;
        }
        break;
    case 4:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = 1.0f;
            P[i + i * ldp] = 1.0f;
        }
        break;
    case 5:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = 1.0f;
            P[i + i * ldp] = 1.0f;
            if (i < n - 1) {
                S[i + (i + 1) * lds] = 1.0f;
                P[i + (i + 1) * ldp] = 1.0f;
            }
        }
        break;
    case 6:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = (float)(i + 1) / n;
            P[i + i * ldp] = 1.0f;
        }
        break;
    case 7:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = 1.0f;
            P[i + i * ldp] = (float)(i + 1) / n;
        }
        break;
    case 8:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = (float)(i + 1) / n;
            P[i + i * ldp] = (float)(n - i) / n;
        }
        break;
    case 9:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = big * (i + 1) / n;
            P[i + i * ldp] = small;
        }
        break;
    case 10:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = small * (i + 1) / n;
            P[i + i * ldp] = big;
        }
        break;
    case 11:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = big;
            P[i + i * ldp] = small * (i + 1) / n;
        }
        break;
    case 12:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = small;
            P[i + i * ldp] = big * (i + 1) / n;
        }
        break;
    case 13:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = big * (i + 1) / n;
            P[i + i * ldp] = big;
        }
        break;
    case 14:
        for (i = 0; i < n; i++) {
            S[i + i * lds] = small * (i + 1) / n;
            P[i + i * ldp] = small;
        }
        break;
    case 15:
        for (i = 0; i < n; i++) {
            if (i == 0 || i == 1 || i == n - 1) {
                S[i + i * lds] = 0.0f;
            }
            else {
                S[i + i * lds] = (float)(i - 1);
            }

            if (i == 0 || i == n - 2 || i == n - 1) {
                P[i + i * ldp] = 0.0f;
            }
            else {
                P[i + i * ldp] = (float)(n - i - 1);
            }
        }
        break;
    default:
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                S[i + j * lds] = dist(gen);
                P[i + j * ldp] = dist(gen);
            }
        }

        if (type % 2 == 0) {
            for (i = 0; i < n - 1; i += 2) {
                if (dist_int(gen) > 50) {
                    a = dist(gen);
                    b = dist(gen);
                    if (b == 0.0f) {
                        b = 1.0f;
                    }
                    p = std::abs(dist(gen)) + 0.1f;

                    S[i + i * lds] = a;
                    S[i + 1 + (i + 1) * lds] = a;
                    S[i + (i + 1) * lds] = b;
                    S[i + 1 + i * lds] = -b;

                    P[i + i * ldp] = p;
                    P[i + 1 + (i + 1) * ldp] = p;
                    P[i + (i + 1) * ldp] = 0.0f;
                    P[i + 1 + i * ldp] = 0.0f;
                }
            }
        }
        break;
    }

    k = 0;
    while (k < n) {
        if (k < n - 1 && S[k + 1 + k * lds] != 0.0f) {
            alphar[k] = S[k + k * lds];
            alphar[k + 1] = S[k + 1 + (k + 1) * lds];
            alphai[k] = std::abs(S[k + (k + 1) * lds]);
            alphai[k + 1] = -alphai[k];
            beta[k] = P[k + k * ldp];
            beta[k + 1] = P[k + 1 + (k + 1) * ldp];
            k += 2;
        }
        else {
            alphar[k] = S[k + k * lds];
            alphai[k] = 0.0f;
            beta[k] = P[k + k * ldp];
            k++;
        }
    }
}

void test_stgevc()
{
    int n = 4, info[1], max_bsize, work_size, i, j, m_out;
    float *work, *work_lapack, res3[2], resL[2];
    float S[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    float P[16] = { 2, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3, 2, -1, 2 };
    float alphar[4] = { 1.0f, 2.0f, 2.0f, 3.0f };
    float alphai[4] = { 0.0f, 5.0f, -5.0f, 0.0f };
    float beta[4] = { 2.0f, 1.0f, 1.0f, 2.0f };
    float VR[16], VL[16];

    fmt::print("--- Testing 4x4 Static Matrix (Both Left/Right GEVP Evecs) ---\n");

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (float *)malloc(work_size * sizeof(float));
    work_lapack = (float *)malloc(6 * n * sizeof(float));

    // Test stgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0f;
            VL[i * n + j] = 0.0f;
        }
        VR[i * n + i] = 1.0f;
        VL[i * n + i] = 1.0f;
    }
    stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
    sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, res3);

    // Test LAPACK stgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0f;
            VL[i * n + j] = 0.0f;
        }
        VR[i * n + i] = 1.0f;
        VL[i * n + i] = 1.0f;
    }
    stgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info);
    sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, resL);

    fmt::print("[stgevc3] sget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[stgevc3] sget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] sget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] sget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(work);
    free(work_lapack);
}

void test_stgevc_infinite()
{
    int n = 4, info[1], max_bsize, work_size, i, j, m_out;
    float *work, *work_lapack, res3[2], resL[2];
    float S[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    float P[16] = { 2, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3, 2, -1, 0 };
    float alphar[4] = { 1.0f, 2.0f, 2.0f, 3.0f };
    float alphai[4] = { 0.0f, 5.0f, -5.0f, 0.0f };
    float beta[4] = { 2.0f, 1.0f, 1.0f, 0.0f };
    float VR[16], VL[16];

    fmt::print("--- Testing 4x4 Static Matrix (Infinite Eigenvalue Case) ---\n");

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (float *)malloc(work_size * sizeof(float));
    work_lapack = (float *)malloc(6 * n * sizeof(float));

    // Test stgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0f;
            VL[i * n + j] = 0.0f;
        }
        VR[i * n + i] = 1.0f;
        VL[i * n + i] = 1.0f;
    }
    stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
    sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, res3);

    // Test LAPACK stgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0f;
            VL[i * n + j] = 0.0f;
        }
        VR[i * n + i] = 1.0f;
        VL[i * n + i] = 1.0f;
    }
    stgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info);
    sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, resL);

    fmt::print("[stgevc3] sget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[stgevc3] sget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] sget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] sget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(work);
    free(work_lapack);
}

void test_stgevc_scaling()
{
    int n = 4, info[1], max_bsize, work_size, i, j, test_idx, m_out;
    float s, *work, *work_lapack, res3[2], resL[2];
    float S_base[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    float P_base[16] = { 2, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3, 2, -1, 2 };
    float alphar_base[4] = { 1.0f, 2.0f, 2.0f, 3.0f };
    float alphai_base[4] = { 0.0f, 5.0f, -5.0f, 0.0f };
    float beta_base[4] = { 2.0f, 1.0f, 1.0f, 2.0f };
    float S[16], P[16], alphar[4], alphai[4], beta[4], VR[16], VL[16];
    float scales[2] = { 1e30f, 1e-30f };
    const char *scale_names[2] = { "Overflow Risk (1e30)", "Underflow Risk (1e-30)" };

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (float *)malloc(work_size * sizeof(float));
    work_lapack = (float *)malloc(6 * n * sizeof(float));

    for (test_idx = 0; test_idx < 2; test_idx++) {
        fmt::print("--- Testing 4x4 Static Matrix (Scaling: {}) ---\n", scale_names[test_idx]);

        s = scales[test_idx];

        for (i = 0; i < 16; i++) {
            S[i] = S_base[i] * s;
            P[i] = P_base[i] * s;
        }
        for (i = 0; i < 4; i++) {
            alphar[i] = alphar_base[i] * s;
            alphai[i] = alphai_base[i] * s;
            beta[i] = beta_base[i] * s;
        }

        // Test stgevc3
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = 0.0f;
                VL[i * n + j] = 0.0f;
            }
            VR[i * n + i] = 1.0f;
            VL[i * n + i] = 1.0f;
        }
        stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, work_size, info);
        sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, res3);

        // Test LAPACK stgevc
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = 0.0f;
                VL[i * n + j] = 0.0f;
            }
            VR[i * n + i] = 1.0f;
            VL[i * n + i] = 1.0f;
        }
        stgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info);
        sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, resL);

        fmt::print("[stgevc3] sget52 Max Right Error Ratio: {:e}\n", res3[0]);
        fmt::print("[stgevc3] sget52 Max Left Error Ratio:  {:e}\n", res3[1]);
        fmt::print("[lapack ] sget52 Max Right Error Ratio: {:e}\n", resL[0]);
        fmt::print("[lapack ] sget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);
    }

    free(work);
    free(work_lapack);
}

void test_dynamic_size(int n)
{
    float *S, *P, *VR, *VL, *alphar, *alphai, *beta, *work, *work_lapack, dummy[1], res3[2], resL[2];
    int info[1], lwork, i, j, m_out;

    fmt::println("--- Testing {0}x{0} Dynamic Random Matrix ---", n);

    S = (float *)malloc(n * n * sizeof(float));
    P = (float *)malloc(n * n * sizeof(float));
    VR = (float *)malloc(n * n * sizeof(float));
    VL = (float *)malloc(n * n * sizeof(float));
    alphar = (float *)malloc(n * sizeof(float));
    alphai = (float *)malloc(n * sizeof(float));
    beta = (float *)malloc(n * sizeof(float));

    generate_generalized_quasi_triangular(n, S, n, P, n, alphar, alphai, beta);
    stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);

    lwork = static_cast<int>(dummy[0]);
    work = (float *)malloc(lwork * sizeof(float));
    work_lapack = (float *)malloc(6 * n * sizeof(float));

    // Test stgevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0f;
            VL[i * n + j] = 0.0f;
        }
        VR[i * n + i] = 1.0f;
        VL[i * n + i] = 1.0f;
    }
    time_code("stgevc3 ('B')", [&]() { stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, lwork, info); });
    sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, res3);

    // Test LAPACK stgevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0f;
            VL[i * n + j] = 0.0f;
        }
        VR[i * n + i] = 1.0f;
        VL[i * n + i] = 1.0f;
    }
    time_code("LAPACK stgevc ('B')", [&]() { stgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info); });
    sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, resL);

    fmt::print("[stgevc3] sget52 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[stgevc3] sget52 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] sget52 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] sget52 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(S);
    free(P);
    free(VR);
    free(VL);
    free(alphar);
    free(alphai);
    free(beta);
    free(work);
    free(work_lapack);
}

void test_lapack_types(int n)
{
    float *S, *P, *VR, *VL, *alphar, *alphai, *beta, *work, *work_lapack, dummy[1], res3[2], resL[2];
    int info[1], lwork, i, j, type, m_out;

    fmt::println("--- Testing {}x{} Matrices over 26 LAPACK Matrix Types ---", n, n);

    S = (float *)malloc(n * n * sizeof(float));
    P = (float *)malloc(n * n * sizeof(float));
    VR = (float *)malloc(n * n * sizeof(float));
    VL = (float *)malloc(n * n * sizeof(float));
    alphar = (float *)malloc(n * sizeof(float));
    alphai = (float *)malloc(n * sizeof(float));
    beta = (float *)malloc(n * sizeof(float));
    work_lapack = (float *)malloc(6 * n * sizeof(float));

    for (type = 1; type <= 26; type++) {
        generate_lapack_matrix_type(type, n, S, n, P, n, alphar, alphai, beta);

        stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);
        lwork = static_cast<int>(dummy[0]);
        if (lwork <= 0) {
            lwork = 2 * n * (32 + 1) + 4 * (32 + 1) * (32 + 1) + 2 * (32 + 1);
        }
        work = (float *)malloc(lwork * sizeof(float));

        // Test stgevc3
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = 0.0f;
                VL[i * n + j] = 0.0f;
            }
            VR[i * n + i] = 1.0f;
            VL[i * n + i] = 1.0f;
        }
        stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, lwork, info);
        sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, res3);

        // Test LAPACK stgevc
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = 0.0f;
                VL[i * n + j] = 0.0f;
            }
            VR[i * n + i] = 1.0f;
            VL[i * n + i] = 1.0f;
        }
        stgevc_("B", "B", nullptr, &n, S, &n, P, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info);
        sget52(true, true, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, resL);

        fmt::println("Type {:2}:", type);
        fmt::println("  [stgevc3] Max Right = {:e}, Max Left = {:e}", res3[0], res3[1]);
        fmt::println("  [lapack ] Max Right = {:e}, Max Left = {:e}", resL[0], resL[1]);

        free(work);
    }

    fmt::print("\n");

    free(S);
    free(P);
    free(VR);
    free(VL);
    free(alphar);
    free(alphai);
    free(beta);
    free(work_lapack);
}

int main()
{
    test_stgevc();
    test_stgevc_infinite();
    test_stgevc_scaling();

    test_dynamic_size(10);
    test_dynamic_size(500);

    test_lapack_types(10);
    test_lapack_types(50);

    return 0;
}