// test_stgevc3.cpp
#include "fmt/core.h"
#include "helpers.h"
#include "tgevc3.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

static std::mt19937 gen(0);
static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
static std::uniform_int_distribution<int> dist_int(0, 99);

template <typename F>
void time_code(const char *name, F &&f)
{
    f();
}

float check_right_eigenvector_residual_gevp_s(int n, const float *S, int lds, const float *P, int ldp, const float *VR, int ldvr, const float *alphar, const float *alphai, const float *beta)
{
    float normS = 0.0f, normP = 0.0f, max_err = 0.0f;
    int c, r, i, col, row, incx = 1, incy = 1;
    float *r_vec;

    for (c = 0; c < n; c++) {
        for (r = 0; r <= std::min(c + 1, n - 1); r++)
            normS += S[r + c * lds] * S[r + c * lds];
        for (r = 0; r <= c; r++)
            normP += P[r + c * ldp] * P[r + c * ldp];
    }
    normS = std::sqrt(normS);
    normP = std::sqrt(normP);
    if (normS == 0.0f)
        normS = 1.0f;
    if (normP == 0.0f)
        normP = 1.0f;

    r_vec = (float *)malloc(n * sizeof(float));

    c = 0;
    while (c < n) {
        if (alphai[c] == 0.0f) {
            float a = alphar[c], b = beta[c], alpha_term = -a, beta_term = b, alpha_zero = 0.0f, norm_r, norm_v;
            for (i = 0; i < n; i++)
                r_vec[i] = 0.0f;
            sgemv_("N", &n, &n, &beta_term, S, &lds, &VR[c * ldvr], &incx, &alpha_zero, r_vec, &incy);
            beta_term = 1.0f;
            sgemv_("N", &n, &n, &alpha_term, P, &ldp, &VR[c * ldvr], &incx, &beta_term, r_vec, &incy);

            for (i = 0; i < n; i++)
                r_vec[i] = 0.0f;
            for (col = 0; col < n; col++) {
                for (row = 0; row <= std::min(col + 1, n - 1); row++) {
                    r_vec[row] += b * S[row + col * lds] * VR[col + c * ldvr];
                    r_vec[row] -= a * P[row + col * ldp] * VR[col + c * ldvr];
                }
            }

            norm_r = snrm2_(&n, r_vec, &incx);
            norm_v = snrm2_(&n, &VR[c * ldvr], &incx);
            max_err = std::max(max_err, norm_r / ((std::abs(b) * normS + std::abs(a) * normP) * norm_v));
            c++;
        }
        else {
            float *rr = (float *)malloc(n * sizeof(float));
            float *ri = (float *)malloc(n * sizeof(float));
            float aR = alphar[c], aI = alphai[c], b = beta[c], s_val, p_val, vr_r, vr_i, norm_rr, norm_ri, norm_v, den;

            for (i = 0; i < n; i++) {
                rr[i] = 0;
                ri[i] = 0;
            }
            for (col = 0; col < n; col++) {
                for (row = 0; row <= std::min(col + 1, n - 1); row++) {
                    s_val = S[row + col * lds];
                    p_val = P[row + col * ldp];
                    vr_r = VR[col + c * ldvr];
                    vr_i = VR[col + (c + 1) * ldvr];
                    rr[row] += b * s_val * vr_r - aR * p_val * vr_r + aI * p_val * vr_i;
                    ri[row] += b * s_val * vr_i - aR * p_val * vr_i - aI * p_val * vr_r;
                }
            }

            norm_rr = snrm2_(&n, rr, &incx);
            norm_ri = snrm2_(&n, ri, &incx);
            norm_v = std::sqrt(std::pow(snrm2_(&n, &VR[c * ldvr], &incx), 2) + std::pow(snrm2_(&n, &VR[(c + 1) * ldvr], &incx), 2));
            den = (std::abs(b) * normS + std::sqrt(aR * aR + aI * aI) * normP) * norm_v;
            max_err = std::max({ max_err, norm_rr / den, norm_ri / den });

            free(rr);
            free(ri);
            c += 2;
        }
    }
    free(r_vec);
    return max_err;
}

float check_left_eigenvector_residual_gevp_s(int n, const float *S, int lds, const float *P, int ldp, const float *VL, int ldvl, const float *alphar, const float *alphai, const float *beta)
{
    float normS = 0.0f, normP = 0.0f, max_err = 0.0f;
    int c, r, i, col, row, incx = 1;
    float *r_vec;

    for (c = 0; c < n; c++) {
        for (r = 0; r <= std::min(c + 1, n - 1); r++)
            normS += S[r + c * lds] * S[r + c * lds];
        for (r = 0; r <= c; r++)
            normP += P[r + c * ldp] * P[r + c * ldp];
    }
    normS = std::sqrt(normS);
    normP = std::sqrt(normP);
    if (normS == 0.0f)
        normS = 1.0f;
    if (normP == 0.0f)
        normP = 1.0f;

    r_vec = (float *)malloc(n * sizeof(float));

    c = 0;
    while (c < n) {
        if (alphai[c] == 0.0f) {
            float a = alphar[c], b = beta[c], norm_r, norm_v;
            for (i = 0; i < n; i++)
                r_vec[i] = 0.0f;

            for (col = 0; col < n; col++) {
                for (row = 0; row <= std::min(col + 1, n - 1); row++) {
                    r_vec[col] += b * S[row + col * lds] * VL[row + c * ldvl];
                    r_vec[col] -= a * P[row + col * ldp] * VL[row + c * ldvl];
                }
            }

            norm_r = snrm2_(&n, r_vec, &incx);
            norm_v = snrm2_(&n, &VL[c * ldvl], &incx);
            max_err = std::max(max_err, norm_r / ((std::abs(b) * normS + std::abs(a) * normP) * norm_v));
            c++;
        }
        else {
            float *rr = (float *)malloc(n * sizeof(float));
            float *ri = (float *)malloc(n * sizeof(float));
            float aR = alphar[c], aI = alphai[c], b = beta[c], s_val, p_val, vl_r, vl_i, norm_rr, norm_ri, norm_v, den;

            for (i = 0; i < n; i++) {
                rr[i] = 0;
                ri[i] = 0;
            }
            for (col = 0; col < n; col++) {
                for (row = 0; row <= std::min(col + 1, n - 1); row++) {
                    s_val = S[row + col * lds];
                    p_val = P[row + col * ldp];
                    vl_r = VL[row + c * ldvl];
                    vl_i = VL[row + (c + 1) * ldvl];
                    rr[col] += b * s_val * vl_r - aR * p_val * vl_r - aI * p_val * vl_i;
                    ri[col] += b * s_val * vl_i - aR * p_val * vl_i + aI * p_val * vl_r;
                }
            }

            norm_rr = snrm2_(&n, rr, &incx);
            norm_ri = snrm2_(&n, ri, &incx);
            norm_v = std::sqrt(std::pow(snrm2_(&n, &VL[c * ldvl], &incx), 2) + std::pow(snrm2_(&n, &VL[(c + 1) * ldvl], &incx), 2));
            den = (std::abs(b) * normS + std::sqrt(aR * aR + aI * aI) * normP) * norm_v;
            max_err = std::max({ max_err, norm_rr / den, norm_ri / den });

            free(rr);
            free(ri);
            c += 2;
        }
    }
    free(r_vec);
    return max_err;
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
            if (b == 0.0f)
                b = 1.0f;
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

void test_stgevc()
{
    int n = 4, info[1], max_bsize = 32, work_size, i, m_out;
    float errR, errL, *work;

    float S[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    float P[16] = { 2, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3, 2, -1, 2 };
    float alphar[4] = { 1.0f, 2.0f, 2.0f, 3.0f };
    float alphai[4] = { 0.0f, 5.0f, -5.0f, 0.0f };
    float beta[4] = { 2.0f, 1.0f, 1.0f, 2.0f };
    float VR[16] = { 0 }, VL[16] = { 0 };

    fmt::print("--- Testing 4x4 Static Matrix (Float) ---\n");
    for (i = 0; i < n; i++) {
        VR[i * n + i] = 1.0f;
        VL[i * n + i] = 1.0f;
    }

    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (float *)malloc(work_size * sizeof(float));

    stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, work_size, info);

    errR = check_right_eigenvector_residual_gevp_s(n, S, n, P, n, VR, n, alphar, alphai, beta);
    errL = check_left_eigenvector_residual_gevp_s(n, S, n, P, n, VL, n, alphar, alphai, beta);
    fmt::print("Max Right Backward Error: {:e}\nMax Left Backward Error: {:e}\n\n", errR, errL);

    free(work);
}

void test_stgevc_scaling()
{
    int n = 4, info[1], max_bsize = 32, work_size, i, test_idx, m_out;
    float errR, errL, *work, s;

    float S_base[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    float P_base[16] = { 2, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3, 2, -1, 2 };
    float alphar_base[4] = { 1.0f, 2.0f, 2.0f, 3.0f };
    float alphai_base[4] = { 0.0f, 5.0f, -5.0f, 0.0f };
    float beta_base[4] = { 2.0f, 1.0f, 1.0f, 2.0f };
    float S[16], P[16], alphar[4], alphai[4], beta[4], VR[16], VL[16];

    float scales[2] = { 1e30f, 1e-30f };
    const char *scale_names[2] = { "Overflow Risk (1e30)", "Underflow Risk (1e-30)" };

    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (float *)malloc(work_size * sizeof(float));

    for (test_idx = 0; test_idx < 2; test_idx++) {
        fmt::print("--- Testing 4x4 Static Matrix (Scaling Float: {}) ---\n", scale_names[test_idx]);
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
        for (i = 0; i < 16; i++) {
            VR[i] = 0.0f;
            VL[i] = 0.0f;
        }
        for (i = 0; i < n; i++) {
            VR[i * n + i] = 1.0f;
            VL[i * n + i] = 1.0f;
        }

        stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, work_size, info);

        errR = check_right_eigenvector_residual_gevp_s(n, S, n, P, n, VR, n, alphar, alphai, beta);
        errL = check_left_eigenvector_residual_gevp_s(n, S, n, P, n, VL, n, alphar, alphai, beta);
        fmt::print("Max Right Backward Error: {:e}\nMax Left Backward Error: {:e}\n\n", errR, errL);
    }
    free(work);
}

void test_dynamic_size_s(int n)
{
    float *S, *P, *VR, *VL, *alphar, *alphai, *beta, *work, dummy[1];
    int info[1], lwork, i, j, m_out;
    float errR, errL;

    fmt::println("--- Testing {0}x{0} Dynamic Random Matrix (Float) ---", n);

    S = (float *)malloc(n * n * sizeof(float));
    P = (float *)malloc(n * n * sizeof(float));
    VR = (float *)malloc(n * n * sizeof(float));
    VL = (float *)malloc(n * n * sizeof(float));
    alphar = (float *)malloc(n * sizeof(float));
    alphai = (float *)malloc(n * sizeof(float));
    beta = (float *)malloc(n * sizeof(float));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = (i == j) ? 1.0f : 0.0f;
            VL[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    generate_generalized_quasi_triangular_s(n, S, n, P, n, alphar, alphai, beta);
    stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);

    lwork = static_cast<int>(*dummy);
    work = (float *)malloc(lwork * sizeof(float));

    stgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, lwork, info);

    errR = check_right_eigenvector_residual_gevp_s(n, S, n, P, n, VR, n, alphar, alphai, beta);
    errL = check_left_eigenvector_residual_gevp_s(n, S, n, P, n, VL, n, alphar, alphai, beta);

    fmt::print("Maximum Right Backward Error: {:e}\nMaximum Left Backward Error: {:e}\n\n", errR, errL);

    free(S);
    free(P);
    free(VR);
    free(VL);
    free(alphar);
    free(alphai);
    free(beta);
    free(work);
}

int main()
{
    test_stgevc();
    test_stgevc_scaling();
    test_dynamic_size_s(10);
    test_dynamic_size_s(500);

    return 0;
}