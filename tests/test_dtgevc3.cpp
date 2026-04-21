// test_dtgevc3.cpp
#include <algorithm>
#include <cstdlib>
#include <random>

#include "fmt/core.h"
#include "helpers.h"
#include "tgevc3.h"

static std::mt19937 gen(0);
static std::uniform_real_distribution<double> dist(-1.0, 1.0);
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

double check_right_eigenvector_residual_gevp(int n, const double *S, int lds, const double *P, int ldp, const double *VR, int ldvr, const double *alphar, const double *alphai, const double *beta)
{
    double normS, normP, max_err, a, b, alpha_term, beta_term, alpha_zero, norm_r, norm_v, aR, aI, s_val, p_val, vr_r, vr_i, norm_rr, norm_ri, den;
    int c, r, i, col, row, incx, incy;
    double *r_vec, *rr, *ri;

    normS = 0.0;
    normP = 0.0;
    for (c = 0; c < n; c++) {
        for (r = 0; r <= std::min(c + 1, n - 1); r++) {
            normS += S[r + c * lds] * S[r + c * lds];
        }
        for (r = 0; r <= c; r++) {
            normP += P[r + c * ldp] * P[r + c * ldp];
        }
    }

    normS = std::sqrt(normS);
    normP = std::sqrt(normP);
    if (normS == 0.0) {
        normS = 1.0;
    }
    if (normP == 0.0) {
        normP = 1.0;
    }

    max_err = 0.0;
    r_vec = (double *)malloc(n * sizeof(double));
    incx = 1;
    incy = 1;

    c = 0;
    while (c < n) {
        if (alphai[c] == 0.0) {
            a = alphar[c];
            b = beta[c];
            alpha_term = -a;
            beta_term = b;
            alpha_zero = 0.0;

            for (i = 0; i < n; i++) {
                r_vec[i] = 0.0;
            }
            dgemv_("N", &n, &n, &beta_term, S, &lds, &VR[c * ldvr], &incx, &alpha_zero, r_vec, &incy);

            beta_term = 1.0;
            dgemv_("N", &n, &n, &alpha_term, P, &ldp, &VR[c * ldvr], &incx, &beta_term, r_vec, &incy);

            for (i = 0; i < n; i++) {
                r_vec[i] = 0.0;
            }
            for (col = 0; col < n; col++) {
                for (row = 0; row <= std::min(col + 1, n - 1); row++) {
                    r_vec[row] += b * S[row + col * lds] * VR[col + c * ldvr];
                    r_vec[row] -= a * P[row + col * ldp] * VR[col + c * ldvr];
                }
            }

            norm_r = dnrm2_(&n, r_vec, &incx);
            norm_v = dnrm2_(&n, &VR[c * ldvr], &incx);

            max_err = std::max(max_err, norm_r / ((std::abs(b) * normS + std::abs(a) * normP) * norm_v));
            c++;
        }
        else {
            rr = (double *)malloc(n * sizeof(double));
            ri = (double *)malloc(n * sizeof(double));
            aR = alphar[c];
            aI = alphai[c];
            b = beta[c];

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

            norm_rr = dnrm2_(&n, rr, &incx);
            norm_ri = dnrm2_(&n, ri, &incx);
            norm_v = std::sqrt(std::pow(dnrm2_(&n, &VR[c * ldvr], &incx), 2) + std::pow(dnrm2_(&n, &VR[(c + 1) * ldvr], &incx), 2));

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

double check_left_eigenvector_residual_gevp(int n, const double *S, int lds, const double *P, int ldp, const double *VL, int ldvl, const double *alphar, const double *alphai, const double *beta)
{
    double normS, normP, max_err, a, b, norm_r, norm_v, aR, aI, s_val, p_val, vl_r, vl_i, norm_rr, norm_ri, den;
    int c, r, i, col, row, incx;
    double *r_vec, *rr, *ri;

    normS = 0.0;
    normP = 0.0;
    for (c = 0; c < n; c++) {
        for (r = 0; r <= std::min(c + 1, n - 1); r++) {
            normS += S[r + c * lds] * S[r + c * lds];
        }
        for (r = 0; r <= c; r++) {
            normP += P[r + c * ldp] * P[r + c * ldp];
        }
    }

    normS = std::sqrt(normS);
    normP = std::sqrt(normP);
    if (normS == 0.0) {
        normS = 1.0;
    }
    if (normP == 0.0) {
        normP = 1.0;
    }

    max_err = 0.0;
    r_vec = (double *)malloc(n * sizeof(double));
    incx = 1;

    c = 0;
    while (c < n) {
        if (alphai[c] == 0.0) {
            a = alphar[c];
            b = beta[c];
            for (i = 0; i < n; i++) {
                r_vec[i] = 0.0;
            }

            for (col = 0; col < n; col++) {
                for (row = 0; row <= std::min(col + 1, n - 1); row++) {
                    r_vec[col] += b * S[row + col * lds] * VL[row + c * ldvl];
                    r_vec[col] -= a * P[row + col * ldp] * VL[row + c * ldvl];
                }
            }

            norm_r = dnrm2_(&n, r_vec, &incx);
            norm_v = dnrm2_(&n, &VL[c * ldvl], &incx);

            max_err = std::max(max_err, norm_r / ((std::abs(b) * normS + std::abs(a) * normP) * norm_v));
            c++;
        }
        else {
            rr = (double *)malloc(n * sizeof(double));
            ri = (double *)malloc(n * sizeof(double));
            aR = alphar[c];
            aI = alphai[c];
            b = beta[c];

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

            norm_rr = dnrm2_(&n, rr, &incx);
            norm_ri = dnrm2_(&n, ri, &incx);
            norm_v = std::sqrt(std::pow(dnrm2_(&n, &VL[c * ldvl], &incx), 2) + std::pow(dnrm2_(&n, &VL[(c + 1) * ldvl], &incx), 2));

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

void generate_quasi_triangular(int n, double *T, int ldt, double *wr, double *wi)
{
    int i, c, r, k;
    double a, b;

    for (i = 0; i < n * n; ++i) {
        T[i] = 0.0;
    }

    for (c = 0; c < n; ++c) {
        for (r = 0; r <= c; ++r) {
            T[r + c * ldt] = dist(gen);
        }
    }

    k = 0;
    while (k < n) {
        if (k < n - 1 && dist_int(gen) < 40) {
            a = T[k + k * ldt];
            T[(k + 1) + (k + 1) * ldt] = a;

            b = T[k + (k + 1) * ldt];
            if (b == 0.0) {
                b = 1.0;
            }
            T[(k + 1) + k * ldt] = -b;

            wr[k] = a;
            wr[k + 1] = a;
            wi[k] = std::abs(b);
            wi[k + 1] = -std::abs(b);
            k += 2;
        }
        else {
            wr[k] = T[k + k * ldt];
            wi[k] = 0.0;
            k += 1;
        }
    }
}

void generate_generalized_quasi_triangular(int n, double *S, int lds, double *P, int ldp, double *alphar, double *alphai, double *beta)
{
    int i, c, r, k;
    double a, b;

    for (i = 0; i < n * n; i++) {
        S[i] = 0.0;
        P[i] = 0.0;
    }

    for (c = 0; c < n; c++) {
        for (r = 0; r <= c; r++) {
            S[r + c * lds] = dist(gen);
            P[r + c * ldp] = dist(gen);
        }
        P[c + c * ldp] += (P[c + c * ldp] >= 0 ? 1.0 : -1.0);
    }

    k = 0;
    while (k < n) {
        if (k < n - 1 && dist_int(gen) < 40) {
            P[k + k * ldp] = 1.0;
            P[k + (k + 1) * ldp] = 0.0;
            P[k + 1 + k * ldp] = 0.0;
            P[k + 1 + (k + 1) * ldp] = 1.0;

            a = S[k + k * lds];
            S[(k + 1) + (k + 1) * lds] = a;

            b = S[k + (k + 1) * lds];
            if (b == 0.0) {
                b = 1.0;
            }
            S[(k + 1) + k * lds] = -b;

            alphar[k] = a;
            alphar[k + 1] = a;
            alphai[k] = std::abs(b);
            alphai[k + 1] = -std::abs(b);
            beta[k] = 1.0;
            beta[k + 1] = 1.0;
            k += 2;
        }
        else {
            alphar[k] = S[k + k * lds];
            alphai[k] = 0.0;
            beta[k] = P[k + k * ldp];
            k += 1;
        }
    }
}

void test_dtgevc()
{
    int n = 4, info[1], max_bsize, work_size, i, m_out;
    double errR, errL, *work;
    double S[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    double P[16] = { 2, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3, 2, -1, 2 };
    double alphar[4] = { 1.0, 2.0, 2.0, 3.0 };
    double alphai[4] = { 0.0, 5.0, -5.0, 0.0 };
    double beta[4] = { 2.0, 1.0, 1.0, 2.0 };
    double VR[16] = { 0 }, VL[16] = { 0 };

    fmt::print("--- Testing 4x4 Static Matrix (Both Left/Right GEVP Evecs) ---\n");

    for (i = 0; i < n; i++) {
        VR[i * n + i] = 1.0;
        VL[i * n + i] = 1.0;
    }

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (double *)malloc(work_size * sizeof(double));

    dtgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, work_size, info);

    errR = check_right_eigenvector_residual_gevp(n, S, n, P, n, VR, n, alphar, alphai, beta);
    errL = check_left_eigenvector_residual_gevp(n, S, n, P, n, VL, n, alphar, alphai, beta);

    fmt::print("Max Right Backward Error: {:e}\n", errR);
    fmt::print("Max Left Backward Error: {:e}\n\n", errL);

    free(work);
}

void test_dtgevc_infinite()
{
    int n = 4, info[1], max_bsize, work_size, i, m_out;
    double errR, errL, *work;
    double S[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    double P[16] = { 2, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3, 2, -1, 0 };
    double alphar[4] = { 1.0, 2.0, 2.0, 3.0 };
    double alphai[4] = { 0.0, 5.0, -5.0, 0.0 };
    double beta[4] = { 2.0, 1.0, 1.0, 0.0 };
    double VR[16] = { 0 }, VL[16] = { 0 };

    fmt::print("--- Testing 4x4 Static Matrix (Infinite Eigenvalue Case) ---\n");

    for (i = 0; i < n; i++) {
        VR[i * n + i] = 1.0;
        VL[i * n + i] = 1.0;
    }

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (double *)malloc(work_size * sizeof(double));

    dtgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, work_size, info);

    errR = check_right_eigenvector_residual_gevp(n, S, n, P, n, VR, n, alphar, alphai, beta);
    errL = check_left_eigenvector_residual_gevp(n, S, n, P, n, VL, n, alphar, alphai, beta);

    fmt::print("Max Right Backward Error: {:e}\n", errR);
    fmt::print("Max Left Backward Error: {:e}\n\n", errL);

    free(work);
}

void test_dtgevc_scaling()
{
    int n = 4, info[1], max_bsize, work_size, i, test_idx, m_out;
    double errR, errL, s, *work;
    double S_base[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    double P_base[16] = { 2, 0, 0, 0, 1, 1, 0, 0, -1, 0, 1, 0, 3, 2, -1, 2 };
    double alphar_base[4] = { 1.0, 2.0, 2.0, 3.0 };
    double alphai_base[4] = { 0.0, 5.0, -5.0, 0.0 };
    double beta_base[4] = { 2.0, 1.0, 1.0, 2.0 };
    double S[16], P[16], alphar[4], alphai[4], beta[4], VR[16], VL[16];
    double scales[2] = { 1e150, 1e-150 };
    const char *scale_names[2] = { "Overflow Risk (1e150)", "Underflow Risk (1e-150)" };

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (double *)malloc(work_size * sizeof(double));

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
        for (i = 0; i < 16; i++) {
            VR[i] = 0.0;
            VL[i] = 0.0;
        }
        for (i = 0; i < n; i++) {
            VR[i * n + i] = 1.0;
            VL[i * n + i] = 1.0;
        }

        dtgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, work_size, info);

        errR = check_right_eigenvector_residual_gevp(n, S, n, P, n, VR, n, alphar, alphai, beta);
        errL = check_left_eigenvector_residual_gevp(n, S, n, P, n, VL, n, alphar, alphai, beta);

        fmt::print("Max Right Backward Error: {:e}\n", errR);
        fmt::print("Max Left Backward Error: {:e}\n\n", errL);
    }

    free(work);
}

void test_dynamic_size(int n)
{
    double *S, *P, *VR, *VL, *alphar, *alphai, *beta, *work, dummy[1], errR, errL;
    int info[1], lwork, i, j, m_out;

    fmt::println("--- Testing {0}x{0} Dynamic Random Matrix ---", n);

    S = (double *)malloc(n * n * sizeof(double));
    P = (double *)malloc(n * n * sizeof(double));
    VR = (double *)malloc(n * n * sizeof(double));
    VL = (double *)malloc(n * n * sizeof(double));
    alphar = (double *)malloc(n * sizeof(double));
    alphai = (double *)malloc(n * sizeof(double));
    beta = (double *)malloc(n * sizeof(double));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = (i == j) ? 1.0 : 0.0;
            VL[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    generate_generalized_quasi_triangular(n, S, n, P, n, alphar, alphai, beta);
    dtgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);

    lwork = static_cast<int>(*dummy);
    fmt::println("lwork = {}", lwork);

    work = (double *)malloc(lwork * sizeof(double));
    time_code("dtgevc_cpp ('B')", [&]() { dtgevc3('B', 'B', nullptr, n, S, n, P, n, alphar, alphai, beta, VL, n, VR, n, n, &m_out, work, lwork, info); });

    errR = check_right_eigenvector_residual_gevp(n, S, n, P, n, VR, n, alphar, alphai, beta);
    errL = check_left_eigenvector_residual_gevp(n, S, n, P, n, VL, n, alphar, alphai, beta);

    fmt::print("Maximum Right Backward Error: {:e}\n", errR);
    fmt::print("Maximum Left Backward Error: {:e}\n\n", errL);

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
    test_dtgevc();
    test_dtgevc_infinite();
    test_dtgevc_scaling();
    test_dynamic_size(10);
    test_dynamic_size(500);

    return 0;
}