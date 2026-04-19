// test_ztgevc3.cpp
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <random>

#include "fmt/core.h"
#include "helpers.h"
#include "tgevc3.h"

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

double check_right_eigenvector_residual_gevp(int n, const std::complex<double> *S, int lds, const std::complex<double> *P, int ldp, const std::complex<double> *VR, int ldvr, const std::complex<double> *alpha, const std::complex<double> *beta)
{
    double normS = 0.0, normP = 0.0, max_err = 0.0, norm_r, norm_v;
    int c, r, i, col, row, incx = 1;
    std::complex<double> a, b;
    std::complex<double> *r_vec;

    for (c = 0; c < n; c++) {
        for (r = 0; r <= c; r++) {
            normS += std::norm(S[r + c * lds]);
            normP += std::norm(P[r + c * ldp]);
        }
    }

    normS = std::sqrt(normS);
    normP = std::sqrt(normP);
    if (normS == 0.0)
        normS = 1.0;
    if (normP == 0.0)
        normP = 1.0;

    r_vec = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));

    for (c = 0; c < n; c++) {
        a = alpha[c];
        b = beta[c];

        for (i = 0; i < n; i++)
            r_vec[i] = { 0.0, 0.0 };

        for (col = 0; col < n; col++) {
            for (row = 0; row <= col; row++) {
                r_vec[row] += b * S[row + col * lds] * VR[col + c * ldvr];
                r_vec[row] -= a * P[row + col * ldp] * VR[col + c * ldvr];
            }
        }

        norm_r = dznrm2_(&n, r_vec, &incx);
        norm_v = dznrm2_(&n, &VR[c * ldvr], &incx);

        max_err = std::max(max_err, norm_r / ((std::abs(b) * normS + std::abs(a) * normP) * norm_v));
    }

    free(r_vec);
    return max_err;
}

double check_left_eigenvector_residual_gevp(int n, const std::complex<double> *S, int lds, const std::complex<double> *P, int ldp, const std::complex<double> *VL, int ldvl, const std::complex<double> *alpha, const std::complex<double> *beta)
{
    double normS = 0.0, normP = 0.0, max_err = 0.0, norm_r, norm_v;
    int c, r, col, row, incx = 1;
    std::complex<double> a, b, sum;
    std::complex<double> *r_vec;

    for (c = 0; c < n; c++) {
        for (r = 0; r <= c; r++) {
            normS += std::norm(S[r + c * lds]);
            normP += std::norm(P[r + c * ldp]);
        }
    }

    normS = std::sqrt(normS);
    normP = std::sqrt(normP);
    if (normS == 0.0)
        normS = 1.0;
    if (normP == 0.0)
        normP = 1.0;

    r_vec = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));

    for (c = 0; c < n; c++) {
        a = alpha[c];
        b = beta[c];

        for (col = 0; col < n; col++) {
            sum = { 0.0, 0.0 };
            for (row = 0; row <= col; row++) {
                sum += std::conj(VL[row + c * ldvl]) * (b * S[row + col * lds] - a * P[row + col * ldp]);
            }
            r_vec[col] = sum;
        }

        norm_r = dznrm2_(&n, r_vec, &incx);
        norm_v = dznrm2_(&n, &VL[c * ldvl], &incx);

        max_err = std::max(max_err, norm_r / ((std::abs(b) * normS + std::abs(a) * normP) * norm_v));
    }

    free(r_vec);
    return max_err;
}

void generate_generalized_upper_triangular(int n, std::complex<double> *S, int lds, std::complex<double> *P, int ldp, std::complex<double> *alpha, std::complex<double> *beta)
{
    int i, c, r;

    for (i = 0; i < n * n; i++) {
        S[i] = { 0.0, 0.0 };
        P[i] = { 0.0, 0.0 };
    }

    for (c = 0; c < n; c++) {
        for (r = 0; r <= c; r++) {
            S[r + c * lds] = std::complex<double>(dist(gen), dist(gen));
            P[r + c * ldp] = std::complex<double>(dist(gen), dist(gen));
        }
        P[c + c * ldp] += std::complex<double>((P[c + c * ldp].real() >= 0 ? 1.0 : -1.0), 0.0);
        alpha[c] = S[c + c * lds];
        beta[c] = P[c + c * ldp];
    }
}

void test_ztgevc_static()
{
    int n = 4, info[1], max_bsize, work_size, m_out, i;
    double errR, errL;
    std::complex<double> *work;

    std::complex<double> S[16] = { { 0, 0 } };
    std::complex<double> P[16] = { { 0, 0 } };
    std::complex<double> alpha[4], beta[4];
    std::complex<double> VR[16] = { { 0, 0 } };
    std::complex<double> VL[16] = { { 0, 0 } };

    S[0 + 0 * 4] = { 1.0, 1.0 };
    P[0 + 0 * 4] = { 2.0, 0.0 };
    S[0 + 1 * 4] = { 2.0, 0.0 };
    P[0 + 1 * 4] = { 1.0, 1.0 };
    S[1 + 1 * 4] = { 2.0, -1.0 };
    P[1 + 1 * 4] = { 1.0, -1.0 };
    S[0 + 2 * 4] = { -5.0, 1.0 };
    P[0 + 2 * 4] = { 0.0, 0.0 };
    S[1 + 2 * 4] = { 0.0, 2.0 };
    P[1 + 2 * 4] = { -1.0, 0.0 };
    S[2 + 2 * 4] = { 3.0, 0.0 };
    P[2 + 2 * 4] = { 1.0, 1.0 };
    S[0 + 3 * 4] = { 5.0, 0.0 };
    P[0 + 3 * 4] = { 0.0, 0.0 };
    S[1 + 3 * 4] = { 2.0, 1.0 };
    P[1 + 3 * 4] = { 0.0, 2.0 };
    S[2 + 3 * 4] = { 4.0, -1.0 };
    P[2 + 3 * 4] = { 3.0, 0.0 };
    S[3 + 3 * 4] = { 3.0, 2.0 };
    P[3 + 3 * 4] = { 2.0, -1.0 };

    for (i = 0; i < n; i++) {
        alpha[i] = S[i + i * n];
        beta[i] = P[i + i * n];
        VR[i * n + i] = { 1.0, 0.0 };
        VL[i * n + i] = { 1.0, 0.0 };
    }

    fmt::print("--- Testing 4x4 Complex Static Matrix ---\n");

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (std::complex<double> *)malloc(work_size * sizeof(std::complex<double>));

    ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, work_size, info);

    errR = check_right_eigenvector_residual_gevp(n, S, n, P, n, VR, n, alpha, beta);
    errL = check_left_eigenvector_residual_gevp(n, S, n, P, n, VL, n, alpha, beta);

    fmt::print("Max Right Backward Error: {:e}\n", errR);
    fmt::print("Max Left Backward Error:  {:e}\n\n", errL);

    free(work);
}

void test_ztgevc_dynamic_size(int n)
{
    int info[1], m_out, lwork, i, j;
    double errR, errL;
    std::complex<double> *S, *P, *VR, *VL, *alpha, *beta, *work;
    std::complex<double> dummy[1];

    fmt::println("--- Testing {0}x{0} Dynamic Complex Matrix ---", n);

    S = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    P = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VR = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    VL = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
    alpha = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));
    beta = (std::complex<double> *)malloc(n * sizeof(std::complex<double>));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = (i == j) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
            VL[i * n + j] = (i == j) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
        }
    }

    generate_generalized_upper_triangular(n, S, n, P, n, alpha, beta);

    ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);

    lwork = static_cast<int>(dummy[0].real());
    fmt::println("lwork = {}", lwork);

    work = (std::complex<double> *)malloc(lwork * sizeof(std::complex<double>));
    time_code("ztgevc3 ('B')", [&]() { ztgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, lwork, info); });

    errR = check_right_eigenvector_residual_gevp(n, S, n, P, n, VR, n, alpha, beta);
    errL = check_left_eigenvector_residual_gevp(n, S, n, P, n, VL, n, alpha, beta);

    fmt::print("Maximum Right Backward Error: {:e}\n", errR);
    fmt::print("Maximum Left Backward Error:  {:e}\n\n", errL);

    free(S);
    free(P);
    free(VR);
    free(VL);
    free(alpha);
    free(beta);
    free(work);
}

int main()
{
    test_ztgevc_static();
    test_ztgevc_dynamic_size(10);
    test_ztgevc_dynamic_size(500);

    return 0;
}