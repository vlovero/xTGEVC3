// test_ctgevc3.cpp
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <random>

#include "fmt/core.h"
#include "helpers.h"
#include "tgevc3.h"

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

float check_right_eigenvector_residual_gevp(int n, const std::complex<float> *S, int lds, const std::complex<float> *P, int ldp, const std::complex<float> *VR, int ldvr, const std::complex<float> *alpha, const std::complex<float> *beta)
{
    float normS = 0.0f, normP = 0.0f, max_err = 0.0f, norm_r, norm_v;
    int c, r, i, col, row, incx = 1;
    std::complex<float> a, b;
    std::complex<float> *r_vec;

    for (c = 0; c < n; c++) {
        for (r = 0; r <= c; r++) {
            normS += std::norm(S[r + c * lds]);
            normP += std::norm(P[r + c * ldp]);
        }
    }

    normS = std::sqrt(normS);
    normP = std::sqrt(normP);
    if (normS == 0.0f) {
        normS = 1.0f;
    }
    if (normP == 0.0f) {
        normP = 1.0f;
    }

    r_vec = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));

    for (c = 0; c < n; c++) {
        a = alpha[c];
        b = beta[c];

        for (i = 0; i < n; i++) {
            r_vec[i] = { 0.0f, 0.0f };
        }

        for (col = 0; col < n; col++) {
            for (row = 0; row <= col; row++) {
                r_vec[row] += b * S[row + col * lds] * VR[col + c * ldvr];
                r_vec[row] -= a * P[row + col * ldp] * VR[col + c * ldvr];
            }
        }

        norm_r = scnrm2_(&n, r_vec, &incx);
        norm_v = scnrm2_(&n, &VR[c * ldvr], &incx);

        max_err = std::max(max_err, norm_r / ((std::abs(b) * normS + std::abs(a) * normP) * norm_v));
    }

    free(r_vec);
    return max_err;
}

float check_left_eigenvector_residual_gevp(int n, const std::complex<float> *S, int lds, const std::complex<float> *P, int ldp, const std::complex<float> *VL, int ldvl, const std::complex<float> *alpha, const std::complex<float> *beta)
{
    float normS = 0.0f, normP = 0.0f, max_err = 0.0f, norm_r, norm_v;
    int c, r, col, row, incx = 1;
    std::complex<float> a, b, sum;
    std::complex<float> *r_vec;

    for (c = 0; c < n; c++) {
        for (r = 0; r <= c; r++) {
            normS += std::norm(S[r + c * lds]);
            normP += std::norm(P[r + c * ldp]);
        }
    }

    normS = std::sqrt(normS);
    normP = std::sqrt(normP);
    if (normS == 0.0f) {
        normS = 1.0f;
    }
    if (normP == 0.0f) {
        normP = 1.0f;
    }

    r_vec = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));

    for (c = 0; c < n; c++) {
        a = alpha[c];
        b = beta[c];

        for (col = 0; col < n; col++) {
            sum = { 0.0f, 0.0f };
            for (row = 0; row <= col; row++) {
                sum += std::conj(VL[row + c * ldvl]) * (b * S[row + col * lds] - a * P[row + col * ldp]);
            }
            r_vec[col] = sum;
        }

        norm_r = scnrm2_(&n, r_vec, &incx);
        norm_v = scnrm2_(&n, &VL[c * ldvl], &incx);

        max_err = std::max(max_err, norm_r / ((std::abs(b) * normS + std::abs(a) * normP) * norm_v));
    }

    free(r_vec);
    return max_err;
}

void generate_generalized_upper_triangular(int n, std::complex<float> *S, int lds, std::complex<float> *P, int ldp, std::complex<float> *alpha, std::complex<float> *beta)
{
    int i, c, r;

    for (i = 0; i < n * n; i++) {
        S[i] = { 0.0f, 0.0f };
        P[i] = { 0.0f, 0.0f };
    }

    for (c = 0; c < n; c++) {
        for (r = 0; r <= c; r++) {
            S[r + c * lds] = std::complex<float>(dist(gen), dist(gen));
            P[r + c * ldp] = std::complex<float>(dist(gen), dist(gen));
        }
        P[c + c * ldp] += std::complex<float>((P[c + c * ldp].real() >= 0 ? 1.0f : -1.0f), 0.0f);
        alpha[c] = S[c + c * lds];
        beta[c] = P[c + c * ldp];
    }
}

void test_ctgevc_static()
{
    int n = 4, info[1], max_bsize, work_size, m_out, i;
    float errR, errL;
    std::complex<float> *work;

    std::complex<float> S[16] = { { 0, 0 } };
    std::complex<float> P[16] = { { 0, 0 } };
    std::complex<float> alpha[4], beta[4];
    std::complex<float> VR[16] = { { 0, 0 } };
    std::complex<float> VL[16] = { { 0, 0 } };

    // Col 0
    S[0 + 0 * 4] = { 1.0f, 1.0f };
    P[0 + 0 * 4] = { 2.0f, 0.0f };
    // Col 1
    S[0 + 1 * 4] = { 2.0f, 0.0f };
    P[0 + 1 * 4] = { 1.0f, 1.0f };
    S[1 + 1 * 4] = { 2.0f, -1.0f };
    P[1 + 1 * 4] = { 1.0f, -1.0f };
    // Col 2
    S[0 + 2 * 4] = { -5.0f, 1.0f };
    P[0 + 2 * 4] = { 0.0f, 0.0f };
    S[1 + 2 * 4] = { 0.0f, 2.0f };
    P[1 + 2 * 4] = { -1.0f, 0.0f };
    S[2 + 2 * 4] = { 3.0f, 0.0f };
    P[2 + 2 * 4] = { 1.0f, 1.0f };
    // Col 3
    S[0 + 3 * 4] = { 5.0f, 0.0f };
    P[0 + 3 * 4] = { 0.0f, 0.0f };
    S[1 + 3 * 4] = { 2.0f, 1.0f };
    P[1 + 3 * 4] = { 0.0f, 2.0f };
    S[2 + 3 * 4] = { 4.0f, -1.0f };
    P[2 + 3 * 4] = { 3.0f, 0.0f };
    S[3 + 3 * 4] = { 3.0f, 2.0f };
    P[3 + 3 * 4] = { 2.0f, -1.0f };

    for (i = 0; i < n; i++) {
        alpha[i] = S[i + i * n];
        beta[i] = P[i + i * n];
        VR[i * n + i] = { 1.0f, 0.0f };
        VL[i * n + i] = { 1.0f, 0.0f };
    }

    fmt::print("--- Testing 4x4 Complex Static Matrix ---\n");

    max_bsize = 32;
    work_size = 2 * n * (max_bsize + 1) + 4 * (max_bsize + 1) * (max_bsize + 1) + 2 * (max_bsize + 1);
    work = (std::complex<float> *)malloc(work_size * sizeof(std::complex<float>));

    ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, work_size, info);

    errR = check_right_eigenvector_residual_gevp(n, S, n, P, n, VR, n, alpha, beta);
    errL = check_left_eigenvector_residual_gevp(n, S, n, P, n, VL, n, alpha, beta);

    fmt::print("Max Right Backward Error: {:e}\n", errR);
    fmt::print("Max Left Backward Error:  {:e}\n\n", errL);

    free(work);
}

void test_ctgevc_dynamic_size(int n)
{
    int info[1], m_out, lwork, i, j;
    float errR, errL;
    std::complex<float> *S, *P, *VR, *VL, *alpha, *beta, *work;
    std::complex<float> dummy[1];

    fmt::println("--- Testing {0}x{0} Dynamic Complex Matrix ---", n);

    S = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    P = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VR = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    VL = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
    alpha = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));
    beta = (std::complex<float> *)malloc(n * sizeof(std::complex<float>));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = (i == j) ? std::complex<float>(1.0f, 0.0f) : std::complex<float>(0.0f, 0.0f);
            VL[i * n + j] = (i == j) ? std::complex<float>(1.0f, 0.0f) : std::complex<float>(0.0f, 0.0f);
        }
    }

    generate_generalized_upper_triangular(n, S, n, P, n, alpha, beta);

    ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, dummy, -1, info);

    lwork = static_cast<int>(dummy[0].real());
    fmt::println("lwork = {}", lwork);

    work = (std::complex<float> *)malloc(lwork * sizeof(std::complex<float>));
    time_code("ctgevc3 ('B')", [&]() { ctgevc3('B', 'B', nullptr, n, S, n, P, n, alpha, beta, VL, n, VR, n, n, &m_out, work, lwork, info); });

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
    test_ctgevc_static();
    test_ctgevc_dynamic_size(10);
    test_ctgevc_dynamic_size(500);

    return 0;
}