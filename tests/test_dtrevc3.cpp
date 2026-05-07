// test_dtrevc3.cpp
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

extern "C" {
    void dtrevc_(const char *side, const char *howmny, const int *select, const int *n, double *t, const int *ldt, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, int *info);
}

extern void dtrevc3(char side, char howmny, const int *select, int n, const double *T, int ldt, const double *wr, const double *wi, double *VL, int ldvl, double *VR, int ldvr, int mm, int *m, double *work, int lwork, int *info);

static std::mt19937 gen(0);
static std::uniform_real_distribution<double> dist(-1.0, 1.0);
static std::uniform_int_distribution<int> dist_int(0, 99);

template <int ndigit = 3>
void print_mat(const char *name, const double *A, const ptrdiff_t ldA, const ptrdiff_t n, const ptrdiff_t m, const bool forder = true)
{
    ptrdiff_t i, j;

    if (name != nullptr) {
        fmt::println("{} = ", name);
    }
    if (forder) {
        for (j = 0; j < m; j++) {
            for (i = 0; i < n; i++) {
                fmt::println("\x1B[{}C{: .3e}", j * (9 + 3), A[i + j * ldA]);
            }
            if (j != (m - 1)) {
                for (i = 0; i < n; i++) {
                    fmt::print("\x1b[1A");
                }
            }
        }
    }
    else {
        for (i = 0; i < n; i++) {
            fmt::println("{: .{}e}", fmt::join(&A[i * ldA], &A[i * ldA + m], " "), ndigit);
        }
    }
}

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

// Replicates the LAPACK test routine dget22.
// Evaluates the scaled backward error for left and right standard eigenvectors.
// result[0] = max right error ratio
// result[1] = max left error ratio
void dget22(bool compute_left, bool compute_right, int n, const double *T, int ldt, const double *wr, const double *wi, const double *VL, int ldvl, const double *VR, int ldvr, double *result)
{
    double t_norm, local_max_res, v_norm;
    double dot, diff, expected_r, expected_i;
    double local_max_res_r, local_max_res_i, dot_r, dot_i;
    double den, ulp, safmin, max_err_r, max_err_l;
    int c, r, i;

    ulp = std::numeric_limits<double>::epsilon();
    safmin = std::numeric_limits<double>::min();

    t_norm = 0.0;
    for (c = 0; c < n; ++c) {
        dot = 0.0;
        for (r = 0; r <= std::min(n - 1, c + 1); ++r) {
            dot += std::abs(T[r + c * ldt]);
        }
        t_norm = std::max(t_norm, dot);
    }
    t_norm = std::max(t_norm, safmin);

    max_err_r = 0.0;
    max_err_l = 0.0;

    // Right Eigenvector Error
    if (compute_right) {
        i = 0;
        while (i < n) {
            if (wi[i] == 0.0) {
                local_max_res = 0.0;
                v_norm = 0.0;
                for (r = 0; r < n; ++r) {
                    dot = 0.0;
                    for (c = (r > 0 ? r - 1 : 0); c < n; ++c) {
                        dot += T[r + c * ldt] * VR[c + i * ldvr];
                    }
                    diff = std::abs(dot - wr[i] * VR[r + i * ldvr]);
                    local_max_res = std::max(local_max_res, diff);
                    v_norm = std::max(v_norm, std::abs(VR[r + i * ldvr]));
                }
                den = t_norm * std::max(v_norm, safmin);
                max_err_r = std::max(max_err_r, local_max_res / (den * ulp));
                i += 1;
            }
            else {
                local_max_res_r = 0.0;
                local_max_res_i = 0.0;
                v_norm = 0.0;
                for (r = 0; r < n; ++r) {
                    dot_r = 0.0;
                    dot_i = 0.0;
                    for (c = (r > 0 ? r - 1 : 0); c < n; ++c) {
                        dot_r += T[r + c * ldt] * VR[c + i * ldvr];
                        dot_i += T[r + c * ldt] * VR[c + (i + 1) * ldvr];
                    }
                    expected_r = wr[i] * VR[r + i * ldvr] - wi[i] * VR[r + (i + 1) * ldvr];
                    expected_i = wr[i] * VR[r + (i + 1) * ldvr] + wi[i] * VR[r + i * ldvr];

                    local_max_res_r = std::max(local_max_res_r, std::abs(dot_r - expected_r));
                    local_max_res_i = std::max(local_max_res_i, std::abs(dot_i - expected_i));
                    v_norm = std::max({ v_norm, std::abs(VR[r + i * ldvr]), std::abs(VR[r + (i + 1) * ldvr]) });
                }
                den = t_norm * std::max(v_norm, safmin);
                max_err_r = std::max(max_err_r, std::max(local_max_res_r, local_max_res_i) / (den * ulp));
                i += 2;
            }
        }
    }

    // Left Eigenvector Error
    if (compute_left) {
        i = 0;
        while (i < n) {
            if (wi[i] == 0.0) {
                local_max_res = 0.0;
                v_norm = 0.0;
                for (r = 0; r < n; ++r) {
                    dot = 0.0;
                    for (c = 0; c <= std::min(n - 1, r + 1); ++c) {
                        dot += T[c + r * ldt] * VL[c + i * ldvl];
                    }
                    diff = std::abs(dot - wr[i] * VL[r + i * ldvl]);
                    local_max_res = std::max(local_max_res, diff);
                    v_norm = std::max(v_norm, std::abs(VL[r + i * ldvl]));
                }
                den = t_norm * std::max(v_norm, safmin);
                max_err_l = std::max(max_err_l, local_max_res / (den * ulp));
                i += 1;
            }
            else {
                local_max_res_r = 0.0;
                local_max_res_i = 0.0;
                v_norm = 0.0;
                for (r = 0; r < n; ++r) {
                    dot_r = 0.0;
                    dot_i = 0.0;
                    for (c = 0; c <= std::min(n - 1, r + 1); ++c) {
                        dot_r += T[c + r * ldt] * VL[c + i * ldvl];
                        dot_i += T[c + r * ldt] * VL[c + (i + 1) * ldvl];
                    }
                    expected_r = wr[i] * VL[r + i * ldvl] + wi[i] * VL[r + (i + 1) * ldvl];
                    expected_i = wr[i] * VL[r + (i + 1) * ldvl] - wi[i] * VL[r + i * ldvl];

                    local_max_res_r = std::max(local_max_res_r, std::abs(dot_r - expected_r));
                    local_max_res_i = std::max(local_max_res_i, std::abs(dot_i - expected_i));
                    v_norm = std::max({ v_norm, std::abs(VL[r + i * ldvl]), std::abs(VL[r + (i + 1) * ldvl]) });
                }
                den = t_norm * std::max(v_norm, safmin);
                max_err_l = std::max(max_err_l, std::max(local_max_res_r, local_max_res_i) / (den * ulp));
                i += 2;
            }
        }
    }

    result[0] = max_err_r;
    result[1] = max_err_l;
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

void generate_lapack_matrix_type(int type, int n, double *T, int ldt, double *wr, double *wi)
{
    int i, j, k;
    double big, small, a, b;

    big = 1e100;
    small = 1e-100;

    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            T[i + j * ldt] = 0.0;
        }
        wi[j] = 0.0;
    }

    switch (type) {
    case 1:
        break;
    case 2:
        for (i = 0; i < n; i++) {
            T[i + i * ldt] = 1.0;
        }
        break;
    case 3:
        for (i = 0; i < n; i++) {
            T[i + i * ldt] = 1.0;
            if (i < n - 1) {
                T[i + (i + 1) * ldt] = 1.0;
            }
        }
        break;
    case 4:
        for (i = 0; i < n; i++) {
            T[i + i * ldt] = (double)(i + 1) / n;
        }
        break;
    case 5:
        for (i = 0; i < n; i++) {
            T[i + i * ldt] = big * (i + 1) / n;
        }
        break;
    case 6:
        for (i = 0; i < n; i++) {
            T[i + i * ldt] = small * (i + 1) / n;
        }
        break;
    case 7:
        for (i = 0; i < n; i++) {
            if (i == 0 || i == 1 || i == n - 1) {
                T[i + i * ldt] = 0.0;
            }
            else {
                T[i + i * ldt] = (double)(i - 1);
            }
        }
        break;
    default:
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                T[i + j * ldt] = dist(gen);
            }
        }

        if (type % 2 == 0) {
            for (i = 0; i < n - 1; i += 2) {
                if (dist_int(gen) > 50) {
                    a = dist(gen);
                    b = dist(gen);
                    if (b == 0.0) {
                        b = 1.0;
                    }
                    T[i + i * ldt] = a;
                    T[i + 1 + (i + 1) * ldt] = a;
                    T[i + (i + 1) * ldt] = b;
                    T[i + 1 + i * ldt] = -b;
                }
            }
        }
        break;
    }

    k = 0;
    while (k < n) {
        if (k < n - 1 && T[k + 1 + k * ldt] != 0.0) {
            wr[k] = T[k + k * ldt];
            wr[k + 1] = T[k + 1 + (k + 1) * ldt];
            wi[k] = std::abs(T[k + (k + 1) * ldt]);
            wi[k + 1] = -wi[k];
            k += 2;
        }
        else {
            wr[k] = T[k + k * ldt];
            wi[k] = 0.0;
            k++;
        }
    }
}

void test_dtrevc()
{
    int n = 4, info[1], m_out, i, j;
    double *work_lapack, res3[2], resL[2];
    double T[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    double wr[4] = { 1.0, 2.0, 2.0, 3.0 };
    double wi[4] = { 0.0, 5.0, -5.0, 0.0 };
    double VR[16], VL[16];

    fmt::print("--- Testing 4x4 Static Matrix (Both Left/Right Evecs) ---\n");

    std::vector<double> work(1);
    work_lapack = (double *)malloc(3 * n * sizeof(double));

    // Test dtrevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0;
            VL[i * n + j] = 0.0;
        }
        VR[i * n + i] = 1.0;
        VL[i * n + i] = 1.0;
    }

    // Using backtransformation B by populating identity matrices VL/VR initially
    // print_mat("T", T, n, n, n);
    dtrevc3('B', 'B', nullptr, n, T, n, wr, wi, VL, n, VR, n, n, &m_out, work.data(), -1, info);
    work.resize(work[0]);
    dtrevc3('B', 'B', nullptr, n, T, n, wr, wi, VL, n, VR, n, n, &m_out, work.data(), work.size(), info);
    dget22(true, true, n, T, n, wr, wi, VL, n, VR, n, res3);
    // print_mat("VL", VL, n, n, n);
    // print_mat("VR", VR, n, n, n);

    // Test LAPACK dtrevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0;
            VL[i * n + j] = 0.0;
        }
        VR[i * n + i] = 1.0;
        VL[i * n + i] = 1.0;
    }
    dtrevc_("B", "B", nullptr, &n, T, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info);
    dget22(true, true, n, T, n, wr, wi, VL, n, VR, n, resL);
    // print_mat("VL", VL, n, n, n);
    // print_mat("VR", VR, n, n, n);

    fmt::print("[dtrevc3] dget22 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[dtrevc3] dget22 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] dget22 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] dget22 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(work_lapack);

    // exit(0);
}

void test_dtrevc_scaling()
{
    int n = 4, info[1], i, j, test_idx, m_out;
    double s, *work_lapack, res3[2], resL[2];
    double T_base[16] = { 1, 0, 0, 0, 2, 2, -5, 0, 3, 5, 2, 0, 4, -1, 2, 3 };
    double wr_base[4] = { 1.0, 2.0, 2.0, 3.0 };
    double wi_base[4] = { 0.0, 5.0, -5.0, 0.0 };
    double T[16], wr[4], wi[4], VR[16], VL[16];
    double scales[2] = { 1e150, 1e-150 };
    const char *scale_names[2] = { "Overflow Risk (1e150)", "Underflow Risk (1e-150)" };

    std::vector<double> work(3 * n);
    work_lapack = (double *)malloc(3 * n * sizeof(double));

    dtrevc3('B', 'B', nullptr, n, T, n, wr, wi, VL, n, VR, n, n, &m_out, work.data(), -1, info);
    work.resize(work[0]);

    for (test_idx = 0; test_idx < 2; test_idx++) {
        fmt::print("--- Testing 4x4 Static Matrix (Scaling: {}) ---\n", scale_names[test_idx]);

        s = scales[test_idx];

        for (i = 0; i < 16; i++) {
            T[i] = T_base[i] * s;
        }
        for (i = 0; i < 4; i++) {
            wr[i] = wr_base[i] * s;
            wi[i] = wi_base[i] * s;
        }

        // Test dtrevc3
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = 0.0;
                VL[i * n + j] = 0.0;
            }
            VR[i * n + i] = 1.0;
            VL[i * n + i] = 1.0;
        }

        dtrevc3('B', 'B', nullptr, n, T, n, wr, wi, VL, n, VR, n, n, &m_out, work.data(), work.size(), info);
        dget22(true, true, n, T, n, wr, wi, VL, n, VR, n, res3);

        // Test LAPACK dtrevc
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = 0.0;
                VL[i * n + j] = 0.0;
            }
            VR[i * n + i] = 1.0;
            VL[i * n + i] = 1.0;
        }
        dtrevc_("B", "B", nullptr, &n, T, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info);
        dget22(true, true, n, T, n, wr, wi, VL, n, VR, n, resL);

        fmt::print("[dtrevc3] dget22 Max Right Error Ratio: {:e}\n", res3[0]);
        fmt::print("[dtrevc3] dget22 Max Left Error Ratio:  {:e}\n", res3[1]);
        fmt::print("[lapack ] dget22 Max Right Error Ratio: {:e}\n", resL[0]);
        fmt::print("[lapack ] dget22 Max Left Error Ratio:  {:e}\n\n", resL[1]);
    }

    free(work_lapack);
}

void test_dynamic_size(int n)
{
    double *T, *VR, *VL, *wr, *wi, *work, *work_lapack, dummy[1], res3[2], resL[2];
    int info[1], lwork, i, j, m_out;

    fmt::println("--- Testing {0}x{0} Dynamic Random Matrix ---", n);

    T = (double *)malloc(n * n * sizeof(double));
    VR = (double *)malloc(n * n * sizeof(double));
    VL = (double *)malloc(n * n * sizeof(double));
    wr = (double *)malloc(n * sizeof(double));
    wi = (double *)malloc(n * sizeof(double));

    generate_quasi_triangular(n, T, n, wr, wi);
    dtrevc3('B', 'B', nullptr, n, T, n, wr, wi, VL, n, VR, n, n, &m_out, dummy, -1, info);

    lwork = static_cast<int>(dummy[0]);
    if (lwork <= 0) {
        lwork = 3 * n; // Fallback
    }
    work = (double *)malloc(lwork * sizeof(double));
    work_lapack = (double *)malloc(3 * n * sizeof(double));

    // Test dtrevc3
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0;
            VL[i * n + j] = 0.0;
        }
        VR[i * n + i] = 1.0;
        VL[i * n + i] = 1.0;
    }
    time_code("dtrevc3 ('B')", [&]() { dtrevc3('B', 'B', nullptr, n, T, n, wr, wi, VL, n, VR, n, n, &m_out, work, lwork, info); });
    dget22(true, true, n, T, n, wr, wi, VL, n, VR, n, res3);

    // Test LAPACK dtrevc
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            VR[i * n + j] = 0.0;
            VL[i * n + j] = 0.0;
        }
        VR[i * n + i] = 1.0;
        VL[i * n + i] = 1.0;
    }
    time_code("LAPACK dtrevc ('B')", [&]() { dtrevc_("B", "B", nullptr, &n, T, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info); });
    dget22(true, true, n, T, n, wr, wi, VL, n, VR, n, resL);

    fmt::print("[dtrevc3] dget22 Max Right Error Ratio: {:e}\n", res3[0]);
    fmt::print("[dtrevc3] dget22 Max Left Error Ratio:  {:e}\n", res3[1]);
    fmt::print("[lapack ] dget22 Max Right Error Ratio: {:e}\n", resL[0]);
    fmt::print("[lapack ] dget22 Max Left Error Ratio:  {:e}\n\n", resL[1]);

    free(T);
    free(VR);
    free(VL);
    free(wr);
    free(wi);
    free(work);
    free(work_lapack);
}

void test_lapack_types(int n)
{
    double *T, *VR, *VL, *wr, *wi, *work, *work_lapack, dummy[1], res3[2], resL[2];
    int info[1], lwork, i, j, type, m_out;

    fmt::println("--- Testing {}x{} Matrices over Various LAPACK Matrix Types ---", n, n);

    T = (double *)malloc(n * n * sizeof(double));
    VR = (double *)malloc(n * n * sizeof(double));
    VL = (double *)malloc(n * n * sizeof(double));
    wr = (double *)malloc(n * sizeof(double));
    wi = (double *)malloc(n * sizeof(double));
    work_lapack = (double *)malloc(3 * n * sizeof(double));

    for (type = 1; type <= 15; type++) {
        generate_lapack_matrix_type(type, n, T, n, wr, wi);

        dtrevc3('B', 'B', nullptr, n, T, n, wr, wi, VL, n, VR, n, n, &m_out, dummy, -1, info);
        lwork = static_cast<int>(dummy[0]);
        if (lwork <= 0) {
            lwork = 3 * n; // Fallback workspace for small inputs or query mismatch
        }
        work = (double *)malloc(lwork * sizeof(double));

        // Test dtrevc3
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = 0.0;
                VL[i * n + j] = 0.0;
            }
            VR[i * n + i] = 1.0;
            VL[i * n + i] = 1.0;
        }
        dtrevc3('B', 'B', nullptr, n, T, n, wr, wi, VL, n, VR, n, n, &m_out, work, lwork, info);
        dget22(true, true, n, T, n, wr, wi, VL, n, VR, n, res3);

        // Test LAPACK dtrevc
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                VR[i * n + j] = 0.0;
                VL[i * n + j] = 0.0;
            }
            VR[i * n + i] = 1.0;
            VL[i * n + i] = 1.0;
        }
        dtrevc_("B", "B", nullptr, &n, T, &n, VL, &n, VR, &n, &n, &m_out, work_lapack, info);
        dget22(true, true, n, T, n, wr, wi, VL, n, VR, n, resL);

        fmt::println("Type {:2}:", type);
        fmt::println("  [dtrevc3] Max Right = {:e}, Max Left = {:e}", res3[0], res3[1]);
        fmt::println("  [lapack ] Max Right = {:e}, Max Left = {:e}", resL[0], resL[1]);

        free(work);
    }

    fmt::print("\n");

    free(T);
    free(VR);
    free(VL);
    free(wr);
    free(wi);
    free(work_lapack);
}

int main()
{
    test_dtrevc();
    test_dtrevc_scaling();

    test_dynamic_size(10);
    test_dynamic_size(500);

    test_lapack_types(2);
    test_lapack_types(50);

    return 0;
}