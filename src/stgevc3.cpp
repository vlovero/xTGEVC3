#include "helpers.h"
#include <algorithm>
#include <cmath>
#include <limits>

/*
C++ Function,Fortran Routine Name,Breakdown
[ ] solve_upper_hessenberg,SLAUHS / DLAUHS,"Single/Double, LA (Auxiliary), Upper Hessenberg Solve"
[ ] solve_lower_hessenberg,SLALHS / DLALHS,"Single/Double, LA (Auxiliary), Lower Hessenberg Solve"
[ ] C++ Function,Fortran Routine Name,Breakdown
[ ] solve_lower_hessenberg_2,SLAL2S / DLAL2S,"Single/Double, LA (Auxiliary), Lower, 2 Superdiagonals"
[ ] solve_two_subdiag,SLAU2S / DLAU2S,"Single/Double, LA (Auxiliary), Upper, 2 Subdiagonals"
[ ] get_prev_boundary,ILAPBD,"Integer, LA (Auxiliary), Prev BounDary"
[ ] get_next_boundary,ILANBD,"Integer, LA (Auxiliary), Next BounDary"
[ ] local_solve_right,SLALSR / DLALSR,"Single/Double, LA (Auxiliary), Local Solve Right"
[ ] local_solve_left,SLALSL / DLALSL,"Single/Double, LA (Auxiliary), Local Solve Left"
*/


inline int islapb(const float *S, int lds, int curr, int bsize)
{
    int idx = std::max(0, curr - bsize);
    if (idx > 0 && S[idx + (idx - 1) * lds] != 0.0f) {
        idx -= 1;
    }
    return idx;
}

inline int islanb(const float *S, int n, int lds, int curr, int bsize)
{
    int idx = std::min(n, curr + bsize);
    if (idx < n && S[idx + (idx - 1) * lds] != 0.0f) {
        idx += 1;
    }
    return idx;
}

int slauhs(int n, int nrhs, RP(float) A, int lda, RP(float) B, int ldb)
{
    int j, k, pivot_row;
    float mult, alpha;
    char side = 'L', uplo = 'U', transa = 'N', diag = 'N';

    if (n < 0)
        return -1;
    if (nrhs < 0)
        return -2;
    if (lda < std::max(1, n))
        return -4;
    if (ldb < std::max(1, n))
        return -6;
    if (n == 0)
        return 0;

    for (j = 0; j < n - 1; j++) {
        pivot_row = j;
        if (std::abs(A[(j + 1) + j * lda]) > std::abs(A[j + j * lda])) {
            pivot_row = j + 1;
        }

        if (A[pivot_row + j * lda] == 0.0f)
            return j + 1;

        if (pivot_row != j) {
            for (k = j; k < n; k++)
                std::swap(A[j + k * lda], A[pivot_row + k * lda]);
            for (k = 0; k < nrhs; k++)
                std::swap(B[j + k * ldb], B[pivot_row + k * ldb]);
        }

        mult = A[(j + 1) + j * lda] / A[j + j * lda];
        A[(j + 1) + j * lda] = 0.0f;

        for (k = j + 1; k < n; k++)
            A[(j + 1) + k * lda] -= mult * A[j + k * lda];
        for (k = 0; k < nrhs; k++)
            B[(j + 1) + k * ldb] -= mult * B[j + k * ldb];
    }

    if (A[(n - 1) + (n - 1) * lda] == 0.0f)
        return n;

    if (nrhs > 0) {
        alpha = 1.0f;
        strsm_(&side, &uplo, &transa, &diag, &n, &nrhs, &alpha, A, &lda, B, &ldb);
    }
    return 0;
}

int slau2s(int n, int nrhs, RP(float) A, int lda, RP(float) B, int ldb)
{
    int j, k, pivot_row;
    float max_val, mult1, mult2, alpha;
    char side = 'L', uplo = 'U', transa = 'N', diag = 'N';

    if (n < 0)
        return -1;
    if (nrhs < 0)
        return -2;
    if (lda < std::max(1, n))
        return -4;
    if (ldb < std::max(1, n))
        return -6;
    if (n == 0)
        return 0;

    for (j = 0; j < n - 1; j++) {
        pivot_row = j;
        max_val = std::abs(A[j + j * lda]);

        if (std::abs(A[(j + 1) + j * lda]) > max_val) {
            max_val = std::abs(A[(j + 1) + j * lda]);
            pivot_row = j + 1;
        }
        if (j + 2 < n && std::abs(A[(j + 2) + j * lda]) > max_val) {
            max_val = std::abs(A[(j + 2) + j * lda]);
            pivot_row = j + 2;
        }

        if (max_val == 0.0f)
            return j + 1;

        if (pivot_row != j) {
            for (k = j; k < n; k++)
                std::swap(A[j + k * lda], A[pivot_row + k * lda]);
            for (k = 0; k < nrhs; k++)
                std::swap(B[j + k * ldb], B[pivot_row + k * ldb]);
        }

        mult1 = A[(j + 1) + j * lda] / A[j + j * lda];
        A[(j + 1) + j * lda] = 0.0f;

        for (k = j + 1; k < n; k++)
            A[(j + 1) + k * lda] -= mult1 * A[j + k * lda];
        for (k = 0; k < nrhs; k++)
            B[(j + 1) + k * ldb] -= mult1 * B[j + k * ldb];

        if (j + 2 < n) {
            mult2 = A[(j + 2) + j * lda] / A[j + j * lda];
            A[(j + 2) + j * lda] = 0.0f;
            for (k = j + 1; k < n; k++)
                A[(j + 2) + k * lda] -= mult2 * A[j + k * lda];
            for (k = 0; k < nrhs; k++)
                B[(j + 2) + k * ldb] -= mult2 * B[j + k * ldb];
        }
    }

    if (A[(n - 1) + (n - 1) * lda] == 0.0f)
        return n;

    if (nrhs > 0) {
        alpha = 1.0f;
        strsm_(&side, &uplo, &transa, &diag, &n, &nrhs, &alpha, A, &lda, B, &ldb);
    }
    return 0;
}

int slalhs(int n, int nrhs, RP(float) A, int lda, int *jpiv, RP(float) B, int ldb)
{
    int k, p, i, c;
    float max_val, pivot, m, alpha;
    char side = 'L', uplo = 'L', transa = 'N', diag = 'N';

    for (k = 0; k < n - 1; k++) {
        p = k;
        max_val = std::abs(A[k + k * lda]);

        if (std::abs(A[k + (k + 1) * lda]) > max_val) {
            p = k + 1;
        }
        jpiv[k] = p;

        if (p != k) {
            for (i = k; i < n; i++)
                std::swap(A[i + k * lda], A[i + p * lda]);
        }

        pivot = A[k + k * lda];
        if (pivot == 0.0f)
            return k + 1;

        m = A[k + (k + 1) * lda] / pivot;
        A[k + (k + 1) * lda] = m;

        for (i = k + 1; i < n; i++)
            A[i + (k + 1) * lda] -= m * A[i + k * lda];
    }

    alpha = 1.0f;
    strsm_(&side, &uplo, &transa, &diag, &n, &nrhs, &alpha, A, &lda, B, &ldb);

    for (k = n - 2; k >= 0; --k) {
        m = A[k + (k + 1) * lda];
        p = jpiv[k];
        for (c = 0; c < nrhs; c++) {
            B[k + c * ldb] -= m * B[(k + 1) + c * ldb];
            if (p != k)
                std::swap(B[k + c * ldb], B[p + c * ldb]);
        }
    }
    return 0;
}

int slal2s(int n, int nrhs, RP(float) A, int lda, int *jpiv, RP(float) B, int ldb)
{
    int k, p, i, c;
    float max_val, pivot, m1, m2, alpha;
    char side = 'L', uplo = 'L', transa = 'N', diag = 'N';

    for (k = 0; k < n - 1; k++) {
        p = k;
        max_val = std::abs(A[k + k * lda]);

        if (std::abs(A[k + (k + 1) * lda]) > max_val) {
            max_val = std::abs(A[k + (k + 1) * lda]);
            p = k + 1;
        }
        if (k + 2 < n && std::abs(A[k + (k + 2) * lda]) > max_val) {
            p = k + 2;
        }
        jpiv[k] = p;

        if (p != k) {
            for (i = k; i < n; i++)
                std::swap(A[i + k * lda], A[i + p * lda]);
        }

        pivot = A[k + k * lda];
        if (pivot == 0.0f)
            return k + 1;

        m1 = A[k + (k + 1) * lda] / pivot;
        A[k + (k + 1) * lda] = m1;
        for (i = k + 1; i < n; i++)
            A[i + (k + 1) * lda] -= m1 * A[i + k * lda];

        if (k + 2 < n) {
            m2 = A[k + (k + 2) * lda] / pivot;
            A[k + (k + 2) * lda] = m2;
            for (i = k + 1; i < n; i++)
                A[i + (k + 2) * lda] -= m2 * A[i + k * lda];
        }
    }

    alpha = 1.0f;
    strsm_(&side, &uplo, &transa, &diag, &n, &nrhs, &alpha, A, &lda, B, &ldb);

    for (k = n - 2; k >= 0; --k) {
        m1 = A[k + (k + 1) * lda];
        m2 = (k + 2 < n) ? A[k + (k + 2) * lda] : 0.0f;
        p = jpiv[k];

        for (c = 0; c < nrhs; c++) {
            B[k + c * ldb] -= m1 * B[(k + 1) + c * ldb];
            if (k + 2 < n)
                B[k + c * ldb] -= m2 * B[(k + 2) + c * ldb];
            if (p != k)
                std::swap(B[k + c * ldb], B[p + c * ldb]);
        }
    }
    return 0;
}
/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves the local  eigenvalue problem for the right
 * eigenvectors. Resolves the equations (beta*S - alpha*P) * x = rhs.
 * Incorporates ACOEFF/BCOEFF scaling and overflow protections.
 * =====================================================================
 */
void slalsr(int ldS, const float *S, int ldP, const float *P, int m_size, int ldV, RP(float) rhs_loc, RP(float) X_panel_base, int panel_rows, int nb, const float *alphar, const float *alphai, const float *beta, int is_diag, RP(float) work, float ascale, float bscale, float safemin, float bignum, const int *col_map, int nb_sel)
{
    int k, cur_m, c, r, dim2, i, c_scale, r_scale, c_packed;
    float sk_kp1, pk_kp1, s_kk, p_kk;
    float aR, aI, b_val, val_real, val_imag;
    float sr1, sr2, pr1, pr2, ar1, ai1, ar2, ai2;
    float x1r, x1i, x2r, x2i;
    float t, acoeff, bcoeffR, bcoeffI;
    float rhs_max, scale;
    float *work_rhs;

    k = 0;
    while (k < nb) {
        c_packed = col_map[k];
        if (c_packed < 0) {
            k += (alphai[k] == 0.0f ? 1 : 2);
            continue;
        }

        aR = alphar[k];
        aI = alphai[k];
        b_val = beta[k];

        // Real eigenvalue case
        if (aI == 0.0f) {
            cur_m = is_diag ? k : m_size;

            t = 1.0f / std::max({ std::abs(aR) * ascale, std::abs(b_val) * bscale, safemin });
            acoeff = (t * b_val * bscale) * ascale;
            bcoeffR = (t * aR * ascale) * bscale;

            // Handle infinite eigenvalue (LAPACK Equivalent)
            if (std::abs(b_val) <= safemin && std::abs(aR) > safemin) {
                acoeff = 0.0f;
                bcoeffR = 1.0f;
            }

            if (is_diag) {
                rhs_loc[k + c_packed * ldV] = 1.0f;
            }

            if (cur_m > 0) {
                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        work[r + c * cur_m] = acoeff * S[r + c * ldS] - bcoeffR * P[r + c * ldP];
                    }
                }

                work_rhs = work + cur_m * cur_m;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = -(acoeff * S[r + k * ldS] - bcoeffR * P[r + k * ldP]) * 1.0f;
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = rhs_loc[r + c_packed * ldV];
                    }
                }

                // --- OVERFLOW GUARD: Pre-Local Solve ---
                rhs_max = 0.0f;
                for (r = 0; r < cur_m; r++) {
                    rhs_max = std::max(rhs_max, std::abs(work_rhs[r]));
                }

                if (rhs_max > bignum / 10.0f) {
                    scale = (bignum / 10.0f) / rhs_max;
                    // Scale down the ENTIRE corresponding panel block (only active ones)
                    for (c_scale = 0; c_scale < nb_sel; c_scale++) {
                        for (r_scale = 0; r_scale < panel_rows; r_scale++) {
                            X_panel_base[r_scale + c_scale * ldV] *= scale;
                        }
                    }
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] *= scale;
                    }
                }
                // ----------------------------------------

                slauhs(cur_m, 1, work, cur_m, work_rhs, cur_m);

                for (r = 0; r < cur_m; r++) {
                    rhs_loc[r + c_packed * ldV] = work_rhs[r];
                }
            }
            k += 1;
        }
        // Complex eigenvalue case (2x2 block)
        else {
            cur_m = is_diag ? k : m_size;

            t = 1.0f / std::max({ std::abs(aR) * ascale + std::abs(aI) * ascale, std::abs(b_val) * bscale, safemin });
            acoeff = (t * b_val * bscale) * ascale;
            bcoeffR = (t * aR * ascale) * bscale;
            bcoeffI = (t * aI * ascale) * bscale;

            // Handle infinite eigenvalue (LAPACK Equivalent)
            if (std::abs(b_val) <= safemin && (std::abs(aR) + std::abs(aI)) > safemin) {
                acoeff = 0.0f;
                bcoeffR = 1.0f;
                bcoeffI = 0.0f;
            }

            if (is_diag) {
                sk_kp1 = S[k + (k + 1) * ldS];
                pk_kp1 = P[k + (k + 1) * ldP];

                s_kk = S[k + k * ldS];
                p_kk = P[k + k * ldP];

                rhs_loc[k + c_packed * ldV] = -(acoeff * sk_kp1 - bcoeffR * pk_kp1);
                rhs_loc[k + (c_packed + 1) * ldV] = bcoeffI * pk_kp1;

                rhs_loc[k + 1 + c_packed * ldV] = acoeff * s_kk - bcoeffR * p_kk;
                rhs_loc[k + 1 + (c_packed + 1) * ldV] = -bcoeffI * p_kk;
            }

            if (cur_m > 0) {
                dim2 = 2 * cur_m;

                for (i = 0; i < dim2 * dim2; i++) {
                    work[i] = 0.0f;
                }

                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        val_real = acoeff * S[r + c * ldS] - bcoeffR * P[r + c * ldP];
                        val_imag = bcoeffI * P[r + c * ldP];

                        work[(2 * r + 0) + (2 * c + 0) * dim2] = val_real;
                        work[(2 * r + 1) + (2 * c + 1) * dim2] = val_real;
                        work[(2 * r + 0) + (2 * c + 1) * dim2] = val_imag;
                        work[(2 * r + 1) + (2 * c + 0) * dim2] = -val_imag;
                    }
                }

                work_rhs = work + dim2 * dim2;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        sr1 = S[r + (k + 0) * ldS];
                        sr2 = S[r + (k + 1) * ldS];
                        pr1 = P[r + (k + 0) * ldP];
                        pr2 = P[r + (k + 1) * ldP];

                        ar1 = acoeff * sr1 - bcoeffR * pr1;
                        ai1 = -bcoeffI * pr1;
                        ar2 = acoeff * sr2 - bcoeffR * pr2;
                        ai2 = -bcoeffI * pr2;

                        x1r = rhs_loc[k + c_packed * ldV];
                        x1i = rhs_loc[k + (c_packed + 1) * ldV];
                        x2r = rhs_loc[k + 1 + c_packed * ldV];
                        x2i = rhs_loc[k + 1 + (c_packed + 1) * ldV];

                        work_rhs[2 * r + 0] = -((ar1 * x1r - ai1 * x1i) + (ar2 * x2r - ai2 * x2i));
                        work_rhs[2 * r + 1] = -((ar1 * x1i + ai1 * x1r) + (ar2 * x2i + ai2 * x2r));
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[2 * r + 0] = rhs_loc[r + (c_packed + 0) * ldV];
                        work_rhs[2 * r + 1] = rhs_loc[r + (c_packed + 1) * ldV];
                    }
                }

                // --- OVERFLOW GUARD: Pre-Local Solve (Complex Block) ---
                rhs_max = 0.0f;
                for (r = 0; r < dim2; r++) {
                    rhs_max = std::max(rhs_max, std::abs(work_rhs[r]));
                }

                if (rhs_max > bignum / 10.0f) {
                    scale = (bignum / 10.0f) / rhs_max;
                    for (c_scale = 0; c_scale < nb_sel; c_scale++) {
                        for (r_scale = 0; r_scale < panel_rows; r_scale++) {
                            X_panel_base[r_scale + c_scale * ldV] *= scale;
                        }
                    }
                    for (r = 0; r < dim2; r++) {
                        work_rhs[r] *= scale;
                    }
                }
                // --------------------------------------------------------

                slau2s(dim2, 1, work, dim2, work_rhs, dim2);

                for (r = 0; r < cur_m; r++) {
                    rhs_loc[r + (c_packed + 0) * ldV] = work_rhs[2 * r + 0];
                    rhs_loc[r + (c_packed + 1) * ldV] = work_rhs[2 * r + 1];
                }
            }
            k += 2;
        }
    }
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves the local  eigenvalue problem for the left
 * eigenvectors. Handles equations corresponding to y^H * (beta*S - alpha*P) = 0.
 * Incorporates ACOEFF/BCOEFF scaling and overflow protections.
 * =====================================================================
 */
void slalsl(int ldS, const float *S, int ldP, const float *P, int m_size, int ldV, RP(float) rhs_loc, RP(float) X_panel_base, int panel_rows, int nb, const float *alphar, const float *alphai, const float *beta, int is_diag, RP(float) work, float ascale, float bscale, float safemin, float bignum, const int *col_map, int nb_sel)
{
    int k, cur_m, row_offset, c, r, i, dim2, c_scale, r_scale, c_packed;
    int jpiv[128];
    float s11, s21, p11;
    float aR, aI, b_val, val_real, val_imag;
    float sr1, sr2, pr1, pr2, ar1, ai1, ar2, ai2;
    float y1r, y1i, y2r, y2i;
    float t, acoeff, bcoeffR, bcoeffI;
    float rhs_max, scale;
    float *work_rhs;

    k = 0;
    while (k < nb) {
        c_packed = col_map[k];
        if (c_packed < 0) {
            k += (alphai[k] == 0.0f ? 1 : 2);
            continue;
        }

        aR = alphar[k];
        aI = alphai[k];
        b_val = beta[k];

        // Real eigenvalue case
        if (aI == 0.0f) {
            cur_m = is_diag ? nb - 1 - k : m_size;
            row_offset = is_diag ? k + 1 : 0;

            t = 1.0f / std::max({ std::abs(aR) * ascale, std::abs(b_val) * bscale, safemin });
            acoeff = (t * b_val * bscale) * ascale;
            bcoeffR = (t * aR * ascale) * bscale;

            // Handle infinite eigenvalue (LAPACK Equivalent)
            if (std::abs(b_val) <= safemin && std::abs(aR) > safemin) {
                acoeff = 0.0f;
                bcoeffR = 1.0f;
            }

            if (is_diag) {
                rhs_loc[k + c_packed * ldV] = 1.0f;
            }

            if (cur_m > 0) {
                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        work[r + c * cur_m] = acoeff * S[(c + row_offset) + (r + row_offset) * ldS] - bcoeffR * P[(c + row_offset) + (r + row_offset) * ldP];
                    }
                }

                work_rhs = work + cur_m * cur_m;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        sr1 = S[(k) + (r + row_offset) * ldS];
                        pr1 = P[(k) + (r + row_offset) * ldP];
                        work_rhs[r] = -(acoeff * sr1 - bcoeffR * pr1) * 1.0f;
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = rhs_loc[(r + row_offset) + c_packed * ldV];
                    }
                }

                // --- OVERFLOW GUARD: Pre-Local Solve ---
                rhs_max = 0.0f;
                for (r = 0; r < cur_m; r++) {
                    rhs_max = std::max(rhs_max, std::abs(work_rhs[r]));
                }

                if (rhs_max > bignum / 10.0f) {
                    scale = (bignum / 10.0f) / rhs_max;
                    for (c_scale = 0; c_scale < nb_sel; c_scale++) {
                        for (r_scale = 0; r_scale < panel_rows; r_scale++) {
                            X_panel_base[r_scale + c_scale * ldV] *= scale;
                        }
                    }
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] *= scale;
                    }
                }
                // ----------------------------------------

                slalhs(cur_m, 1, work, cur_m, jpiv, work_rhs, cur_m);

                for (r = 0; r < cur_m; r++) {
                    rhs_loc[(r + row_offset) + c_packed * ldV] = work_rhs[r];
                }
            }
            k += 1;
        }
        // Complex eigenvalue case (2x2 block)
        else {
            cur_m = is_diag ? nb - 2 - k : m_size;
            row_offset = is_diag ? k + 2 : 0;

            t = 1.0f / std::max({ std::abs(aR) * ascale + std::abs(aI) * ascale, std::abs(b_val) * bscale, safemin });
            acoeff = (t * b_val * bscale) * ascale;
            bcoeffR = (t * aR * ascale) * bscale;
            bcoeffI = (t * aI * ascale) * bscale;

            // Handle infinite eigenvalue (LAPACK Equivalent)
            if (std::abs(b_val) <= safemin && (std::abs(aR) + std::abs(aI)) > safemin) {
                acoeff = 0.0f;
                bcoeffR = 1.0f;
                bcoeffI = 0.0f;
            }

            if (is_diag) {
                s11 = S[k + k * ldS];
                s21 = S[k + 1 + k * ldS];
                p11 = P[k + k * ldP];

                rhs_loc[k + c_packed * ldV] = -acoeff * s21;
                rhs_loc[k + (c_packed + 1) * ldV] = 0.0f;

                rhs_loc[k + 1 + c_packed * ldV] = acoeff * s11 - bcoeffR * p11;
                rhs_loc[k + 1 + (c_packed + 1) * ldV] = bcoeffI * p11;
            }

            if (cur_m > 0) {
                dim2 = 2 * cur_m;

                for (i = 0; i < dim2 * dim2; i++) {
                    work[i] = 0.0f;
                }

                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        val_real = acoeff * S[(c + row_offset) + (r + row_offset) * ldS] - bcoeffR * P[(c + row_offset) + (r + row_offset) * ldP];
                        val_imag = bcoeffI * P[(c + row_offset) + (r + row_offset) * ldP];

                        work[(2 * r + 0) + (2 * c + 0) * dim2] = val_real;
                        work[(2 * r + 1) + (2 * c + 1) * dim2] = val_real;
                        work[(2 * r + 0) + (2 * c + 1) * dim2] = -val_imag;
                        work[(2 * r + 1) + (2 * c + 0) * dim2] = val_imag;
                    }
                }

                work_rhs = work + dim2 * dim2;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        sr1 = S[(k + 0) + (r + row_offset) * ldS];
                        sr2 = S[(k + 1) + (r + row_offset) * ldS];
                        pr1 = P[(k + 0) + (r + row_offset) * ldP];
                        pr2 = P[(k + 1) + (r + row_offset) * ldP];

                        ar1 = acoeff * sr1 - bcoeffR * pr1;
                        ai1 = bcoeffI * pr1;
                        ar2 = acoeff * sr2 - bcoeffR * pr2;
                        ai2 = bcoeffI * pr2;

                        y1r = rhs_loc[k + c_packed * ldV];
                        y1i = rhs_loc[k + (c_packed + 1) * ldV];
                        y2r = rhs_loc[k + 1 + c_packed * ldV];
                        y2i = rhs_loc[k + 1 + (c_packed + 1) * ldV];

                        work_rhs[2 * r + 0] = -((ar1 * y1r - ai1 * y1i) + (ar2 * y2r - ai2 * y2i));
                        work_rhs[2 * r + 1] = -((ar1 * y1i + ai1 * y1r) + (ar2 * y2i + ai2 * y2r));
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[2 * r + 0] = rhs_loc[(r + row_offset) + (c_packed + 0) * ldV];
                        work_rhs[2 * r + 1] = rhs_loc[(r + row_offset) + (c_packed + 1) * ldV];
                    }
                }

                // --- OVERFLOW GUARD: Pre-Local Solve (Complex Block) ---
                rhs_max = 0.0f;
                for (r = 0; r < dim2; r++) {
                    rhs_max = std::max(rhs_max, std::abs(work_rhs[r]));
                }

                if (rhs_max > bignum / 10.0f) {
                    scale = (bignum / 10.0f) / rhs_max;
                    for (c_scale = 0; c_scale < nb_sel; c_scale++) {
                        for (r_scale = 0; r_scale < panel_rows; r_scale++) {
                            X_panel_base[r_scale + c_scale * ldV] *= scale;
                        }
                    }
                    for (r = 0; r < dim2; r++) {
                        work_rhs[r] *= scale;
                    }
                }
                // --------------------------------------------------------

                slal2s(dim2, 1, work, dim2, jpiv, work_rhs, dim2);

                for (r = 0; r < cur_m; r++) {
                    rhs_loc[(r + row_offset) + (c_packed + 0) * ldV] = work_rhs[2 * r + 0];
                    rhs_loc[(r + row_offset) + (c_packed + 1) * ldV] = work_rhs[2 * r + 1];
                }
            }
            k += 2;
        }
    }
}


void stgevc3(char side, char howmny, const int *select, int n, const float *S, int lds, const float *P, int ldp, const float *alphar, const float *alphai, const float *beta, RP(float) VL, int ldvl, RP(float) VR, int ldvr, int mm, int *m, RP(float) work, int lwork, int *info)
{
    bool compute_right, compute_left, do_all, do_back, do_sel, selected;
    int req_lwork;
    float *X_panel, *Temp, *work_local;
    int curr_col, i, nb, ld_x, c, r, curr_row, j, j_nb, is_diag, i_next, rem_rows, j_next, rem;
    float *TempS, *TempP;
    float b_val, aR, aI, xr, xi;
    float alpha_m1, beta_1, alpha_1, one, zero;
    float safemin, eps, smlnum, bignum;
    float anorm, bnorm, sum_s, sum_p;
    int col, row, row_limit;
    float ascale, bscale;
    float t, acoeff, bcoeffR, bcoeffI;
    float update_max, safe_limit, scale;
    int total_elements, idx, c_idx, r_idx;

    int bsize = 32; // Default ideal block size
    int num_sel, k_idx, c_packed, out_col, current_out_col;
    int col_map[128]; // Safe stack array covering maximum block size
    int nb_sel;

    compute_right = (side == 'R' || side == 'B');
    compute_left = (side == 'L' || side == 'B');
    do_all = (howmny == 'A' || howmny == 'B');
    do_back = (howmny == 'B');
    do_sel = (howmny == 'S');

    if (!do_all && !do_sel) {
        *info = -2;
        return;
    }

    if (n == 0) {
        *info = 0;
        if (m != nullptr)
            *m = 0;
        return;
    }

    // Pass 1: Count total selected eigenvalues without allocations
    num_sel = 0;
    for (k_idx = 0; k_idx < n;) {
        selected = do_all;
        if (do_sel && select != nullptr) {
            selected = select[k_idx];
            if (alphai[k_idx] != 0.0f && k_idx + 1 < n) {
                selected = (selected || select[k_idx + 1]);
            }
        }

        if (selected) {
            num_sel += (alphai[k_idx] != 0.0f && k_idx + 1 < n) ? 2 : 1;
        }
        k_idx += (alphai[k_idx] != 0.0f && k_idx + 1 < n) ? 2 : 1;
    }

    if (m != nullptr) {
        *m = num_sel;
    }

    if (mm < num_sel) {
        *info = -16;
        return;
    }

    req_lwork = 2 * n * (bsize + 1) + 4 * (bsize + 1) * (bsize + 1) + 2 * (bsize + 1);

    // Dynamic block size fallback to prevent Info = -19 crash
    if (lwork != -1 && lwork < req_lwork) {
        for (bsize = 63; bsize >= 1; bsize--) {
            req_lwork = 2 * n * (bsize + 1) + 4 * (bsize + 1) * (bsize + 1) + 2 * (bsize + 1);
            if (lwork >= req_lwork)
                break;
        }
    }

    if (lwork == -1) {
        work[0] = static_cast<float>(req_lwork); // Request optimal workspace
        *info = 0;
        return;
    }

    if (lwork < req_lwork) {
        *info = -19;
        return;
    }

    *info = 0;

    safemin = std::numeric_limits<float>::min();
    eps = std::numeric_limits<float>::epsilon();
    smlnum = safemin / eps;
    bignum = 1.0f / smlnum;

    anorm = 0.0f;
    bnorm = 0.0f;
    for (col = 0; col < n; ++col) {
        sum_s = 0.0f;
        sum_p = 0.0f;
        row_limit = std::min(n - 1, col + 1);
        for (row = 0; row <= row_limit; ++row) {
            sum_s += std::abs(S[row + col * lds]);
            sum_p += std::abs(P[row + col * ldp]);
        }
        anorm = std::max(anorm, sum_s);
        bnorm = std::max(bnorm, sum_p);
    }

    ascale = 1.0f / std::max(anorm, safemin);
    bscale = 1.0f / std::max(bnorm, safemin);

    X_panel = work;
    Temp = X_panel + n * (bsize + 1);
    work_local = Temp + n * (bsize + 1);

    if (compute_right && VR != nullptr) {
        current_out_col = num_sel; // Right solver operates backwards
        curr_col = n;

        while (curr_col > 0) {
            i = islapb(S, lds, curr_col, bsize);
            nb = curr_col - i;
            ld_x = curr_col;

            nb_sel = 0;
            for (c = 0; c < nb;) {
                selected = do_all;
                if (do_sel && select != nullptr) {
                    selected = select[i + c];
                    if (alphai[i + c] != 0.0f && i + c + 1 < n) {
                        selected = (selected || select[i + c + 1]);
                    }
                }

                if (selected) {
                    col_map[c] = nb_sel++;
                    if (alphai[i + c] != 0.0f && i + c + 1 < n) {
                        col_map[c + 1] = nb_sel++;
                        c += 2;
                    }
                    else
                        c += 1;
                }
                else {
                    col_map[c] = -1;
                    if (alphai[i + c] != 0.0f && i + c + 1 < n) {
                        col_map[c + 1] = -1;
                        c += 2;
                    }
                    else
                        c += 1;
                }
            }

            current_out_col -= nb_sel;

            if (nb_sel == 0) {
                curr_col = i;
                continue;
            }

            for (c = 0; c < nb_sel; c++) {
                for (r = 0; r < curr_col; r++)
                    X_panel[r + c * ld_x] = 0.0f;
            }

            curr_row = curr_col;
            while (curr_row > 0) {
                j = islapb(S, lds, curr_row, bsize);
                j_nb = curr_row - j;
                is_diag = (curr_row == curr_col) ? 1 : 0;

                slalsr(lds, &S[j + j * lds], ldp, &P[j + j * ldp], j_nb, ld_x, &X_panel[j], X_panel, ld_x, nb, alphar + i, alphai + i, beta + i, is_diag, work_local, ascale, bscale, safemin, bignum, col_map, nb_sel);

                if (j > 0) {
                    TempS = work_local;
                    TempP = work_local + j_nb * nb_sel;

                    for (c = 0; c < nb;) {
                        c_packed = col_map[c];
                        if (c_packed < 0) {
                            c += (alphai[i + c] == 0.0f ? 1 : 2);
                            continue;
                        }

                        b_val = beta[i + c];
                        aR = alphar[i + c];
                        aI = alphai[i + c];

                        t = 1.0f / std::max({ std::abs(aR) * ascale + std::abs(aI) * ascale, std::abs(b_val) * bscale, safemin });
                        acoeff = (t * b_val * bscale) * ascale;
                        bcoeffR = (t * aR * ascale) * bscale;
                        bcoeffI = (t * aI * ascale) * bscale;

                        if (std::abs(b_val) <= safemin && (std::abs(aR) + std::abs(aI)) > safemin) {
                            acoeff = 0.0f;
                            bcoeffR = 1.0f;
                            bcoeffI = 0.0f;
                        }

                        if (aI == 0.0f) {
                            for (r = 0; r < j_nb; r++) {
                                xr = X_panel[(j + r) + c_packed * ld_x];
                                TempS[r + c_packed * j_nb] = acoeff * xr;
                                TempP[r + c_packed * j_nb] = bcoeffR * xr;
                            }
                            c++;
                        }
                        else {
                            for (r = 0; r < j_nb; r++) {
                                xr = X_panel[(j + r) + c_packed * ld_x];
                                xi = X_panel[(j + r) + (c_packed + 1) * ld_x];
                                TempS[r + c_packed * j_nb] = acoeff * xr;
                                TempS[r + (c_packed + 1) * j_nb] = acoeff * xi;
                                TempP[r + c_packed * j_nb] = bcoeffR * xr - bcoeffI * xi;
                                TempP[r + (c_packed + 1) * j_nb] = bcoeffR * xi + bcoeffI * xr;
                            }
                            c += 2;
                        }
                    }

                    update_max = 0.0f;
                    total_elements = j_nb * nb_sel;
                    for (idx = 0; idx < total_elements; ++idx) {
                        update_max = std::max({ update_max, std::abs(TempS[idx]), std::abs(TempP[idx]) });
                    }

                    safe_limit = bignum / static_cast<float>(std::max(1, nb_sel));

                    if (update_max > safe_limit) {
                        scale = safe_limit / update_max;
                        for (idx = 0; idx < total_elements; ++idx) {
                            TempS[idx] *= scale;
                            TempP[idx] *= scale;
                        }
                        for (c_idx = 0; c_idx < nb_sel; ++c_idx) {
                            for (r_idx = 0; r_idx < ld_x; ++r_idx)
                                X_panel[r_idx + c_idx * ld_x] *= scale;
                        }
                    }

                    alpha_m1 = -1.0f;
                    beta_1 = 1.0f;
                    alpha_1 = 1.0f;
                    sgemm_("N", "N", &j, &nb_sel, &j_nb, &alpha_m1, &S[0 + j * lds], &lds, TempS, &j_nb, &beta_1, &X_panel[0], &ld_x);
                    sgemm_("N", "N", &j, &nb_sel, &j_nb, &alpha_1, &P[0 + j * ldp], &ldp, TempP, &j_nb, &beta_1, &X_panel[0], &ld_x);
                }
                curr_row = j;
            }

            if (do_back) {
                one = 1.0f;
                zero = 0.0f;
                sgemm_("N", "N", &n, &nb_sel, &curr_col, &one, VR, &ldvr, X_panel, &ld_x, &zero, Temp, &n);

                for (c = 0; c < nb_sel; c++) {
                    out_col = current_out_col + c;
                    for (r = 0; r < n; r++)
                        VR[r + out_col * ldvr] = Temp[r + c * n];
                }
            }
            else {
                for (c = 0; c < nb_sel; c++) {
                    out_col = current_out_col + c;
                    for (r = 0; r < curr_col; r++)
                        VR[r + out_col * ldvr] = X_panel[r + c * ld_x];
                    for (r = curr_col; r < n; r++)
                        VR[r + out_col * ldvr] = 0.0f;
                }
            }
            curr_col = i;
        }
    }

    if (compute_left && VL != nullptr) {
        current_out_col = 0; // Left solver operates forwards
        curr_col = 0;

        while (curr_col < n) {
            i_next = islanb(S, n, lds, curr_col, bsize);
            nb = i_next - curr_col;
            i = curr_col;
            rem_rows = n - i;
            ld_x = rem_rows;

            nb_sel = 0;
            for (c = 0; c < nb;) {
                selected = do_all;
                if (do_sel && select != nullptr) {
                    selected = select[i + c];
                    if (alphai[i + c] != 0.0f && i + c + 1 < n) {
                        selected = (selected || select[i + c + 1]);
                    }
                }

                if (selected) {
                    col_map[c] = nb_sel++;
                    if (alphai[i + c] != 0.0f && i + c + 1 < n) {
                        col_map[c + 1] = nb_sel++;
                        c += 2;
                    }
                    else
                        c += 1;
                }
                else {
                    col_map[c] = -1;
                    if (alphai[i + c] != 0.0f && i + c + 1 < n) {
                        col_map[c + 1] = -1;
                        c += 2;
                    }
                    else
                        c += 1;
                }
            }

            if (nb_sel == 0) {
                curr_col = i_next;
                continue;
            }

            for (c = 0; c < nb_sel; c++) {
                for (r = 0; r < rem_rows; r++)
                    X_panel[r + c * ld_x] = 0.0f;
            }

            curr_row = i;
            while (curr_row < n) {
                j_next = islanb(S, n, lds, curr_row, bsize);
                j_nb = j_next - curr_row;
                is_diag = (curr_row == i) ? 1 : 0;

                slalsl(lds, &S[curr_row + curr_row * lds], ldp, &P[curr_row + curr_row * ldp], j_nb, ld_x, &X_panel[curr_row - i], X_panel, ld_x, nb, alphar + i, alphai + i, beta + i, is_diag, work_local, ascale, bscale, safemin, bignum, col_map, nb_sel);

                if (j_next < n) {
                    TempS = work_local;
                    TempP = work_local + j_nb * nb_sel;

                    for (c = 0; c < nb;) {
                        c_packed = col_map[c];
                        if (c_packed < 0) {
                            c += (alphai[i + c] == 0.0f ? 1 : 2);
                            continue;
                        }

                        b_val = beta[i + c];
                        aR = alphar[i + c];
                        aI = alphai[i + c];

                        t = 1.0f / std::max({ std::abs(aR) * ascale + std::abs(aI) * ascale, std::abs(b_val) * bscale, safemin });
                        acoeff = (t * b_val * bscale) * ascale;
                        bcoeffR = (t * aR * ascale) * bscale;
                        bcoeffI = (t * aI * ascale) * bscale;

                        if (std::abs(b_val) <= safemin && (std::abs(aR) + std::abs(aI)) > safemin) {
                            acoeff = 0.0f;
                            bcoeffR = 1.0f;
                            bcoeffI = 0.0f;
                        }

                        if (aI == 0.0f) {
                            for (r = 0; r < j_nb; r++) {
                                xr = X_panel[(curr_row - i + r) + c_packed * ld_x];
                                TempS[r + c_packed * j_nb] = acoeff * xr;
                                TempP[r + c_packed * j_nb] = bcoeffR * xr;
                            }
                            c += 1;
                        }
                        else {
                            for (r = 0; r < j_nb; r++) {
                                xr = X_panel[(curr_row - i + r) + c_packed * ld_x];
                                xi = X_panel[(curr_row - i + r) + (c_packed + 1) * ld_x];
                                TempS[r + c_packed * j_nb] = acoeff * xr;
                                TempS[r + (c_packed + 1) * j_nb] = acoeff * xi;
                                TempP[r + c_packed * j_nb] = bcoeffR * xr + bcoeffI * xi;
                                TempP[r + (c_packed + 1) * j_nb] = bcoeffR * xi - bcoeffI * xr;
                            }
                            c += 2;
                        }
                    }

                    rem = n - j_next;
                    update_max = 0.0f;
                    total_elements = j_nb * nb_sel;
                    for (idx = 0; idx < total_elements; ++idx) {
                        update_max = std::max({ update_max, std::abs(TempS[idx]), std::abs(TempP[idx]) });
                    }

                    safe_limit = bignum / static_cast<float>(std::max(1, nb_sel));

                    if (update_max > safe_limit) {
                        scale = safe_limit / update_max;
                        for (idx = 0; idx < total_elements; ++idx) {
                            TempS[idx] *= scale;
                            TempP[idx] *= scale;
                        }
                        for (c_idx = 0; c_idx < nb_sel; ++c_idx) {
                            for (r_idx = 0; r_idx < ld_x; ++r_idx)
                                X_panel[r_idx + c_idx * ld_x] *= scale;
                        }
                    }

                    alpha_m1 = -1.0f;
                    beta_1 = 1.0f;
                    alpha_1 = 1.0f;
                    sgemm_("T", "N", &rem, &nb_sel, &j_nb, &alpha_m1, &S[curr_row + j_next * lds], &lds, TempS, &j_nb, &beta_1, &X_panel[j_next - i], &ld_x);
                    sgemm_("T", "N", &rem, &nb_sel, &j_nb, &alpha_1, &P[curr_row + j_next * ldp], &ldp, TempP, &j_nb, &beta_1, &X_panel[j_next - i], &ld_x);
                }
                curr_row = j_next;
            }

            if (do_back) {
                one = 1.0f;
                zero = 0.0f;
                sgemm_("N", "N", &n, &nb_sel, &rem_rows, &one, &VL[i * ldvl], &ldvl, X_panel, &ld_x, &zero, Temp, &n);

                for (c = 0; c < nb_sel; c++) {
                    out_col = current_out_col + c;
                    for (r = 0; r < n; r++)
                        VL[r + out_col * ldvl] = Temp[r + c * n];
                }
            }
            else {
                for (c = 0; c < nb_sel; c++) {
                    out_col = current_out_col + c;
                    for (r = 0; r < i; r++)
                        VL[r + out_col * ldvl] = 0.0f;
                    for (r = 0; r < rem_rows; r++)
                        VL[(i + r) + out_col * ldvl] = X_panel[r + c * ld_x];
                }
            }
            current_out_col += nb_sel;
            curr_col = i_next;
        }
    }
}