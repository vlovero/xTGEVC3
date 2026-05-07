#include "helpers.h"
#include <algorithm>
#include <cmath>
#include <limits>


int dlauhs(int n, int nrhs, double *A, int lda, double *B, int ldb);
int dlau2s(int n, int nrhs, double *A, int lda, double *B, int ldb);
int dlalhs(int n, int nrhs, double *A, int lda, int *jpiv, double *B, int ldb);
int dlal2s(int n, int nrhs, double *A, int lda, int *jpiv, double *B, int ldb);


/*
 * =====================================================================
 * Purpose:
 * =======
 * Helper function to determine the start index of a block for backward
 * traversal. It ensures that a 2x2 diagonal block (which corresponds to
 * a complex conjugate eigenvalue pair) is not split across boundaries.
 * =====================================================================
 */
inline int idlapb(const double *T, int ldt, int curr, int bsize)
{
    int idx;
    idx = std::max(0, curr - bsize);

    if (idx > 0 && T[idx + (idx - 1) * ldt] != 0.0) {
        idx -= 1;
    }
    return idx;
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Helper function to determine the end index of a block for forward
 * traversal, ensuring that a 2x2 diagonal block is not split across
 * block boundaries.
 * =====================================================================
 */
inline int idlanb(const double *T, int n, int ldt, int curr, int bsize)
{
    int idx;
    idx = std::min(n, curr + bsize);

    if (idx < n && T[idx + (idx - 1) * ldt] != 0.0) {
        idx += 1;
    }
    return idx;
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves the local panel eigenvalue problem for right eigenvectors
 * corresponding to the equations (T - w * I) * x = rhs.
 * =====================================================================
 */
void dlatsr(int ldT, const double *T, int m_size, int ldV, double *rhs_loc, double *X_panel_base, int panel_rows, int nb, const double *wr, const double *wi, int is_diag, double *work, double bignum, const int *col_map, int nb_sel)
{
    int k, cur_m, c, r, dim2, i, c_scale, r_scale, c_packed;
    double wR, wI, val_real, val_imag;
    double x1r, x1i, x2r, x2i;
    double tr1, tr2;
    double rhs_max, scale;
    double *work_rhs;

    k = 0;
    while (k < nb) {
        c_packed = col_map[k];

        if (c_packed < 0) {
            k += (wi[k] == 0.0 ? 1 : 2);
            continue;
        }

        wR = wr[k];
        wI = wi[k];

        if (wI == 0.0) {
            // --- REAL EIGENVALUE ---
            cur_m = is_diag ? k : m_size;

            if (is_diag) {
                rhs_loc[k + c_packed * ldV] = 1.0;
            }

            if (cur_m > 0) {
                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        work[r + c * cur_m] = T[r + c * ldT];
                        if (r == c) {
                            work[r + c * cur_m] -= wR;
                        }
                    }
                }

                work_rhs = work + cur_m * cur_m;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = -T[r + k * ldT];
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = rhs_loc[r + c_packed * ldV];
                    }
                }

                rhs_max = 0.0;
                for (r = 0; r < cur_m; r++) {
                    rhs_max = std::max(rhs_max, std::abs(work_rhs[r]));
                }

                if (rhs_max > bignum / 10.0) {
                    scale = (bignum / 10.0) / rhs_max;
                    for (c_scale = 0; c_scale < nb_sel; c_scale++) {
                        for (r_scale = 0; r_scale < panel_rows; r_scale++) {
                            X_panel_base[r_scale + c_scale * ldV] *= scale;
                        }
                    }
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] *= scale;
                    }
                }

                dlauhs(cur_m, 1, work, cur_m, work_rhs, cur_m);

                for (r = 0; r < cur_m; r++) {
                    rhs_loc[r + c_packed * ldV] = work_rhs[r];
                }
            }
            k += 1;
        }
        else {
            // --- COMPLEX CONJUGATE PAIR ---
            cur_m = is_diag ? k : m_size;

            if (is_diag) {
                rhs_loc[k + c_packed * ldV] = -T[k + (k + 1) * ldT];
                rhs_loc[k + (c_packed + 1) * ldV] = 0.0;

                rhs_loc[k + 1 + c_packed * ldV] = T[k + k * ldT] - wR;
                rhs_loc[k + 1 + (c_packed + 1) * ldV] = -wI;
            }

            if (cur_m > 0) {
                dim2 = 2 * cur_m;
                for (i = 0; i < dim2 * dim2; i++) {
                    work[i] = 0.0;
                }

                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        val_real = T[r + c * ldT];
                        if (r == c) {
                            val_real -= wR;
                        }
                        val_imag = (r == c) ? wI : 0.0;

                        work[(2 * r + 0) + (2 * c + 0) * dim2] = val_real;
                        work[(2 * r + 1) + (2 * c + 1) * dim2] = val_real;
                        work[(2 * r + 0) + (2 * c + 1) * dim2] = val_imag;
                        work[(2 * r + 1) + (2 * c + 0) * dim2] = -val_imag;
                    }
                }

                work_rhs = work + dim2 * dim2;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        tr1 = T[r + (k + 0) * ldT];
                        tr2 = T[r + (k + 1) * ldT];

                        x1r = rhs_loc[k + c_packed * ldV];
                        x1i = rhs_loc[k + (c_packed + 1) * ldV];
                        x2r = rhs_loc[k + 1 + c_packed * ldV];
                        x2i = rhs_loc[k + 1 + (c_packed + 1) * ldV];

                        work_rhs[2 * r + 0] = -(tr1 * x1r + tr2 * x2r);
                        work_rhs[2 * r + 1] = -(tr1 * x1i + tr2 * x2i);
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[2 * r + 0] = rhs_loc[r + (c_packed + 0) * ldV];
                        work_rhs[2 * r + 1] = rhs_loc[r + (c_packed + 1) * ldV];
                    }
                }

                rhs_max = 0.0;
                for (r = 0; r < dim2; r++) {
                    rhs_max = std::max(rhs_max, std::abs(work_rhs[r]));
                }

                if (rhs_max > bignum / 10.0) {
                    scale = (bignum / 10.0) / rhs_max;
                    for (c_scale = 0; c_scale < nb_sel; c_scale++) {
                        for (r_scale = 0; r_scale < panel_rows; r_scale++) {
                            X_panel_base[r_scale + c_scale * ldV] *= scale;
                        }
                    }
                    for (r = 0; r < dim2; r++) {
                        work_rhs[r] *= scale;
                    }
                }

                dlau2s(dim2, 1, work, dim2, work_rhs, dim2);

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
 * Solves the local panel eigenvalue problem for left eigenvectors
 * corresponding to the equations y^H * (T - w * I) = 0.
 * =====================================================================
 */
void dlatsl(int ldT, const double *T, int m_size, int ldV, double *rhs_loc, double *X_panel_base, int panel_rows, int nb, const double *wr, const double *wi, int is_diag, double *work, double bignum, const int *col_map, int nb_sel)
{
    int k, cur_m, row_offset, c, r, i, dim2, c_scale, r_scale, c_packed;
    int jpiv[128]; // Reverted back to stack allocated array
    double wR, wI, val_real, val_imag;
    double y1r, y1i, y2r, y2i;
    double tr1, tr2;
    double rhs_max, scale;
    double *work_rhs;

    k = 0;
    while (k < nb) {
        c_packed = col_map[k];
        if (c_packed < 0) {
            k += (wi[k] == 0.0 ? 1 : 2);
            continue;
        }

        wR = wr[k];
        wI = wi[k];

        if (wI == 0.0) {
            // --- REAL EIGENVALUE ---
            cur_m = is_diag ? nb - 1 - k : m_size;
            row_offset = is_diag ? k + 1 : 0;

            if (is_diag) {
                rhs_loc[k + c_packed * ldV] = 1.0;
            }

            if (cur_m > 0) {
                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        work[r + c * cur_m] = T[(c + row_offset) + (r + row_offset) * ldT];
                        if (r == c) {
                            work[r + c * cur_m] -= wR;
                        }
                    }
                }

                work_rhs = work + cur_m * cur_m;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = -T[k + (r + row_offset) * ldT];
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = rhs_loc[(r + row_offset) + c_packed * ldV];
                    }
                }

                rhs_max = 0.0;
                for (r = 0; r < cur_m; r++) {
                    rhs_max = std::max(rhs_max, std::abs(work_rhs[r]));
                }

                if (rhs_max > bignum / 10.0) {
                    scale = (bignum / 10.0) / rhs_max;
                    for (c_scale = 0; c_scale < nb_sel; c_scale++) {
                        for (r_scale = 0; r_scale < panel_rows; r_scale++) {
                            X_panel_base[r_scale + c_scale * ldV] *= scale;
                        }
                    }
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] *= scale;
                    }
                }

                dlalhs(cur_m, 1, work, cur_m, jpiv, work_rhs, cur_m);

                for (r = 0; r < cur_m; r++) {
                    rhs_loc[(r + row_offset) + c_packed * ldV] = work_rhs[r];
                }
            }
            k += 1;
        }
        else {
            // --- COMPLEX CONJUGATE PAIR ---
            cur_m = is_diag ? nb - 2 - k : m_size;
            row_offset = is_diag ? k + 2 : 0;

            if (is_diag) {
                rhs_loc[k + c_packed * ldV] = -T[k + 1 + k * ldT];
                rhs_loc[k + (c_packed + 1) * ldV] = 0.0;

                rhs_loc[k + 1 + c_packed * ldV] = T[k + k * ldT] - wR;
                rhs_loc[k + 1 + (c_packed + 1) * ldV] = wI;
            }

            if (cur_m > 0) {
                dim2 = 2 * cur_m;
                for (i = 0; i < dim2 * dim2; i++) {
                    work[i] = 0.0;
                }

                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        val_real = T[(c + row_offset) + (r + row_offset) * ldT];
                        if (r == c) {
                            val_real -= wR;
                        }
                        val_imag = (r == c) ? wI : 0.0;

                        work[(2 * r + 0) + (2 * c + 0) * dim2] = val_real;
                        work[(2 * r + 1) + (2 * c + 1) * dim2] = val_real;
                        work[(2 * r + 0) + (2 * c + 1) * dim2] = -val_imag;
                        work[(2 * r + 1) + (2 * c + 0) * dim2] = val_imag;
                    }
                }

                work_rhs = work + dim2 * dim2;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        tr1 = T[(k + 0) + (r + row_offset) * ldT];
                        tr2 = T[(k + 1) + (r + row_offset) * ldT];

                        y1r = rhs_loc[k + c_packed * ldV];
                        y1i = rhs_loc[k + (c_packed + 1) * ldV];
                        y2r = rhs_loc[k + 1 + c_packed * ldV];
                        y2i = rhs_loc[k + 1 + (c_packed + 1) * ldV];

                        work_rhs[2 * r + 0] = -(tr1 * y1r + tr2 * y2r);
                        work_rhs[2 * r + 1] = -(tr1 * y1i + tr2 * y2i);
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[2 * r + 0] = rhs_loc[(r + row_offset) + (c_packed + 0) * ldV];
                        work_rhs[2 * r + 1] = rhs_loc[(r + row_offset) + (c_packed + 1) * ldV];
                    }
                }

                rhs_max = 0.0;
                for (r = 0; r < dim2; r++) {
                    rhs_max = std::max(rhs_max, std::abs(work_rhs[r]));
                }

                if (rhs_max > bignum / 10.0) {
                    scale = (bignum / 10.0) / rhs_max;
                    for (c_scale = 0; c_scale < nb_sel; c_scale++) {
                        for (r_scale = 0; r_scale < panel_rows; r_scale++) {
                            X_panel_base[r_scale + c_scale * ldV] *= scale;
                        }
                    }
                    for (r = 0; r < dim2; r++) {
                        work_rhs[r] *= scale;
                    }
                }

                dlal2s(dim2, 1, work, dim2, jpiv, work_rhs, dim2);

                for (r = 0; r < cur_m; r++) {
                    rhs_loc[(r + row_offset) + (c_packed + 0) * ldV] = work_rhs[2 * r + 0];
                    rhs_loc[(r + row_offset) + (c_packed + 1) * ldV] = work_rhs[2 * r + 1];
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
 * DTREVC3 computes some or all of the right and/or left eigenvectors of
 * a real upper quasi-triangular matrix T.
 *
 * This is a blocked algorithm that utilizes Level 3 BLAS for performance.
 *
 * Arguments:
 * =========
 * side            (input) char
 * = 'R': compute right eigenvectors only;
 * = 'L': compute left eigenvectors only;
 * = 'B': compute both right and left eigenvectors.
 *
 * howmny          (input) char
 * = 'A': compute all right and/or left eigenvectors;
 * = 'B': compute all eigenvectors, backtransformed by VL/VR;
 * = 'S': compute selected eigenvectors specified by SELECT.
 *
 * select          (input) const int*
 * Logical array specifying selected eigenvectors.
 *
 * n               (input) int
 * The order of the matrix T.
 *
 * T               (input) const double*
 * The upper quasi-triangular matrix T.
 *
 * ldt             (input) int
 * The leading dimension of the array T.
 *
 * wr, wi          (input) const double*
 * Real and imaginary parts of the eigenvalues.
 *
 * VL              (input/output) double*
 * Left eigenvector storage.
 *
 * ldvl            (input) int
 * Leading dimension of array VL.
 *
 * VR              (input/output) double*
 * Right eigenvector storage.
 *
 * ldvr            (input) int
 * Leading dimension of array VR.
 *
 * mm              (input) int
 * Number of columns in VL/VR.
 *
 * m               (output) int*
 * Number of columns actually used in VL/VR.
 *
 * work            (workspace) double*
 * Workspace array.
 *
 * lwork           (input) int
 * Dimension of WORK. -1 for workspace query.
 *
 * info            (output) int*
 * Exit code: 0 on success, <0 for invalid args.
 * =====================================================================
 */
void dtrevc3(char side, char howmny, const int *select, int n, const double *T, int ldt, const double *wr, const double *wi, double *VL, int ldvl, double *VR, int ldvr, int mm, int *m, double *work, int lwork, int *info)
{
    bool compute_right, compute_left, do_all, do_back, do_sel, selected;
    int req_lwork;
    double *X_panel, *Temp, *work_local;
    int curr_col, i, nb, ld_x, c, r, curr_row, j, j_nb, is_diag, i_next, rem_rows, j_next, rem;
    double alpha_m1, beta_1, one, zero;
    double safemin, eps, smlnum, bignum;
    double anorm, sum_t;
    int col, row, row_limit;
    double update_max, safe_limit, scale, xmax;
    int c_idx, r_idx;

    int bsize = 32;
    int num_sel, k_idx, c_packed, out_col, current_out_col;
    int col_map[128]; // Stack allocated array bounded safely for sizes up to 64
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
        if (m != nullptr) {
            *m = 0;
        }
        return;
    }

    num_sel = 0;
    for (k_idx = 0; k_idx < n;) {
        selected = do_all;
        if (do_sel && select != nullptr) {
            selected = select[k_idx];
            if (wi[k_idx] != 0.0 && k_idx + 1 < n) {
                selected = (selected || select[k_idx + 1]);
            }
        }

        if (selected) {
            num_sel += (wi[k_idx] != 0.0 && k_idx + 1 < n) ? 2 : 1;
        }
        k_idx += (wi[k_idx] != 0.0 && k_idx + 1 < n) ? 2 : 1;
    }

    if (m != nullptr) {
        *m = num_sel;
    }
    if (mm < num_sel) {
        *info = -13;
        return;
    }

    req_lwork = 2 * n * (bsize + 1) + 4 * (bsize + 1) * (bsize + 1) + 2 * (bsize + 1);

    if (lwork != -1 && lwork < req_lwork) {
        for (bsize = bsize - 1; bsize >= 1; bsize--) {
            req_lwork = 2 * n * (bsize + 1) + 4 * (bsize + 1) * (bsize + 1) + 2 * (bsize + 1);
            if (lwork >= req_lwork) {
                break;
            }
        }
    }

    if (lwork == -1) {
        work[0] = static_cast<double>(req_lwork);
        *info = 0;
        return;
    }

    if (lwork < req_lwork) {
        *info = -16;
        return;
    }

    *info = 0;

    safemin = std::numeric_limits<double>::min();
    eps = std::numeric_limits<double>::epsilon();
    smlnum = safemin / eps;
    bignum = 1.0 / smlnum;

    anorm = 0.0;
    for (col = 0; col < n; ++col) {
        sum_t = 0.0;
        row_limit = std::min(n - 1, col + 1);
        for (row = 0; row <= row_limit; ++row) {
            sum_t += std::abs(T[row + col * ldt]);
        }
        anorm = std::max(anorm, sum_t);
    }
    anorm = std::max(anorm, safemin);

    X_panel = work;
    Temp = X_panel + n * (bsize + 1);
    work_local = Temp + n * (bsize + 1);

    // ==================================================================
    // Right Eigenvector Computation
    // ==================================================================
    if (compute_right && VR != nullptr) {
        current_out_col = num_sel;
        curr_col = n;

        while (curr_col > 0) {
            i = idlapb(T, ldt, curr_col, bsize);
            nb = curr_col - i;
            ld_x = curr_col;

            nb_sel = 0;
            for (c = 0; c < nb;) {
                selected = do_all;
                if (do_sel && select != nullptr) {
                    selected = select[i + c];
                    if (wi[i + c] != 0.0 && i + c + 1 < n) {
                        selected = (selected || select[i + c + 1]);
                    }
                }

                if (selected) {
                    col_map[c] = nb_sel++;
                    if (wi[i + c] != 0.0 && i + c + 1 < n) {
                        col_map[c + 1] = nb_sel++;
                        c += 2;
                    }
                    else {
                        c += 1;
                    }
                }
                else {
                    col_map[c] = -1;
                    if (wi[i + c] != 0.0 && i + c + 1 < n) {
                        col_map[c + 1] = -1;
                        c += 2;
                    }
                    else {
                        c += 1;
                    }
                }
            }

            current_out_col -= nb_sel;
            if (nb_sel == 0) {
                curr_col = i;
                continue;
            }

            for (c = 0; c < nb_sel; c++) {
                for (r = 0; r < curr_col; r++) {
                    X_panel[r + c * ld_x] = 0.0;
                }
            }

            curr_row = curr_col;
            while (curr_row > 0) {
                j = idlapb(T, ldt, curr_row, bsize);
                j_nb = curr_row - j;
                is_diag = (curr_row == curr_col) ? 1 : 0;

                dlatsr(ldt, &T[j + j * ldt], j_nb, ld_x, &X_panel[j], X_panel, ld_x, nb, wr + i, wi + i, is_diag, work_local, bignum, col_map, nb_sel);

                if (j > 0) {
                    update_max = 0.0;
                    for (c_idx = 0; c_idx < nb_sel; ++c_idx) {
                        for (r_idx = 0; r_idx < j_nb; ++r_idx) {
                            update_max = std::max(update_max, std::abs(X_panel[j + r_idx + c_idx * ld_x]));
                        }
                    }

                    safe_limit = bignum / static_cast<double>(std::max(1, j_nb));
                    if (update_max * anorm > safe_limit) {
                        scale = safe_limit / (update_max * anorm);
                        for (c_idx = 0; c_idx < nb_sel; ++c_idx) {
                            for (r_idx = 0; r_idx < ld_x; ++r_idx) {
                                X_panel[r_idx + c_idx * ld_x] *= scale;
                            }
                        }
                    }

                    alpha_m1 = -1.0;
                    beta_1 = 1.0;
                    dgemm_("N", "N", &j, &nb_sel, &j_nb, &alpha_m1, const_cast<double *>(&T[0 + j * ldt]), &ldt, &X_panel[j], &ld_x, &beta_1, &X_panel[0], &ld_x);
                }
                curr_row = j;
            }

            if (do_back) {
                one = 1.0;
                zero = 0.0;
                dgemm_("N", "N", &n, &nb_sel, &curr_col, &one, VR, &ldvr, X_panel, &ld_x, &zero, Temp, &n);
                for (c = 0; c < nb_sel; c++) {
                    out_col = current_out_col + c;
                    for (r = 0; r < n; r++) {
                        VR[r + out_col * ldvr] = Temp[r + c * n];
                    }
                }
            }
            else {
                for (c = 0; c < nb_sel; c++) {
                    out_col = current_out_col + c;
                    for (r = 0; r < curr_col; r++) {
                        VR[r + out_col * ldvr] = X_panel[r + c * ld_x];
                    }
                    for (r = curr_col; r < n; r++) {
                        VR[r + out_col * ldvr] = 0.0;
                    }
                }
            }

            for (c = 0; c < nb;) {
                c_packed = col_map[c];
                if (c_packed < 0) {
                    c += (wi[i + c] == 0.0 ? 1 : 2);
                    continue;
                }
                out_col = current_out_col + c_packed;

                if (wi[i + c] == 0.0) {
                    xmax = 0.0;
                    for (r = 0; r < n; r++) {
                        xmax = std::max(xmax, std::abs(VR[r + out_col * ldvr]));
                    }
                    if (xmax > safemin) {
                        xmax = 1.0 / xmax;
                        for (r = 0; r < n; r++) {
                            VR[r + out_col * ldvr] *= xmax;
                        }
                    }
                    c += 1;
                }
                else {
                    xmax = 0.0;
                    for (r = 0; r < n; r++) {
                        xmax = std::max(xmax, std::abs(VR[r + out_col * ldvr]) + std::abs(VR[r + (out_col + 1) * ldvr]));
                    }
                    if (xmax > safemin) {
                        xmax = 1.0 / xmax;
                        for (r = 0; r < n; r++) {
                            VR[r + out_col * ldvr] *= xmax;
                            VR[r + (out_col + 1) * ldvr] *= xmax;
                        }
                    }
                    c += 2;
                }
            }
            curr_col = i;
        }
    }

    // ==================================================================
    // Left Eigenvector Computation
    // ==================================================================
    if (compute_left && VL != nullptr) {
        current_out_col = 0;
        curr_col = 0;

        while (curr_col < n) {
            i_next = idlanb(T, n, ldt, curr_col, bsize);
            nb = i_next - curr_col;
            i = curr_col;
            rem_rows = n - i;
            ld_x = rem_rows;

            nb_sel = 0;
            for (c = 0; c < nb;) {
                selected = do_all;
                if (do_sel && select != nullptr) {
                    selected = select[i + c];
                    if (wi[i + c] != 0.0 && i + c + 1 < n) {
                        selected = (selected || select[i + c + 1]);
                    }
                }

                if (selected) {
                    col_map[c] = nb_sel++;
                    if (wi[i + c] != 0.0 && i + c + 1 < n) {
                        col_map[c + 1] = nb_sel++;
                        c += 2;
                    }
                    else {
                        c += 1;
                    }
                }
                else {
                    col_map[c] = -1;
                    if (wi[i + c] != 0.0 && i + c + 1 < n) {
                        col_map[c + 1] = -1;
                        c += 2;
                    }
                    else {
                        c += 1;
                    }
                }
            }

            if (nb_sel == 0) {
                curr_col = i_next;
                continue;
            }

            for (c = 0; c < nb_sel; c++) {
                for (r = 0; r < rem_rows; r++) {
                    X_panel[r + c * ld_x] = 0.0;
                }
            }

            curr_row = i;
            while (curr_row < n) {
                j_next = idlanb(T, n, ldt, curr_row, bsize);
                j_nb = j_next - curr_row;
                is_diag = (curr_row == i) ? 1 : 0;

                dlatsl(ldt, &T[curr_row + curr_row * ldt], j_nb, ld_x, &X_panel[curr_row - i], X_panel, ld_x, nb, wr + i, wi + i, is_diag, work_local, bignum, col_map, nb_sel);

                if (j_next < n) {
                    update_max = 0.0;
                    for (c_idx = 0; c_idx < nb_sel; ++c_idx) {
                        for (r_idx = 0; r_idx < j_nb; ++r_idx) {
                            update_max = std::max(update_max, std::abs(X_panel[(curr_row - i + r_idx) + c_idx * ld_x]));
                        }
                    }

                    safe_limit = bignum / static_cast<double>(std::max(1, j_nb));
                    if (update_max * anorm > safe_limit) {
                        scale = safe_limit / (update_max * anorm);
                        for (c_idx = 0; c_idx < nb_sel; ++c_idx) {
                            for (r_idx = 0; r_idx < ld_x; ++r_idx) {
                                X_panel[r_idx + c_idx * ld_x] *= scale;
                            }
                        }
                    }

                    rem = n - j_next;
                    alpha_m1 = -1.0;
                    beta_1 = 1.0;
                    dgemm_("T", "N", &rem, &nb_sel, &j_nb, &alpha_m1, const_cast<double *>(&T[curr_row + j_next * ldt]), &ldt, &X_panel[curr_row - i], &ld_x, &beta_1, &X_panel[j_next - i], &ld_x);
                }
                curr_row = j_next;
            }

            if (do_back) {
                one = 1.0;
                zero = 0.0;
                dgemm_("N", "N", &n, &nb_sel, &rem_rows, &one, &VL[i * ldvl], &ldvl, X_panel, &ld_x, &zero, Temp, &n);
                for (c = 0; c < nb_sel; c++) {
                    out_col = current_out_col + c;
                    for (r = 0; r < n; r++) {
                        VL[r + out_col * ldvl] = Temp[r + c * n];
                    }
                }
            }
            else {
                for (c = 0; c < nb_sel; c++) {
                    out_col = current_out_col + c;
                    for (r = 0; r < i; r++) {
                        VL[r + out_col * ldvl] = 0.0;
                    }
                    for (r = 0; r < rem_rows; r++) {
                        VL[(i + r) + out_col * ldvl] = X_panel[r + c * ld_x];
                    }
                }
            }

            for (c = 0; c < nb;) {
                c_packed = col_map[c];
                if (c_packed < 0) {
                    c += (wi[i + c] == 0.0 ? 1 : 2);
                    continue;
                }
                out_col = current_out_col + c_packed;

                if (wi[i + c] == 0.0) {
                    xmax = 0.0;
                    for (r = 0; r < n; r++) {
                        xmax = std::max(xmax, std::abs(VL[r + out_col * ldvl]));
                    }
                    if (xmax > safemin) {
                        xmax = 1.0 / xmax;
                        for (r = 0; r < n; r++) {
                            VL[r + out_col * ldvl] *= xmax;
                        }
                    }
                    c += 1;
                }
                else {
                    xmax = 0.0;
                    for (r = 0; r < n; r++) {
                        xmax = std::max(xmax, std::abs(VL[r + out_col * ldvl]) + std::abs(VL[r + (out_col + 1) * ldvl]));
                    }
                    if (xmax > safemin) {
                        xmax = 1.0 / xmax;
                        for (r = 0; r < n; r++) {
                            VL[r + out_col * ldvl] *= xmax;
                            VL[r + (out_col + 1) * ldvl] *= xmax;
                        }
                    }
                    c += 2;
                }
            }
            current_out_col += nb_sel;
            curr_col = i_next;
        }
    }
}