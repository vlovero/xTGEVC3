#include "helpers.h"
#include <algorithm>
#include <cmath>
#include <limits>

/*
 * =====================================================================
 * Purpose:
 * =======
 * Helper function to determine the start index of a block for backward
 * traversal. For complex matrices, diagonal blocks are purely 1x1,
 * so no block splitting checks are required.
 *
 * Arguments:
 * =========
 * curr    (input) int
 * The current column index.
 *
 * bsize   (input) int
 * The desired block size.
 *
 * Returns:
 * =======
 * int     The starting index for the block.
 * =====================================================================
 */
inline int iclapb(int curr, int bsize)
{
    int result;

    result = std::max(0, curr - bsize);
    return result;
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Helper function to determine the end index of a block for forward
 * traversal. For complex matrices, diagonal blocks are purely 1x1,
 * so no block splitting checks are required.
 *
 * Arguments:
 * =========
 * n       (input) int
 * The order of the matrix S.
 *
 * curr    (input) int
 * The current row index.
 *
 * bsize   (input) int
 * The desired block size.
 *
 * Returns:
 * =======
 * int     The ending index for the block.
 * =====================================================================
 */
inline int iclanb(int n, int curr, int bsize)
{
    int result;

    result = std::min(n, curr + bsize);
    return result;
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves the local panel eigenvalue problem for the right eigenvectors.
 * Resolves the equations (beta*S - alpha*P) * x = rhs for a local block.
 * Incorporates rigorous scaling logic to prevent numeric overflow.
 *
 * Arguments:
 * =========
 * ldS             (input) int
 * The leading dimension of S.
 * S               (input) const std::complex<float>*
 * The block of matrix S.
 * ldP             (input) int
 * The leading dimension of P.
 * P               (input) const std::complex<float>*
 * The block of matrix P.
 * m_size          (input) int
 * Size of the current block solve.
 * ldV             (input) int
 * The leading dimension of the eigenvector arrays.
 * rhs_loc         (input/output) std::complex<float>*
 * Right-hand side matrix for the local solve. Overwritten
 * by the solution.
 * X_panel_base    (input/output) std::complex<float>*
 * Base pointer for the global panel (used for scaling).
 * panel_rows      (input) int
 * Total rows in the working panel.
 * nb              (input) int
 * Number of columns in the block.
 * alpha           (input) const std::complex<float>*
 * Array of complex eigenvalues alpha.
 * beta            (input) const std::complex<float>*
 * The scaling factor array beta for the eigenvalues.
 * is_diag         (input) int
 * Flag indicating if we are on the diagonal block.
 * work            (workspace) std::complex<float>*
 * Local workspace array.
 * ascale, bscale  (input) float
 * Global normalization scales to prevent overflow.
 * safemin, bignum (input) float
 * Machine parameters representing minimum safe and maximum bounds.
 * col_map         (input) const int*
 * Maps block indices to the active working set indices.
 * nb_sel          (input) int
 * Number of actively selected eigenvectors in this block.
 * =====================================================================
 */
void clalsr(int ldS, const std::complex<float> *S, int ldP, const std::complex<float> *P, int m_size, int ldV, std::complex<float> *rhs_loc, std::complex<float> *X_panel_base, int panel_rows, int nb, const std::complex<float> *alpha, const std::complex<float> *beta, int is_diag, std::complex<float> *work, float ascale, float bscale, float safemin, float bignum, const int *col_map, int nb_sel)
{
    int k, cur_m, c, r, c_scale, r_scale, c_packed;
    std::complex<float> a, b_val, acoeff, bcoeff;
    float t, rhs_max, scale;
    std::complex<float> *work_rhs;

    std::complex<float> one;
    int one_int;
    char side, uplo, transa, diag;

    one = std::complex<float>(1.0f, 0.0f);
    one_int = 1;
    side = 'L';
    uplo = 'U';
    transa = 'N';
    diag = 'N';

    for (k = 0; k < nb; k++) {
        c_packed = col_map[k];

        // Skip non-selected eigenvalues
        if (c_packed < 0) {
            continue;
        }

        a = alpha[k];
        b_val = beta[k];
        cur_m = is_diag ? k : m_size;

        // Calculate a local scaling factor 't' so that evaluating the local
        // operator (beta * S - alpha * P) does not overflow.
        t = 1.0f / std::max({ std::abs(a) * ascale, std::abs(b_val) * bscale, safemin });
        acoeff = std::complex<float>(t * ascale, 0.0f) * b_val * std::complex<float>(bscale, 0.0f);
        bcoeff = std::complex<float>(t * bscale, 0.0f) * a * std::complex<float>(ascale, 0.0f);

        // Limit handling for infinite eigenvalues
        if (std::abs(b_val) <= safemin && std::abs(a) > safemin) {
            acoeff = { 0.0f, 0.0f };
            bcoeff = { 1.0f, 0.0f };
        }

        if (is_diag) {
            rhs_loc[k + c_packed * ldV] = { 1.0f, 0.0f };
        }

        if (cur_m > 0) {
            // Form the scaled local operator matrix
            for (c = 0; c < cur_m; c++) {
                for (r = 0; r <= c; r++) {
                    work[r + c * cur_m] = acoeff * S[r + c * ldS] - bcoeff * P[r + c * ldP];
                }
            }

            // Prepare the right-hand side for the local solve
            work_rhs = work + cur_m * cur_m;
            if (is_diag) {
                for (r = 0; r < cur_m; r++) {
                    work_rhs[r] = -(acoeff * S[r + k * ldS] - bcoeff * P[r + k * ldP]);
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
                rhs_max = std::max(rhs_max, std::abs(work_rhs[r].real()) + std::abs(work_rhs[r].imag()));
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

            // Solve the local triangular system
            ctrsm_(&side, &uplo, &transa, &diag, &cur_m, &one_int, &one, work, &cur_m, work_rhs, &cur_m);

            // Write the solution back to the block
            for (r = 0; r < cur_m; r++) {
                rhs_loc[r + c_packed * ldV] = work_rhs[r];
            }
        }
    }
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves the local panel eigenvalue problem for the left eigenvectors.
 * Handles equations corresponding to y^H * (beta*S - alpha*P) = 0.
 * Operates forward across the upper triangular representations (transposed).
 * Incorporates scaling to prevent overflow during formulation and solve.
 *
 * Arguments:
 * =========
 * ldS             (input) int
 * The leading dimension of S.
 * S               (input) const std::complex<float>*
 * The block of matrix S.
 * ldP             (input) int
 * The leading dimension of P.
 * P               (input) const std::complex<float>*
 * The block of matrix P.
 * m_size          (input) int
 * Size of the current block solve.
 * ldV             (input) int
 * The leading dimension of the eigenvector arrays.
 * rhs_loc         (input/output) std::complex<float>*
 * Right-hand side matrix for the local solve.
 * X_panel_base    (input/output) std::complex<float>*
 * Base pointer for the global panel (used for scaling).
 * panel_rows      (input) int
 * Total rows in the working panel.
 * nb              (input) int
 * Number of columns in the block.
 * alpha           (input) const std::complex<float>*
 * Array of complex eigenvalues alpha.
 * beta            (input) const std::complex<float>*
 * The scaling factor array beta for the eigenvalues.
 * is_diag         (input) int
 * Flag indicating if we are on the diagonal block.
 * work            (workspace) std::complex<float>*
 * Local workspace array.
 * ascale, bscale  (input) float
 * Global normalization scales to prevent overflow.
 * safemin, bignum (input) float
 * Machine parameters for safe limit checking.
 * col_map         (input) const int*
 * Maps block indices to active working set indices.
 * nb_sel          (input) int
 * Number of actively selected eigenvectors in this block.
 * =====================================================================
 */
void clalsl(int ldS, const std::complex<float> *S, int ldP, const std::complex<float> *P, int m_size, int ldV, std::complex<float> *rhs_loc, std::complex<float> *X_panel_base, int panel_rows, int nb, const std::complex<float> *alpha, const std::complex<float> *beta, int is_diag, std::complex<float> *work, float ascale, float bscale, float safemin, float bignum, const int *col_map, int nb_sel)
{
    int k, cur_m, row_offset, c, r, c_scale, r_scale, c_packed;
    std::complex<float> a, b_val, acoeff, bcoeff, sr1, pr1;
    float t, rhs_max, scale;
    std::complex<float> *work_rhs;

    std::complex<float> one;
    int one_int;
    char side, uplo, transa, diag;

    one = std::complex<float>(1.0f, 0.0f);
    one_int = 1;
    side = 'L';
    uplo = 'U';
    transa = 'C';
    diag = 'N';

    for (k = 0; k < nb; k++) {
        c_packed = col_map[k];

        // Skip non-selected eigenvalues
        if (c_packed < 0) {
            continue;
        }

        a = alpha[k];
        b_val = beta[k];
        cur_m = is_diag ? nb - 1 - k : m_size;
        row_offset = is_diag ? k + 1 : 0;

        // Compute scaling to ensure that evaluating the local operator
        // strictly avoids NaN or Inf propagation.
        t = 1.0f / std::max({ std::abs(a) * ascale, std::abs(b_val) * bscale, safemin });
        acoeff = std::complex<float>(t * ascale, 0.0f) * b_val * std::complex<float>(bscale, 0.0f);
        bcoeff = std::complex<float>(t * bscale, 0.0f) * a * std::complex<float>(ascale, 0.0f);

        // Limit handling for infinite eigenvalues
        if (std::abs(b_val) <= safemin && std::abs(a) > safemin) {
            acoeff = { 0.0f, 0.0f };
            bcoeff = { 1.0f, 0.0f };
        }

        if (is_diag) {
            rhs_loc[k + c_packed * ldV] = { 1.0f, 0.0f };
        }

        if (cur_m > 0) {
            // Populate the local scaled operator matrix
            for (c = 0; c < cur_m; c++) {
                for (r = 0; r <= c; r++) {
                    work[r + c * cur_m] = acoeff * S[(r + row_offset) + (c + row_offset) * ldS] - bcoeff * P[(r + row_offset) + (c + row_offset) * ldP];
                }
            }

            work_rhs = work + cur_m * cur_m;
            if (is_diag) {
                for (r = 0; r < cur_m; r++) {
                    sr1 = S[k + (r + row_offset) * ldS];
                    pr1 = P[k + (r + row_offset) * ldP];
                    work_rhs[r] = -std::conj(acoeff * sr1 - bcoeff * pr1);
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
                rhs_max = std::max(rhs_max, std::abs(work_rhs[r].real()) + std::abs(work_rhs[r].imag()));
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

            // Solve the local triangular system (conjugate transpose)
            ctrsm_(&side, &uplo, &transa, &diag, &cur_m, &one_int, &one, work, &cur_m, work_rhs, &cur_m);

            for (r = 0; r < cur_m; r++) {
                rhs_loc[(r + row_offset) + c_packed * ldV] = work_rhs[r];
            }
        }
    }
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * CTGEVC3 computes some or all of the right and/or left eigenvectors of
 * a pair of complex matrices (S,P), where S is an upper triangular matrix
 * and P is upper triangular. Matrix pairs of this type are produced by
 * the generalized Schur factorization of a matrix pair (A,B).
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
 * The order of the matrices S and P.
 *
 * S               (input) const std::complex<float>*
 * The upper triangular matrix S.
 *
 * lds             (input) int
 * The leading dimension of the array S.
 *
 * P               (input) const std::complex<float>*
 * The upper triangular matrix P.
 *
 * ldp             (input) int
 * The leading dimension of the array P.
 *
 * alpha           (input) const std::complex<float>*
 * Array of complex eigenvalues alpha.
 *
 * beta            (input) const std::complex<float>*
 * Array of complex scalars beta defining eigenvalues.
 *
 * VL              (input/output) std::complex<float>*
 * Left eigenvector storage.
 *
 * ldvl            (input) int
 * Leading dimension of array VL.
 *
 * VR              (input/output) std::complex<float>*
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
 * work            (workspace) std::complex<float>*
 * Workspace array.
 *
 * lwork           (input) int
 * Dimension of WORK. -1 for workspace query.
 *
 * info            (output) int*
 * Exit code: 0 on success, <0 for invalid args.
 * =====================================================================
 */
void ctgevc3(char side, char howmny, const int *select, int n, const std::complex<float> *S, int lds, const std::complex<float> *P, int ldp, const std::complex<float> *alpha, const std::complex<float> *beta, std::complex<float> *VL, int ldvl, std::complex<float> *VR, int ldvr, int mm, int *m, std::complex<float> *work, int lwork, int *info)
{
    bool compute_right, compute_left, do_all, do_back, do_sel;
    int num_sel, k_idx, bsize, req_lwork;
    float safemin, eps, smlnum, bignum;
    float anorm, bnorm, sum_s, sum_p;
    int col, row;
    float ascale, bscale;
    std::complex<float> *X_panel, *Temp, *work_local;
    int col_map[128];
    std::complex<float> alpha_m1, beta_1, alpha_1, one, zero;
    int current_out_col, curr_col, i, nb, ld_x, nb_sel, c, r;
    int curr_row, j, j_nb, is_diag, out_col, c_packed;
    std::complex<float> *TempS, *TempP;
    std::complex<float> a, b_val, acoeff, bcoeff, xr;
    float t, update_max, safe_limit, scale;
    int total_elements, idx, c_idx, r_idx;
    int i_next, rem_rows, j_next, rem;

    // Decode and validate parameters
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

    // Pass 1: Count total selected eigenvalues without allocating storage.
    num_sel = 0;
    for (k_idx = 0; k_idx < n; k_idx++) {
        if (do_all || (do_sel && select && select[k_idx])) {
            num_sel++;
        }
    }

    if (m != nullptr) {
        *m = num_sel;
    }
    if (mm < num_sel) {
        *info = -16;
        return;
    }

    bsize = 32; // Default ideal block size for Level 3 BLAS optimization
    req_lwork = 2 * n * (bsize + 1) + 4 * (bsize + 1) * (bsize + 1) + 2 * (bsize + 1);

    // Dynamic block size fallback. If the provided memory is insufficient
    // for bsize = 32, shrink the block size to fit the constraints.
    if (lwork != -1 && lwork < req_lwork) {
        for (bsize = 63; bsize >= 1; bsize--) {
            req_lwork = 2 * n * (bsize + 1) + 4 * (bsize + 1) * (bsize + 1) + 2 * (bsize + 1);
            if (lwork >= req_lwork) {
                break;
            }
        }
    }

    // Workspace query branch
    if (lwork == -1) {
        work[0] = std::complex<float>(static_cast<float>(req_lwork), 0.0f);
        *info = 0;
        return;
    }

    if (lwork < req_lwork) {
        *info = -19;
        return;
    }

    *info = 0;

    // Retrieve machine constants for evaluating safe scaling boundaries
    safemin = std::numeric_limits<float>::min();
    eps = std::numeric_limits<float>::epsilon();
    smlnum = safemin / eps;
    bignum = 1.0f / smlnum;

    // Compute the 1-norm of S and P to generate global scaling factors ascale
    // and bscale. This ensures combinations of S and P remain properly bounded.
    anorm = 0.0f;
    bnorm = 0.0f;
    for (col = 0; col < n; ++col) {
        sum_s = 0.0f;
        sum_p = 0.0f;
        for (row = 0; row <= col; ++row) {
            sum_s += std::abs(S[row + col * lds]);
            sum_p += std::abs(P[row + col * ldp]);
        }
        anorm = std::max(anorm, sum_s);
        bnorm = std::max(bnorm, sum_p);
    }

    ascale = 1.0f / std::max(anorm, safemin);
    bscale = 1.0f / std::max(bnorm, safemin);

    // Partition the allocated workspace into the active vector panel,
    // intermediate temp buffers, and local solver workspaces.
    X_panel = work;
    Temp = X_panel + n * (bsize + 1);
    work_local = Temp + n * (bsize + 1);

    alpha_m1 = std::complex<float>(-1.0f, 0.0f);
    beta_1 = std::complex<float>(1.0f, 0.0f);
    alpha_1 = std::complex<float>(1.0f, 0.0f);
    one = std::complex<float>(1.0f, 0.0f);
    zero = std::complex<float>(0.0f, 0.0f);

    // ==================================================================
    // Right Eigenvector Computation
    // Proceed backward from the bottom-right of the matrix to the top-left
    // ==================================================================
    if (compute_right && VR != nullptr) {
        current_out_col = num_sel;
        curr_col = n;

        while (curr_col > 0) {
            i = iclapb(curr_col, bsize);
            nb = curr_col - i;
            ld_x = curr_col;

            nb_sel = 0;
            // Build the block to active subset column map
            for (c = 0; c < nb; c++) {
                if (do_all || (do_sel && select && select[i + c])) {
                    col_map[c] = nb_sel++;
                }
                else {
                    col_map[c] = -1;
                }
            }

            current_out_col -= nb_sel;
            if (nb_sel == 0) {
                curr_col = i;
                continue;
            }

            // Zero out the active matrix panel slice before beginning updates
            for (c = 0; c < nb_sel; c++) {
                for (r = 0; r < curr_col; r++) {
                    X_panel[r + c * ld_x] = { 0.0f, 0.0f };
                }
            }

            // Iterate sequentially backward up the matrix to resolve components
            curr_row = curr_col;
            while (curr_row > 0) {
                j = iclapb(curr_row, bsize);
                j_nb = curr_row - j;
                is_diag = (curr_row == curr_col) ? 1 : 0;

                // Obtain the local piece of the eigenvector matrix
                clalsr(lds, &S[j + j * lds], ldp, &P[j + j * ldp], j_nb, ld_x, &X_panel[j], X_panel, ld_x, nb, alpha + i, beta + i, is_diag, work_local, ascale, bscale, safemin, bignum, col_map, nb_sel);

                if (j > 0) {
                    TempS = work_local;
                    TempP = work_local + j_nb * nb_sel;

                    // Form the scaled matrix components that act as right-side
                    // inputs for the upcoming Level 3 BLAS operations
                    for (c = 0; c < nb; c++) {
                        c_packed = col_map[c];
                        if (c_packed < 0) {
                            continue;
                        }

                        a = alpha[i + c];
                        b_val = beta[i + c];
                        t = 1.0f / std::max({ std::abs(a) * ascale, std::abs(b_val) * bscale, safemin });
                        acoeff = std::complex<float>(t * ascale, 0.0f) * b_val * std::complex<float>(bscale, 0.0f);
                        bcoeff = std::complex<float>(t * bscale, 0.0f) * a * std::complex<float>(ascale, 0.0f);

                        if (std::abs(b_val) <= safemin && std::abs(a) > safemin) {
                            acoeff = { 0.0f, 0.0f };
                            bcoeff = { 1.0f, 0.0f };
                        }

                        for (r = 0; r < j_nb; r++) {
                            xr = X_panel[(j + r) + c_packed * ld_x];
                            TempS[r + c_packed * j_nb] = acoeff * xr;
                            TempP[r + c_packed * j_nb] = bcoeff * xr;
                        }
                    }

                    // Before running BLAS updates, ensure the internal matrix inner products
                    // will not trigger numeric overflow boundaries.
                    update_max = 0.0f;
                    total_elements = j_nb * nb_sel;
                    for (idx = 0; idx < total_elements; ++idx) {
                        update_max = std::max({ update_max, std::abs(TempS[idx].real()) + std::abs(TempS[idx].imag()), std::abs(TempP[idx].real()) + std::abs(TempP[idx].imag()) });
                    }

                    safe_limit = bignum / static_cast<float>(std::max(1, nb_sel));
                    if (update_max > safe_limit) {
                        scale = safe_limit / update_max;
                        for (idx = 0; idx < total_elements; ++idx) {
                            TempS[idx] *= scale;
                            TempP[idx] *= scale;
                        }
                        for (c_idx = 0; c_idx < nb_sel; ++c_idx) {
                            for (r_idx = 0; r_idx < ld_x; ++r_idx) {
                                X_panel[r_idx + c_idx * ld_x] *= scale;
                            }
                        }
                    }

                    // Perform block update accumulating results into X_panel via CGEMM
                    cgemm_("N", "N", &j, &nb_sel, &j_nb, &alpha_m1, &S[0 + j * lds], &lds, TempS, &j_nb, &beta_1, &X_panel[0], &ld_x);
                    cgemm_("N", "N", &j, &nb_sel, &j_nb, &alpha_1, &P[0 + j * ldp], &ldp, TempP, &j_nb, &beta_1, &X_panel[0], &ld_x);
                }
                curr_row = j;
            }

            // Copy results back to the VR matrix.
            // If do_back is true, the vectors require backtransformation via matrix multiplication.
            if (do_back) {
                cgemm_("N", "N", &n, &nb_sel, &curr_col, &one, VR, &ldvr, X_panel, &ld_x, &zero, Temp, &n);
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
                        VR[r + out_col * ldvr] = { 0.0f, 0.0f };
                    }
                }
            }
            curr_col = i;
        }
    }

    // ==================================================================
    // Left Eigenvector Computation
    // Proceed forward from the top-left of the matrix to the bottom-right
    // ==================================================================
    if (compute_left && VL != nullptr) {
        current_out_col = 0;
        curr_col = 0;

        while (curr_col < n) {
            i_next = iclanb(n, curr_col, bsize);
            nb = i_next - curr_col;
            i = curr_col;
            rem_rows = n - i;
            ld_x = rem_rows;

            nb_sel = 0;
            // Build the block to active subset column map
            for (c = 0; c < nb; c++) {
                if (do_all || (do_sel && select && select[i + c])) {
                    col_map[c] = nb_sel++;
                }
                else {
                    col_map[c] = -1;
                }
            }

            if (nb_sel == 0) {
                curr_col = i_next;
                continue;
            }

            // Initialize active submatrix panel to zeroes
            for (c = 0; c < nb_sel; c++) {
                for (r = 0; r < rem_rows; r++) {
                    X_panel[r + c * ld_x] = { 0.0f, 0.0f };
                }
            }

            // Process forward down the matrix representations
            curr_row = i;
            while (curr_row < n) {
                j_next = iclanb(n, curr_row, bsize);
                j_nb = j_next - curr_row;
                is_diag = (curr_row == i) ? 1 : 0;

                // Obtain the local piece of the left eigenvector matrix
                clalsl(lds, &S[curr_row + curr_row * lds], ldp, &P[curr_row + curr_row * ldp], j_nb, ld_x, &X_panel[curr_row - i], X_panel, ld_x, nb, alpha + i, beta + i, is_diag, work_local, ascale, bscale, safemin, bignum, col_map, nb_sel);

                if (j_next < n) {
                    TempS = work_local;
                    TempP = work_local + j_nb * nb_sel;

                    // Compute scaled input vectors for downward block GEMM
                    for (c = 0; c < nb; c++) {
                        c_packed = col_map[c];
                        if (c_packed < 0) {
                            continue;
                        }

                        a = alpha[i + c];
                        b_val = beta[i + c];
                        t = 1.0f / std::max({ std::abs(a) * ascale, std::abs(b_val) * bscale, safemin });
                        acoeff = std::complex<float>(t * ascale, 0.0f) * b_val * std::complex<float>(bscale, 0.0f);
                        bcoeff = std::complex<float>(t * bscale, 0.0f) * a * std::complex<float>(ascale, 0.0f);

                        if (std::abs(b_val) <= safemin && std::abs(a) > safemin) {
                            acoeff = { 0.0f, 0.0f };
                            bcoeff = { 1.0f, 0.0f };
                        }

                        for (r = 0; r < j_nb; r++) {
                            xr = X_panel[(curr_row - i + r) + c_packed * ld_x];
                            // Conjugate applied for the left eigenvector solver
                            TempS[r + c_packed * j_nb] = std::conj(acoeff) * xr;
                            TempP[r + c_packed * j_nb] = std::conj(bcoeff) * xr;
                        }
                    }

                    rem = n - j_next;
                    // Shield updating matrix product from overflow situations
                    update_max = 0.0f;
                    total_elements = j_nb * nb_sel;
                    for (idx = 0; idx < total_elements; ++idx) {
                        update_max = std::max({ update_max, std::abs(TempS[idx].real()) + std::abs(TempS[idx].imag()), std::abs(TempP[idx].real()) + std::abs(TempP[idx].imag()) });
                    }

                    safe_limit = bignum / static_cast<float>(std::max(1, nb_sel));
                    if (update_max > safe_limit) {
                        scale = safe_limit / update_max;
                        for (idx = 0; idx < total_elements; ++idx) {
                            TempS[idx] *= scale;
                            TempP[idx] *= scale;
                        }
                        for (c_idx = 0; c_idx < nb_sel; ++c_idx) {
                            for (r_idx = 0; r_idx < ld_x; ++r_idx) {
                                X_panel[r_idx + c_idx * ld_x] *= scale;
                            }
                        }
                    }

                    // Level 3 BLAS sequence to push local outcomes downwards
                    cgemm_("C", "N", &rem, &nb_sel, &j_nb, &alpha_m1, &S[curr_row + j_next * lds], &lds, TempS, &j_nb, &beta_1, &X_panel[j_next - i], &ld_x);
                    cgemm_("C", "N", &rem, &nb_sel, &j_nb, &alpha_1, &P[curr_row + j_next * ldp], &ldp, TempP, &j_nb, &beta_1, &X_panel[j_next - i], &ld_x);
                }
                curr_row = j_next;
            }

            // Transfer result back to the user matrix VL. Apply base transformation
            // via cgemm if computing original coordinates (howmny = 'B')
            if (do_back) {
                cgemm_("N", "N", &n, &nb_sel, &rem_rows, &one, &VL[i * ldvl], &ldvl, X_panel, &ld_x, &zero, Temp, &n);
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
                        VL[r + out_col * ldvl] = { 0.0f, 0.0f };
                    }
                    for (r = 0; r < rem_rows; r++) {
                        VL[(i + r) + out_col * ldvl] = X_panel[r + c * ld_x];
                    }
                }
            }
            current_out_col += nb_sel;
            curr_col = i_next;
        }
    }
}