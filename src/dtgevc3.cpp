#include "helpers.h"
#include <algorithm>
#include <cmath>

/*
 * =====================================================================
 * Purpose:
 * =======
 * Helper function to determine the start index of a block for backward
 * traversal. It ensures that a 2x2 diagonal block (which corresponds to
 * a complex conjugate eigenvalue pair) is not split across boundaries.
 *
 * Arguments:
 * =========
 * S       (input) const double*
 * The upper quasi-triangular matrix S.
 *
 * lds     (input) int
 * The leading dimension of the array S.
 *
 * curr    (input) int
 * The current column index.
 *
 * bsize   (input) int
 * The desired block size.
 *
 * Returns:
 * =======
 * int     The adjusted starting index for the block.
 * =====================================================================
 */
inline int idlapb(const double *S, int lds, int curr, int bsize)
{
    int idx;

    idx = std::max(0, curr - bsize);

    // Check if the proposed boundary splits a 2x2 block.
    // S[idx + (idx - 1)*lds] is the subdiagonal element. If it is non-zero,
    // idx and idx-1 belong to the same block, so we decrement to keep them together.
    if (idx > 0 && S[idx + (idx - 1) * lds] != 0.0) {
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
 *
 * Arguments:
 * =========
 * S       (input) const double*
 * The upper quasi-triangular matrix S.
 *
 * n       (input) int
 * The order of the matrix S.
 *
 * lds     (input) int
 * The leading dimension of the array S.
 *
 * curr    (input) int
 * The current row index.
 *
 * bsize   (input) int
 * The desired block size.
 *
 * Returns:
 * =======
 * int     The adjusted ending index for the block.
 * =====================================================================
 */
inline int idlanb(const double *S, int n, int lds, int curr, int bsize)
{
    int idx;

    idx = std::min(n, curr + bsize);

    // Check if the boundary splits a 2x2 block.
    // If the subdiagonal element at the boundary is non-zero, increment
    // the index to include the rest of the 2x2 block.
    if (idx < n && S[idx + (idx - 1) * lds] != 0.0) {
        idx += 1;
    }

    return idx;
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves a linear system A * X = B where A is an upper Hessenberg matrix
 * using Gaussian elimination with partial pivoting.
 *
 * Arguments:
 * =========
 * n       (input) int
 * The order of the matrix A.
 *
 * nrhs    (input) int
 * The number of right hand sides.
 *
 * A       (input/output) double*
 * On entry, the upper Hessenberg matrix A.
 * On exit, the upper triangular matrix after elimination.
 *
 * lda     (input) int
 * The leading dimension of the array A.
 *
 * B       (input/output) double*
 * On entry, the right hand side matrix B.
 * On exit, the solution matrix X.
 *
 * ldb     (input) int
 * The leading dimension of the array B.
 *
 * Returns:
 * =======
 * int     0 on success.
 * > 0 if A is singular (returns the 1-based index of the zero pivot).
 * < 0 if an argument is invalid.
 * =====================================================================
 */
int dlauhs(int n, int nrhs, double *A, int lda, double *B, int ldb)
{
    int j, k, pivot_row;
    double mult, alpha;
    char side, uplo, transa, diag;

    if (n < 0) {
        return -1;
    }
    if (nrhs < 0) {
        return -2;
    }
    if (lda < std::max(1, n)) {
        return -4;
    }
    if (ldb < std::max(1, n)) {
        return -6;
    }
    if (n == 0) {
        return 0;
    }

    // Forward elimination: traverse columns to eliminate subdiagonal elements
    for (j = 0; j < n - 1; j++) {

        // Partial pivoting: Select the largest element between A(j,j) and A(j+1,j)
        pivot_row = j;
        if (std::abs(A[(j + 1) + j * lda]) > std::abs(A[j + j * lda])) {
            pivot_row = j + 1;
        }

        // Return early if the exact zero pivot indicates a singular matrix
        if (A[pivot_row + j * lda] == 0.0) {
            return j + 1;
        }

        // Apply row swaps to both the operator A and the right-hand side B
        if (pivot_row != j) {
            for (k = j; k < n; k++) {
                std::swap(A[j + k * lda], A[pivot_row + k * lda]);
            }
            for (k = 0; k < nrhs; k++) {
                std::swap(B[j + k * ldb], B[pivot_row + k * ldb]);
            }
        }

        // Compute the multiplier to eliminate the subdiagonal element A(j+1, j)
        mult = A[(j + 1) + j * lda] / A[j + j * lda];
        A[(j + 1) + j * lda] = 0.0;

        // Apply the multiplier to the remaining elements in the row
        for (k = j + 1; k < n; k++) {
            A[(j + 1) + k * lda] -= mult * A[j + k * lda];
        }
        for (k = 0; k < nrhs; k++) {
            B[(j + 1) + k * ldb] -= mult * B[j + k * ldb];
        }
    }

    // Check the final diagonal element for singularity
    if (A[(n - 1) + (n - 1) * lda] == 0.0) {
        return n;
    }

    // The system is now upper triangular. Solve via TRSM.
    if (nrhs > 0) {
        side = 'L';
        uplo = 'U';
        transa = 'N';
        diag = 'N';
        alpha = 1.0;
        dtrsm_(&side, &uplo, &transa, &diag, &n, &nrhs, &alpha, A, &lda, B, &ldb);
    }

    return 0;
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves a linear system A * X = B with an upper Hessenberg matrix A
 * having TWO subdiagonals using Gaussian elimination with partial pivoting.
 * This arises when solving for complex eigenvector pairs.
 *
 * Arguments:
 * =========
 * n       (input) int
 * The order of the matrix A.
 *
 * nrhs    (input) int
 * The number of right hand sides.
 *
 * A       (input/output) double*
 * On entry, the matrix A with 2 subdiagonals.
 * On exit, the upper triangular matrix after elimination.
 *
 * lda     (input) int
 * The leading dimension of the array A.
 *
 * B       (input/output) double*
 * On entry, the right hand side matrix B.
 * On exit, the solution matrix X.
 *
 * ldb     (input) int
 * The leading dimension of the array B.
 *
 * Returns:
 * =======
 * int     0 on success.
 * > 0 if A is singular (returns the 1-based index of the zero pivot).
 * < 0 if an argument is invalid.
 * =====================================================================
 */
int dlau2s(int n, int nrhs, double *A, int lda, double *B, int ldb)
{
    int j, k, pivot_row;
    double max_val, mult1, mult2, alpha;
    char side, uplo, transa, diag;

    if (n < 0) {
        return -1;
    }
    if (nrhs < 0) {
        return -2;
    }
    if (lda < std::max(1, n)) {
        return -4;
    }
    if (ldb < std::max(1, n)) {
        return -6;
    }
    if (n == 0) {
        return 0;
    }

    // Forward elimination: process both subdiagonals
    for (j = 0; j < n - 1; j++) {

        // Partial pivoting across the diagonal and the two elements below it
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

        if (max_val == 0.0) {
            return j + 1;
        }

        // Swap the selected pivot row into the current row position
        if (pivot_row != j) {
            for (k = j; k < n; k++) {
                std::swap(A[j + k * lda], A[pivot_row + k * lda]);
            }
            for (k = 0; k < nrhs; k++) {
                std::swap(B[j + k * ldb], B[pivot_row + k * ldb]);
            }
        }

        // Eliminate the first subdiagonal element
        mult1 = A[(j + 1) + j * lda] / A[j + j * lda];
        A[(j + 1) + j * lda] = 0.0;

        for (k = j + 1; k < n; k++) {
            A[(j + 1) + k * lda] -= mult1 * A[j + k * lda];
        }
        for (k = 0; k < nrhs; k++) {
            B[(j + 1) + k * ldb] -= mult1 * B[j + k * ldb];
        }

        // Eliminate the second subdiagonal element, if it is within bounds
        if (j + 2 < n) {
            mult2 = A[(j + 2) + j * lda] / A[j + j * lda];
            A[(j + 2) + j * lda] = 0.0;

            for (k = j + 1; k < n; k++) {
                A[(j + 2) + k * lda] -= mult2 * A[j + k * lda];
            }
            for (k = 0; k < nrhs; k++) {
                B[(j + 2) + k * ldb] -= mult2 * B[j + k * ldb];
            }
        }
    }

    if (A[(n - 1) + (n - 1) * lda] == 0.0) {
        return n;
    }

    // Solve the resulting upper triangular system
    if (nrhs > 0) {
        side = 'L';
        uplo = 'U';
        transa = 'N';
        diag = 'N';
        alpha = 1.0;
        dtrsm_(&side, &uplo, &transa, &diag, &n, &nrhs, &alpha, A, &lda, B, &ldb);
    }

    return 0;
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves a linear system A * X = B with a lower Hessenberg matrix A
 * (arising in left eigenvector computations) using Gaussian elimination
 * with partial pivoting.
 *
 * Arguments:
 * =========
 * n       (input) int
 * The order of the matrix A.
 *
 * nrhs    (input) int
 * The number of right hand sides.
 *
 * A       (input/output) double*
 * On entry, the lower Hessenberg matrix A.
 * On exit, the lower triangular matrix after elimination.
 *
 * lda     (input) int
 * The leading dimension of the array A.
 *
 * jpiv    (output) int*
 * Integer array tracking the pivot permutations.
 *
 * B       (input/output) double*
 * On entry, the right hand side matrix B.
 * On exit, the solution matrix X.
 *
 * ldb     (input) int
 * The leading dimension of the array B.
 *
 * Returns:
 * =======
 * int     0 on success.
 * > 0 if A is singular (returns the 1-based index of the zero pivot).
 * =====================================================================
 */
int dlalhs(int n, int nrhs, double *A, int lda, int *jpiv, double *B, int ldb)
{
    int k, p, i, c;
    double max_val, pivot, m, alpha;
    char side, uplo, transa, diag;

    // Forward elimination: process the single superdiagonal
    for (k = 0; k < n - 1; k++) {

        // Find the maximum element for partial pivoting
        p = k;
        max_val = std::abs(A[k + k * lda]);

        if (std::abs(A[k + (k + 1) * lda]) > max_val) {
            p = k + 1;
        }

        // Store the pivot index so it can be applied to B after solving A
        jpiv[k] = p;

        // Swap columns if necessary
        if (p != k) {
            for (i = k; i < n; i++) {
                std::swap(A[i + k * lda], A[i + p * lda]);
            }
        }

        pivot = A[k + k * lda];
        if (pivot == 0.0) {
            return k + 1;
        }

        // Compute multiplier and eliminate the superdiagonal element
        m = A[k + (k + 1) * lda] / pivot;
        A[k + (k + 1) * lda] = m;

        // Apply multiplier to the rest of the column
        for (i = k + 1; i < n; i++) {
            A[i + (k + 1) * lda] -= m * A[i + k * lda];
        }
    }

    // Solve the resulting lower triangular system A * X = B
    side = 'L';
    uplo = 'L';
    transa = 'N';
    diag = 'N';
    alpha = 1.0;
    dtrsm_(&side, &uplo, &transa, &diag, &n, &nrhs, &alpha, A, &lda, B, &ldb);

    // Backward permutation update: Apply the pivoting history to the solution
    for (k = n - 2; k >= 0; --k) {
        m = A[k + (k + 1) * lda];
        p = jpiv[k];
        for (c = 0; c < nrhs; c++) {
            B[k + c * ldb] -= m * B[(k + 1) + c * ldb];
            if (p != k) {
                std::swap(B[k + c * ldb], B[p + c * ldb]);
            }
        }
    }

    return 0;
}

/*
 * =====================================================================
 * Purpose:
 * =======
 * Solves a linear system A * X = B with a lower Hessenberg matrix A
 * having TWO superdiagonals using Gaussian elimination with partial pivoting.
 *
 * Arguments:
 * =========
 * n       (input) int
 * The order of the matrix A.
 *
 * nrhs    (input) int
 * The number of right hand sides.
 *
 * A       (input/output) double*
 * On entry, the matrix A with 2 superdiagonals.
 * On exit, the lower triangular matrix after elimination.
 *
 * lda     (input) int
 * The leading dimension of the array A.
 *
 * jpiv    (output) int*
 * Integer array tracking the pivot permutations.
 *
 * B       (input/output) double*
 * On entry, the right hand side matrix B.
 * On exit, the solution matrix X.
 *
 * ldb     (input) int
 * The leading dimension of the array B.
 *
 * Returns:
 * =======
 * int     0 on success.
 * > 0 if A is singular (returns the 1-based index of the zero pivot).
 * =====================================================================
 */
int dlal2s(int n, int nrhs, double *A, int lda, int *jpiv, double *B, int ldb)
{
    int k, p, i, c;
    double max_val, pivot, m1, m2, alpha;
    char side, uplo, transa, diag;

    // Forward elimination: process both superdiagonals
    for (k = 0; k < n - 1; k++) {

        // Find the maximum element across the current column and the two superdiagonals
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
            for (i = k; i < n; i++) {
                std::swap(A[i + k * lda], A[i + p * lda]);
            }
        }

        pivot = A[k + k * lda];
        if (pivot == 0.0) {
            return k + 1;
        }

        // Eliminate the first superdiagonal
        m1 = A[k + (k + 1) * lda] / pivot;
        A[k + (k + 1) * lda] = m1;
        for (i = k + 1; i < n; i++) {
            A[i + (k + 1) * lda] -= m1 * A[i + k * lda];
        }

        // Eliminate the second superdiagonal if it exists
        if (k + 2 < n) {
            m2 = A[k + (k + 2) * lda] / pivot;
            A[k + (k + 2) * lda] = m2;
            for (i = k + 1; i < n; i++) {
                A[i + (k + 2) * lda] -= m2 * A[i + k * lda];
            }
        }
    }

    // Solve the lower triangular system
    side = 'L';
    uplo = 'L';
    transa = 'N';
    diag = 'N';
    alpha = 1.0;
    dtrsm_(&side, &uplo, &transa, &diag, &n, &nrhs, &alpha, A, &lda, B, &ldb);

    // Apply the pivoting and multiplier updates back to the solution
    for (k = n - 2; k >= 0; --k) {
        m1 = A[k + (k + 1) * lda];
        m2 = (k + 2 < n) ? A[k + (k + 2) * lda] : 0.0;
        p = jpiv[k];

        for (c = 0; c < nrhs; c++) {
            B[k + c * ldb] -= m1 * B[(k + 1) + c * ldb];
            if (k + 2 < n) {
                B[k + c * ldb] -= m2 * B[(k + 2) + c * ldb];
            }
            if (p != k) {
                std::swap(B[k + c * ldb], B[p + c * ldb]);
            }
        }
    }

    return 0;
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
 * S               (input) const double*
 * The block of matrix S.
 * ldP             (input) int
 * The leading dimension of P.
 * P               (input) const double*
 * The block of matrix P.
 * m_size          (input) int
 * Size of the current block solve.
 * ldV             (input) int
 * The leading dimension of the eigenvector arrays.
 * rhs_loc         (input/output) double*
 * Right-hand side matrix for the local solve. Overwritten
 * by the solution.
 * X_panel_base    (input/output) double*
 * Base pointer for the global panel (used for scaling).
 * panel_rows      (input) int
 * Total rows in the working panel.
 * nb              (input) int
 * Number of columns in the block.
 * alphar, alphai  (input) const double*
 * Real and imaginary parts of the eigenvalues alpha.
 * beta            (input) const double*
 * The scaling factor beta for the eigenvalues.
 * is_diag         (input) int
 * Flag indicating if we are on the diagonal block.
 * work            (workspace) double*
 * Local workspace array.
 * ascale, bscale  (input) double
 * Global normalization scales to prevent overflow.
 * safemin, bignum (input) double
 * Machine parameters representing minimum safe and maximum bounds.
 * col_map         (input) const int*
 * Maps block indices to the active working set indices.
 * nb_sel          (input) int
 * Number of actively selected eigenvectors in this block.
 * =====================================================================
 */
void dlalsr(int ldS, const double *S, int ldP, const double *P, int m_size, int ldV, double *rhs_loc, double *X_panel_base, int panel_rows, int nb, const double *alphar, const double *alphai, const double *beta, int is_diag, double *work, double ascale, double bscale, double safemin, double bignum, const int *col_map, int nb_sel)
{
    int k, cur_m, c, r, dim2, i, c_scale, r_scale, c_packed;
    double sk_kp1, pk_kp1, s_kk, p_kk;
    double aR, aI, b_val, val_real, val_imag;
    double sr1, sr2, pr1, pr2, ar1, ai1, ar2, ai2;
    double x1r, x1i, x2r, x2i;
    double t, acoeff, bcoeffR, bcoeffI;
    double rhs_max, scale;
    double *work_rhs;

    k = 0;
    while (k < nb) {
        c_packed = col_map[k];

        // Skip non-selected eigenvalues
        if (c_packed < 0) {
            k += (alphai[k] == 0.0 ? 1 : 2);
            continue;
        }

        aR = alphar[k];
        aI = alphai[k];
        b_val = beta[k];

        // ---------------------------------------------------------
        // Case 1: Real eigenvalue block
        // ---------------------------------------------------------
        if (aI == 0.0) {
            cur_m = is_diag ? k : m_size;

            // Calculate a local scaling factor 't' so that evaluating the local
            // operator (beta * S - alpha * P) does not overflow. 'ascale' and 'bscale'
            // are factored in here to ensure matrix norms are respected.
            t = 1.0 / std::max({ std::abs(aR) * ascale, std::abs(b_val) * bscale, safemin });
            acoeff = (t * b_val * bscale) * ascale;
            bcoeffR = (t * aR * ascale) * bscale;

            // Limit handling for infinite eigenvalues
            if (std::abs(b_val) <= safemin && std::abs(aR) > safemin) {
                acoeff = 0.0;
                bcoeffR = 1.0;
            }

            if (is_diag) {
                rhs_loc[k + c_packed * ldV] = 1.0;
            }

            if (cur_m > 0) {
                // Form the scaled local operator matrix
                for (c = 0; c < cur_m; c++) {
                    for (r = 0; r < cur_m; r++) {
                        work[r + c * cur_m] = acoeff * S[r + c * ldS] - bcoeffR * P[r + c * ldP];
                    }
                }

                // Prepare the right-hand side for the local solve
                work_rhs = work + cur_m * cur_m;
                if (is_diag) {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = -(acoeff * S[r + k * ldS] - bcoeffR * P[r + k * ldP]) * 1.0;
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = rhs_loc[r + c_packed * ldV];
                    }
                }

                // Check for potential overflow in the local linear system solver.
                // If the maximum RHS value exceeds the safe limit (bignum / 10.0),
                // systematically scale down the RHS block AND the previously accumulated
                // X_panel columns to maintain numeric stability.
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

                // Solve the local Hessenberg system
                dlauhs(cur_m, 1, work, cur_m, work_rhs, cur_m);

                // Write the solution back to the block
                for (r = 0; r < cur_m; r++) {
                    rhs_loc[r + c_packed * ldV] = work_rhs[r];
                }
            }
            k += 1;
        }
        // ---------------------------------------------------------
        // Case 2: Complex eigenvalue block (2x2 pair)
        // ---------------------------------------------------------
        else {
            cur_m = is_diag ? k : m_size;

            // Factor in both the real (aR) and imaginary (aI) components of the
            // eigenvalue to prevent overflow in the complex operator matrix formation.
            t = 1.0 / std::max({ std::abs(aR) * ascale + std::abs(aI) * ascale, std::abs(b_val) * bscale, safemin });
            acoeff = (t * b_val * bscale) * ascale;
            bcoeffR = (t * aR * ascale) * bscale;
            bcoeffI = (t * aI * ascale) * bscale;

            if (std::abs(b_val) <= safemin && (std::abs(aR) + std::abs(aI)) > safemin) {
                acoeff = 0.0;
                bcoeffR = 1.0;
                bcoeffI = 0.0;
            }

            if (is_diag) {
                sk_kp1 = S[k + (k + 1) * ldS];
                pk_kp1 = P[k + (k + 1) * ldP];

                s_kk = S[k + k * ldS];
                p_kk = P[k + k * ldP];

                // Set up the RHS for the 2x2 diagonal component
                rhs_loc[k + c_packed * ldV] = -(acoeff * sk_kp1 - bcoeffR * pk_kp1);
                rhs_loc[k + (c_packed + 1) * ldV] = bcoeffI * pk_kp1;

                rhs_loc[k + 1 + c_packed * ldV] = acoeff * s_kk - bcoeffR * p_kk;
                rhs_loc[k + 1 + (c_packed + 1) * ldV] = -bcoeffI * p_kk;
            }

            if (cur_m > 0) {
                dim2 = 2 * cur_m;

                for (i = 0; i < dim2 * dim2; i++) {
                    work[i] = 0.0;
                }

                // Populate the workspace with interleaved real and imaginary components
                // to form the 2-subdiagonal matrix for the complex solve.
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

                // Construct the RHS vector for the complex solve
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

                // Validate RHS magnitude to avoid overflow during complex block solve
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

                // Solve the local 2-subdiagonal Hessenberg system
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
 * Solves the local panel eigenvalue problem for the left eigenvectors.
 * Handles equations corresponding to y^H * (beta*S - alpha*P) = 0.
 * Operates forward across the lower Hessenberg representations.
 * Incorporates scaling to prevent overflow during formulation and solve.
 *
 * Arguments:
 * =========
 * ldS             (input) int
 * The leading dimension of S.
 * S               (input) const double*
 * The block of matrix S.
 * ldP             (input) int
 * The leading dimension of P.
 * P               (input) const double*
 * The block of matrix P.
 * m_size          (input) int
 * Size of the current block solve.
 * ldV             (input) int
 * The leading dimension of the eigenvector arrays.
 * rhs_loc         (input/output) double*
 * Right-hand side matrix for the local solve.
 * X_panel_base    (input/output) double*
 * Base pointer for the global panel (used for scaling).
 * panel_rows      (input) int
 * Total rows in the working panel.
 * nb              (input) int
 * Number of columns in the block.
 * alphar, alphai  (input) const double*
 * Real and imaginary parts of the eigenvalues alpha.
 * beta            (input) const double*
 * The scaling factor beta for the eigenvalues.
 * is_diag         (input) int
 * Flag indicating if we are on the diagonal block.
 * work            (workspace) double*
 * Local workspace array.
 * ascale, bscale  (input) double
 * Global normalization scales to prevent overflow.
 * safemin, bignum (input) double
 * Machine parameters for safe limit checking.
 * col_map         (input) const int*
 * Maps block indices to active working set indices.
 * nb_sel          (input) int
 * Number of actively selected eigenvectors in this block.
 * =====================================================================
 */
void dlalsl(int ldS, const double *S, int ldP, const double *P, int m_size, int ldV, double *rhs_loc, double *X_panel_base, int panel_rows, int nb, const double *alphar, const double *alphai, const double *beta, int is_diag, double *work, double ascale, double bscale, double safemin, double bignum, const int *col_map, int nb_sel)
{
    int k, cur_m, row_offset, c, r, i, dim2, c_scale, r_scale, c_packed;
    int jpiv[128];
    double s11, s21, p11;
    double aR, aI, b_val, val_real, val_imag;
    double sr1, sr2, pr1, pr2, ar1, ai1, ar2, ai2;
    double y1r, y1i, y2r, y2i;
    double t, acoeff, bcoeffR, bcoeffI;
    double rhs_max, scale;
    double *work_rhs;

    k = 0;
    while (k < nb) {
        c_packed = col_map[k];
        if (c_packed < 0) {
            k += (alphai[k] == 0.0 ? 1 : 2);
            continue;
        }

        aR = alphar[k];
        aI = alphai[k];
        b_val = beta[k];

        // ---------------------------------------------------------
        // Case 1: Real eigenvalue block
        // ---------------------------------------------------------
        if (aI == 0.0) {
            cur_m = is_diag ? nb - 1 - k : m_size;
            row_offset = is_diag ? k + 1 : 0;

            // Compute scaling to ensure that evaluating the local operator
            // strictly avoids NaN or Inf propagation.
            t = 1.0 / std::max({ std::abs(aR) * ascale, std::abs(b_val) * bscale, safemin });
            acoeff = (t * b_val * bscale) * ascale;
            bcoeffR = (t * aR * ascale) * bscale;

            if (std::abs(b_val) <= safemin && std::abs(aR) > safemin) {
                acoeff = 0.0;
                bcoeffR = 1.0;
            }

            if (is_diag) {
                rhs_loc[k + c_packed * ldV] = 1.0;
            }

            if (cur_m > 0) {
                // Populate the local scaled operator matrix
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
                        work_rhs[r] = -(acoeff * sr1 - bcoeffR * pr1) * 1.0;
                    }
                }
                else {
                    for (r = 0; r < cur_m; r++) {
                        work_rhs[r] = rhs_loc[(r + row_offset) + c_packed * ldV];
                    }
                }

                // Assess the RHS magnitude and scale proportionally if the
                // solution risks numeric overflow.
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

                // Solve the local lower Hessenberg system
                dlalhs(cur_m, 1, work, cur_m, jpiv, work_rhs, cur_m);

                for (r = 0; r < cur_m; r++) {
                    rhs_loc[(r + row_offset) + c_packed * ldV] = work_rhs[r];
                }
            }
            k += 1;
        }
        // ---------------------------------------------------------
        // Case 2: Complex eigenvalue block (2x2 pair)
        // ---------------------------------------------------------
        else {
            cur_m = is_diag ? nb - 2 - k : m_size;
            row_offset = is_diag ? k + 2 : 0;

            t = 1.0 / std::max({ std::abs(aR) * ascale + std::abs(aI) * ascale, std::abs(b_val) * bscale, safemin });
            acoeff = (t * b_val * bscale) * ascale;
            bcoeffR = (t * aR * ascale) * bscale;
            bcoeffI = (t * aI * ascale) * bscale;

            if (std::abs(b_val) <= safemin && (std::abs(aR) + std::abs(aI)) > safemin) {
                acoeff = 0.0;
                bcoeffR = 1.0;
                bcoeffI = 0.0;
            }

            if (is_diag) {
                s11 = S[k + k * ldS];
                s21 = S[k + 1 + k * ldS];
                p11 = P[k + k * ldP];

                rhs_loc[k + c_packed * ldV] = -acoeff * s21;
                rhs_loc[k + (c_packed + 1) * ldV] = 0.0;

                rhs_loc[k + 1 + c_packed * ldV] = acoeff * s11 - bcoeffR * p11;
                rhs_loc[k + 1 + (c_packed + 1) * ldV] = bcoeffI * p11;
            }

            if (cur_m > 0) {
                dim2 = 2 * cur_m;

                for (i = 0; i < dim2 * dim2; i++) {
                    work[i] = 0.0;
                }

                // Populate workspace with interleaved real and imaginary parts
                // for the 2-superdiagonal matrix solve.
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

                // Check complex block RHS array to avoid overflow limits
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
 * DTGEVC3 computes some or all of the right and/or left eigenvectors of
 * a pair of real matrices (S,P), where S is a quasi-triangular matrix
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
 * S               (input) const double*
 * The upper quasi-triangular matrix S.
 *
 * lds             (input) int
 * The leading dimension of the array S.
 *
 * P               (input) const double*
 * The upper triangular matrix P.
 *
 * ldp             (input) int
 * The leading dimension of the array P.
 *
 * alphar, alphai  (input) const double*
 * Real and imaginary parts of scalar alphas.
 *
 * beta            (input) const double*
 * Scalars beta defining eigenvalues.
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
void dtgevc3(char side, char howmny, const int *select, int n, const double *S, int lds, const double *P, int ldp, const double *alphar, const double *alphai, const double *beta, double *VL, int ldvl, double *VR, int ldvr, int mm, int *m, double *work, int lwork, int *info)
{
    bool compute_right, compute_left, do_all, do_back, do_sel, selected;
    int req_lwork;
    double *X_panel, *Temp, *work_local;
    int curr_col, i, nb, ld_x, c, r, curr_row, j, j_nb, is_diag, i_next, rem_rows, j_next, rem;
    double *TempS, *TempP;
    double b_val, aR, aI, xr, xi;
    double alpha_m1, beta_1, alpha_1, one, zero;
    double safemin, eps, smlnum, bignum;
    double anorm, bnorm, sum_s, sum_p;
    int col, row, row_limit;
    double ascale, bscale;
    double t, acoeff, bcoeffR, bcoeffI;
    double update_max, safe_limit, scale;
    int total_elements, idx, c_idx, r_idx;

    int bsize = 32; // Default ideal block size for Level 3 BLAS optimization
    int num_sel, k_idx, c_packed, out_col, current_out_col;
    int col_map[128]; // Safe stack array bounding the maximum block size
    int nb_sel;

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
    for (k_idx = 0; k_idx < n;) {
        selected = do_all;
        if (do_sel && select != nullptr) {
            selected = select[k_idx];
            if (alphai[k_idx] != 0.0 && k_idx + 1 < n) {
                selected = (selected || select[k_idx + 1]);
            }
        }

        if (selected) {
            num_sel += (alphai[k_idx] != 0.0 && k_idx + 1 < n) ? 2 : 1;
        }
        k_idx += (alphai[k_idx] != 0.0 && k_idx + 1 < n) ? 2 : 1;
    }

    if (m != nullptr) {
        *m = num_sel;
    }

    if (mm < num_sel) {
        *info = -16;
        return;
    }

    // Determine required workspace size
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
        work[0] = static_cast<double>(req_lwork);
        *info = 0;
        return;
    }

    if (lwork < req_lwork) {
        *info = -19;
        return;
    }

    *info = 0;

    // Retrieve machine constants for evaluating safe scaling boundaries
    safemin = std::numeric_limits<double>::min();
    eps = std::numeric_limits<double>::epsilon();
    smlnum = safemin / eps;
    bignum = 1.0 / smlnum;

    // Compute the 1-norm of S and P to generate global scaling factors ascale
    // and bscale. This ensures combinations of S and P remain properly bounded.
    anorm = 0.0;
    bnorm = 0.0;
    for (col = 0; col < n; ++col) {
        sum_s = 0.0;
        sum_p = 0.0;
        row_limit = std::min(n - 1, col + 1);
        for (row = 0; row <= row_limit; ++row) {
            sum_s += std::abs(S[row + col * lds]);
            sum_p += std::abs(P[row + col * ldp]);
        }
        anorm = std::max(anorm, sum_s);
        bnorm = std::max(bnorm, sum_p);
    }

    ascale = 1.0 / std::max(anorm, safemin);
    bscale = 1.0 / std::max(bnorm, safemin);

    // Partition the allocated workspace into the active vector panel,
    // intermediate temp buffers, and local solver workspaces.
    X_panel = work;
    Temp = X_panel + n * (bsize + 1);
    work_local = Temp + n * (bsize + 1);

    // ==================================================================
    // Right Eigenvector Computation
    // Proceed backward from the bottom-right of the matrix to the top-left
    // ==================================================================
    if (compute_right && VR != nullptr) {
        current_out_col = num_sel;
        curr_col = n;

        while (curr_col > 0) {
            i = idlapb(S, lds, curr_col, bsize);
            nb = curr_col - i;
            ld_x = curr_col;

            nb_sel = 0;
            // Build the block to active subset column map
            for (c = 0; c < nb;) {
                selected = do_all;
                if (do_sel && select != nullptr) {
                    selected = select[i + c];
                    if (alphai[i + c] != 0.0 && i + c + 1 < n) {
                        selected = (selected || select[i + c + 1]);
                    }
                }

                if (selected) {
                    col_map[c] = nb_sel++;
                    if (alphai[i + c] != 0.0 && i + c + 1 < n) {
                        col_map[c + 1] = nb_sel++;
                        c += 2;
                    }
                    else {
                        c += 1;
                    }
                }
                else {
                    col_map[c] = -1;
                    if (alphai[i + c] != 0.0 && i + c + 1 < n) {
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

            // Zero out the active matrix panel slice before beginning updates
            for (c = 0; c < nb_sel; c++) {
                for (r = 0; r < curr_col; r++) {
                    X_panel[r + c * ld_x] = 0.0;
                }
            }

            // Iterate sequentially backward up the matrix to resolve components
            curr_row = curr_col;
            while (curr_row > 0) {
                j = idlapb(S, lds, curr_row, bsize);
                j_nb = curr_row - j;
                is_diag = (curr_row == curr_col) ? 1 : 0;

                // Obtain the local piece of the eigenvector matrix
                dlalsr(lds, &S[j + j * lds], ldp, &P[j + j * ldp], j_nb, ld_x, &X_panel[j], X_panel, ld_x, nb, alphar + i, alphai + i, beta + i, is_diag, work_local, ascale, bscale, safemin, bignum, col_map, nb_sel);

                if (j > 0) {
                    TempS = work_local;
                    TempP = work_local + j_nb * nb_sel;

                    // Form the scaled matrix components that act as right-side
                    // inputs for the upcoming Level 3 BLAS operations
                    for (c = 0; c < nb;) {
                        c_packed = col_map[c];
                        if (c_packed < 0) {
                            c += (alphai[i + c] == 0.0 ? 1 : 2);
                            continue;
                        }

                        b_val = beta[i + c];
                        aR = alphar[i + c];
                        aI = alphai[i + c];

                        t = 1.0 / std::max({ std::abs(aR) * ascale + std::abs(aI) * ascale, std::abs(b_val) * bscale, safemin });
                        acoeff = (t * b_val * bscale) * ascale;
                        bcoeffR = (t * aR * ascale) * bscale;
                        bcoeffI = (t * aI * ascale) * bscale;

                        if (std::abs(b_val) <= safemin && (std::abs(aR) + std::abs(aI)) > safemin) {
                            acoeff = 0.0;
                            bcoeffR = 1.0;
                            bcoeffI = 0.0;
                        }

                        if (aI == 0.0) {
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

                    // Before running BLAS updates, ensure the internal matrix inner products
                    // will not trigger numeric overflow boundaries.
                    update_max = 0.0;
                    total_elements = j_nb * nb_sel;
                    for (idx = 0; idx < total_elements; ++idx) {
                        update_max = std::max({ update_max, std::abs(TempS[idx]), std::abs(TempP[idx]) });
                    }

                    safe_limit = bignum / static_cast<double>(std::max(1, nb_sel));

                    // Scale proportionately down to prevent calculation explosions
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

                    // Perform block update accumulating results into X_panel via DGEMM
                    alpha_m1 = -1.0;
                    beta_1 = 1.0;
                    alpha_1 = 1.0;
                    dgemm_("N", "N", &j, &nb_sel, &j_nb, &alpha_m1, &S[0 + j * lds], &lds, TempS, &j_nb, &beta_1, &X_panel[0], &ld_x);
                    dgemm_("N", "N", &j, &nb_sel, &j_nb, &alpha_1, &P[0 + j * ldp], &ldp, TempP, &j_nb, &beta_1, &X_panel[0], &ld_x);
                }
                curr_row = j;
            }

            // Copy results back to the VR matrix.
            // If do_back is true, the vectors require backtransformation via matrix multiplication.
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
            i_next = idlanb(S, n, lds, curr_col, bsize);
            nb = i_next - curr_col;
            i = curr_col;
            rem_rows = n - i;
            ld_x = rem_rows;

            nb_sel = 0;
            // Build the block to active subset column map
            for (c = 0; c < nb;) {
                selected = do_all;
                if (do_sel && select != nullptr) {
                    selected = select[i + c];
                    if (alphai[i + c] != 0.0 && i + c + 1 < n) {
                        selected = (selected || select[i + c + 1]);
                    }
                }

                if (selected) {
                    col_map[c] = nb_sel++;
                    if (alphai[i + c] != 0.0 && i + c + 1 < n) {
                        col_map[c + 1] = nb_sel++;
                        c += 2;
                    }
                    else {
                        c += 1;
                    }
                }
                else {
                    col_map[c] = -1;
                    if (alphai[i + c] != 0.0 && i + c + 1 < n) {
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

            // Initialize active submatrix panel to zeroes
            for (c = 0; c < nb_sel; c++) {
                for (r = 0; r < rem_rows; r++) {
                    X_panel[r + c * ld_x] = 0.0;
                }
            }

            // Process forward down the matrix representations
            curr_row = i;
            while (curr_row < n) {
                j_next = idlanb(S, n, lds, curr_row, bsize);
                j_nb = j_next - curr_row;
                is_diag = (curr_row == i) ? 1 : 0;

                // Obtain the local piece of the left eigenvector matrix
                dlalsl(lds, &S[curr_row + curr_row * lds], ldp, &P[curr_row + curr_row * ldp], j_nb, ld_x, &X_panel[curr_row - i], X_panel, ld_x, nb, alphar + i, alphai + i, beta + i, is_diag, work_local, ascale, bscale, safemin, bignum, col_map, nb_sel);

                if (j_next < n) {
                    TempS = work_local;
                    TempP = work_local + j_nb * nb_sel;

                    // Compute scaled input vectors for downward block GEMM
                    for (c = 0; c < nb;) {
                        c_packed = col_map[c];
                        if (c_packed < 0) {
                            c += (alphai[i + c] == 0.0 ? 1 : 2);
                            continue;
                        }

                        b_val = beta[i + c];
                        aR = alphar[i + c];
                        aI = alphai[i + c];

                        t = 1.0 / std::max({ std::abs(aR) * ascale + std::abs(aI) * ascale, std::abs(b_val) * bscale, safemin });
                        acoeff = (t * b_val * bscale) * ascale;
                        bcoeffR = (t * aR * ascale) * bscale;
                        bcoeffI = (t * aI * ascale) * bscale;

                        if (std::abs(b_val) <= safemin && (std::abs(aR) + std::abs(aI)) > safemin) {
                            acoeff = 0.0;
                            bcoeffR = 1.0;
                            bcoeffI = 0.0;
                        }

                        if (aI == 0.0) {
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

                    // Shield updating matrix product from overflow situations
                    update_max = 0.0;
                    total_elements = j_nb * nb_sel;
                    for (idx = 0; idx < total_elements; ++idx) {
                        update_max = std::max({ update_max, std::abs(TempS[idx]), std::abs(TempP[idx]) });
                    }

                    safe_limit = bignum / static_cast<double>(std::max(1, nb_sel));

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
                    alpha_m1 = -1.0;
                    beta_1 = 1.0;
                    alpha_1 = 1.0;
                    dgemm_("T", "N", &rem, &nb_sel, &j_nb, &alpha_m1, &S[curr_row + j_next * lds], &lds, TempS, &j_nb, &beta_1, &X_panel[j_next - i], &ld_x);
                    dgemm_("T", "N", &rem, &nb_sel, &j_nb, &alpha_1, &P[curr_row + j_next * ldp], &ldp, TempP, &j_nb, &beta_1, &X_panel[j_next - i], &ld_x);
                }
                curr_row = j_next;
            }

            // Transfer result back to the user matrix VL. Apply base transformation
            // via dgemm if computing original coordinates (howmny = 'B')
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
            current_out_col += nb_sel;
            curr_col = i_next;
        }
    }
}