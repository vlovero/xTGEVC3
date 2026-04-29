      SUBROUTINE SLAL2S(N, NRHS, A, LDA, JPIV, B, LDB, INFO)
      IMPLICIT NONE
      INTEGER N, NRHS, LDA, LDB, INFO
      INTEGER JPIV(*)
      REAL A(LDA, *), B(LDB, *)
      INTEGER K, P, I, C
      REAL MAX_VAL, PIVOT, M1, M2, ALPHA, TMP
      CHARACTER SIDE, UPLO, TRANSA, DIAG

      INFO = 0

*     1. Forward Elimination (Process both superdiagonals)
      DO K = 1, N - 1

*        --- Partial Pivoting ---
         P = K
         MAX_VAL = ABS(A(K, K))

         IF (ABS(A(K, K + 1)) .GT. MAX_VAL) THEN
            MAX_VAL = ABS(A(K, K + 1))
            P = K + 1
         END IF
         IF (K + 2 .LE. N) THEN
            IF (ABS(A(K, K + 2)) .GT. MAX_VAL) THEN
               P = K + 2
            END IF
         END IF

         JPIV(K) = P

*        --- Column Swap ---
         IF (P .NE. K) THEN
            DO I = K, N
               TMP = A(I, K)
               A(I, K) = A(I, P)
               A(I, P) = TMP
            END DO
         END IF

         PIVOT = A(K, K)
         IF (PIVOT .EQ. 0.0E0) THEN
            INFO = K
            RETURN
         END IF

*        --- Row Elimination ---
         M1 = A(K, K + 1) / PIVOT
         A(K, K + 1) = M1
         DO I = K + 1, N
            A(I, K + 1) = A(I, K + 1) - M1 * A(I, K)
         END DO

*        Eliminate the second superdiagonal if it exists
         IF (K + 2 .LE. N) THEN
            M2 = A(K, K + 2) / PIVOT
            A(K, K + 2) = M2
            DO I = K + 1, N
               A(I, K + 2) = A(I, K + 2) - M2 * A(I, K)
            END DO
         END IF
      END DO

*     2. Forward-substitution (Solve Lower Triangular System)
      SIDE = 'L'
      UPLO = 'L'
      TRANSA = 'N'
      DIAG = 'N'
      ALPHA = 1.0E0
      CALL STRSM(SIDE, UPLO, TRANSA, DIAG, N, NRHS, ALPHA, 
     $           A, LDA, B, LDB)

*     3. Backward permutation update
      DO K = N - 1, 1, -1
         M1 = A(K, K + 1)
         IF (K + 2 .LE. N) THEN
            M2 = A(K, K + 2)
         ELSE
            M2 = 0.0E0
         END IF
         P = JPIV(K)

         DO C = 1, NRHS
            B(K, C) = B(K, C) - M1 * B(K + 1, C)
            IF (K + 2 .LE. N) THEN
               B(K, C) = B(K, C) - M2 * B(K + 2, C)
            END IF
            IF (P .NE. K) THEN
               TMP = B(K, C)
               B(K, C) = B(P, C)
               B(P, C) = TMP
            END IF
         END DO
      END DO

      RETURN
      END
