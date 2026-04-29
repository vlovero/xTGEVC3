      SUBROUTINE SLALHS(N, NRHS, A, LDA, JPIV, B, LDB, INFO)
      IMPLICIT NONE
      INTEGER N, NRHS, LDA, LDB, INFO
      INTEGER JPIV(*)
      REAL A(LDA, *), B(LDB, *)
      INTEGER K, P, I, C
      REAL MAX_VAL, PIVOT, M, ALPHA, TMP
      CHARACTER SIDE, UPLO, TRANSA, DIAG

      INFO = 0

*     1. Forward Elimination (Process the single superdiagonal)
      DO K = 1, N - 1

*        --- Partial Pivoting ---
         P = K
         MAX_VAL = ABS(A(K, K))

         IF (ABS(A(K, K + 1)) .GT. MAX_VAL) THEN
            P = K + 1
         END IF

*        Store the pivot index so it can be applied to B after solving A
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
         M = A(K, K + 1) / PIVOT
         A(K, K + 1) = M

         DO I = K + 1, N
            A(I, K + 1) = A(I, K + 1) - M * A(I, K)
         END DO
      END DO

*     2. Forward-substitution (Solve Lower Triangular System A * X = B)
      SIDE = 'L'
      UPLO = 'L'
      TRANSA = 'N'
      DIAG = 'N'
      ALPHA = 1.0E0
      CALL STRSM(SIDE, UPLO, TRANSA, DIAG, N, NRHS, ALPHA, 
     $           A, LDA, B, LDB)

*     3. Backward permutation update: Apply pivoting history to solution
      DO K = N - 1, 1, -1
         M = A(K, K + 1)
         P = JPIV(K)
         DO C = 1, NRHS
            B(K, C) = B(K, C) - M * B(K + 1, C)
            IF (P .NE. K) THEN
               TMP = B(K, C)
               B(K, C) = B(P, C)
               B(P, C) = TMP
            END IF
         END DO
      END DO

      RETURN
      END
