      SUBROUTINE DLAU2S(N, NRHS, A, LDA, B, LDB, INFO)
      IMPLICIT NONE
      INTEGER N, NRHS, LDA, LDB, INFO
      DOUBLE PRECISION A(LDA, *), B(LDB, *)
      INTEGER J, K, PIVOT_ROW
      DOUBLE PRECISION MAX_VAL, MULT1, MULT2, ALPHA, TMP, EPS, SAFEMIN
      DOUBLE PRECISION LOCAL_MAX, PERTURB
      CHARACTER SIDE, UPLO, TRANSA, DIAG
      DOUBLE PRECISION DLAMCH
      EXTERNAL DLAMCH

      INFO = 0

      IF (N .LT. 0) THEN
         INFO = -1
         RETURN
      END IF
      IF (NRHS .LT. 0) THEN
         INFO = -2
         RETURN
      END IF
      IF (LDA .LT. MAX(1, N)) THEN
         INFO = -4
         RETURN
      END IF
      IF (LDB .LT. MAX(1, N)) THEN
         INFO = -6
         RETURN
      END IF
      IF (N .EQ. 0) THEN
         RETURN
      END IF

      EPS = DLAMCH('E')
      SAFEMIN = DLAMCH('S')
      LOCAL_MAX = 0.0D0

      DO J = 1, N
         DO K = J, MIN(J + 2, N)
            LOCAL_MAX = MAX(LOCAL_MAX, ABS(A(K, J)))
         END DO
      END DO
      PERTURB = MAX(SAFEMIN, EPS * LOCAL_MAX)

* 1. Forward Elimination
      DO J = 1, N - 1

* --- Partial Pivoting ---
         PIVOT_ROW = J
         MAX_VAL = ABS(A(J, J))

         IF (ABS(A(J + 1, J)) .GT. MAX_VAL) THEN
            MAX_VAL = ABS(A(J + 1, J))
            PIVOT_ROW = J + 1
         END IF
         IF (J + 2 .LE. N) THEN
            IF (ABS(A(J + 2, J)) .GT. MAX_VAL) THEN
               MAX_VAL = ABS(A(J + 2, J))
               PIVOT_ROW = J + 2
            END IF
         END IF

* --- Row Swap ---
         IF (PIVOT_ROW .NE. J) THEN
            DO K = J, N
               TMP = A(J, K)
               A(J, K) = A(PIVOT_ROW, K)
               A(PIVOT_ROW, K) = TMP
            END DO
            DO K = 1, NRHS
               TMP = B(J, K)
               B(J, K) = B(PIVOT_ROW, K)
               B(PIVOT_ROW, K) = TMP
            END DO
         END IF

* --- Zero Pivot Perturbation ---
         IF (ABS(A(J, J)) .LT. PERTURB) THEN
            IF (A(J, J) .LT. 0.0D0) THEN
               A(J, J) = -PERTURB
            ELSE
               A(J, J) = PERTURB
            END IF
         END IF

* --- Row Elimination ---
         MULT1 = A(J + 1, J) / A(J, J)
         A(J + 1, J) = 0.0D0

         DO K = J + 1, N
            A(J + 1, K) = A(J + 1, K) - MULT1 * A(J, K)
         END DO
         DO K = 1, NRHS
            B(J + 1, K) = B(J + 1, K) - MULT1 * B(J, K)
         END DO

* Eliminate the second subdiagonal element
         IF (J + 2 .LE. N) THEN
            MULT2 = A(J + 2, J) / A(J, J)
            A(J + 2, J) = 0.0D0

            DO K = J + 1, N
               A(J + 2, K) = A(J + 2, K) - MULT2 * A(J, K)
            END DO
            DO K = 1, NRHS
               B(J + 2, K) = B(J + 2, K) - MULT2 * B(J, K)
            END DO
         END IF
      END DO

      IF (ABS(A(N, N)) .LT. PERTURB) THEN
         IF (A(N, N) .LT. 0.0D0) THEN
            A(N, N) = -PERTURB
         ELSE
            A(N, N) = PERTURB
         END IF
      END IF

* 2. Back-substitution (Solve Upper Triangular System)
      IF (NRHS .GT. 0) THEN
         SIDE = 'L'
         UPLO = 'U'
         TRANSA = 'N'
         DIAG = 'N'
         ALPHA = 1.0D0
         CALL DTRSM(SIDE, UPLO, TRANSA, DIAG, N, NRHS, ALPHA, 
     $              A, LDA, B, LDB)
      END IF

      RETURN
      END