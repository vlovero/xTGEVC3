      SUBROUTINE DLAUHS(N, NRHS, A, LDA, B, LDB, INFO)
      IMPLICIT NONE
      INTEGER N, NRHS, LDA, LDB, INFO
      DOUBLE PRECISION A(LDA, *), B(LDB, *)
      INTEGER J, K, PIVOT_ROW
      DOUBLE PRECISION MULT, ALPHA, TMP, EPS, SAFEMIN, LOCAL_MAX
      DOUBLE PRECISION PERTURB
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
         DO K = J, MIN(N, J + 1)
            LOCAL_MAX = MAX(LOCAL_MAX, ABS(A(K, J)))
         END DO
      END DO
      PERTURB = MAX(SAFEMIN, EPS * LOCAL_MAX)

* 1. Forward Elimination (Column-oriented traversal)
      DO J = 1, N - 1

* --- Partial Pivoting ---
         PIVOT_ROW = J
         IF (ABS(A(J + 1, J)) .GT. ABS(A(J, J))) THEN
            PIVOT_ROW = J + 1
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
         MULT = A(J + 1, J) / A(J, J)
         A(J + 1, J) = 0.0D0

         DO K = J + 1, N
            A(J + 1, K) = A(J + 1, K) - MULT * A(J, K)
         END DO
         DO K = 1, NRHS
            B(J + 1, K) = B(J + 1, K) - MULT * B(J, K)
         END DO
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