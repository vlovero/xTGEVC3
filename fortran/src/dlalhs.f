      SUBROUTINE DLALHS(N, NRHS, A, LDA, JPIV, B, LDB, INFO)
      IMPLICIT NONE
      INTEGER N, NRHS, LDA, LDB, INFO
      INTEGER JPIV(*)
      DOUBLE PRECISION A(LDA, *), B(LDB, *)
      INTEGER K, P, I, C
      DOUBLE PRECISION MAX_VAL, PIVOT, M, ALPHA, TMP, EPS, SAFEMIN
      DOUBLE PRECISION LOCAL_MAX, PERTURB
      CHARACTER SIDE, UPLO, TRANSA, DIAG
      DOUBLE PRECISION DLAMCH
      EXTERNAL DLAMCH

      INFO = 0
      IF (N .EQ. 0) RETURN

      EPS = DLAMCH('E')
      SAFEMIN = DLAMCH('S')
      LOCAL_MAX = 0.0D0

      DO I = 1, N
         DO K = MAX(1, I - 1), N
            LOCAL_MAX = MAX(LOCAL_MAX, ABS(A(K, I)))
         END DO
      END DO
      PERTURB = MAX(SAFEMIN, EPS * LOCAL_MAX)

* 1. Forward Elimination
      DO K = 1, N - 1

* --- Partial Pivoting ---
         P = K
         MAX_VAL = ABS(A(K, K))

         IF (ABS(A(K, K + 1)) .GT. MAX_VAL) THEN
            P = K + 1
         END IF

         JPIV(K) = P

* --- Column Swap ---
         IF (P .NE. K) THEN
            DO I = K, N
               TMP = A(I, K)
               A(I, K) = A(I, P)
               A(I, P) = TMP
            END DO
         END IF

* --- Zero Pivot Perturbation ---
         PIVOT = A(K, K)
         IF (ABS(PIVOT) .LT. PERTURB) THEN
            IF (PIVOT .LT. 0.0D0) THEN
               PIVOT = -PERTURB
            ELSE
               PIVOT = PERTURB
            END IF
            A(K, K) = PIVOT
         END IF

* --- Row Elimination ---
         M = A(K, K + 1) / PIVOT
         A(K, K + 1) = M

         DO I = K + 1, N
            A(I, K + 1) = A(I, K + 1) - M * A(I, K)
         END DO
      END DO

      IF (ABS(A(N, N)) .LT. PERTURB) THEN
         IF (A(N, N) .LT. 0.0D0) THEN
            A(N, N) = -PERTURB
         ELSE
            A(N, N) = PERTURB
         END IF
      END IF

* 2. Forward-substitution
      IF (NRHS .GT. 0) THEN
         SIDE = 'L'
         UPLO = 'L'
         TRANSA = 'N'
         DIAG = 'N'
         ALPHA = 1.0D0
         CALL DTRSM(SIDE, UPLO, TRANSA, DIAG, N, NRHS, ALPHA, 
     $              A, LDA, B, LDB)
      END IF

* 3. Backward permutation update
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