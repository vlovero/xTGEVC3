      SUBROUTINE CLALSL(LDS, S, LDP, P, M_SIZE, LDV, RHS_LOC,
     $                  X_PANEL_BASE, PANEL_ROWS, NB, ALPHA, BETA,
     $                  IS_DIAG, WORK, ASCALE, BSCALE, SAFEMIN, BIGNUM,
     $                  COL_MAP, NB_SEL)
      IMPLICIT NONE
      INTEGER LDS, LDP, M_SIZE, LDV, PANEL_ROWS, NB, IS_DIAG, NB_SEL
      REAL ASCALE, BSCALE, SAFEMIN, BIGNUM
      COMPLEX S(LDS, *), P(LDP, *), RHS_LOC(LDV, *)
      COMPLEX X_PANEL_BASE(LDV, *)
      COMPLEX WORK(*)
      COMPLEX ALPHA(*), BETA(*)
      INTEGER COL_MAP(*)

      INTEGER K, CUR_M, ROW_OFFSET, C, R, C_SCALE, R_SCALE, C_PACKED
      INTEGER WORK_RHS_IDX
      COMPLEX A, B_VAL, ACOEFF, BCOEFF, SR1, PR1
      REAL T, RHS_MAX, SCALE_VAL
      COMPLEX ONE
      INTEGER ONE_INT
      CHARACTER SIDE, UPLO, TRANSA, DIAG

      ONE = CMPLX(1.0E0, 0.0E0)
      ONE_INT = 1
      SIDE = 'L'
      UPLO = 'U'
      TRANSA = 'C'
      DIAG = 'N'

      DO K = 1, NB
          C_PACKED = COL_MAP(K)
*         Skip non-selected eigenvalues
          IF (C_PACKED .GT. 0) THEN
              A = ALPHA(K)
              B_VAL = BETA(K)
              IF (IS_DIAG .EQ. 1) THEN
                  CUR_M = NB - K
                  ROW_OFFSET = K
              ELSE
                  CUR_M = M_SIZE
                  ROW_OFFSET = 0
              END IF

              T = 1.0E0 / MAX(ABS(A) * ASCALE, ABS(B_VAL) * BSCALE,
     $                        SAFEMIN)
              ACOEFF = CMPLX(T * ASCALE, 0.0E0) * B_VAL *
     $                 CMPLX(BSCALE, 0.0E0)
              BCOEFF = CMPLX(T * BSCALE, 0.0E0) * A *
     $                 CMPLX(ASCALE, 0.0E0)

*             Limit handling for infinite eigenvalues
              IF (ABS(B_VAL) .LE. SAFEMIN .AND. 
     $            ABS(A) .GT. SAFEMIN) THEN
                  ACOEFF = CMPLX(0.0E0, 0.0E0)
                  BCOEFF = CMPLX(1.0E0, 0.0E0)
              END IF

              IF (IS_DIAG .EQ. 1) THEN
                  RHS_LOC(K, C_PACKED) = CMPLX(1.0E0, 0.0E0)
              END IF

              IF (CUR_M .GT. 0) THEN
                  DO C = 1, CUR_M
                      DO R = 1, C
                          WORK(R + (C - 1) * CUR_M) =
     $                      ACOEFF * S(R + ROW_OFFSET, C + ROW_OFFSET) -
     $                      BCOEFF * P(R + ROW_OFFSET, C + ROW_OFFSET)
                      END DO
                  END DO

                  WORK_RHS_IDX = CUR_M * CUR_M
                  IF (IS_DIAG .EQ. 1) THEN
                      DO R = 1, CUR_M
                          SR1 = S(K, R + ROW_OFFSET)
                          PR1 = P(K, R + ROW_OFFSET)
                          WORK(WORK_RHS_IDX + R) = -CONJG(ACOEFF * SR1-
     $                                                     BCOEFF * PR1)
                      END DO
                  ELSE
                      DO R = 1, CUR_M
                          WORK(WORK_RHS_IDX + R) =
     $                      RHS_LOC(R + ROW_OFFSET, C_PACKED)
                      END DO
                  END IF

                  RHS_MAX = 0.0E0
                  DO R = 1, CUR_M
                      RHS_MAX = MAX(RHS_MAX,
     $                  ABS(REAL(WORK(WORK_RHS_IDX + R))) +
     $                  ABS(AIMAG(WORK(WORK_RHS_IDX + R))))
                  END DO

                  IF (RHS_MAX .GT. BIGNUM / 10.0D0) THEN
                      SCALE_VAL = (BIGNUM / 10.0D0) / RHS_MAX
                      DO C_SCALE = 1, NB_SEL
                          DO R_SCALE = 1, PANEL_ROWS
                              X_PANEL_BASE(R_SCALE, C_SCALE) =
     $                          X_PANEL_BASE(R_SCALE, C_SCALE) *
     $                          CMPLX(SCALE_VAL, 0.0E0)
                          END DO
                      END DO
                      DO R = 1, CUR_M
                          WORK(WORK_RHS_IDX + R) =
     $                      WORK(WORK_RHS_IDX + R) *
     $                      CMPLX(SCALE_VAL, 0.0E0)
                      END DO
                  END IF

                  CALL CTRSM(SIDE, UPLO, TRANSA, DIAG, CUR_M, ONE_INT,
     $                       ONE, WORK, CUR_M, WORK(WORK_RHS_IDX + 1),
     $                       CUR_M)

                  DO R = 1, CUR_M
                      RHS_LOC(R + ROW_OFFSET, C_PACKED) =
     $                  WORK(WORK_RHS_IDX + R)
                  END DO
              END IF
          END IF
      END DO
      RETURN
      END SUBROUTINE CLALSL