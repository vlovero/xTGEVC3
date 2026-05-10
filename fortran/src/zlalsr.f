        SUBROUTINE ZLALSR(LDS, S, LDP, P, M_SIZE, LDV, RHS_LOC,
     $                  X_PANEL_BASE, PANEL_ROWS, NB, ALPHA, BETA,
     $                  IS_DIAG, WORK, ASCALE, BSCALE, SAFEMIN, BIGNUM,
     $                  COL_MAP, NB_SEL)
      IMPLICIT NONE
      INTEGER LDS, LDP, M_SIZE, LDV, PANEL_ROWS, NB, IS_DIAG, NB_SEL
      DOUBLE PRECISION ASCALE, BSCALE, SAFEMIN, BIGNUM
      COMPLEX*16 S(LDS, *), P(LDP, *), RHS_LOC(LDV, *)
      COMPLEX*16 X_PANEL_BASE(LDV, *)
      COMPLEX*16 WORK(*)
      COMPLEX*16 ALPHA(*), BETA(*)
      INTEGER COL_MAP(*)

      INTEGER K, CUR_M, C, R, C_SCALE, R_SCALE, C_PACKED
      INTEGER WORK_RHS_IDX
      COMPLEX*16 A, B_VAL, ACOEFF, BCOEFF
      DOUBLE PRECISION T, RHS_MAX, SCALE_VAL
      COMPLEX*16 ONE
      INTEGER ONE_INT
      CHARACTER SIDE, UPLO, TRANSA, DIAG

      INTEGER D_IDX, R_IDX_P, C_IDX
      DOUBLE PRECISION EPS, LOCAL_MAX, PERTURB, R_VAL, I_VAL
      DOUBLE PRECISION DLAMCH
      EXTERNAL DLAMCH

      ONE = DCMPLX(1.0D0, 0.0D0)
      ONE_INT = 1
      SIDE = 'L'
      UPLO = 'U'
      TRANSA = 'N'
      DIAG = 'N'

      EPS = DLAMCH('E')

      DO K = 1, NB
          C_PACKED = COL_MAP(K)
* Skip non-selected eigenvalues (using 0 as unselected)
          IF (C_PACKED .GT. 0) THEN
              A = ALPHA(K)
              B_VAL = BETA(K)
              IF (IS_DIAG .EQ. 1) THEN
                  CUR_M = K - 1
              ELSE
                  CUR_M = M_SIZE
              END IF

              T = 1.0D0 / MAX(ABS(A) * ASCALE, ABS(B_VAL) * BSCALE,
     $                        SAFEMIN)
              ACOEFF = DCMPLX(T * ASCALE, 0.0D0) * B_VAL *
     $                 DCMPLX(BSCALE, 0.0D0)
              BCOEFF = DCMPLX(T * BSCALE, 0.0D0) * A *
     $                 DCMPLX(ASCALE, 0.0D0)

* Limit handling for infinite eigenvalues
              IF (ABS(B_VAL) .LE. SAFEMIN .AND. 
     $            ABS(A) .GT. SAFEMIN) THEN
                  ACOEFF = DCMPLX(0.0D0, 0.0D0)
                  BCOEFF = DCMPLX(1.0D0, 0.0D0)
              END IF

              IF (IS_DIAG .EQ. 1) THEN
                  RHS_LOC(K, C_PACKED) = DCMPLX(1.0D0, 0.0D0)
              END IF

              IF (CUR_M .GT. 0) THEN
                  DO C = 1, CUR_M
                      DO R = 1, C
                          WORK(R + (C - 1) * CUR_M) = ACOEFF * S(R, C) -
     $                                                BCOEFF * P(R, C)
                      END DO
                  END DO

                  WORK_RHS_IDX = CUR_M * CUR_M
                  IF (IS_DIAG .EQ. 1) THEN
                      DO R = 1, CUR_M
                          WORK(WORK_RHS_IDX + R) = -(ACOEFF * S(R, K) -
     $                                               BCOEFF * P(R, K))
                      END DO
                  ELSE
                      DO R = 1, CUR_M
                          WORK(WORK_RHS_IDX + R) = RHS_LOC(R, C_PACKED)
                      END DO
                  END IF

                  RHS_MAX = 0.0D0
                  DO R = 1, CUR_M
                      RHS_MAX = MAX(RHS_MAX,
     $                  ABS(DBLE(WORK(WORK_RHS_IDX + R))) +
     $                  ABS(DIMAG(WORK(WORK_RHS_IDX + R))))
                  END DO

                  IF (RHS_MAX .GT. BIGNUM / 10.0D0) THEN
                      SCALE_VAL = (BIGNUM / 10.0D0) / RHS_MAX
                      DO C_SCALE = 1, NB_SEL
                          DO R_SCALE = 1, PANEL_ROWS
                              X_PANEL_BASE(R_SCALE, C_SCALE) =
     $                          X_PANEL_BASE(R_SCALE, C_SCALE) *
     $                          DCMPLX(SCALE_VAL, 0.0D0)
                          END DO
                      END DO
                      DO R = 1, CUR_M
                          WORK(WORK_RHS_IDX + R) =
     $                      WORK(WORK_RHS_IDX + R) *
     $                      DCMPLX(SCALE_VAL, 0.0D0)
                      END DO
                  END IF

* Perturbation check for local 1x1 or mxm generalized block
                  LOCAL_MAX = 0.0D0
                  DO C_IDX = 1, CUR_M
                      DO R_IDX_P = 1, C_IDX
                          LOCAL_MAX = MAX(LOCAL_MAX,
     $                   ABS(DBLE(WORK(R_IDX_P + (C_IDX - 1)*CUR_M)))
     $                 + ABS(DIMAG(WORK(R_IDX_P + (C_IDX - 1)*CUR_M))))
                      END DO
                  END DO
                  PERTURB = MAX(SAFEMIN, EPS * LOCAL_MAX)

                  DO D_IDX = 1, CUR_M
                      IF (ABS(DBLE(WORK(D_IDX + (D_IDX - 1)*CUR_M))) +
     $                    ABS(DIMAG(WORK(D_IDX + (D_IDX - 1)*CUR_M)))
     $                    .LT. PERTURB) THEN
                          R_VAL = DBLE(WORK(D_IDX + (D_IDX - 1)*CUR_M))
                          I_VAL = DIMAG(WORK(D_IDX + (D_IDX - 1)*CUR_M))
                          IF (R_VAL .LT. 0.0D0) THEN
                              R_VAL = -PERTURB
                          ELSE
                              R_VAL = PERTURB
                          END IF
                          WORK(D_IDX + (D_IDX - 1)*CUR_M) = 
     $                        DCMPLX(R_VAL, I_VAL)
                      END IF
                  END DO

                  CALL ZTRSM(SIDE, UPLO, TRANSA, DIAG, CUR_M, ONE_INT,
     $                       ONE, WORK, CUR_M, WORK(WORK_RHS_IDX + 1),
     $                       CUR_M)

                  DO R = 1, CUR_M
                      RHS_LOC(R, C_PACKED) = WORK(WORK_RHS_IDX + R)
                  END DO
              END IF
          END IF
      END DO
      RETURN
      END SUBROUTINE ZLALSR