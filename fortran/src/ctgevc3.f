      SUBROUTINE CTGEVC3(SIDE, HOWMNY, SELECT, N, S, LDS, P, LDP,
     $                   ALPHA, BETA, VL, LDVL, VR, LDVR, MM, M, WORK,
     $                   LWORK, INFO)
      IMPLICIT NONE
      CHARACTER SIDE, HOWMNY
      INTEGER N, LDS, LDP, LDVL, LDVR, MM, M, LWORK, INFO
      LOGICAL SELECT(*)
      COMPLEX S(LDS, *), P(LDP, *), VL(LDVL, *), VR(LDVR, *)
      COMPLEX ALPHA(*), BETA(*), WORK(*)

      LOGICAL COMP_R, COMP_L, DO_ALL, DO_BACK, DO_SEL
      INTEGER NUM_SEL, K_IDX, REQ_LWORK
      REAL SAFEMIN, EPS, SMLNUM, BIGNUM
      REAL ANORM, BNORM, SUM_S, SUM_P
      INTEGER COL, ROW
      REAL ASCALE, BSCALE
      INTEGER XPANEL_PTR, TEMP_PTR, WORK_LOC_PTR, TS_PTR, TP_PTR
      INTEGER COL_MAP(128)
      COMPLEX ALPHA_M1, BETA_1, ALPHA_1, ONE, ZERO
      INTEGER CUR_OUT_COL, CURR_COL, I, NB, LD_X, NB_SEL, C, R
      INTEGER CURR_ROW, J, J_NB, IS_DIAG, OUT_COL, C_PACKED
      COMPLEX A, B_VAL, ACOEFF, BCOEFF, XR
      REAL T, UPD_MAX, SAFELIM, SCALE_VAL
      INTEGER TOT_ELEM, IDX, C_IDX, R_IDX
      INTEGER I_NEXT, REM_ROWS, J_NEXT, REM
      INTEGER BSIZE

*     External functions
      REAL DLAMCH
      LOGICAL LSAME
      INTEGER ICLAPB, ICLANB
      EXTERNAL DLAMCH, LSAME, ICLAPB, ICLANB, CGEMM, CLALSR, CLALSL

      BSIZE = 32

      COMP_R = LSAME(SIDE, 'R') .OR. LSAME(SIDE, 'B')
      COMP_L = LSAME(SIDE, 'L') .OR. LSAME(SIDE, 'B')
      DO_ALL = LSAME(HOWMNY, 'A') .OR. LSAME(HOWMNY, 'B')
      DO_BACK = LSAME(HOWMNY, 'B')
      DO_SEL = LSAME(HOWMNY, 'S')

      IF (.NOT. DO_ALL .AND. .NOT. DO_SEL) THEN
          INFO = -2
          RETURN
      END IF

      IF (N .EQ. 0) THEN
          INFO = 0
          M = 0
          RETURN
      END IF

      NUM_SEL = 0
      DO K_IDX = 1, N
          IF (DO_ALL .OR. (DO_SEL .AND. SELECT(K_IDX))) THEN
              NUM_SEL = NUM_SEL + 1
          END IF
      END DO

      M = NUM_SEL
      IF (MM .LT. NUM_SEL) THEN
          INFO = -16
          RETURN
      END IF

      REQ_LWORK = 2 * N * (BSIZE + 1) + 4 * (BSIZE + 1) * (BSIZE + 1) +
     $            2 * (BSIZE + 1)

      IF (LWORK .NE. -1 .AND. LWORK .LT. REQ_LWORK) THEN
          DO BSIZE = BSIZE - 1, 1, -1
              REQ_LWORK = 2 * N * (BSIZE + 1) +
     $                    4 * (BSIZE + 1) * (BSIZE + 1) +
     $                    2 * (BSIZE + 1)
              IF (LWORK .GE. REQ_LWORK) EXIT
          END DO
      END IF

      IF (LWORK .EQ. -1) THEN
          WORK(1) = CMPLX(REAL(REQ_LWORK), 0.0E0)
          INFO = 0
          RETURN
      END IF

      IF (LWORK .LT. REQ_LWORK) THEN
          INFO = -19
          RETURN
      END IF

      INFO = 0

      SAFEMIN = DLAMCH('S')
      EPS = DLAMCH('E')
      SMLNUM = SAFEMIN / EPS
      BIGNUM = 1.0E0 / SMLNUM

      ANORM = 0.0E0
      BNORM = 0.0E0
      DO COL = 1, N
          SUM_S = 0.0E0
          SUM_P = 0.0E0
          DO ROW = 1, COL
              SUM_S = SUM_S + ABS(S(ROW, COL))
              SUM_P = SUM_P + ABS(P(ROW, COL))
          END DO
          ANORM = MAX(ANORM, SUM_S)
          BNORM = MAX(BNORM, SUM_P)
      END DO

      ASCALE = 1.0E0 / MAX(ANORM, SAFEMIN)
      BSCALE = 1.0E0 / MAX(BNORM, SAFEMIN)

      XPANEL_PTR = 1
      TEMP_PTR = XPANEL_PTR + N * (BSIZE + 1)
      WORK_LOC_PTR = TEMP_PTR + N * (BSIZE + 1)

      ALPHA_M1 = CMPLX(-1.0E0, 0.0E0)
      BETA_1 = CMPLX(1.0E0, 0.0E0)
      ALPHA_1 = CMPLX(1.0E0, 0.0E0)
      ONE = CMPLX(1.0E0, 0.0E0)
      ZERO = CMPLX(0.0E0, 0.0E0)

      IF (COMP_R .AND. LDVR .GT. 0) THEN
          CUR_OUT_COL = NUM_SEL
          CURR_COL = N

          DO WHILE (CURR_COL .GT. 0)
              I = ICLAPB(CURR_COL, BSIZE)
              NB = CURR_COL - I + 1
              LD_X = CURR_COL

              NB_SEL = 0
              DO C = 1, NB
                  IF (DO_ALL .OR. (DO_SEL .AND. SELECT(I + C - 1))) THEN
                      NB_SEL = NB_SEL + 1
                      COL_MAP(C) = NB_SEL
                  ELSE
                      COL_MAP(C) = 0
                  END IF
              END DO

              CUR_OUT_COL = CUR_OUT_COL - NB_SEL
              IF (NB_SEL .EQ. 0) THEN
                  CURR_COL = I - 1
                  CYCLE
              END IF

              DO C = 1, NB_SEL
                  DO R = 1, CURR_COL
                      WORK(XPANEL_PTR + R - 1 + (C - 1) * LD_X) = ZERO
                  END DO
              END DO

              CURR_ROW = CURR_COL
              DO WHILE (CURR_ROW .GT. 0)
                  J = ICLAPB(CURR_ROW, BSIZE)
                  J_NB = CURR_ROW - J + 1

                  IF (CURR_ROW .EQ. CURR_COL) THEN
                      IS_DIAG = 1
                  ELSE
                      IS_DIAG = 0
                  END IF

                  CALL CLALSR(LDS, S(J, J), LDP, P(J, J), J_NB, LD_X,
     $                        WORK(XPANEL_PTR + J - 1),
     $                        WORK(XPANEL_PTR), LD_X, NB,
     $                        ALPHA(I), BETA(I), IS_DIAG,
     $                        WORK(WORK_LOC_PTR), ASCALE, BSCALE,
     $                        SAFEMIN, BIGNUM, COL_MAP, NB_SEL)

                  IF (J .GT. 1) THEN
                      TS_PTR = WORK_LOC_PTR
                      TP_PTR = WORK_LOC_PTR + J_NB * NB_SEL

                      DO C = 1, NB
                          C_PACKED = COL_MAP(C)
                          IF (C_PACKED .GT. 0) THEN
                              A = ALPHA(I + C - 1)
                              B_VAL = BETA(I + C - 1)
                              T = 1.0E0 / MAX(ABS(A) * ASCALE,
     $                                        ABS(B_VAL) * BSCALE,
     $                                        SAFEMIN)
                              ACOEFF = CMPLX(T * ASCALE, 0.0E0) *
     $                                 B_VAL * CMPLX(BSCALE, 0.0E0)
                              BCOEFF = CMPLX(T * BSCALE, 0.0E0) *
     $                                 A * CMPLX(ASCALE, 0.0E0)

                              IF (ABS(B_VAL) .LE. SAFEMIN .AND.
     $                            ABS(A) .GT. SAFEMIN) THEN
                                  ACOEFF = ZERO
                                  BCOEFF = ONE
                              END IF

                              DO R = 1, J_NB
                                  XR = WORK(XPANEL_PTR + J + R - 2 +
     $                                      (C_PACKED - 1) * LD_X)
                                  WORK(TS_PTR + R - 1 +
     $                                 (C_PACKED - 1) * J_NB) =
     $                              ACOEFF * XR
                                  WORK(TP_PTR + R - 1 +
     $                                 (C_PACKED - 1) * J_NB) =
     $                              BCOEFF * XR
                              END DO
                          END IF
                      END DO

                      UPD_MAX = 0.0E0
                      TOT_ELEM = J_NB * NB_SEL
                      DO IDX = 1, TOT_ELEM
                          UPD_MAX = MAX(UPD_MAX,
     $                      ABS(REAL(WORK(TS_PTR + IDX - 1))) +
     $                      ABS(AIMAG(WORK(TS_PTR + IDX - 1))),
     $                      ABS(REAL(WORK(TP_PTR + IDX - 1))) +
     $                      ABS(AIMAG(WORK(TP_PTR + IDX - 1))))
                      END DO

                      SAFELIM = BIGNUM / REAL(MAX(1, NB_SEL))

                      IF (UPD_MAX .GT. SAFELIM) THEN
                          SCALE_VAL = SAFELIM / UPD_MAX
                          DO IDX = 1, TOT_ELEM
                              WORK(TS_PTR + IDX - 1) =
     $                          WORK(TS_PTR + IDX - 1) *
     $                          CMPLX(SCALE_VAL, 0.0E0)
                              WORK(TP_PTR + IDX - 1) =
     $                          WORK(TP_PTR + IDX - 1) *
     $                          CMPLX(SCALE_VAL, 0.0E0)
                          END DO
                          DO C_IDX = 1, NB_SEL
                              DO R_IDX = 1, LD_X
                                  WORK(XPANEL_PTR + R_IDX - 1 +
     $                                 (C_IDX - 1) * LD_X) =
     $                              WORK(XPANEL_PTR + R_IDX - 1 +
     $                                   (C_IDX - 1) * LD_X) *
     $                              CMPLX(SCALE_VAL, 0.0E0)
                              END DO
                          END DO
                      END IF

                      CALL CGEMM('N', 'N', J - 1, NB_SEL, J_NB,
     $                           ALPHA_M1, S(1, J), LDS,
     $                           WORK(TS_PTR), J_NB, BETA_1,
     $                           WORK(XPANEL_PTR), LD_X)
                      CALL CGEMM('N', 'N', J - 1, NB_SEL, J_NB,
     $                           ALPHA_1, P(1, J), LDP,
     $                           WORK(TP_PTR), J_NB, BETA_1,
     $                           WORK(XPANEL_PTR), LD_X)
                  END IF

                  CURR_ROW = J - 1
              END DO

              IF (DO_BACK) THEN
                  CALL CGEMM('N', 'N', N, NB_SEL, CURR_COL, ONE,
     $                       VR, LDVR, WORK(XPANEL_PTR), LD_X, ZERO,
     $                       WORK(TEMP_PTR), N)
                  DO C = 1, NB_SEL
                      OUT_COL = CUR_OUT_COL + C
                      DO R = 1, N
                          VR(R, OUT_COL) = WORK(TEMP_PTR + R - 1 +
     $                                          (C - 1) * N)
                      END DO
                  END DO
              ELSE
                  DO C = 1, NB_SEL
                      OUT_COL = CUR_OUT_COL + C
                      DO R = 1, CURR_COL
                          VR(R, OUT_COL) = WORK(XPANEL_PTR + R - 1 +
     $                                          (C - 1) * LD_X)
                      END DO
                      DO R = CURR_COL + 1, N
                          VR(R, OUT_COL) = ZERO
                      END DO
                  END DO
              END IF

              CURR_COL = I - 1
          END DO
      END IF

      IF (COMP_L .AND. LDVL .GT. 0) THEN
          CUR_OUT_COL = 0
          CURR_COL = 1

          DO WHILE (CURR_COL .LE. N)
              I_NEXT = ICLANB(N, CURR_COL, BSIZE)
              NB = I_NEXT - CURR_COL + 1
              I = CURR_COL
              REM_ROWS = N - I + 1
              LD_X = REM_ROWS

              NB_SEL = 0
              DO C = 1, NB
                  IF (DO_ALL .OR. (DO_SEL .AND. SELECT(I + C - 1))) THEN
                      NB_SEL = NB_SEL + 1
                      COL_MAP(C) = NB_SEL
                  ELSE
                      COL_MAP(C) = 0
                  END IF
              END DO

              IF (NB_SEL .EQ. 0) THEN
                  CURR_COL = I_NEXT + 1
                  CYCLE
              END IF

              DO C = 1, NB_SEL
                  DO R = 1, REM_ROWS
                      WORK(XPANEL_PTR + R - 1 + (C - 1) * LD_X) = ZERO
                  END DO
              END DO

              CURR_ROW = I
              DO WHILE (CURR_ROW .LE. N)
                  J_NEXT = ICLANB(N, CURR_ROW, BSIZE)
                  J_NB = J_NEXT - CURR_ROW + 1

                  IF (CURR_ROW .EQ. I) THEN
                      IS_DIAG = 1
                  ELSE
                      IS_DIAG = 0
                  END IF

                  CALL CLALSL(LDS, S(CURR_ROW, CURR_ROW), LDP,
     $                        P(CURR_ROW, CURR_ROW), J_NB, LD_X,
     $                        WORK(XPANEL_PTR + CURR_ROW - I),
     $                        WORK(XPANEL_PTR), LD_X, NB,
     $                        ALPHA(I), BETA(I), IS_DIAG,
     $                        WORK(WORK_LOC_PTR), ASCALE, BSCALE,
     $                        SAFEMIN, BIGNUM, COL_MAP, NB_SEL)

                  IF (J_NEXT .LT. N) THEN
                      TS_PTR = WORK_LOC_PTR
                      TP_PTR = WORK_LOC_PTR + J_NB * NB_SEL

                      DO C = 1, NB
                          C_PACKED = COL_MAP(C)
                          IF (C_PACKED .GT. 0) THEN
                              A = ALPHA(I + C - 1)
                              B_VAL = BETA(I + C - 1)
                              T = 1.0E0 / MAX(ABS(A) * ASCALE,
     $                                        ABS(B_VAL) * BSCALE,
     $                                        SAFEMIN)
                              ACOEFF = CMPLX(T * ASCALE, 0.0E0) *
     $                                 B_VAL * CMPLX(BSCALE, 0.0E0)
                              BCOEFF = CMPLX(T * BSCALE, 0.0E0) *
     $                                 A * CMPLX(ASCALE, 0.0E0)

                              IF (ABS(B_VAL) .LE. SAFEMIN .AND.
     $                            ABS(A) .GT. SAFEMIN) THEN
                                  ACOEFF = ZERO
                                  BCOEFF = ONE
                              END IF

                              DO R = 1, J_NB
                                  XR = WORK(XPANEL_PTR + CURR_ROW - I +
     $                                      R - 1 +
     $                                      (C_PACKED - 1) * LD_X)
                                  WORK(TS_PTR + R - 1 +
     $                                 (C_PACKED - 1) * J_NB) =
     $                              CONJG(ACOEFF) * XR
                                  WORK(TP_PTR + R - 1 +
     $                                 (C_PACKED - 1) * J_NB) =
     $                              CONJG(BCOEFF) * XR
                              END DO
                          END IF
                      END DO

                      REM = N - J_NEXT
                      UPD_MAX = 0.0E0
                      TOT_ELEM = J_NB * NB_SEL
                      DO IDX = 1, TOT_ELEM
                          UPD_MAX = MAX(UPD_MAX,
     $                      ABS(REAL(WORK(TS_PTR + IDX - 1))) +
     $                      ABS(AIMAG(WORK(TS_PTR + IDX - 1))),
     $                      ABS(REAL(WORK(TP_PTR + IDX - 1))) +
     $                      ABS(AIMAG(WORK(TP_PTR + IDX - 1))))
                      END DO

                      SAFELIM = BIGNUM / REAL(MAX(1, NB_SEL))

                      IF (UPD_MAX .GT. SAFELIM) THEN
                          SCALE_VAL = SAFELIM / UPD_MAX
                          DO IDX = 1, TOT_ELEM
                              WORK(TS_PTR + IDX - 1) =
     $                          WORK(TS_PTR + IDX - 1) *
     $                          CMPLX(SCALE_VAL, 0.0E0)
                              WORK(TP_PTR + IDX - 1) =
     $                          WORK(TP_PTR + IDX - 1) *
     $                          CMPLX(SCALE_VAL, 0.0E0)
                          END DO
                          DO C_IDX = 1, NB_SEL
                              DO R_IDX = 1, LD_X
                                  WORK(XPANEL_PTR + R_IDX - 1 +
     $                                 (C_IDX - 1) * LD_X) =
     $                              WORK(XPANEL_PTR + R_IDX - 1 +
     $                                   (C_IDX - 1) * LD_X) *
     $                              CMPLX(SCALE_VAL, 0.0E0)
                              END DO
                          END DO
                      END IF

                      CALL CGEMM('C', 'N', REM, NB_SEL, J_NB,
     $                           ALPHA_M1, S(CURR_ROW, J_NEXT + 1), LDS,
     $                           WORK(TS_PTR), J_NB, BETA_1,
     $                           WORK(XPANEL_PTR + J_NEXT - I + 1),
     $                           LD_X)
                      CALL CGEMM('C', 'N', REM, NB_SEL, J_NB,
     $                           ALPHA_1, P(CURR_ROW, J_NEXT + 1), LDP,
     $                           WORK(TP_PTR), J_NB, BETA_1,
     $                           WORK(XPANEL_PTR + J_NEXT - I + 1),
     $                           LD_X)
                  END IF

                  CURR_ROW = J_NEXT + 1
              END DO

              IF (DO_BACK) THEN
                  CALL CGEMM('N', 'N', N, NB_SEL, REM_ROWS, ONE,
     $                       VL(1, I), LDVL, WORK(XPANEL_PTR), LD_X,
     $                       ZERO, WORK(TEMP_PTR), N)
                  DO C = 1, NB_SEL
                      OUT_COL = CUR_OUT_COL + C
                      DO R = 1, N
                          VL(R, OUT_COL) = WORK(TEMP_PTR + R - 1 +
     $                                          (C - 1) * N)
                      END DO
                  END DO
              ELSE
                  DO C = 1, NB_SEL
                      OUT_COL = CUR_OUT_COL + C
                      DO R = 1, I - 1
                          VL(R, OUT_COL) = ZERO
                      END DO
                      DO R = 1, REM_ROWS
                          VL(I + R - 1, OUT_COL) =
     $                      WORK(XPANEL_PTR + R - 1 + (C - 1) * LD_X)
                      END DO
                  END DO
              END IF
              CUR_OUT_COL = CUR_OUT_COL + NB_SEL

              CURR_COL = I_NEXT + 1
          END DO
      END IF

      RETURN
      END SUBROUTINE CTGEVC3