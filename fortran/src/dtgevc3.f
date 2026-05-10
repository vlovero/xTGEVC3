      SUBROUTINE DTGEVC3(SIDE, HOWMNY, SELECT, N, S, LDS, P, LDP,
     $                   ALPHAR, ALPHAI, BETA, VL, LDVL, VR, LDVR,
     $                   MM, M, WORK, LWORK, INFO)
      IMPLICIT NONE
      CHARACTER SIDE, HOWMNY
      LOGICAL SELECT(*)
      INTEGER N, LDS, LDP, LDVL, LDVR, MM, M, LWORK, INFO
      DOUBLE PRECISION S(LDS, *), P(LDP, *)
      DOUBLE PRECISION ALPHAR(*), ALPHAI(*), BETA(*)
      DOUBLE PRECISION VL(LDVL, *), VR(LDVR, *)
      DOUBLE PRECISION WORK(*)

      LOGICAL COMPUTE_RIGHT, COMPUTE_LEFT, DO_ALL, DO_BACK, DO_SEL
      LOGICAL SELECTED
      INTEGER REQ_LWORK
      INTEGER X_PANEL_OFFSET, TEMP_OFFSET, WORK_LOCAL_OFFSET
      INTEGER CURR_COL, I, NB, LD_X, C, R, CURR_ROW, J, J_NB, IS_DIAG
      INTEGER I_NEXT, REM_ROWS, J_NEXT, REM
      DOUBLE PRECISION B_VAL, AR, AI, XR, XI, XMAX
      DOUBLE PRECISION ALPHA_M1, BETA_1, ALPHA_1, ONE, ZERO
      DOUBLE PRECISION SAFEMIN, EPS, SMLNUM, BIGNUM
      DOUBLE PRECISION ANORM, BNORM, SUM_S, SUM_P
      INTEGER COL, ROW, ROW_LIMIT
      DOUBLE PRECISION ASCALE, BSCALE
      DOUBLE PRECISION T, ACOEFF, BCOEFFR, BCOEFFI
      DOUBLE PRECISION UPDATE_MAX, SAFE_LIMIT, SCALE
      INTEGER TOTAL_ELEMENTS, IDX, C_IDX, R_IDX
      INTEGER BSIZE, NUM_SEL, K_IDX, C_PACKED, OUT_COL
      INTEGER CURRENT_OUT_COL
      INTEGER COL_MAP(128)
      INTEGER NB_SEL

      INTEGER IDLAPB, IDLANB
      EXTERNAL IDLAPB, IDLANB
      EXTERNAL DLAUHS, DLAU2S, DLALHS, DLAL2S, DLALSR, DLALSL
      EXTERNAL DGEMM

      DOUBLE PRECISION DLAMCH
      EXTERNAL DLAMCH
      LOGICAL LSAME
      EXTERNAL LSAME

      BSIZE = 32

* Decode and validate parameters
      COMPUTE_RIGHT = LSAME(SIDE, 'R') .OR. LSAME(SIDE, 'B')
      COMPUTE_LEFT = LSAME(SIDE, 'L') .OR. LSAME(SIDE, 'B')
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

* Pass 1: Count total selected eigenvalues
      NUM_SEL = 0
      K_IDX = 1
      DO WHILE (K_IDX .LE. N)
         SELECTED = DO_ALL
         IF (DO_SEL) THEN
            SELECTED = SELECT(K_IDX)
            IF (ALPHAI(K_IDX) .NE. 0.0D0 .AND. K_IDX .LT. N) THEN
               SELECTED = (SELECTED .OR. SELECT(K_IDX + 1))
            END IF
         END IF

         IF (SELECTED) THEN
            IF (ALPHAI(K_IDX) .NE. 0.0D0 .AND. K_IDX .LT. N) THEN
               NUM_SEL = NUM_SEL + 2
            ELSE
               NUM_SEL = NUM_SEL + 1
            END IF
         END IF
         IF (ALPHAI(K_IDX) .NE. 0.0D0 .AND. K_IDX .LT. N) THEN
            K_IDX = K_IDX + 2
         ELSE
            K_IDX = K_IDX + 1
         END IF
      END DO

      M = NUM_SEL

      IF (MM .LT. NUM_SEL) THEN
         INFO = -16
         RETURN
      END IF

* Determine required workspace size
      REQ_LWORK = 2 * N * (BSIZE + 1) + 4 * (BSIZE + 1) * (BSIZE + 1) + 
     $            2 * (BSIZE + 1)

      IF (LWORK .NE. -1 .AND. LWORK .LT. REQ_LWORK) THEN
         DO BSIZE = 31, 1, -1
            REQ_LWORK = 2 * N * (BSIZE + 1) + 
     $                  4 * (BSIZE + 1) * (BSIZE + 1) + 2 * (BSIZE + 1)
            IF (LWORK .GE. REQ_LWORK) THEN
               EXIT
            END IF
         END DO
      END IF

      IF (LWORK .EQ. -1) THEN
         WORK(1) = DBLE(REQ_LWORK)
         INFO = 0
         RETURN
      END IF

      IF (LWORK .LT. REQ_LWORK) THEN
         INFO = -19
         RETURN
      END IF

      INFO = 0

* Retrieve machine constants
      SAFEMIN = DLAMCH('S')
      EPS = DLAMCH('E')
      SMLNUM = SAFEMIN / EPS
      BIGNUM = 1.0D0 / SMLNUM

* Compute the 1-norm of S and P
      ANORM = 0.0D0
      BNORM = 0.0D0
      DO COL = 1, N
         SUM_S = 0.0D0
         SUM_P = 0.0D0
         ROW_LIMIT = MIN(N, COL + 1)
         DO ROW = 1, ROW_LIMIT
            SUM_S = SUM_S + ABS(S(ROW, COL))
            SUM_P = SUM_P + ABS(P(ROW, COL))
         END DO
         ANORM = MAX(ANORM, SUM_S)
         BNORM = MAX(BNORM, SUM_P)
      END DO

      ASCALE = 1.0D0 / MAX(ANORM, SAFEMIN)
      BSCALE = 1.0D0 / MAX(BNORM, SAFEMIN)

* Partition the workspace
      X_PANEL_OFFSET = 0
      TEMP_OFFSET = X_PANEL_OFFSET + N * (BSIZE + 1)
      WORK_LOCAL_OFFSET = TEMP_OFFSET + N * (BSIZE + 1)

* ==================================================================
* Right Eigenvector Computation
* ==================================================================
      IF (COMPUTE_RIGHT) THEN
         CURRENT_OUT_COL = NUM_SEL
         CURR_COL = N

         DO WHILE (CURR_COL .GT. 0)
            I = IDLAPB(S, LDS, CURR_COL, BSIZE)
            NB = CURR_COL - I + 1
            LD_X = CURR_COL

            NB_SEL = 0
            C = 1
            DO WHILE (C .LE. NB)
               SELECTED = DO_ALL
               IF (DO_SEL) THEN
                  SELECTED = SELECT(I + C - 1)
                  IF (ALPHAI(I + C - 1) .NE. 0.0D0 .AND. 
     $                I + C - 1 .LT. N) THEN
                     SELECTED = (SELECTED .OR. SELECT(I + C))
                  END IF
               END IF

               IF (SELECTED) THEN
                  NB_SEL = NB_SEL + 1
                  COL_MAP(C) = NB_SEL
                  IF (ALPHAI(I + C - 1) .NE. 0.0D0 .AND. 
     $                I + C - 1 .LT. N) THEN
                     NB_SEL = NB_SEL + 1
                     COL_MAP(C + 1) = NB_SEL
                     C = C + 2
                  ELSE
                     C = C + 1
                  END IF
               ELSE
                  COL_MAP(C) = -1
                  IF (ALPHAI(I + C - 1) .NE. 0.0D0 .AND. 
     $                I + C - 1 .LT. N) THEN
                     COL_MAP(C + 1) = -1
                     C = C + 2
                  ELSE
                     C = C + 1
                  END IF
               END IF
            END DO

            CURRENT_OUT_COL = CURRENT_OUT_COL - NB_SEL

            IF (NB_SEL .EQ. 0) THEN
               CURR_COL = I - 1
               CYCLE
            END IF

            DO C = 1, NB_SEL
               DO R = 1, CURR_COL
                  WORK(X_PANEL_OFFSET + R + (C - 1) * LD_X) = 0.0D0
               END DO
            END DO

            CURR_ROW = CURR_COL
            DO WHILE (CURR_ROW .GT. 0)
               J = IDLAPB(S, LDS, CURR_ROW, BSIZE)
               J_NB = CURR_ROW - J + 1

               IF (CURR_ROW .EQ. CURR_COL) THEN
                  IS_DIAG = 1
               ELSE
                  IS_DIAG = 0
               END IF

               CALL DLALSR(LDS, S(J, J), LDP, P(J, J), J_NB, LD_X,
     $                     WORK(X_PANEL_OFFSET + J), 
     $                     WORK(X_PANEL_OFFSET + 1), LD_X, NB,
     $                     ALPHAR(I), ALPHAI(I), BETA(I), IS_DIAG,
     $                     WORK(WORK_LOCAL_OFFSET + 1), ASCALE, BSCALE,
     $                     SAFEMIN, BIGNUM, COL_MAP, NB_SEL)

               IF (J .GT. 1) THEN
                  TOTAL_ELEMENTS = J_NB * NB_SEL
                  
                  C = 1
                  DO WHILE (C .LE. NB)
                     C_PACKED = COL_MAP(C)
                     IF (C_PACKED .LT. 1) THEN
                        IF (ALPHAI(I + C - 1) .EQ. 0.0D0) THEN
                           C = C + 1
                        ELSE
                           C = C + 2
                        END IF
                        CYCLE
                     END IF

                     B_VAL = BETA(I + C - 1)
                     AR = ALPHAR(I + C - 1)
                     AI = ALPHAI(I + C - 1)

                     T = 1.0D0 / MAX(ABS(AR) * ASCALE + ABS(AI) * 
     $                               ASCALE, ABS(B_VAL) * BSCALE, 
     $                               SAFEMIN)
                     ACOEFF = (T * B_VAL * BSCALE) * ASCALE
                     BCOEFFR = (T * AR * ASCALE) * BSCALE
                     BCOEFFI = (T * AI * ASCALE) * BSCALE

                     IF (ABS(B_VAL) .LE. SAFEMIN .AND. 
     $                   (ABS(AR) + ABS(AI)) .GT. SAFEMIN) THEN
                        ACOEFF = 0.0D0
                        BCOEFFR = 1.0D0
                        BCOEFFI = 0.0D0
                     END IF

                     IF (AI .EQ. 0.0D0) THEN
                        DO R = 1, J_NB
                           XR = WORK(X_PANEL_OFFSET + (J - 1 + R) + 
     $                               (C_PACKED - 1) * LD_X)
                           WORK(WORK_LOCAL_OFFSET + R + 
     $                          (C_PACKED - 1) * J_NB) = ACOEFF * XR
                           WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                          R + (C_PACKED - 1) * J_NB) = 
     $                          BCOEFFR * XR
                        END DO
                        C = C + 1
                     ELSE
                        DO R = 1, J_NB
                           XR = WORK(X_PANEL_OFFSET + (J - 1 + R) + 
     $                               (C_PACKED - 1) * LD_X)
                           XI = WORK(X_PANEL_OFFSET + (J - 1 + R) + 
     $                               C_PACKED * LD_X)
                           WORK(WORK_LOCAL_OFFSET + R + 
     $                          (C_PACKED - 1) * J_NB) = ACOEFF * XR
                           WORK(WORK_LOCAL_OFFSET + R + 
     $                          C_PACKED * J_NB) = ACOEFF * XI
                           WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                          R + (C_PACKED - 1) * J_NB) = 
     $                          BCOEFFR * XR - BCOEFFI * XI
                           WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                          R + C_PACKED * J_NB) = 
     $                          BCOEFFR * XI + BCOEFFI * XR
                        END DO
                        C = C + 2
                     END IF
                  END DO

                  UPDATE_MAX = 0.0D0
                  DO IDX = 1, TOTAL_ELEMENTS
                     UPDATE_MAX = MAX(UPDATE_MAX, 
     $                  ABS(WORK(WORK_LOCAL_OFFSET + IDX)),
     $                  ABS(WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                  IDX)))
                  END DO

                  SAFE_LIMIT = BIGNUM / DBLE(MAX(1, NB_SEL))

                  IF (UPDATE_MAX .GT. SAFE_LIMIT) THEN
                     SCALE = SAFE_LIMIT / UPDATE_MAX
                     DO IDX = 1, TOTAL_ELEMENTS
                        WORK(WORK_LOCAL_OFFSET + IDX) = 
     $                     WORK(WORK_LOCAL_OFFSET + IDX) * SCALE
                        WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                     IDX) = WORK(WORK_LOCAL_OFFSET + 
     $                     TOTAL_ELEMENTS + IDX) * SCALE
                     END DO
                     DO C_IDX = 1, NB_SEL
                        DO R_IDX = 1, LD_X
                           WORK(X_PANEL_OFFSET + R_IDX + 
     $                          (C_IDX - 1) * LD_X) = 
     $                     WORK(X_PANEL_OFFSET + R_IDX + 
     $                          (C_IDX - 1) * LD_X) * SCALE
                        END DO
                     END DO
                  END IF

                  ALPHA_M1 = -1.0D0
                  BETA_1 = 1.0D0
                  ALPHA_1 = 1.0D0
                  CALL DGEMM('N', 'N', J - 1, NB_SEL, J_NB, ALPHA_M1,
     $                       S(1, J), LDS, 
     $                       WORK(WORK_LOCAL_OFFSET + 1), J_NB, BETA_1,
     $                       WORK(X_PANEL_OFFSET + 1), LD_X)
                  CALL DGEMM('N', 'N', J - 1, NB_SEL, J_NB, ALPHA_1,
     $                       P(1, J), LDP, 
     $                       WORK(WORK_LOCAL_OFFSET + 
     $                            TOTAL_ELEMENTS + 1), J_NB, BETA_1,
     $                       WORK(X_PANEL_OFFSET + 1), LD_X)
               END IF
               CURR_ROW = J - 1
            END DO

            IF (DO_BACK) THEN
               ONE = 1.0D0
               ZERO = 0.0D0
               CALL DGEMM('N', 'N', N, NB_SEL, CURR_COL, ONE, VR, LDVR,
     $                    WORK(X_PANEL_OFFSET + 1), LD_X, ZERO,
     $                    WORK(TEMP_OFFSET + 1), N)

               DO C = 1, NB_SEL
                  OUT_COL = CURRENT_OUT_COL + C
                  DO R = 1, N
                     VR(R, OUT_COL) = WORK(TEMP_OFFSET + R + 
     $                                (C - 1) * N)
                  END DO
               END DO
            ELSE
               DO C = 1, NB_SEL
                  OUT_COL = CURRENT_OUT_COL + C
                  DO R = 1, CURR_COL
                     VR(R, OUT_COL) = WORK(X_PANEL_OFFSET + R + 
     $                                (C - 1) * LD_X)
                  END DO
                  DO R = CURR_COL + 1, N
                     VR(R, OUT_COL) = 0.0D0
                  END DO
               END DO
            END IF

* 5. scale eigenvectors
            C = 1
            DO WHILE (C .LE. NB)
               C_PACKED = COL_MAP(C)
               IF (C_PACKED .LT. 1) THEN
                  IF (ALPHAI(I + C - 1) .EQ. 0.0D0) THEN
                     C = C + 1
                  ELSE
                     C = C + 2
                  END IF
                  CYCLE
               END IF

               OUT_COL = CURRENT_OUT_COL + C_PACKED

               IF (ALPHAI(I + C - 1) .EQ. 0.0D0) THEN
                  XMAX = 0.0D0
                  DO R = 1, N
                     XMAX = MAX(XMAX, ABS(VR(R, OUT_COL)))
                  END DO
                  IF (XMAX .GT. SAFEMIN) THEN
                     XMAX = 1.0D0 / XMAX
                     DO R = 1, N
                        VR(R, OUT_COL) = VR(R, OUT_COL) * XMAX
                     END DO
                  END IF
                  C = C + 1
               ELSE
                  XMAX = 0.0D0
                  DO R = 1, N
                     XMAX = MAX(XMAX, ABS(VR(R, OUT_COL)) +
     $                                ABS(VR(R, OUT_COL + 1)))
                  END DO
                  IF (XMAX .GT. SAFEMIN) THEN
                     XMAX = 1.0D0 / XMAX
                     DO R = 1, N
                        VR(R, OUT_COL) = VR(R, OUT_COL) * XMAX
                        VR(R, OUT_COL + 1) = VR(R, OUT_COL + 1) * XMAX
                     END DO
                  END IF
                  C = C + 2
               END IF
            END DO

            CURR_COL = I - 1
         END DO
      END IF

* ==================================================================
* Left Eigenvector Computation
* ==================================================================
      IF (COMPUTE_LEFT) THEN
         CURRENT_OUT_COL = 0
         CURR_COL = 1

         DO WHILE (CURR_COL .LE. N)
            I_NEXT = IDLANB(S, N, LDS, CURR_COL, BSIZE)
            NB = I_NEXT - CURR_COL + 1
            I = CURR_COL
            REM_ROWS = N - I + 1
            LD_X = REM_ROWS

            NB_SEL = 0
            C = 1
            DO WHILE (C .LE. NB)
               SELECTED = DO_ALL
               IF (DO_SEL) THEN
                  SELECTED = SELECT(I + C - 1)
                  IF (ALPHAI(I + C - 1) .NE. 0.0D0 .AND. 
     $                I + C - 1 .LT. N) THEN
                     SELECTED = (SELECTED .OR. SELECT(I + C))
                  END IF
               END IF

               IF (SELECTED) THEN
                  NB_SEL = NB_SEL + 1
                  COL_MAP(C) = NB_SEL
                  IF (ALPHAI(I + C - 1) .NE. 0.0D0 .AND. 
     $                I + C - 1 .LT. N) THEN
                     NB_SEL = NB_SEL + 1
                     COL_MAP(C + 1) = NB_SEL
                     C = C + 2
                  ELSE
                     C = C + 1
                  END IF
               ELSE
                  COL_MAP(C) = -1
                  IF (ALPHAI(I + C - 1) .NE. 0.0D0 .AND. 
     $                I + C - 1 .LT. N) THEN
                     COL_MAP(C + 1) = -1
                     C = C + 2
                  ELSE
                     C = C + 1
                  END IF
               END IF
            END DO

            IF (NB_SEL .EQ. 0) THEN
               CURR_COL = I_NEXT + 1
               CYCLE
            END IF

            DO C = 1, NB_SEL
               DO R = 1, REM_ROWS
                  WORK(X_PANEL_OFFSET + R + (C - 1) * LD_X) = 0.0D0
               END DO
            END DO

            CURR_ROW = I
            DO WHILE (CURR_ROW .LE. N)
               J_NEXT = IDLANB(S, N, LDS, CURR_ROW, BSIZE)
               J_NB = J_NEXT - CURR_ROW + 1
               IF (CURR_ROW .EQ. I) THEN
                  IS_DIAG = 1
               ELSE
                  IS_DIAG = 0
               END IF

               CALL DLALSL(LDS, S(CURR_ROW, CURR_ROW), LDP, 
     $                     P(CURR_ROW, CURR_ROW), J_NB, LD_X,
     $                     WORK(X_PANEL_OFFSET + CURR_ROW - I + 1), 
     $                     WORK(X_PANEL_OFFSET + 1), LD_X, NB,
     $                     ALPHAR(I), ALPHAI(I), BETA(I), IS_DIAG,
     $                     WORK(WORK_LOCAL_OFFSET + 1), ASCALE, BSCALE,
     $                     SAFEMIN, BIGNUM, COL_MAP, NB_SEL)

               IF (J_NEXT .LT. N) THEN
                  TOTAL_ELEMENTS = J_NB * NB_SEL

                  C = 1
                  DO WHILE (C .LE. NB)
                     C_PACKED = COL_MAP(C)
                     IF (C_PACKED .LT. 1) THEN
                        IF (ALPHAI(I + C - 1) .EQ. 0.0D0) THEN
                           C = C + 1
                        ELSE
                           C = C + 2
                        END IF
                        CYCLE
                     END IF

                     B_VAL = BETA(I + C - 1)
                     AR = ALPHAR(I + C - 1)
                     AI = ALPHAI(I + C - 1)

                     T = 1.0D0 / MAX(ABS(AR) * ASCALE + ABS(AI) *
     $                               ASCALE, ABS(B_VAL) * BSCALE, 
     $                               SAFEMIN)
                     ACOEFF = (T * B_VAL * BSCALE) * ASCALE
                     BCOEFFR = (T * AR * ASCALE) * BSCALE
                     BCOEFFI = (T * AI * ASCALE) * BSCALE

                     IF (ABS(B_VAL) .LE. SAFEMIN .AND. 
     $                   (ABS(AR) + ABS(AI)) .GT. SAFEMIN) THEN
                        ACOEFF = 0.0D0
                        BCOEFFR = 1.0D0
                        BCOEFFI = 0.0D0
                     END IF

                     IF (AI .EQ. 0.0D0) THEN
                        DO R = 1, J_NB
                           XR = WORK(X_PANEL_OFFSET + (CURR_ROW - I + 
     $                               R) + (C_PACKED - 1) * LD_X)
                           WORK(WORK_LOCAL_OFFSET + R + 
     $                          (C_PACKED - 1) * J_NB) = ACOEFF * XR
                           WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                          R + (C_PACKED - 1) * J_NB) = 
     $                          BCOEFFR * XR
                        END DO
                        C = C + 1
                     ELSE
                        DO R = 1, J_NB
                           XR = WORK(X_PANEL_OFFSET + (CURR_ROW - I + 
     $                               R) + (C_PACKED - 1) * LD_X)
                           XI = WORK(X_PANEL_OFFSET + (CURR_ROW - I + 
     $                               R) + C_PACKED * LD_X)
                           WORK(WORK_LOCAL_OFFSET + R + 
     $                          (C_PACKED - 1) * J_NB) = ACOEFF * XR
                           WORK(WORK_LOCAL_OFFSET + R + 
     $                          C_PACKED * J_NB) = ACOEFF * XI
                           WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                          R + (C_PACKED - 1) * J_NB) = 
     $                          BCOEFFR * XR + BCOEFFI * XI
                           WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                          R + C_PACKED * J_NB) = 
     $                          BCOEFFR * XI - BCOEFFI * XR
                        END DO
                        C = C + 2
                     END IF
                  END DO

                  REM = N - J_NEXT

                  UPDATE_MAX = 0.0D0
                  DO IDX = 1, TOTAL_ELEMENTS
                     UPDATE_MAX = MAX(UPDATE_MAX, 
     $                  ABS(WORK(WORK_LOCAL_OFFSET + IDX)),
     $                  ABS(WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                  IDX)))
                  END DO

                  SAFE_LIMIT = BIGNUM / DBLE(MAX(1, NB_SEL))

                  IF (UPDATE_MAX .GT. SAFE_LIMIT) THEN
                     SCALE = SAFE_LIMIT / UPDATE_MAX
                     DO IDX = 1, TOTAL_ELEMENTS
                        WORK(WORK_LOCAL_OFFSET + IDX) = 
     $                     WORK(WORK_LOCAL_OFFSET + IDX) * SCALE
                        WORK(WORK_LOCAL_OFFSET + TOTAL_ELEMENTS + 
     $                     IDX) = WORK(WORK_LOCAL_OFFSET + 
     $                     TOTAL_ELEMENTS + IDX) * SCALE
                     END DO
                     DO C_IDX = 1, NB_SEL
                        DO R_IDX = 1, LD_X
                           WORK(X_PANEL_OFFSET + R_IDX + 
     $                          (C_IDX - 1) * LD_X) = 
     $                     WORK(X_PANEL_OFFSET + R_IDX + 
     $                          (C_IDX - 1) * LD_X) * SCALE
                        END DO
                     END DO
                  END IF

                  ALPHA_M1 = -1.0D0
                  BETA_1 = 1.0D0
                  ALPHA_1 = 1.0D0
                  CALL DGEMM('T', 'N', REM, NB_SEL, J_NB, ALPHA_M1,
     $                       S(CURR_ROW, J_NEXT + 1), LDS, 
     $                       WORK(WORK_LOCAL_OFFSET + 1), J_NB, BETA_1,
     $                       WORK(X_PANEL_OFFSET + J_NEXT - I + 2), 
     $                       LD_X)
                  CALL DGEMM('T', 'N', REM, NB_SEL, J_NB, ALPHA_1,
     $                       P(CURR_ROW, J_NEXT + 1), LDP, 
     $                       WORK(WORK_LOCAL_OFFSET + 
     $                       TOTAL_ELEMENTS + 1), J_NB, BETA_1,
     $                       WORK(X_PANEL_OFFSET + J_NEXT - I + 2), 
     $                       LD_X)
               END IF
               CURR_ROW = J_NEXT + 1
            END DO

            IF (DO_BACK) THEN
               ONE = 1.0D0
               ZERO = 0.0D0
               CALL DGEMM('N', 'N', N, NB_SEL, REM_ROWS, ONE, 
     $                    VL(1, I), LDVL, WORK(X_PANEL_OFFSET + 1), 
     $                    LD_X, ZERO, WORK(TEMP_OFFSET + 1), N)

               DO C = 1, NB_SEL
                  OUT_COL = CURRENT_OUT_COL + C
                  DO R = 1, N
                     VL(R, OUT_COL) = WORK(TEMP_OFFSET + R + 
     $                                     (C - 1) * N)
                  END DO
               END DO
            ELSE
               DO C = 1, NB_SEL
                  OUT_COL = CURRENT_OUT_COL + C
                  DO R = 1, I - 1
                     VL(R, OUT_COL) = 0.0D0
                  END DO
                  DO R = 1, REM_ROWS
                     VL(I - 1 + R, OUT_COL) = WORK(X_PANEL_OFFSET + 
     $                                             R + (C - 1) * LD_X)
                  END DO
               END DO
            END IF

* 5. scale eigenvectors
            C = 1
            DO WHILE (C .LE. NB)
               C_PACKED = COL_MAP(C)
               IF (C_PACKED .LT. 1) THEN
                  IF (ALPHAI(I + C - 1) .EQ. 0.0D0) THEN
                     C = C + 1
                  ELSE
                     C = C + 2
                  END IF
                  CYCLE
               END IF

               OUT_COL = CURRENT_OUT_COL + C_PACKED

               IF (ALPHAI(I + C - 1) .EQ. 0.0D0) THEN
                  XMAX = 0.0D0
                  DO R = 1, N
                     XMAX = MAX(XMAX, ABS(VL(R, OUT_COL)))
                  END DO
                  IF (XMAX .GT. SAFEMIN) THEN
                     XMAX = 1.0D0 / XMAX
                     DO R = 1, N
                        VL(R, OUT_COL) = VL(R, OUT_COL) * XMAX
                     END DO
                  END IF
                  C = C + 1
               ELSE
                  XMAX = 0.0D0
                  DO R = 1, N
                     XMAX = MAX(XMAX, ABS(VL(R, OUT_COL)) +
     $                                ABS(VL(R, OUT_COL + 1)))
                  END DO
                  IF (XMAX .GT. SAFEMIN) THEN
                     XMAX = 1.0D0 / XMAX
                     DO R = 1, N
                        VL(R, OUT_COL) = VL(R, OUT_COL) * XMAX
                        VL(R, OUT_COL + 1) = VL(R, OUT_COL + 1) * XMAX
                     END DO
                  END IF
                  C = C + 2
               END IF
            END DO

            CURRENT_OUT_COL = CURRENT_OUT_COL + NB_SEL
            CURR_COL = I_NEXT + 1
         END DO
      END IF

      RETURN
      END