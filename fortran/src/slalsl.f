      SUBROUTINE SLALSL(LDS, S, LDP, P, M_SIZE, LDV, RHS_LOC,
     $                  X_PANEL_BASE, PANEL_ROWS, NB, ALPHAR, ALPHAI,
     $                  BETA, IS_DIAG, WORK, ASCALE, BSCALE, SAFEMIN,
     $                  BIGNUM, COL_MAP, NB_SEL)
      IMPLICIT NONE
      INTEGER LDS, LDP, M_SIZE, LDV, PANEL_ROWS, NB, IS_DIAG, NB_SEL
      REAL S(LDS, *), P(LDP, *)
      REAL RHS_LOC(LDV, *), X_PANEL_BASE(LDV, *)
      REAL ALPHAR(*), ALPHAI(*), BETA(*)
      REAL WORK(*)
      REAL ASCALE, BSCALE, SAFEMIN, BIGNUM
      INTEGER COL_MAP(*)

      INTEGER K, CUR_M, ROW_OFFSET, C, R, I, DIM2
      INTEGER C_SCALE, R_SCALE, C_PACKED
      INTEGER JPIV(128)
      REAL S11, S21, P11
      REAL AR, AI, B_VAL, VAL_REAL, VAL_IMAG
      REAL SR1, SR2, PR1, PR2, AR1, AI1, AR2, AI2
      REAL Y1R, Y1I, Y2R, Y2I
      REAL T, ACOEFF, BCOEFFR, BCOEFFI
      REAL RHS_MAX, SCALE
      INTEGER WORK_RHS_OFFSET
      INTEGER INFO

      K = 1
      DO WHILE (K .LE. NB)
         C_PACKED = COL_MAP(K)
         IF (C_PACKED .LT. 1) THEN
            IF (ALPHAI(K) .EQ. 0.0E0) THEN
               K = K + 1
            ELSE
               K = K + 2
            END IF
            CYCLE
         END IF

         AR = ALPHAR(K)
         AI = ALPHAI(K)
         B_VAL = BETA(K)

         IF (AI .EQ. 0.0E0) THEN
*           --- REAL EIGENVALUE (1x1 Block) ---
            IF (IS_DIAG .NE. 0) THEN
               CUR_M = NB - K
               ROW_OFFSET = K
            ELSE
               CUR_M = M_SIZE
               ROW_OFFSET = 0
            END IF

            T = 1.0E0 / MAX(ABS(AR) * ASCALE, ABS(B_VAL) * BSCALE, 
     $                      SAFEMIN)
            ACOEFF = (T * B_VAL * BSCALE) * ASCALE
            BCOEFFR = (T * AR * ASCALE) * BSCALE

            IF (ABS(B_VAL) .LE. SAFEMIN .AND. ABS(AR) .GT. SAFEMIN) THEN
               ACOEFF = 0.0E0
               BCOEFFR = 1.0E0
            END IF

            IF (IS_DIAG .NE. 0) THEN
               RHS_LOC(K, C_PACKED) = 1.0E0
            END IF

            IF (CUR_M .GT. 0) THEN
               DO C = 1, CUR_M
                  DO R = 1, CUR_M
                     WORK(R + (C - 1) * CUR_M) = 
     $                  ACOEFF * S(C + ROW_OFFSET, R + ROW_OFFSET) - 
     $                  BCOEFFR * P(C + ROW_OFFSET, R + ROW_OFFSET)
                  END DO
               END DO

               WORK_RHS_OFFSET = CUR_M * CUR_M
               IF (IS_DIAG .NE. 0) THEN
                  DO R = 1, CUR_M
                     SR1 = S(K, R + ROW_OFFSET)
                     PR1 = P(K, R + ROW_OFFSET)
                     WORK(WORK_RHS_OFFSET + R) = 
     $                  -(ACOEFF * SR1 - BCOEFFR * PR1)
                  END DO
               ELSE
                  DO R = 1, CUR_M
                     WORK(WORK_RHS_OFFSET + R) = 
     $                  RHS_LOC(R + ROW_OFFSET, C_PACKED)
                  END DO
               END IF

               RHS_MAX = 0.0E0
               DO R = 1, CUR_M
                  RHS_MAX = MAX(RHS_MAX, ABS(WORK(WORK_RHS_OFFSET + R)))
               END DO

               IF (RHS_MAX .GT. BIGNUM / 10.0) THEN
                  SCALE = (BIGNUM / 10.0) / RHS_MAX
                  DO C_SCALE = 1, NB_SEL
                     DO R_SCALE = 1, PANEL_ROWS
                        X_PANEL_BASE(R_SCALE, C_SCALE) = 
     $                     X_PANEL_BASE(R_SCALE, C_SCALE) * SCALE
                     END DO
                  END DO
                  DO R = 1, CUR_M
                     WORK(WORK_RHS_OFFSET + R) = 
     $                  WORK(WORK_RHS_OFFSET + R) * SCALE
                  END DO
               END IF

               CALL SLALHS(CUR_M, 1, WORK, CUR_M, JPIV, 
     $                     WORK(WORK_RHS_OFFSET + 1), CUR_M, INFO)

               DO R = 1, CUR_M
                  RHS_LOC(R + ROW_OFFSET, C_PACKED) = 
     $               WORK(WORK_RHS_OFFSET + R)
               END DO
            END IF
            K = K + 1
         ELSE
*           --- COMPLEX CONJUGATE PAIR (2x2 Block) ---
            IF (IS_DIAG .NE. 0) THEN
               CUR_M = NB - 1 - K
               ROW_OFFSET = K + 1
            ELSE
               CUR_M = M_SIZE
               ROW_OFFSET = 0
            END IF

            T = 1.0E0 / MAX(ABS(AR) * ASCALE + ABS(AI) * ASCALE, 
     $                      ABS(B_VAL) * BSCALE, SAFEMIN)
            ACOEFF = (T * B_VAL * BSCALE) * ASCALE
            BCOEFFR = (T * AR * ASCALE) * BSCALE
            BCOEFFI = (T * AI * ASCALE) * BSCALE

            IF (ABS(B_VAL) .LE. SAFEMIN .AND. 
     $          (ABS(AR) + ABS(AI)) .GT. SAFEMIN) THEN
               ACOEFF = 0.0E0
               BCOEFFR = 1.0E0
               BCOEFFI = 0.0E0
            END IF

            IF (IS_DIAG .NE. 0) THEN
               S11 = S(K, K)
               S21 = S(K + 1, K)
               P11 = P(K, K)

               RHS_LOC(K, C_PACKED) = -ACOEFF * S21
               RHS_LOC(K, C_PACKED + 1) = 0.0E0

               RHS_LOC(K + 1, C_PACKED) = ACOEFF * S11 - BCOEFFR * P11
               RHS_LOC(K + 1, C_PACKED + 1) = BCOEFFI * P11
            END IF

            IF (CUR_M .GT. 0) THEN
               DIM2 = 2 * CUR_M

               DO I = 1, DIM2 * DIM2
                  WORK(I) = 0.0E0
               END DO

               DO C = 1, CUR_M
                  DO R = 1, CUR_M
                     VAL_REAL = ACOEFF * S(C + ROW_OFFSET, 
     $                  R + ROW_OFFSET) - BCOEFFR * P(C + ROW_OFFSET, 
     $                  R + ROW_OFFSET)
                     VAL_IMAG = BCOEFFI * P(C + ROW_OFFSET, 
     $                  R + ROW_OFFSET)

                     WORK((2*R-1) + (2*C-2)*DIM2) = VAL_REAL
                     WORK((2*R) + (2*C-1)*DIM2)   = VAL_REAL
                     WORK((2*R-1) + (2*C-1)*DIM2) = -VAL_IMAG
                     WORK((2*R) + (2*C-2)*DIM2)   = VAL_IMAG
                  END DO
               END DO

               WORK_RHS_OFFSET = DIM2 * DIM2
               IF (IS_DIAG .NE. 0) THEN
                  DO R = 1, CUR_M
                     SR1 = S(K, R + ROW_OFFSET)
                     SR2 = S(K + 1, R + ROW_OFFSET)
                     PR1 = P(K, R + ROW_OFFSET)
                     PR2 = P(K + 1, R + ROW_OFFSET)

                     AR1 = ACOEFF * SR1 - BCOEFFR * PR1
                     AI1 = BCOEFFI * PR1
                     AR2 = ACOEFF * SR2 - BCOEFFR * PR2
                     AI2 = BCOEFFI * PR2

                     Y1R = RHS_LOC(K, C_PACKED)
                     Y1I = RHS_LOC(K, C_PACKED + 1)
                     Y2R = RHS_LOC(K + 1, C_PACKED)
                     Y2I = RHS_LOC(K + 1, C_PACKED + 1)

                     WORK(WORK_RHS_OFFSET + 2*R-1) = -((AR1 * Y1R - 
     $                  AI1 * Y1I) + (AR2 * Y2R - AI2 * Y2I))
                     WORK(WORK_RHS_OFFSET + 2*R) = -((AR1 * Y1I + 
     $                  AI1 * Y1R) + (AR2 * Y2I + AI2 * Y2R))
                  END DO
               ELSE
                  DO R = 1, CUR_M
                     WORK(WORK_RHS_OFFSET + 2*R-1) = 
     $                  RHS_LOC(R + ROW_OFFSET, C_PACKED)
                     WORK(WORK_RHS_OFFSET + 2*R) = 
     $                  RHS_LOC(R + ROW_OFFSET, C_PACKED + 1)
                  END DO
               END IF

               RHS_MAX = 0.0E0
               DO R = 1, DIM2
                  RHS_MAX = MAX(RHS_MAX, ABS(WORK(WORK_RHS_OFFSET + R)))
               END DO

               IF (RHS_MAX .GT. BIGNUM / 10.0) THEN
                  SCALE = (BIGNUM / 10.0) / RHS_MAX
                  DO C_SCALE = 1, NB_SEL
                     DO R_SCALE = 1, PANEL_ROWS
                        X_PANEL_BASE(R_SCALE, C_SCALE) = 
     $                     X_PANEL_BASE(R_SCALE, C_SCALE) * SCALE
                     END DO
                  END DO
                  DO R = 1, DIM2
                     WORK(WORK_RHS_OFFSET + R) = 
     $                  WORK(WORK_RHS_OFFSET + R) * SCALE
                  END DO
               END IF

               CALL SLAL2S(DIM2, 1, WORK, DIM2, JPIV, 
     $                     WORK(WORK_RHS_OFFSET + 1), DIM2, INFO)

               DO R = 1, CUR_M
                  RHS_LOC(R + ROW_OFFSET, C_PACKED) = 
     $               WORK(WORK_RHS_OFFSET + 2*R-1)
                  RHS_LOC(R + ROW_OFFSET, C_PACKED + 1) = 
     $               WORK(WORK_RHS_OFFSET + 2*R)
               END DO
            END IF
            K = K + 2
         END IF
      END DO

      RETURN
      END
