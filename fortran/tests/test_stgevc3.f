      PROGRAM TEST_DTGEVC3
      IMPLICIT NONE
      
      CALL TEST_DTGEVC()
      CALL TEST_DTGEVC_INFINITE()
      CALL TEST_DTGEVC_SCALING()
      CALL TEST_DYNAMIC_SIZE(10)
      CALL TEST_DYNAMIC_SIZE(500)
      
      CONTAINS

      SUBROUTINE TEST_DTGEVC()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER :: INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      REAL :: ERR_R, ERR_L
      REAL, ALLOCATABLE :: WORK(:)
      REAL :: S(N,N), P(N,N), ALPHAR(N), ALPHAI(N), BETA(N)
      REAL :: VR(N,N), VL(N,N)
      LOGICAL :: SEL(1)
      
      PRINT *, '--- Testing 4x4 Static Matrix (Both Left/Right) ---'
      
      S(1,1) = 1.0E0; S(1,2) = 2.0D0
      S(1,3) = 3.0D0; S(1,4) = 4.0D0
      S(2,1) = 0.0E0; S(2,2) = 2.0D0
      S(2,3) = 5.0D0; S(2,4) = -1.0E0
      S(3,1) = 0.0E0; S(3,2) = -5.0D0
      S(3,3) = 2.0D0; S(3,4) = 2.0D0
      S(4,1) = 0.0E0; S(4,2) = 0.0E0
      S(4,3) = 0.0E0; S(4,4) = 3.0D0
      
      P(1,1) = 2.0D0; P(1,2) = 1.0E0
      P(1,3) = -1.0E0; P(1,4) = 3.0D0
      P(2,1) = 0.0E0; P(2,2) = 1.0E0
      P(2,3) = 0.0E0; P(2,4) = 2.0D0
      P(3,1) = 0.0E0; P(3,2) = 0.0E0
      P(3,3) = 1.0E0; P(3,4) = -1.0E0
      P(4,1) = 0.0E0; P(4,2) = 0.0E0
      P(4,3) = 0.0E0; P(4,4) = 2.0D0
      
      ALPHAR(1) = 1.0E0; ALPHAR(2) = 2.0D0
      ALPHAR(3) = 2.0D0; ALPHAR(4) = 3.0D0
      ALPHAI(1) = 0.0E0; ALPHAI(2) = 5.0D0
      ALPHAI(3) = -5.0D0; ALPHAI(4) = 0.0E0
      BETA(1) = 2.0D0; BETA(2) = 1.0E0
      BETA(3) = 1.0E0; BETA(4) = 2.0D0
      
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = 0.0E0
            VL(I,J) = 0.0E0
         END DO
         VR(J,J) = 1.0E0
         VL(J,J) = 1.0E0
      END DO
      
      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 + 
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      
      CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $             BETA, VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      
      CALL CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VR,
     $             N, ALPHAR, ALPHAI, BETA, ERR_R)
      CALL CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VL,
     $             N, ALPHAR, ALPHAI, BETA, ERR_L)
      
      PRINT *, 'Max Right Backward Error: ', ERR_R
      PRINT *, 'Max Left Backward Error: ', ERR_L
      PRINT *
      
      DEALLOCATE(WORK)
      END SUBROUTINE TEST_DTGEVC

      SUBROUTINE TEST_DTGEVC_INFINITE()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER :: INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      REAL :: ERR_R, ERR_L
      REAL, ALLOCATABLE :: WORK(:)
      REAL :: S(N,N), P(N,N), ALPHAR(N), ALPHAI(N), BETA(N)
      REAL :: VR(N,N), VL(N,N)
      LOGICAL :: SEL(1)
      
      PRINT *, '--- Testing 4x4 Static Matrix (Infinite Eval) ---'
      
      S(1,1) = 1.0E0; S(1,2) = 2.0D0
      S(1,3) = 3.0D0; S(1,4) = 4.0D0
      S(2,1) = 0.0E0; S(2,2) = 2.0D0
      S(2,3) = 5.0D0; S(2,4) = -1.0E0
      S(3,1) = 0.0E0; S(3,2) = -5.0D0
      S(3,3) = 2.0D0; S(3,4) = 2.0D0
      S(4,1) = 0.0E0; S(4,2) = 0.0E0
      S(4,3) = 0.0E0; S(4,4) = 3.0D0
      
      P(1,1) = 2.0D0; P(1,2) = 1.0E0
      P(1,3) = -1.0E0; P(1,4) = 3.0D0
      P(2,1) = 0.0E0; P(2,2) = 1.0E0
      P(2,3) = 0.0E0; P(2,4) = 2.0D0
      P(3,1) = 0.0E0; P(3,2) = 0.0E0
      P(3,3) = 1.0E0; P(3,4) = -1.0E0
      P(4,1) = 0.0E0; P(4,2) = 0.0E0
      P(4,3) = 0.0E0; P(4,4) = 0.0E0
      
      ALPHAR(1) = 1.0E0; ALPHAR(2) = 2.0D0
      ALPHAR(3) = 2.0D0; ALPHAR(4) = 3.0D0
      ALPHAI(1) = 0.0E0; ALPHAI(2) = 5.0D0
      ALPHAI(3) = -5.0D0; ALPHAI(4) = 0.0E0
      BETA(1) = 2.0D0; BETA(2) = 1.0E0
      BETA(3) = 1.0E0; BETA(4) = 0.0E0
      
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = 0.0E0
            VL(I,J) = 0.0E0
         END DO
         VR(J,J) = 1.0E0
         VL(J,J) = 1.0E0
      END DO
      
      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 + 
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      
      CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $             BETA, VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      
      CALL CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VR,
     $             N, ALPHAR, ALPHAI, BETA, ERR_R)
      CALL CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VL,
     $             N, ALPHAR, ALPHAI, BETA, ERR_L)
      
      PRINT *, 'Max Right Backward Error: ', ERR_R
      PRINT *, 'Max Left Backward Error: ', ERR_L
      PRINT *
      
      DEALLOCATE(WORK)
      END SUBROUTINE TEST_DTGEVC_INFINITE

      SUBROUTINE TEST_DTGEVC_SCALING()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER :: INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT, TEST_IDX
      REAL :: ERR_R, ERR_L, S_VAL
      REAL, ALLOCATABLE :: WORK(:)
      REAL :: S_BASE(N,N), P_BASE(N,N)
      REAL :: ALPHAR_BASE(N), ALPHAI_BASE(N), BETA_BASE(N)
      REAL :: S(N,N), P(N,N), ALPHAR(N), ALPHAI(N), BETA(N)
      REAL :: VR(N,N), VL(N,N)
      REAL :: SCALES(2)
      LOGICAL :: SEL(1)
      
      S_BASE(1,1) = 1.0E0; S_BASE(1,2) = 2.0D0
      S_BASE(1,3) = 3.0D0; S_BASE(1,4) = 4.0D0
      S_BASE(2,1) = 0.0E0; S_BASE(2,2) = 2.0D0
      S_BASE(2,3) = 5.0D0; S_BASE(2,4) = -1.0E0
      S_BASE(3,1) = 0.0E0; S_BASE(3,2) = -5.0D0
      S_BASE(3,3) = 2.0D0; S_BASE(3,4) = 2.0D0
      S_BASE(4,1) = 0.0E0; S_BASE(4,2) = 0.0E0
      S_BASE(4,3) = 0.0E0; S_BASE(4,4) = 3.0D0
      
      P_BASE(1,1) = 2.0D0; P_BASE(1,2) = 1.0E0
      P_BASE(1,3) = -1.0E0; P_BASE(1,4) = 3.0D0
      P_BASE(2,1) = 0.0E0; P_BASE(2,2) = 1.0E0
      P_BASE(2,3) = 0.0E0; P_BASE(2,4) = 2.0D0
      P_BASE(3,1) = 0.0E0; P_BASE(3,2) = 0.0E0
      P_BASE(3,3) = 1.0E0; P_BASE(3,4) = -1.0E0
      P_BASE(4,1) = 0.0E0; P_BASE(4,2) = 0.0E0
      P_BASE(4,3) = 0.0E0; P_BASE(4,4) = 2.0D0
      
      ALPHAR_BASE(1) = 1.0E0; ALPHAR_BASE(2) = 2.0D0
      ALPHAR_BASE(3) = 2.0D0; ALPHAR_BASE(4) = 3.0D0
      ALPHAI_BASE(1) = 0.0E0; ALPHAI_BASE(2) = 5.0D0
      ALPHAI_BASE(3) = -5.0D0; ALPHAI_BASE(4) = 0.0E0
      BETA_BASE(1) = 2.0D0; BETA_BASE(2) = 1.0E0
      BETA_BASE(3) = 1.0E0; BETA_BASE(4) = 2.0D0
      
      SCALES(1) = 1.0E30
      SCALES(2) = 1.0E-30
      
      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 + 
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      
      DO TEST_IDX = 1, 2
         S_VAL = SCALES(TEST_IDX)
         IF (TEST_IDX .EQ. 1) THEN
            PRINT *, '--- Testing 4x4 Matrix (Scaling: 1e30) ---'
         ELSE
            PRINT *, '--- Testing 4x4 Matrix (Scaling: 1e-30) ---'
         END IF
         
         DO J = 1, N
            DO I = 1, N
               S(I,J) = S_BASE(I,J) * S_VAL
               P(I,J) = P_BASE(I,J) * S_VAL
            END DO
         END DO
         
         DO I = 1, N
            ALPHAR(I) = ALPHAR_BASE(I) * S_VAL
            ALPHAI(I) = ALPHAI_BASE(I) * S_VAL
            BETA(I) = BETA_BASE(I) * S_VAL
         END DO
         
         DO J = 1, N
            DO I = 1, N
               VR(I,J) = 0.0E0
               VL(I,J) = 0.0E0
            END DO
            VR(J,J) = 1.0E0
            VL(J,J) = 1.0E0
         END DO
         
         CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $                BETA, VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE,
     $                INFO)
         
         CALL CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, 
     $                VR, N, ALPHAR, ALPHAI, BETA, ERR_R)
         CALL CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, 
     $                VL, N, ALPHAR, ALPHAI, BETA, ERR_L)
         
         PRINT *, 'Max Right Backward Error: ', ERR_R
         PRINT *, 'Max Left Backward Error: ', ERR_L
         PRINT *
      END DO
      
      DEALLOCATE(WORK)
      END SUBROUTINE TEST_DTGEVC_SCALING

      SUBROUTINE TEST_DYNAMIC_SIZE(N)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      REAL, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      REAL, ALLOCATABLE :: ALPHAR(:), ALPHAI(:), BETA(:)
      REAL, ALLOCATABLE :: WORK(:)
      REAL :: DUMMY(1), ERR_R, ERR_L
      INTEGER :: INFO, LWORK, I, J, M_OUT
      LOGICAL :: SEL(1)
      
      PRINT *, '--- Testing Dynamic Size Matrix ---', N
      
      ALLOCATE(S(N,N), P(N,N), VR(N,N), VL(N,N))
      ALLOCATE(ALPHAR(N), ALPHAI(N), BETA(N))
      
      DO J = 1, N
         DO I = 1, N
            IF (I .EQ. J) THEN
               VR(I,J) = 1.0E0
               VL(I,J) = 1.0E0
            ELSE
               VR(I,J) = 0.0E0
               VL(I,J) = 0.0E0
            END IF
         END DO
      END DO
      
      CALL GENERATE_GENERALIZED_QUASI_TRIANGULAR(N, S, N, P, N,
     $     ALPHAR, ALPHAI, BETA)
     
      CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $             BETA, VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)
     
      LWORK = INT(DUMMY(1))
      PRINT *, 'LWORK = ', LWORK
      
      ALLOCATE(WORK(LWORK))
      CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $             BETA, VL, N, VR, N, N, M_OUT, WORK, LWORK, INFO)
     
      CALL CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VR,
     $             N, ALPHAR, ALPHAI, BETA, ERR_R)
      CALL CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VL,
     $             N, ALPHAR, ALPHAI, BETA, ERR_L)
                   
      PRINT *, 'Max Right Backward Error: ', ERR_R
      PRINT *, 'Max Left Backward Error: ', ERR_L
      PRINT *
      
      DEALLOCATE(S, P, VR, VL, ALPHAR, ALPHAI, BETA, WORK)
      END SUBROUTINE TEST_DYNAMIC_SIZE

      SUBROUTINE GENERATE_GENERALIZED_QUASI_TRIANGULAR(N, S, LDS, P, 
     $   LDP, ALPHAR, ALPHAI, BETA)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP
      REAL, INTENT(OUT) :: S(LDS,N), P(LDP,N)
      REAL, INTENT(OUT) :: ALPHAR(N), ALPHAI(N), BETA(N)
      INTEGER :: I, J, K
      REAL :: A, B, RAND_VAL
      
      DO J = 1, N
         DO I = 1, N
            S(I,J) = 0.0E0
            P(I,J) = 0.0E0
         END DO
      END DO
      
      DO J = 1, N
         DO I = 1, J
            CALL RANDOM_NUMBER(RAND_VAL)
            S(I,J) = RAND_VAL * 2.0D0 - 1.0E0
            CALL RANDOM_NUMBER(RAND_VAL)
            P(I,J) = RAND_VAL * 2.0D0 - 1.0E0
         END DO
         IF (P(J,J) .GE. 0.0E0) THEN
            P(J,J) = P(J,J) + 1.0E0
         ELSE
            P(J,J) = P(J,J) - 1.0E0
         END IF
      END DO
      
      K = 1
      DO WHILE (K .LE. N)
         CALL RANDOM_NUMBER(RAND_VAL)
         IF (K .LT. N .AND. RAND_VAL .LT. 0.4D0) THEN
            P(K,K) = 1.0E0
            P(K,K+1) = 0.0E0
            P(K+1,K) = 0.0E0
            P(K+1,K+1) = 1.0E0
            
            A = S(K,K)
            S(K+1,K+1) = A
            
            B = S(K,K+1)
            IF (B .EQ. 0.0E0) B = 1.0E0
            S(K+1,K) = -B
            
            ALPHAR(K) = A
            ALPHAR(K+1) = A
            ALPHAI(K) = ABS(B)
            ALPHAI(K+1) = -ABS(B)
            BETA(K) = 1.0E0
            BETA(K+1) = 1.0E0
            K = K + 2
         ELSE
            ALPHAR(K) = S(K,K)
            ALPHAI(K) = 0.0E0
            BETA(K) = P(K,K)
            K = K + 1
         END IF
      END DO
      END SUBROUTINE GENERATE_GENERALIZED_QUASI_TRIANGULAR

      SUBROUTINE CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, LDS, P,
     $   LDP, VR, LDVR, ALPHAR, ALPHAI, BETA, MAX_ERR)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP, LDVR
      REAL, INTENT(IN) :: S(LDS,N), P(LDP,N), VR(LDVR,N)
      REAL, INTENT(IN) :: ALPHAR(N), ALPHAI(N), BETA(N)
      REAL, INTENT(OUT) :: MAX_ERR
      
      REAL :: NORMS, NORMP, A, B
      REAL :: NORM_R, NORM_V, AR, AI, S_VAL, P_VAL
      REAL :: VR_R, VR_I, NORM_RR, NORM_RI, DEN
      INTEGER :: C, R, I, COL, ROW, INCX
      REAL, ALLOCATABLE :: R_VEC(:), RR(:), RI(:)
      
      REAL :: SNRM2
      EXTERNAL SNRM2
      
      NORMS = 0.0E0
      NORMP = 0.0E0
      DO C = 1, N
         DO R = 1, MIN(C + 1, N)
            NORMS = NORMS + S(R,C)*S(R,C)
         END DO
         DO R = 1, C
            NORMP = NORMP + P(R,C)*P(R,C)
         END DO
      END DO
      NORMS = SQRT(NORMS)
      NORMP = SQRT(NORMP)
      IF (NORMS .EQ. 0.0E0) NORMS = 1.0E0
      IF (NORMP .EQ. 0.0E0) NORMP = 1.0E0
      
      MAX_ERR = 0.0E0
      ALLOCATE(R_VEC(N))
      INCX = 1
      
      C = 1
      DO WHILE (C .LE. N)
         IF (ALPHAI(C) .EQ. 0.0E0) THEN
            A = ALPHAR(C)
            B = BETA(C)
            
            DO I = 1, N
               R_VEC(I) = 0.0E0
            END DO
            
            DO COL = 1, N
               DO ROW = 1, MIN(COL + 1, N)
                  R_VEC(ROW) = R_VEC(ROW) + 
     $                 B * S(ROW,COL) * VR(COL,C) -
     $                 A * P(ROW,COL) * VR(COL,C)
               END DO
            END DO
            
            NORM_R = SNRM2(N, R_VEC, INCX)
            NORM_V = SNRM2(N, VR(1,C), INCX)
            
            MAX_ERR = MAX(MAX_ERR, NORM_R / ((ABS(B)*NORMS + 
     $                    ABS(A)*NORMP) * NORM_V))
            C = C + 1
         ELSE
            ALLOCATE(RR(N), RI(N))
            AR = ALPHAR(C)
            AI = ALPHAI(C)
            B = BETA(C)
            
            DO I = 1, N
               RR(I) = 0.0E0
               RI(I) = 0.0E0
            END DO
            
            DO COL = 1, N
               DO ROW = 1, MIN(COL + 1, N)
                  S_VAL = S(ROW,COL)
                  P_VAL = P(ROW,COL)
                  VR_R = VR(COL,C)
                  VR_I = VR(COL,C+1)
                  
                  RR(ROW) = RR(ROW) + B*S_VAL*VR_R - 
     $                      AR*P_VAL*VR_R + AI*P_VAL*VR_I
                  RI(ROW) = RI(ROW) + B*S_VAL*VR_I -
     $                      AR*P_VAL*VR_I - AI*P_VAL*VR_R
               END DO
            END DO
            
            NORM_RR = SNRM2(N, RR, INCX)
            NORM_RI = SNRM2(N, RI, INCX)
            NORM_V = SQRT(SNRM2(N, VR(1,C), INCX)**2 + 
     $                    SNRM2(N, VR(1,C+1), INCX)**2)
                          
            DEN = (ABS(B)*NORMS + SQRT(AR*AR + AI*AI)*NORMP) * NORM_V
            MAX_ERR = MAX(MAX_ERR, NORM_RR/DEN, NORM_RI/DEN)
            
            DEALLOCATE(RR, RI)
            C = C + 2
         END IF
      END DO
      
      DEALLOCATE(R_VEC)
      END SUBROUTINE CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP

      SUBROUTINE CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, LDS, P,
     $   LDP, VL, LDVL, ALPHAR, ALPHAI, BETA, MAX_ERR)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP, LDVL
      REAL, INTENT(IN) :: S(LDS,N), P(LDP,N), VL(LDVL,N)
      REAL, INTENT(IN) :: ALPHAR(N), ALPHAI(N), BETA(N)
      REAL, INTENT(OUT) :: MAX_ERR
      
      REAL :: NORMS, NORMP, A, B
      REAL :: NORM_R, NORM_V, AR, AI, S_VAL, P_VAL
      REAL :: VL_R, VL_I, NORM_RR, NORM_RI, DEN
      INTEGER :: C, R, I, COL, ROW, INCX
      REAL, ALLOCATABLE :: R_VEC(:), RR(:), RI(:)
      
      REAL :: SNRM2
      EXTERNAL SNRM2
      
      NORMS = 0.0E0
      NORMP = 0.0E0
      DO C = 1, N
         DO R = 1, MIN(C + 1, N)
            NORMS = NORMS + S(R,C)*S(R,C)
         END DO
         DO R = 1, C
            NORMP = NORMP + P(R,C)*P(R,C)
         END DO
      END DO
      NORMS = SQRT(NORMS)
      NORMP = SQRT(NORMP)
      IF (NORMS .EQ. 0.0E0) NORMS = 1.0E0
      IF (NORMP .EQ. 0.0E0) NORMP = 1.0E0
      
      MAX_ERR = 0.0E0
      ALLOCATE(R_VEC(N))
      INCX = 1
      
      C = 1
      DO WHILE (C .LE. N)
         IF (ALPHAI(C) .EQ. 0.0E0) THEN
            A = ALPHAR(C)
            B = BETA(C)
            
            DO I = 1, N
               R_VEC(I) = 0.0E0
            END DO
            
            DO COL = 1, N
               DO ROW = 1, MIN(COL + 1, N)
                  R_VEC(COL) = R_VEC(COL) + 
     $                 B * S(ROW,COL) * VL(ROW,C) -
     $                 A * P(ROW,COL) * VL(ROW,C)
               END DO
            END DO
            
            NORM_R = SNRM2(N, R_VEC, INCX)
            NORM_V = SNRM2(N, VL(1,C), INCX)
            
            MAX_ERR = MAX(MAX_ERR, NORM_R / ((ABS(B)*NORMS + 
     $                    ABS(A)*NORMP) * NORM_V))
            C = C + 1
         ELSE
            ALLOCATE(RR(N), RI(N))
            AR = ALPHAR(C)
            AI = ALPHAI(C)
            B = BETA(C)
            
            DO I = 1, N
               RR(I) = 0.0E0
               RI(I) = 0.0E0
            END DO
            
            DO COL = 1, N
               DO ROW = 1, MIN(COL + 1, N)
                  S_VAL = S(ROW,COL)
                  P_VAL = P(ROW,COL)
                  VL_R = VL(ROW,C)
                  VL_I = VL(ROW,C+1)
                  
                  RR(COL) = RR(COL) + B*S_VAL*VL_R - 
     $                      AR*P_VAL*VL_R - AI*P_VAL*VL_I
                  RI(COL) = RI(COL) + B*S_VAL*VL_I -
     $                      AR*P_VAL*VL_I + AI*P_VAL*VL_R
               END DO
            END DO
            
            NORM_RR = SNRM2(N, RR, INCX)
            NORM_RI = SNRM2(N, RI, INCX)
            NORM_V = SQRT(SNRM2(N, VL(1,C), INCX)**2 + 
     $                    SNRM2(N, VL(1,C+1), INCX)**2)
                          
            DEN = (ABS(B)*NORMS + SQRT(AR*AR + AI*AI)*NORMP) * NORM_V
            MAX_ERR = MAX(MAX_ERR, NORM_RR/DEN, NORM_RI/DEN)
            
            DEALLOCATE(RR, RI)
            C = C + 2
         END IF
      END DO
      
      DEALLOCATE(R_VEC)
      END SUBROUTINE CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP
      
      END PROGRAM TEST_DTGEVC3