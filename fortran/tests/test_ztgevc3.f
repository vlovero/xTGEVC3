      PROGRAM TEST_ZTGEVC3
      IMPLICIT NONE
      
      CALL TEST_ZTGEVC_STATIC()
      CALL TEST_DYNAMIC_SIZE(10)
      CALL TEST_DYNAMIC_SIZE(500)
      
      CONTAINS

      SUBROUTINE TEST_ZTGEVC_STATIC()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER :: INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      DOUBLE PRECISION :: ERR_R, ERR_L
      COMPLEX*16, ALLOCATABLE :: WORK(:)
      COMPLEX*16 :: S(N,N), P(N,N), ALPHA(N), BETA(N)
      COMPLEX*16 :: VR(N,N), VL(N,N)
      LOGICAL :: SEL(1)
      
      PRINT *, '--- Testing 4x4 Complex Static Matrix ---'
      
      DO J = 1, N
         DO I = 1, N
            S(I,J) = DCMPLX(0.0D0, 0.0D0)
            P(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
      END DO
      
      S(1,1) = DCMPLX(1.0D0, 1.0D0)
      P(1,1) = DCMPLX(2.0D0, 0.0D0)
      S(1,2) = DCMPLX(2.0D0, 0.0D0)
      P(1,2) = DCMPLX(1.0D0, 1.0D0)
      S(2,2) = DCMPLX(2.0D0, -1.0D0)
      P(2,2) = DCMPLX(1.0D0, -1.0D0)
      S(1,3) = DCMPLX(-5.0D0, 1.0D0)
      P(1,3) = DCMPLX(0.0D0, 0.0D0)
      S(2,3) = DCMPLX(0.0D0, 2.0D0)
      P(2,3) = DCMPLX(-1.0D0, 0.0D0)
      S(3,3) = DCMPLX(3.0D0, 0.0D0)
      P(3,3) = DCMPLX(1.0D0, 1.0D0)
      S(1,4) = DCMPLX(5.0D0, 0.0D0)
      P(1,4) = DCMPLX(0.0D0, 0.0D0)
      S(2,4) = DCMPLX(2.0D0, 1.0D0)
      P(2,4) = DCMPLX(0.0D0, 2.0D0)
      S(3,4) = DCMPLX(4.0D0, -1.0D0)
      P(3,4) = DCMPLX(3.0D0, 0.0D0)
      S(4,4) = DCMPLX(3.0D0, 2.0D0)
      P(4,4) = DCMPLX(2.0D0, -1.0D0)
      
      DO I = 1, N
         ALPHA(I) = S(I,I)
         BETA(I) = P(I,I)
         DO J = 1, N
            VR(J,I) = DCMPLX(0.0D0, 0.0D0)
            VL(J,I) = DCMPLX(0.0D0, 0.0D0)
         END DO
         VR(I,I) = DCMPLX(1.0D0, 0.0D0)
         VL(I,I) = DCMPLX(1.0D0, 0.0D0)
      END DO
      
      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 + 
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      
      CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      
      CALL CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VR,
     $             N, ALPHA, BETA, ERR_R)
      CALL CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VL,
     $             N, ALPHA, BETA, ERR_L)
      
      PRINT *, 'Max Right Backward Error: ', ERR_R
      PRINT *, 'Max Left Backward Error:  ', ERR_L
      PRINT *
      
      DEALLOCATE(WORK)
      END SUBROUTINE TEST_ZTGEVC_STATIC

      SUBROUTINE TEST_DYNAMIC_SIZE(N)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      COMPLEX*16, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      COMPLEX*16, ALLOCATABLE :: ALPHA(:), BETA(:)
      COMPLEX*16, ALLOCATABLE :: WORK(:)
      COMPLEX*16 :: DUMMY(1)
      DOUBLE PRECISION :: ERR_R, ERR_L
      INTEGER :: INFO, LWORK, I, J, M_OUT
      LOGICAL :: SEL(1)
      
      PRINT *, '--- Testing Dynamic Size Complex Matrix ---', N
      
      ALLOCATE(S(N,N), P(N,N), VR(N,N), VL(N,N))
      ALLOCATE(ALPHA(N), BETA(N))
      
      DO J = 1, N
         DO I = 1, N
            IF (I .EQ. J) THEN
               VR(I,J) = DCMPLX(1.0D0, 0.0D0)
               VL(I,J) = DCMPLX(1.0D0, 0.0D0)
            ELSE
               VR(I,J) = DCMPLX(0.0D0, 0.0D0)
               VL(I,J) = DCMPLX(0.0D0, 0.0D0)
            END IF
         END DO
      END DO
      
      CALL GENERATE_GENERALIZED_UPPER_TRIANGULAR(N, S, N, P, N,
     $     ALPHA, BETA)
     
      CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)
     
      LWORK = INT(REAL(DUMMY(1)))
      PRINT *, 'LWORK = ', LWORK
      
      ALLOCATE(WORK(LWORK))
      CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, WORK, LWORK, INFO)
     
      CALL CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VR,
     $             N, ALPHA, BETA, ERR_R)
      CALL CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, N, P, N, VL,
     $             N, ALPHA, BETA, ERR_L)
                   
      PRINT *, 'Max Right Backward Error: ', ERR_R
      PRINT *, 'Max Left Backward Error:  ', ERR_L
      PRINT *
      
      DEALLOCATE(S, P, VR, VL, ALPHA, BETA, WORK)
      END SUBROUTINE TEST_DYNAMIC_SIZE

      SUBROUTINE GENERATE_GENERALIZED_UPPER_TRIANGULAR(N, S, LDS, P, 
     $   LDP, ALPHA, BETA)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP
      COMPLEX*16, INTENT(OUT) :: S(LDS,N), P(LDP,N)
      COMPLEX*16, INTENT(OUT) :: ALPHA(N), BETA(N)
      INTEGER :: I, J, C, R
      DOUBLE PRECISION :: RAND_R, RAND_I
      
      DO J = 1, N
         DO I = 1, N
            S(I,J) = DCMPLX(0.0D0, 0.0D0)
            P(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
      END DO
      
      DO C = 1, N
         DO R = 1, C
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            S(R,C) = DCMPLX(RAND_R * 2.0D0 - 1.0D0, 
     $                      RAND_I * 2.0D0 - 1.0D0)
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            P(R,C) = DCMPLX(RAND_R * 2.0D0 - 1.0D0, 
     $                      RAND_I * 2.0D0 - 1.0D0)
         END DO
         IF (REAL(P(C,C)) .GE. 0.0D0) THEN
            P(C,C) = P(C,C) + DCMPLX(1.0D0, 0.0D0)
         ELSE
            P(C,C) = P(C,C) - DCMPLX(1.0D0, 0.0D0)
         END IF
         ALPHA(C) = S(C,C)
         BETA(C) = P(C,C)
      END DO
      END SUBROUTINE GENERATE_GENERALIZED_UPPER_TRIANGULAR

      SUBROUTINE CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, LDS, P,
     $   LDP, VR, LDVR, ALPHA, BETA, MAX_ERR)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP, LDVR
      COMPLEX*16, INTENT(IN) :: S(LDS,N), P(LDP,N), VR(LDVR,N)
      COMPLEX*16, INTENT(IN) :: ALPHA(N), BETA(N)
      DOUBLE PRECISION, INTENT(OUT) :: MAX_ERR
      
      DOUBLE PRECISION :: NORMS, NORMP
      DOUBLE PRECISION :: NORM_R, NORM_V
      INTEGER :: C, R, I, COL, ROW, INCX
      COMPLEX*16 :: A, B
      COMPLEX*16, ALLOCATABLE :: R_VEC(:)
      
      DOUBLE PRECISION :: DZNRM2
      EXTERNAL DZNRM2
      
      NORMS = 0.0D0
      NORMP = 0.0D0
      DO C = 1, N
         DO R = 1, C
            NORMS = NORMS + REAL(S(R,C))**2 + AIMAG(S(R,C))**2
            NORMP = NORMP + REAL(P(R,C))**2 + AIMAG(P(R,C))**2
         END DO
      END DO
      NORMS = SQRT(NORMS)
      NORMP = SQRT(NORMP)
      IF (NORMS .EQ. 0.0D0) NORMS = 1.0D0
      IF (NORMP .EQ. 0.0D0) NORMP = 1.0D0
      
      MAX_ERR = 0.0D0
      ALLOCATE(R_VEC(N))
      INCX = 1
      
      DO C = 1, N
         A = ALPHA(C)
         B = BETA(C)
         
         DO I = 1, N
            R_VEC(I) = DCMPLX(0.0D0, 0.0D0)
         END DO
         
         DO COL = 1, N
            DO ROW = 1, COL
               R_VEC(ROW) = R_VEC(ROW) + 
     $              B * S(ROW,COL) * VR(COL,C) - 
     $              A * P(ROW,COL) * VR(COL,C)
            END DO
         END DO
         
         NORM_R = DZNRM2(N, R_VEC, INCX)
         NORM_V = DZNRM2(N, VR(1,C), INCX)
         
         MAX_ERR = MAX(MAX_ERR, NORM_R / ((ABS(B)*NORMS + 
     $                 ABS(A)*NORMP) * NORM_V))
      END DO
      
      DEALLOCATE(R_VEC)
      END SUBROUTINE CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP

      SUBROUTINE CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, LDS, P,
     $   LDP, VL, LDVL, ALPHA, BETA, MAX_ERR)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP, LDVL
      COMPLEX*16, INTENT(IN) :: S(LDS,N), P(LDP,N), VL(LDVL,N)
      COMPLEX*16, INTENT(IN) :: ALPHA(N), BETA(N)
      DOUBLE PRECISION, INTENT(OUT) :: MAX_ERR
      
      DOUBLE PRECISION :: NORMS, NORMP
      DOUBLE PRECISION :: NORM_R, NORM_V
      INTEGER :: C, R, COL, ROW, INCX
      COMPLEX*16 :: A, B, SUM_VAL
      COMPLEX*16, ALLOCATABLE :: R_VEC(:)
      
      DOUBLE PRECISION :: DZNRM2
      EXTERNAL DZNRM2
      
      NORMS = 0.0D0
      NORMP = 0.0D0
      DO C = 1, N
         DO R = 1, C
            NORMS = NORMS + REAL(S(R,C))**2 + AIMAG(S(R,C))**2
            NORMP = NORMP + REAL(P(R,C))**2 + AIMAG(P(R,C))**2
         END DO
      END DO
      NORMS = SQRT(NORMS)
      NORMP = SQRT(NORMP)
      IF (NORMS .EQ. 0.0D0) NORMS = 1.0D0
      IF (NORMP .EQ. 0.0D0) NORMP = 1.0D0
      
      MAX_ERR = 0.0D0
      ALLOCATE(R_VEC(N))
      INCX = 1
      
      DO C = 1, N
         A = ALPHA(C)
         B = BETA(C)
         
         DO COL = 1, N
            SUM_VAL = DCMPLX(0.0D0, 0.0D0)
            DO ROW = 1, COL
               SUM_VAL = SUM_VAL + DCONJG(VL(ROW,C)) * 
     $                   (B * S(ROW,COL) - A * P(ROW,COL))
            END DO
            R_VEC(COL) = SUM_VAL
         END DO
         
         NORM_R = DZNRM2(N, R_VEC, INCX)
         NORM_V = DZNRM2(N, VL(1,C), INCX)
         
         MAX_ERR = MAX(MAX_ERR, NORM_R / ((ABS(B)*NORMS + 
     $                 ABS(A)*NORMP) * NORM_V))
      END DO
      
      DEALLOCATE(R_VEC)
      END SUBROUTINE CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP
      
      END PROGRAM TEST_ZTGEVC3