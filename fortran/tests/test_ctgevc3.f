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
      REAL :: ERR_R, ERR_L
      COMPLEX, ALLOCATABLE :: WORK(:)
      COMPLEX :: S(N,N), P(N,N), ALPHA(N), BETA(N)
      COMPLEX :: VR(N,N), VL(N,N)
      LOGICAL :: SEL(1)
      
      PRINT *, '--- Testing 4x4 Complex Static Matrix ---'
      
      DO J = 1, N
         DO I = 1, N
            S(I,J) = CMPLX(0.0E0, 0.0E0)
            P(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
      END DO
      
      S(1,1) = CMPLX(1.0E0, 1.0E0)
      P(1,1) = CMPLX(2.0D0, 0.0E0)
      S(1,2) = CMPLX(2.0D0, 0.0E0)
      P(1,2) = CMPLX(1.0E0, 1.0E0)
      S(2,2) = CMPLX(2.0D0, -1.0E0)
      P(2,2) = CMPLX(1.0E0, -1.0E0)
      S(1,3) = CMPLX(-5.0D0, 1.0E0)
      P(1,3) = CMPLX(0.0E0, 0.0E0)
      S(2,3) = CMPLX(0.0E0, 2.0D0)
      P(2,3) = CMPLX(-1.0E0, 0.0E0)
      S(3,3) = CMPLX(3.0D0, 0.0E0)
      P(3,3) = CMPLX(1.0E0, 1.0E0)
      S(1,4) = CMPLX(5.0D0, 0.0E0)
      P(1,4) = CMPLX(0.0E0, 0.0E0)
      S(2,4) = CMPLX(2.0D0, 1.0E0)
      P(2,4) = CMPLX(0.0E0, 2.0D0)
      S(3,4) = CMPLX(4.0D0, -1.0E0)
      P(3,4) = CMPLX(3.0D0, 0.0E0)
      S(4,4) = CMPLX(3.0D0, 2.0D0)
      P(4,4) = CMPLX(2.0D0, -1.0E0)
      
      DO I = 1, N
         ALPHA(I) = S(I,I)
         BETA(I) = P(I,I)
         DO J = 1, N
            VR(J,I) = CMPLX(0.0E0, 0.0E0)
            VL(J,I) = CMPLX(0.0E0, 0.0E0)
         END DO
         VR(I,I) = CMPLX(1.0E0, 0.0E0)
         VL(I,I) = CMPLX(1.0E0, 0.0E0)
      END DO
      
      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 + 
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      
      CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
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
      COMPLEX, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      COMPLEX, ALLOCATABLE :: ALPHA(:), BETA(:)
      COMPLEX, ALLOCATABLE :: WORK(:)
      COMPLEX :: DUMMY(1)
      REAL :: ERR_R, ERR_L
      INTEGER :: INFO, LWORK, I, J, M_OUT
      LOGICAL :: SEL(1)
      
      PRINT *, '--- Testing Dynamic Size Complex Matrix ---', N
      
      ALLOCATE(S(N,N), P(N,N), VR(N,N), VL(N,N))
      ALLOCATE(ALPHA(N), BETA(N))
      
      DO J = 1, N
         DO I = 1, N
            IF (I .EQ. J) THEN
               VR(I,J) = CMPLX(1.0E0, 0.0E0)
               VL(I,J) = CMPLX(1.0E0, 0.0E0)
            ELSE
               VR(I,J) = CMPLX(0.0E0, 0.0E0)
               VL(I,J) = CMPLX(0.0E0, 0.0E0)
            END IF
         END DO
      END DO
      
      CALL GENERATE_GENERALIZED_UPPER_TRIANGULAR(N, S, N, P, N,
     $     ALPHA, BETA)
     
      CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)
     
      LWORK = INT(REAL(DUMMY(1)))
      PRINT *, 'LWORK = ', LWORK
      
      ALLOCATE(WORK(LWORK))
      CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
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
      COMPLEX, INTENT(OUT) :: S(LDS,N), P(LDP,N)
      COMPLEX, INTENT(OUT) :: ALPHA(N), BETA(N)
      INTEGER :: I, J, C, R
      REAL :: RAND_R, RAND_I
      
      DO J = 1, N
         DO I = 1, N
            S(I,J) = CMPLX(0.0E0, 0.0E0)
            P(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
      END DO
      
      DO C = 1, N
         DO R = 1, C
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            S(R,C) = CMPLX(RAND_R * 2.0D0 - 1.0E0, 
     $                      RAND_I * 2.0D0 - 1.0E0)
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            P(R,C) = CMPLX(RAND_R * 2.0D0 - 1.0E0, 
     $                      RAND_I * 2.0D0 - 1.0E0)
         END DO
         IF (REAL(P(C,C)) .GE. 0.0E0) THEN
            P(C,C) = P(C,C) + CMPLX(1.0E0, 0.0E0)
         ELSE
            P(C,C) = P(C,C) - CMPLX(1.0E0, 0.0E0)
         END IF
         ALPHA(C) = S(C,C)
         BETA(C) = P(C,C)
      END DO
      END SUBROUTINE GENERATE_GENERALIZED_UPPER_TRIANGULAR

      SUBROUTINE CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP(N, S, LDS, P,
     $   LDP, VR, LDVR, ALPHA, BETA, MAX_ERR)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP, LDVR
      COMPLEX, INTENT(IN) :: S(LDS,N), P(LDP,N), VR(LDVR,N)
      COMPLEX, INTENT(IN) :: ALPHA(N), BETA(N)
      REAL, INTENT(OUT) :: MAX_ERR
      
      REAL :: NORMS, NORMP
      REAL :: NORM_R, NORM_V
      INTEGER :: C, R, I, COL, ROW, INCX
      COMPLEX :: A, B
      COMPLEX, ALLOCATABLE :: R_VEC(:)
      
      REAL :: SCNRM2
      EXTERNAL SCNRM2
      
      NORMS = 0.0E0
      NORMP = 0.0E0
      DO C = 1, N
         DO R = 1, C
            NORMS = NORMS + REAL(S(R,C))**2 + AIMAG(S(R,C))**2
            NORMP = NORMP + REAL(P(R,C))**2 + AIMAG(P(R,C))**2
         END DO
      END DO
      NORMS = SQRT(NORMS)
      NORMP = SQRT(NORMP)
      IF (NORMS .EQ. 0.0E0) NORMS = 1.0E0
      IF (NORMP .EQ. 0.0E0) NORMP = 1.0E0
      
      MAX_ERR = 0.0E0
      ALLOCATE(R_VEC(N))
      INCX = 1
      
      DO C = 1, N
         A = ALPHA(C)
         B = BETA(C)
         
         DO I = 1, N
            R_VEC(I) = CMPLX(0.0E0, 0.0E0)
         END DO
         
         DO COL = 1, N
            DO ROW = 1, COL
               R_VEC(ROW) = R_VEC(ROW) + 
     $              B * S(ROW,COL) * VR(COL,C) - 
     $              A * P(ROW,COL) * VR(COL,C)
            END DO
         END DO
         
         NORM_R = SCNRM2(N, R_VEC, INCX)
         NORM_V = SCNRM2(N, VR(1,C), INCX)
         
         MAX_ERR = MAX(MAX_ERR, NORM_R / ((ABS(B)*NORMS + 
     $                 ABS(A)*NORMP) * NORM_V))
      END DO
      
      DEALLOCATE(R_VEC)
      END SUBROUTINE CHECK_RIGHT_EIGENVECTOR_RESIDUAL_GEVP

      SUBROUTINE CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP(N, S, LDS, P,
     $   LDP, VL, LDVL, ALPHA, BETA, MAX_ERR)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP, LDVL
      COMPLEX, INTENT(IN) :: S(LDS,N), P(LDP,N), VL(LDVL,N)
      COMPLEX, INTENT(IN) :: ALPHA(N), BETA(N)
      REAL, INTENT(OUT) :: MAX_ERR
      
      REAL :: NORMS, NORMP
      REAL :: NORM_R, NORM_V
      INTEGER :: C, R, COL, ROW, INCX
      COMPLEX :: A, B, SUM_VAL
      COMPLEX, ALLOCATABLE :: R_VEC(:)
      
      REAL :: SCNRM2
      EXTERNAL SCNRM2
      
      NORMS = 0.0E0
      NORMP = 0.0E0
      DO C = 1, N
         DO R = 1, C
            NORMS = NORMS + REAL(S(R,C))**2 + AIMAG(S(R,C))**2
            NORMP = NORMP + REAL(P(R,C))**2 + AIMAG(P(R,C))**2
         END DO
      END DO
      NORMS = SQRT(NORMS)
      NORMP = SQRT(NORMP)
      IF (NORMS .EQ. 0.0E0) NORMS = 1.0E0
      IF (NORMP .EQ. 0.0E0) NORMP = 1.0E0
      
      MAX_ERR = 0.0E0
      ALLOCATE(R_VEC(N))
      INCX = 1
      
      DO C = 1, N
         A = ALPHA(C)
         B = BETA(C)
         
         DO COL = 1, N
            SUM_VAL = CMPLX(0.0E0, 0.0E0)
            DO ROW = 1, COL
               SUM_VAL = SUM_VAL + CONJG(VL(ROW,C)) * 
     $                   (B * S(ROW,COL) - A * P(ROW,COL))
            END DO
            R_VEC(COL) = SUM_VAL
         END DO
         
         NORM_R = SCNRM2(N, R_VEC, INCX)
         NORM_V = SCNRM2(N, VL(1,C), INCX)
         
         MAX_ERR = MAX(MAX_ERR, NORM_R / ((ABS(B)*NORMS + 
     $                 ABS(A)*NORMP) * NORM_V))
      END DO
      
      DEALLOCATE(R_VEC)
      END SUBROUTINE CHECK_LEFT_EIGENVECTOR_RESIDUAL_GEVP
      
      END PROGRAM TEST_ZTGEVC3