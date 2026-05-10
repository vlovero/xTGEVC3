      PROGRAM TEST_CTGEVC3
      IMPLICIT NONE

      CALL TEST_CTGEVC_STATIC()
      CALL TEST_CTGEVC_INFINITE()
      CALL TEST_CTGEVC_SCALING()
      CALL TEST_DYNAMIC_SIZE(10)
      CALL TEST_DYNAMIC_SIZE(500)
      CALL TEST_LAPACK_TYPES(10)
      CALL TEST_LAPACK_TYPES(50)

      CONTAINS

      SUBROUTINE TEST_CTGEVC_STATIC()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      COMPLEX, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX S(N,N), P(N,N), ALPHA(N), BETA(N)
      COMPLEX VR(N,N), VL(N,N)
      REAL RES3(2), RESL(2)
      LOGICAL SEL(1)

      PRINT *, '--- Testing 4x4 Complex Static Matrix ---'

      DO J = 1, N
         DO I = 1, N
            S(I,J) = CMPLX(0.0E0, 0.0E0)
            P(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
      END DO

      S(1,1) = CMPLX(1.0E0, 1.0E0)
      P(1,1) = CMPLX(2.0E0, 0.0E0)
      S(1,2) = CMPLX(2.0E0, 0.0E0)
      P(1,2) = CMPLX(1.0E0, 1.0E0)
      S(2,2) = CMPLX(2.0E0, -1.0E0)
      P(2,2) = CMPLX(1.0E0, 0.0E0)
      S(1,3) = CMPLX(-5.0E0, 1.0E0)
      P(1,3) = CMPLX(0.0E0, 0.0E0)
      S(2,3) = CMPLX(0.0E0, 2.0E0)
      P(2,3) = CMPLX(-1.0E0, 0.0E0)
      S(3,3) = CMPLX(3.0E0, 0.0E0)
      P(3,3) = CMPLX(1.0E0, 0.0E0)
      S(1,4) = CMPLX(5.0E0, 0.0E0)
      P(1,4) = CMPLX(0.0E0, 0.0E0)
      S(2,4) = CMPLX(2.0E0, 1.0E0)
      P(2,4) = CMPLX(0.0E0, 2.0E0)
      S(3,4) = CMPLX(4.0E0, -1.0E0)
      P(3,4) = CMPLX(3.0E0, 0.0E0)
      S(4,4) = CMPLX(3.0E0, 2.0E0)
      P(4,4) = CMPLX(2.0E0, 0.0E0)

      DO I = 1, N
         ALPHA(I) = S(I,I)
         BETA(I) = P(I,I)
      END DO

      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 +
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      ALLOCATE(WORK_LAPACK(2*N))
      ALLOCATE(RWORK_LAPACK(2*N))
      SEL(1) = .FALSE.

      ! Test ctgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = CMPLX(0.0E0, 0.0E0)
            VL(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
         VR(J,J) = CMPLX(1.0E0, 0.0E0)
         VL(J,J) = CMPLX(1.0E0, 0.0E0)
      END DO
      CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RES3)

      ! Test LAPACK ctgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = CMPLX(0.0E0, 0.0E0)
            VL(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
         VR(J,J) = CMPLX(1.0E0, 0.0E0)
         VL(J,J) = CMPLX(1.0E0, 0.0E0)
      END DO
      CALL CTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
      CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PE14.6)') 
     $   '[ctgevc3] cget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[ctgevc3] cget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] cget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] cget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(WORK, WORK_LAPACK, RWORK_LAPACK)
      END SUBROUTINE TEST_CTGEVC_STATIC

      SUBROUTINE TEST_CTGEVC_INFINITE()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      COMPLEX, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX S(N,N), P(N,N), ALPHA(N), BETA(N)
      COMPLEX VR(N,N), VL(N,N)
      REAL RES3(2), RESL(2)
      LOGICAL SEL(1)

      PRINT *, '--- Testing 4x4 Complex Matrix (Infinite Eval) ---'

      DO J = 1, N
         DO I = 1, N
            S(I,J) = CMPLX(0.0E0, 0.0E0)
            P(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
      END DO

      S(1,1) = CMPLX(1.0E0, 1.0E0)
      P(1,1) = CMPLX(2.0E0, 0.0E0)
      S(1,2) = CMPLX(2.0E0, 0.0E0)
      P(1,2) = CMPLX(1.0E0, 1.0E0)
      S(2,2) = CMPLX(2.0E0, -1.0E0)
      P(2,2) = CMPLX(1.0E0, 0.0E0)
      S(1,3) = CMPLX(-5.0E0, 1.0E0)
      P(1,3) = CMPLX(0.0E0, 0.0E0)
      S(2,3) = CMPLX(0.0E0, 2.0E0)
      P(2,3) = CMPLX(-1.0E0, 0.0E0)
      S(3,3) = CMPLX(3.0E0, 0.0E0)
      P(3,3) = CMPLX(1.0E0, 0.0E0)
      S(1,4) = CMPLX(5.0E0, 0.0E0)
      P(1,4) = CMPLX(0.0E0, 0.0E0)
      S(2,4) = CMPLX(2.0E0, 1.0E0)
      P(2,4) = CMPLX(0.0E0, 2.0E0)
      S(3,4) = CMPLX(4.0E0, -1.0E0)
      P(3,4) = CMPLX(3.0E0, 0.0E0)
      S(4,4) = CMPLX(3.0E0, 2.0E0)
      P(4,4) = CMPLX(0.0E0, 0.0E0)

      DO I = 1, N
         ALPHA(I) = S(I,I)
         BETA(I) = P(I,I)
      END DO

      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 +
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      ALLOCATE(WORK_LAPACK(2*N))
      ALLOCATE(RWORK_LAPACK(2*N))
      SEL(1) = .FALSE.

      ! Test ctgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = CMPLX(0.0E0, 0.0E0)
            VL(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
         VR(J,J) = CMPLX(1.0E0, 0.0E0)
         VL(J,J) = CMPLX(1.0E0, 0.0E0)
      END DO
      CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RES3)

      ! Test LAPACK ctgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = CMPLX(0.0E0, 0.0E0)
            VL(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
         VR(J,J) = CMPLX(1.0E0, 0.0E0)
         VL(J,J) = CMPLX(1.0E0, 0.0E0)
      END DO
      CALL CTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
      CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PE14.6)') 
     $   '[ctgevc3] cget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[ctgevc3] cget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] cget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] cget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(WORK, WORK_LAPACK, RWORK_LAPACK)
      END SUBROUTINE TEST_CTGEVC_INFINITE

      SUBROUTINE TEST_CTGEVC_SCALING()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT, TEST_IDX
      COMPLEX, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX S_BASE(N,N), P_BASE(N,N), S(N,N), P(N,N)
      COMPLEX ALPHA(N), BETA(N), VR(N,N), VL(N,N)
      REAL RES3(2), RESL(2), SCALES(2), S_VAL
      LOGICAL SEL(1)

      DO J = 1, N
         DO I = 1, N
            S_BASE(I,J) = CMPLX(0.0E0, 0.0E0)
            P_BASE(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
      END DO

      S_BASE(1,1) = CMPLX(1.0E0, 1.0E0)
      P_BASE(1,1) = CMPLX(2.0E0, 0.0E0)
      S_BASE(1,2) = CMPLX(2.0E0, 0.0E0)
      P_BASE(1,2) = CMPLX(1.0E0, 1.0E0)
      S_BASE(2,2) = CMPLX(2.0E0,-1.0E0)
      P_BASE(2,2) = CMPLX(1.0E0, 0.0E0)
      S_BASE(1,3) = CMPLX(-5.0E0,1.0E0)
      P_BASE(1,3) = CMPLX(0.0E0, 0.0E0)
      S_BASE(2,3) = CMPLX(0.0E0, 2.0E0)
      P_BASE(2,3) = CMPLX(-1.0E0,0.0E0)
      S_BASE(3,3) = CMPLX(3.0E0, 0.0E0)
      P_BASE(3,3) = CMPLX(1.0E0, 0.0E0)
      S_BASE(1,4) = CMPLX(5.0E0, 0.0E0)
      P_BASE(1,4) = CMPLX(0.0E0, 0.0E0)
      S_BASE(2,4) = CMPLX(2.0E0, 1.0E0)
      P_BASE(2,4) = CMPLX(0.0E0, 2.0E0)
      S_BASE(3,4) = CMPLX(4.0E0,-1.0E0)
      P_BASE(3,4) = CMPLX(3.0E0, 0.0E0)
      S_BASE(4,4) = CMPLX(3.0E0, 2.0E0)
      P_BASE(4,4) = CMPLX(2.0E0, 0.0E0)

      SCALES(1) = 1.0E30
      SCALES(2) = 1.0E-30

      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 +
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      ALLOCATE(WORK_LAPACK(2*N))
      ALLOCATE(RWORK_LAPACK(2*N))
      SEL(1) = .FALSE.

      DO TEST_IDX = 1, 2
         S_VAL = SCALES(TEST_IDX)
         IF (TEST_IDX .EQ. 1) THEN
            PRINT *, '--- Testing 4x4 Matrix (Scaling: 1e30) ---'
         ELSE
            PRINT *, '--- Testing 4x4 Matrix (Scaling: 1e-30) ---'
         END IF

         DO J = 1, N
            DO I = 1, N
               S(I,J) = S_BASE(I,J) * CMPLX(S_VAL, 0.0E0)
               P(I,J) = P_BASE(I,J) * CMPLX(S_VAL, 0.0E0)
            END DO
         END DO

         DO I = 1, N
            ALPHA(I) = S(I,I)
            BETA(I) = P(I,I)
         END DO

         ! Test ctgevc3
         DO J = 1, N
            DO I = 1, N
               VR(I,J) = CMPLX(0.0E0, 0.0E0)
               VL(I,J) = CMPLX(0.0E0, 0.0E0)
            END DO
            VR(J,J) = CMPLX(1.0E0, 0.0E0)
            VL(J,J) = CMPLX(1.0E0, 0.0E0)
         END DO
         CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $                VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
         CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $               VL, N, VR, N, RES3)

         ! Test LAPACK ctgevc
         DO J = 1, N
            DO I = 1, N
               VR(I,J) = CMPLX(0.0E0, 0.0E0)
               VL(I,J) = CMPLX(0.0E0, 0.0E0)
            END DO
            VR(J,J) = CMPLX(1.0E0, 0.0E0)
            VL(J,J) = CMPLX(1.0E0, 0.0E0)
         END DO
         CALL CTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $               M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
         CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $               VL, N, VR, N, RESL)

         WRITE(*, '(A, 1PE14.6)') 
     $      '[ctgevc3] cget52 Max Right Error: ', RES3(1)
         WRITE(*, '(A, 1PE14.6)') 
     $      '[ctgevc3] cget52 Max Left Error:  ', RES3(2)
         WRITE(*, '(A, 1PE14.6)') 
     $      '[lapack ] cget52 Max Right Error: ', RESL(1)
         WRITE(*, '(A, 1PE14.6)') 
     $      '[lapack ] cget52 Max Left Error:  ', RESL(2)
         PRINT *
      END DO

      DEALLOCATE(WORK, WORK_LAPACK, RWORK_LAPACK)
      END SUBROUTINE TEST_CTGEVC_SCALING

      SUBROUTINE TEST_DYNAMIC_SIZE(N)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      COMPLEX, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      COMPLEX, ALLOCATABLE :: ALPHA(:), BETA(:)
      COMPLEX, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX DUMMY(1)
      REAL RES3(2), RESL(2), T1, T2
      INTEGER INFO, LWORK, I, J, M_OUT
      LOGICAL SEL(1)

      PRINT *, '--- Testing Dynamic Size Matrix ---', N

      ALLOCATE(S(N,N), P(N,N), VR(N,N), VL(N,N))
      ALLOCATE(ALPHA(N), BETA(N))
      ALLOCATE(WORK_LAPACK(2*N), RWORK_LAPACK(2*N))
      SEL(1) = .FALSE.

      CALL GENERATE_GENERALIZED_UPPER_TRIANGULAR(N, S, N, P, N,
     $     ALPHA, BETA)

      CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)

      LWORK = INT(REAL(DUMMY(1)))
      IF (LWORK .LE. 0) THEN
         LWORK = 2*N*(32+1) + 4*(32+1)**2 + 2*(32+1)
      END IF
      ALLOCATE(WORK(LWORK))

      ! Test ctgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = CMPLX(0.0E0, 0.0E0)
            VL(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
         VR(J,J) = CMPLX(1.0E0, 0.0E0)
         VL(J,J) = CMPLX(1.0E0, 0.0E0)
      END DO
      CALL CPU_TIME(T1)
      CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, WORK, LWORK, INFO)
      CALL CPU_TIME(T2)
      PRINT *, 'ctgevc3 (''B'') took ', T2-T1, ' seconds'
      CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RES3)

      ! Test LAPACK ctgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = CMPLX(0.0E0, 0.0E0)
            VL(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
         VR(J,J) = CMPLX(1.0E0, 0.0E0)
         VL(J,J) = CMPLX(1.0E0, 0.0E0)
      END DO
      CALL CPU_TIME(T1)
      CALL CTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
      CALL CPU_TIME(T2)
      PRINT *, 'LAPACK ctgevc (''B'') took ', T2-T1, ' seconds'
      CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PE14.6)') 
     $   '[ctgevc3] cget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[ctgevc3] cget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] cget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] cget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(S, P, VR, VL, ALPHA, BETA, WORK, WORK_LAPACK)
      DEALLOCATE(RWORK_LAPACK)
      END SUBROUTINE TEST_DYNAMIC_SIZE

      SUBROUTINE TEST_LAPACK_TYPES(N)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      COMPLEX, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      COMPLEX, ALLOCATABLE :: ALPHA(:), BETA(:)
      COMPLEX, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX DUMMY(1)
      REAL RES3(2), RESL(2)
      INTEGER INFO, LWORK, I, J, TYPE, M_OUT
      LOGICAL SEL(1)

      PRINT *, '--- Testing ', N, 'x', N, ' LAPACK Matrix Types ---'

      ALLOCATE(S(N,N), P(N,N), VR(N,N), VL(N,N))
      ALLOCATE(ALPHA(N), BETA(N))
      ALLOCATE(WORK_LAPACK(2*N), RWORK_LAPACK(2*N))
      SEL(1) = .FALSE.

      DO TYPE = 1, 26
         CALL GENERATE_LAPACK_MATRIX_TYPE(TYPE, N, S, N, P, N,
     $        ALPHA, BETA)

         CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $                VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)

         LWORK = INT(REAL(DUMMY(1)))
         IF (LWORK .LE. 0) THEN
            LWORK = 2*N*(32+1) + 4*(32+1)**2 + 2*(32+1)
         END IF
         ALLOCATE(WORK(LWORK))

         DO J = 1, N
            DO I = 1, N
               VR(I,J) = CMPLX(0.0E0, 0.0E0)
               VL(I,J) = CMPLX(0.0E0, 0.0E0)
            END DO
            VR(J,J) = CMPLX(1.0E0, 0.0E0)
            VL(J,J) = CMPLX(1.0E0, 0.0E0)
         END DO
         CALL CTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $                VL, N, VR, N, N, M_OUT, WORK, LWORK, INFO)
         CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $               VL, N, VR, N, RES3)

         DO J = 1, N
            DO I = 1, N
               VR(I,J) = CMPLX(0.0E0, 0.0E0)
               VL(I,J) = CMPLX(0.0E0, 0.0E0)
            END DO
            VR(J,J) = CMPLX(1.0E0, 0.0E0)
            VL(J,J) = CMPLX(1.0E0, 0.0E0)
         END DO
         CALL CTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $               M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
         CALL CGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $               VL, N, VR, N, RESL)

         WRITE(*, '(A, I2, A)') 'Type ', TYPE, ':'
         WRITE(*, '(A, 1PE14.6, A, 1PE14.6)') 
     $      '  [ctgevc3] Max R = ', RES3(1), ' L = ', RES3(2)
         WRITE(*, '(A, 1PE14.6, A, 1PE14.6)') 
     $      '  [lapack ] Max R = ', RESL(1), ' L = ', RESL(2)

         DEALLOCATE(WORK)
      END DO
      PRINT *

      DEALLOCATE(S, P, VR, VL, ALPHA, BETA, WORK_LAPACK, RWORK_LAPACK)
      END SUBROUTINE TEST_LAPACK_TYPES

      SUBROUTINE CGET52(COMP_L, COMP_R, N, S, LDS, P, LDP,
     $                  ALPHA, BETA, VL, LDVL, VR, LDVR, RES)
      IMPLICIT NONE
      LOGICAL COMP_L, COMP_R
      INTEGER N, LDS, LDP, LDVL, LDVR
      COMPLEX S(LDS,*), P(LDP,*), ALPHA(*), BETA(*)
      COMPLEX VL(LDVL,*), VR(LDVR,*)
      REAL RES(2)

      REAL NORMS, NORMP, CSS, CSP, NORM_R, NORM_V, DEN
      REAL ULP, SAFMIN, MAX_ERR_R, MAX_ERR_L
      COMPLEX A, B, VAL
      INTEGER C, R, I, J

      REAL SLAMCH
      EXTERNAL SLAMCH

      ULP = SLAMCH('E')
      SAFMIN = SLAMCH('S')

      NORMS = 0.0E0
      NORMP = 0.0E0
      DO C = 1, N
         CSS = 0.0E0
         CSP = 0.0E0
         DO R = 1, N
            CSS = CSS + ABS(S(R,C))
            CSP = CSP + ABS(P(R,C))
         END DO
         NORMS = MAX(NORMS, CSS)
         NORMP = MAX(NORMP, CSP)
      END DO

      IF (NORMS .EQ. 0.0E0) NORMS = 1.0E0
      IF (NORMP .EQ. 0.0E0) NORMP = 1.0E0

      MAX_ERR_R = 0.0E0
      MAX_ERR_L = 0.0E0

      IF (COMP_R) THEN
         DO C = 1, N
            A = ALPHA(C)
            B = BETA(C)
            NORM_R = 0.0E0
            NORM_V = 0.0E0

            DO I = 1, N
               VAL = CMPLX(0.0E0, 0.0E0)
               DO J = 1, N
                  VAL = VAL + B * S(I,J) * VR(J,C) -
     $                        A * P(I,J) * VR(J,C)
               END DO
               NORM_R = NORM_R + ABS(VAL)
               NORM_V = NORM_V + ABS(VR(I,C))
            END DO

            DEN = (ABS(B) * NORMS + ABS(A) * NORMP) * NORM_V
            DEN = MAX(DEN, SAFMIN)
            MAX_ERR_R = MAX(MAX_ERR_R, NORM_R / (DEN * ULP))
         END DO
      END IF

      IF (COMP_L) THEN
         DO C = 1, N
            A = ALPHA(C)
            B = BETA(C)
            NORM_R = 0.0E0
            NORM_V = 0.0E0

            DO J = 1, N
               VAL = CMPLX(0.0E0, 0.0E0)
               DO I = 1, N
                  VAL = VAL + CONJG(VL(I,C)) *
     $                  (B * S(I,J) - A * P(I,J))
               END DO
               NORM_R = NORM_R + ABS(VAL)
            END DO
            DO I = 1, N
               NORM_V = NORM_V + ABS(VL(I,C))
            END DO

            DEN = (ABS(B) * NORMS + ABS(A) * NORMP) * NORM_V
            DEN = MAX(DEN, SAFMIN)
            MAX_ERR_L = MAX(MAX_ERR_L, NORM_R / (DEN * ULP))
         END DO
      END IF

      RES(1) = MAX_ERR_R
      RES(2) = MAX_ERR_L
      RETURN
      END SUBROUTINE CGET52

      SUBROUTINE GENERATE_GENERALIZED_UPPER_TRIANGULAR(N, S, LDS, P,
     $   LDP, ALPHA, BETA)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP
      COMPLEX, INTENT(OUT) :: S(LDS,N), P(LDP,N)
      COMPLEX, INTENT(OUT) :: ALPHA(N), BETA(N)
      INTEGER C, R
      REAL RAND_R, RAND_I

      DO C = 1, N
         DO R = 1, N
            S(R,C) = CMPLX(0.0E0, 0.0E0)
            P(R,C) = CMPLX(0.0E0, 0.0E0)
         END DO
      END DO

      DO C = 1, N
         DO R = 1, C - 1
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            S(R,C) = CMPLX(RAND_R * 2.0E0 - 1.0E0,
     $                     RAND_I * 2.0E0 - 1.0E0)
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            P(R,C) = CMPLX(RAND_R * 2.0E0 - 1.0E0,
     $                     RAND_I * 2.0E0 - 1.0E0)
         END DO
         CALL RANDOM_NUMBER(RAND_R)
         CALL RANDOM_NUMBER(RAND_I)
         S(C,C) = CMPLX(RAND_R * 2.0E0 - 1.0E0, RAND_I * 2.0E0 - 1.0E0)

         CALL RANDOM_NUMBER(RAND_R)
         P(C,C) = CMPLX(ABS(RAND_R * 2.0E0 - 1.0E0) + 0.1E0, 0.0E0)

         ALPHA(C) = S(C,C)
         BETA(C) = P(C,C)
      END DO
      END SUBROUTINE GENERATE_GENERALIZED_UPPER_TRIANGULAR

      SUBROUTINE GENERATE_LAPACK_MATRIX_TYPE(TYPE, N, S, LDS, P, LDP,
     $                                       ALPHA, BETA)
      IMPLICIT NONE
      INTEGER TYPE, N, LDS, LDP
      COMPLEX S(LDS,*), P(LDP,*)
      COMPLEX ALPHA(*), BETA(*)

      INTEGER I, J
      REAL BIG, SMALL, RAND_R, RAND_I

      BIG = 1.0E10
      SMALL = 1.0E-10

      DO J = 1, N
         DO I = 1, N
            S(I,J) = CMPLX(0.0E0, 0.0E0)
            P(I,J) = CMPLX(0.0E0, 0.0E0)
         END DO
      END DO

      SELECT CASE (TYPE)
      CASE (1)
         ! Zero matrix
      CASE (2)
         DO I = 1, N
            S(I,I) = CMPLX(1.0E0, 0.0E0)
         END DO
      CASE (3)
         DO I = 1, N
            P(I,I) = CMPLX(1.0E0, 0.0E0)
         END DO
      CASE (4)
         DO I = 1, N
            S(I,I) = CMPLX(1.0E0, 0.0E0)
            P(I,I) = CMPLX(1.0E0, 0.0E0)
         END DO
      CASE (5)
         DO I = 1, N
            S(I,I) = CMPLX(1.0E0, 0.0E0)
            P(I,I) = CMPLX(1.0E0, 0.0E0)
            IF (I .LT. N) THEN
               S(I,I+1) = CMPLX(1.0E0, 0.0E0)
               P(I,I+1) = CMPLX(1.0E0, 0.0E0)
            END IF
         END DO
      CASE (6)
         DO I = 1, N
            S(I,I) = CMPLX(REAL(I) / REAL(N), 0.0E0)
            P(I,I) = CMPLX(1.0E0, 0.0E0)
         END DO
      CASE (7)
         DO I = 1, N
            S(I,I) = CMPLX(1.0E0, 0.0E0)
            P(I,I) = CMPLX(REAL(I) / REAL(N), 0.0E0)
         END DO
      CASE (8)
         DO I = 1, N
            S(I,I) = CMPLX(REAL(I) / REAL(N), 0.0E0)
            P(I,I) = CMPLX(REAL(N - I + 1) / REAL(N), 0.0E0)
         END DO
      CASE (9)
         DO I = 1, N
            S(I,I) = CMPLX(BIG * REAL(I) / REAL(N), 0.0E0)
            P(I,I) = CMPLX(SMALL, 0.0E0)
         END DO
      CASE (10)
         DO I = 1, N
            S(I,I) = CMPLX(SMALL * REAL(I) / REAL(N), 0.0E0)
            P(I,I) = CMPLX(BIG, 0.0E0)
         END DO
      CASE (11)
         DO I = 1, N
            S(I,I) = CMPLX(BIG, 0.0E0)
            P(I,I) = CMPLX(SMALL * REAL(I) / REAL(N), 0.0E0)
         END DO
      CASE (12)
         DO I = 1, N
            S(I,I) = CMPLX(SMALL, 0.0E0)
            P(I,I) = CMPLX(BIG * REAL(I) / REAL(N), 0.0E0)
         END DO
      CASE (13)
         DO I = 1, N
            S(I,I) = CMPLX(BIG * REAL(I) / REAL(N), 0.0E0)
            P(I,I) = CMPLX(BIG, 0.0E0)
         END DO
      CASE (14)
         DO I = 1, N
            S(I,I) = CMPLX(SMALL * REAL(I) / REAL(N), 0.0E0)
            P(I,I) = CMPLX(SMALL, 0.0E0)
         END DO
      CASE (15)
         DO I = 1, N
            IF (I .EQ. 1 .OR. I .EQ. 2 .OR. I .EQ. N) THEN
               S(I,I) = CMPLX(0.0E0, 0.0E0)
            ELSE
               S(I,I) = CMPLX(REAL(I - 2), 0.0E0)
            END IF

            IF (I .EQ. 1 .OR. I .EQ. N-1 .OR. I .EQ. N) THEN
               P(I,I) = CMPLX(0.0E0, 0.0E0)
            ELSE
               P(I,I) = CMPLX(REAL(N - I), 0.0E0)
            END IF
         END DO
      CASE DEFAULT
         DO J = 1, N
            DO I = 1, J - 1
               CALL RANDOM_NUMBER(RAND_R)
               CALL RANDOM_NUMBER(RAND_I)
               S(I,J) = CMPLX(RAND_R * 2.0E0 - 1.0E0,
     $                        RAND_I * 2.0E0 - 1.0E0)
               CALL RANDOM_NUMBER(RAND_R)
               CALL RANDOM_NUMBER(RAND_I)
               P(I,J) = CMPLX(RAND_R * 2.0E0 - 1.0E0,
     $                        RAND_I * 2.0E0 - 1.0E0)
            END DO
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            S(J,J) = CMPLX(RAND_R * 2.0E0 - 1.0E0,
     $                     RAND_I * 2.0E0 - 1.0E0)

            CALL RANDOM_NUMBER(RAND_R)
            P(J,J) = CMPLX(ABS(RAND_R * 2.0E0 - 1.0E0) + 0.1E0, 0.0E0)
         END DO
      END SELECT

      DO I = 1, N
         ALPHA(I) = S(I,I)
         BETA(I) = P(I,I)
      END DO
      RETURN
      END SUBROUTINE GENERATE_LAPACK_MATRIX_TYPE

      END PROGRAM TEST_CTGEVC3