      PROGRAM TEST_ZTGEVC3
      IMPLICIT NONE

      CALL TEST_ZTGEVC_STATIC()
      CALL TEST_ZTGEVC_INFINITE()
      CALL TEST_ZTGEVC_SCALING()
      CALL TEST_DYNAMIC_SIZE(10)
      CALL TEST_DYNAMIC_SIZE(500)
      CALL TEST_LAPACK_TYPES(10)
      CALL TEST_LAPACK_TYPES(50)

      CONTAINS

      SUBROUTINE TEST_ZTGEVC_STATIC()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      COMPLEX*16, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      DOUBLE PRECISION, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX*16 S(N,N), P(N,N), ALPHA(N), BETA(N)
      COMPLEX*16 VR(N,N), VL(N,N)
      DOUBLE PRECISION RES3(2), RESL(2)
      LOGICAL SEL(1)

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
      P(2,2) = DCMPLX(1.0D0, 0.0D0)
      S(1,3) = DCMPLX(-5.0D0, 1.0D0)
      P(1,3) = DCMPLX(0.0D0, 0.0D0)
      S(2,3) = DCMPLX(0.0D0, 2.0D0)
      P(2,3) = DCMPLX(-1.0D0, 0.0D0)
      S(3,3) = DCMPLX(3.0D0, 0.0D0)
      P(3,3) = DCMPLX(1.0D0, 0.0D0)
      S(1,4) = DCMPLX(5.0D0, 0.0D0)
      P(1,4) = DCMPLX(0.0D0, 0.0D0)
      S(2,4) = DCMPLX(2.0D0, 1.0D0)
      P(2,4) = DCMPLX(0.0D0, 2.0D0)
      S(3,4) = DCMPLX(4.0D0, -1.0D0)
      P(3,4) = DCMPLX(3.0D0, 0.0D0)
      S(4,4) = DCMPLX(3.0D0, 2.0D0)
      P(4,4) = DCMPLX(2.0D0, 0.0D0)

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

      ! Test ztgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = DCMPLX(0.0D0, 0.0D0)
            VL(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
         VR(J,J) = DCMPLX(1.0D0, 0.0D0)
         VL(J,J) = DCMPLX(1.0D0, 0.0D0)
      END DO
      CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RES3)

      ! Test LAPACK ztgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = DCMPLX(0.0D0, 0.0D0)
            VL(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
         VR(J,J) = DCMPLX(1.0D0, 0.0D0)
         VL(J,J) = DCMPLX(1.0D0, 0.0D0)
      END DO
      CALL ZTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
      CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PD14.6)') 
     $   '[ztgevc3] zget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[ztgevc3] zget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[lapack ] zget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[lapack ] zget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(WORK, WORK_LAPACK, RWORK_LAPACK)
      END SUBROUTINE TEST_ZTGEVC_STATIC

      SUBROUTINE TEST_ZTGEVC_INFINITE()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      COMPLEX*16, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      DOUBLE PRECISION, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX*16 S(N,N), P(N,N), ALPHA(N), BETA(N)
      COMPLEX*16 VR(N,N), VL(N,N)
      DOUBLE PRECISION RES3(2), RESL(2)
      LOGICAL SEL(1)

      PRINT *, '--- Testing 4x4 Complex Matrix (Infinite Eval) ---'

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
      P(2,2) = DCMPLX(1.0D0, 0.0D0)
      S(1,3) = DCMPLX(-5.0D0, 1.0D0)
      P(1,3) = DCMPLX(0.0D0, 0.0D0)
      S(2,3) = DCMPLX(0.0D0, 2.0D0)
      P(2,3) = DCMPLX(-1.0D0, 0.0D0)
      S(3,3) = DCMPLX(3.0D0, 0.0D0)
      P(3,3) = DCMPLX(1.0D0, 0.0D0)
      S(1,4) = DCMPLX(5.0D0, 0.0D0)
      P(1,4) = DCMPLX(0.0D0, 0.0D0)
      S(2,4) = DCMPLX(2.0D0, 1.0D0)
      P(2,4) = DCMPLX(0.0D0, 2.0D0)
      S(3,4) = DCMPLX(4.0D0, -1.0D0)
      P(3,4) = DCMPLX(3.0D0, 0.0D0)
      S(4,4) = DCMPLX(3.0D0, 2.0D0)
      P(4,4) = DCMPLX(0.0D0, 0.0D0)

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

      ! Test ztgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = DCMPLX(0.0D0, 0.0D0)
            VL(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
         VR(J,J) = DCMPLX(1.0D0, 0.0D0)
         VL(J,J) = DCMPLX(1.0D0, 0.0D0)
      END DO
      CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RES3)

      ! Test LAPACK ztgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = DCMPLX(0.0D0, 0.0D0)
            VL(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
         VR(J,J) = DCMPLX(1.0D0, 0.0D0)
         VL(J,J) = DCMPLX(1.0D0, 0.0D0)
      END DO
      CALL ZTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
      CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PD14.6)') 
     $   '[ztgevc3] zget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[ztgevc3] zget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[lapack ] zget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[lapack ] zget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(WORK, WORK_LAPACK, RWORK_LAPACK)
      END SUBROUTINE TEST_ZTGEVC_INFINITE

      SUBROUTINE TEST_ZTGEVC_SCALING()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT, TEST_IDX
      COMPLEX*16, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      DOUBLE PRECISION, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX*16 S_BASE(N,N), P_BASE(N,N), S(N,N), P(N,N)
      COMPLEX*16 ALPHA(N), BETA(N), VR(N,N), VL(N,N)
      DOUBLE PRECISION RES3(2), RESL(2), SCALES(2), S_VAL
      LOGICAL SEL(1)

      DO J = 1, N
         DO I = 1, N
            S_BASE(I,J) = DCMPLX(0.0D0, 0.0D0)
            P_BASE(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
      END DO

      S_BASE(1,1) = DCMPLX(1.0D0, 1.0D0)
      P_BASE(1,1) = DCMPLX(2.0D0, 0.0D0)
      S_BASE(1,2) = DCMPLX(2.0D0, 0.0D0)
      P_BASE(1,2) = DCMPLX(1.0D0, 1.0D0)
      S_BASE(2,2) = DCMPLX(2.0D0,-1.0D0)
      P_BASE(2,2) = DCMPLX(1.0D0, 0.0D0)
      S_BASE(1,3) = DCMPLX(-5.0D0,1.0D0)
      P_BASE(1,3) = DCMPLX(0.0D0, 0.0D0)
      S_BASE(2,3) = DCMPLX(0.0D0, 2.0D0)
      P_BASE(2,3) = DCMPLX(-1.0D0,0.0D0)
      S_BASE(3,3) = DCMPLX(3.0D0, 0.0D0)
      P_BASE(3,3) = DCMPLX(1.0D0, 0.0D0)
      S_BASE(1,4) = DCMPLX(5.0D0, 0.0D0)
      P_BASE(1,4) = DCMPLX(0.0D0, 0.0D0)
      S_BASE(2,4) = DCMPLX(2.0D0, 1.0D0)
      P_BASE(2,4) = DCMPLX(0.0D0, 2.0D0)
      S_BASE(3,4) = DCMPLX(4.0D0,-1.0D0)
      P_BASE(3,4) = DCMPLX(3.0D0, 0.0D0)
      S_BASE(4,4) = DCMPLX(3.0D0, 2.0D0)
      P_BASE(4,4) = DCMPLX(2.0D0, 0.0D0)

      SCALES(1) = 1.0D150
      SCALES(2) = 1.0D-150

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
            PRINT *, '--- Testing 4x4 Matrix (Scaling: 1e150) ---'
         ELSE
            PRINT *, '--- Testing 4x4 Matrix (Scaling: 1e-150) ---'
         END IF

         DO J = 1, N
            DO I = 1, N
               S(I,J) = S_BASE(I,J) * DCMPLX(S_VAL, 0.0D0)
               P(I,J) = P_BASE(I,J) * DCMPLX(S_VAL, 0.0D0)
            END DO
         END DO

         DO I = 1, N
            ALPHA(I) = S(I,I)
            BETA(I) = P(I,I)
         END DO

         ! Test ztgevc3
         DO J = 1, N
            DO I = 1, N
               VR(I,J) = DCMPLX(0.0D0, 0.0D0)
               VL(I,J) = DCMPLX(0.0D0, 0.0D0)
            END DO
            VR(J,J) = DCMPLX(1.0D0, 0.0D0)
            VL(J,J) = DCMPLX(1.0D0, 0.0D0)
         END DO
         CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $                VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
         CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $               VL, N, VR, N, RES3)

         ! Test LAPACK ztgevc
         DO J = 1, N
            DO I = 1, N
               VR(I,J) = DCMPLX(0.0D0, 0.0D0)
               VL(I,J) = DCMPLX(0.0D0, 0.0D0)
            END DO
            VR(J,J) = DCMPLX(1.0D0, 0.0D0)
            VL(J,J) = DCMPLX(1.0D0, 0.0D0)
         END DO
         CALL ZTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $               M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
         CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $               VL, N, VR, N, RESL)

         WRITE(*, '(A, 1PD14.6)') 
     $      '[ztgevc3] zget52 Max Right Error: ', RES3(1)
         WRITE(*, '(A, 1PD14.6)') 
     $      '[ztgevc3] zget52 Max Left Error:  ', RES3(2)
         WRITE(*, '(A, 1PD14.6)') 
     $      '[lapack ] zget52 Max Right Error: ', RESL(1)
         WRITE(*, '(A, 1PD14.6)') 
     $      '[lapack ] zget52 Max Left Error:  ', RESL(2)
         PRINT *
      END DO

      DEALLOCATE(WORK, WORK_LAPACK, RWORK_LAPACK)
      END SUBROUTINE TEST_ZTGEVC_SCALING

      SUBROUTINE TEST_DYNAMIC_SIZE(N)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      COMPLEX*16, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      COMPLEX*16, ALLOCATABLE :: ALPHA(:), BETA(:)
      COMPLEX*16, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      DOUBLE PRECISION, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX*16 DUMMY(1)
      DOUBLE PRECISION RES3(2), RESL(2)
      REAL T1, T2
      INTEGER INFO, LWORK, I, J, M_OUT
      LOGICAL SEL(1)

      PRINT *, '--- Testing Dynamic Size Matrix ---', N

      ALLOCATE(S(N,N), P(N,N), VR(N,N), VL(N,N))
      ALLOCATE(ALPHA(N), BETA(N))
      ALLOCATE(WORK_LAPACK(2*N), RWORK_LAPACK(2*N))
      SEL(1) = .FALSE.

      CALL GENERATE_GENERALIZED_UPPER_TRIANGULAR(N, S, N, P, N,
     $     ALPHA, BETA)

      CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)

      LWORK = INT(REAL(DUMMY(1)))
      IF (LWORK .LE. 0) THEN
         LWORK = 2*N*(32+1) + 4*(32+1)**2 + 2*(32+1)
      END IF
      ALLOCATE(WORK(LWORK))

      ! Test ztgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = DCMPLX(0.0D0, 0.0D0)
            VL(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
         VR(J,J) = DCMPLX(1.0D0, 0.0D0)
         VL(J,J) = DCMPLX(1.0D0, 0.0D0)
      END DO
      CALL CPU_TIME(T1)
      CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $             VL, N, VR, N, N, M_OUT, WORK, LWORK, INFO)
      CALL CPU_TIME(T2)
      PRINT *, 'ztgevc3 (''B'') took ', T2-T1, ' seconds'
      CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RES3)

      ! Test LAPACK ztgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = DCMPLX(0.0D0, 0.0D0)
            VL(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
         VR(J,J) = DCMPLX(1.0D0, 0.0D0)
         VL(J,J) = DCMPLX(1.0D0, 0.0D0)
      END DO
      CALL CPU_TIME(T1)
      CALL ZTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
      CALL CPU_TIME(T2)
      PRINT *, 'LAPACK ztgevc (''B'') took ', T2-T1, ' seconds'
      CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $            VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PD14.6)') 
     $   '[ztgevc3] zget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[ztgevc3] zget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[lapack ] zget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PD14.6)') 
     $   '[lapack ] zget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(S, P, VR, VL, ALPHA, BETA, WORK, WORK_LAPACK)
      DEALLOCATE(RWORK_LAPACK)
      END SUBROUTINE TEST_DYNAMIC_SIZE

      SUBROUTINE TEST_LAPACK_TYPES(N)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      COMPLEX*16, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      COMPLEX*16, ALLOCATABLE :: ALPHA(:), BETA(:)
      COMPLEX*16, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      DOUBLE PRECISION, ALLOCATABLE :: RWORK_LAPACK(:)
      COMPLEX*16 DUMMY(1)
      DOUBLE PRECISION RES3(2), RESL(2)
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

         CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $                VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)

         LWORK = INT(REAL(DUMMY(1)))
         IF (LWORK .LE. 0) THEN
            LWORK = 2*N*(32+1) + 4*(32+1)**2 + 2*(32+1)
         END IF
         ALLOCATE(WORK(LWORK))

         DO J = 1, N
            DO I = 1, N
               VR(I,J) = DCMPLX(0.0D0, 0.0D0)
               VL(I,J) = DCMPLX(0.0D0, 0.0D0)
            END DO
            VR(J,J) = DCMPLX(1.0D0, 0.0D0)
            VL(J,J) = DCMPLX(1.0D0, 0.0D0)
         END DO
         CALL ZTGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHA, BETA,
     $                VL, N, VR, N, N, M_OUT, WORK, LWORK, INFO)
         CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $               VL, N, VR, N, RES3)

         DO J = 1, N
            DO I = 1, N
               VR(I,J) = DCMPLX(0.0D0, 0.0D0)
               VL(I,J) = DCMPLX(0.0D0, 0.0D0)
            END DO
            VR(J,J) = DCMPLX(1.0D0, 0.0D0)
            VL(J,J) = DCMPLX(1.0D0, 0.0D0)
         END DO
         CALL ZTGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $               M_OUT, WORK_LAPACK, RWORK_LAPACK, INFO)
         CALL ZGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHA, BETA,
     $               VL, N, VR, N, RESL)

         WRITE(*, '(A, I2, A)') 'Type ', TYPE, ':'
         WRITE(*, '(A, 1PD14.6, A, 1PD14.6)') 
     $      '  [ztgevc3] Max R = ', RES3(1), ' L = ', RES3(2)
         WRITE(*, '(A, 1PD14.6, A, 1PD14.6)') 
     $      '  [lapack ] Max R = ', RESL(1), ' L = ', RESL(2)

         DEALLOCATE(WORK)
      END DO
      PRINT *

      DEALLOCATE(S, P, VR, VL, ALPHA, BETA, WORK_LAPACK, RWORK_LAPACK)
      END SUBROUTINE TEST_LAPACK_TYPES

      SUBROUTINE ZGET52(COMP_L, COMP_R, N, S, LDS, P, LDP,
     $                  ALPHA, BETA, VL, LDVL, VR, LDVR, RES)
      IMPLICIT NONE
      LOGICAL COMP_L, COMP_R
      INTEGER N, LDS, LDP, LDVL, LDVR
      COMPLEX*16 S(LDS,*), P(LDP,*), ALPHA(*), BETA(*)
      COMPLEX*16 VL(LDVL,*), VR(LDVR,*)
      DOUBLE PRECISION RES(2)

      DOUBLE PRECISION NORMS, NORMP, CSS, CSP, NORM_R, NORM_V, DEN
      DOUBLE PRECISION ULP, SAFMIN, MAX_ERR_R, MAX_ERR_L
      COMPLEX*16 A, B, VAL
      INTEGER C, R, I, J

      DOUBLE PRECISION DLAMCH
      EXTERNAL DLAMCH

      ULP = DLAMCH('E')
      SAFMIN = DLAMCH('S')

      NORMS = 0.0D0
      NORMP = 0.0D0
      DO C = 1, N
         CSS = 0.0D0
         CSP = 0.0D0
         DO R = 1, N
            CSS = CSS + ABS(S(R,C))
            CSP = CSP + ABS(P(R,C))
         END DO
         NORMS = MAX(NORMS, CSS)
         NORMP = MAX(NORMP, CSP)
      END DO

      IF (NORMS .EQ. 0.0D0) NORMS = 1.0D0
      IF (NORMP .EQ. 0.0D0) NORMP = 1.0D0

      MAX_ERR_R = 0.0D0
      MAX_ERR_L = 0.0D0

      IF (COMP_R) THEN
         DO C = 1, N
            A = ALPHA(C)
            B = BETA(C)
            NORM_R = 0.0D0
            NORM_V = 0.0D0

            DO I = 1, N
               VAL = DCMPLX(0.0D0, 0.0D0)
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
            NORM_R = 0.0D0
            NORM_V = 0.0D0

            DO J = 1, N
               VAL = DCMPLX(0.0D0, 0.0D0)
               DO I = 1, N
                  VAL = VAL + DCONJG(VL(I,C)) *
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
      END SUBROUTINE ZGET52

      SUBROUTINE GENERATE_GENERALIZED_UPPER_TRIANGULAR(N, S, LDS, P,
     $   LDP, ALPHA, BETA)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP
      COMPLEX*16, INTENT(OUT) :: S(LDS,N), P(LDP,N)
      COMPLEX*16, INTENT(OUT) :: ALPHA(N), BETA(N)
      INTEGER C, R
      DOUBLE PRECISION RAND_R, RAND_I

      DO C = 1, N
         DO R = 1, N
            S(R,C) = DCMPLX(0.0D0, 0.0D0)
            P(R,C) = DCMPLX(0.0D0, 0.0D0)
         END DO
      END DO

      DO C = 1, N
         DO R = 1, C - 1
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            S(R,C) = DCMPLX(RAND_R * 2.0D0 - 1.0D0,
     $                      RAND_I * 2.0D0 - 1.0D0)
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            P(R,C) = DCMPLX(RAND_R * 2.0D0 - 1.0D0,
     $                      RAND_I * 2.0D0 - 1.0D0)
         END DO
         CALL RANDOM_NUMBER(RAND_R)
         CALL RANDOM_NUMBER(RAND_I)
         S(C,C) = DCMPLX(RAND_R * 2.0D0 - 1.0D0, RAND_I * 2.0D0 - 1.0D0)

         CALL RANDOM_NUMBER(RAND_R)
         P(C,C) = DCMPLX(ABS(RAND_R * 2.0D0 - 1.0D0) + 0.1D0, 0.0D0)

         ALPHA(C) = S(C,C)
         BETA(C) = P(C,C)
      END DO
      END SUBROUTINE GENERATE_GENERALIZED_UPPER_TRIANGULAR

      SUBROUTINE GENERATE_LAPACK_MATRIX_TYPE(TYPE, N, S, LDS, P, LDP,
     $                                       ALPHA, BETA)
      IMPLICIT NONE
      INTEGER TYPE, N, LDS, LDP
      COMPLEX*16 S(LDS,*), P(LDP,*)
      COMPLEX*16 ALPHA(*), BETA(*)

      INTEGER I, J
      DOUBLE PRECISION BIG, SMALL, RAND_R, RAND_I

      BIG = 1.0D10
      SMALL = 1.0D-10

      DO J = 1, N
         DO I = 1, N
            S(I,J) = DCMPLX(0.0D0, 0.0D0)
            P(I,J) = DCMPLX(0.0D0, 0.0D0)
         END DO
      END DO

      SELECT CASE (TYPE)
      CASE (1)
         ! Zero matrix
      CASE (2)
         DO I = 1, N
            S(I,I) = DCMPLX(1.0D0, 0.0D0)
         END DO
      CASE (3)
         DO I = 1, N
            P(I,I) = DCMPLX(1.0D0, 0.0D0)
         END DO
      CASE (4)
         DO I = 1, N
            S(I,I) = DCMPLX(1.0D0, 0.0D0)
            P(I,I) = DCMPLX(1.0D0, 0.0D0)
         END DO
      CASE (5)
         DO I = 1, N
            S(I,I) = DCMPLX(1.0D0, 0.0D0)
            P(I,I) = DCMPLX(1.0D0, 0.0D0)
            IF (I .LT. N) THEN
               S(I,I+1) = DCMPLX(1.0D0, 0.0D0)
               P(I,I+1) = DCMPLX(1.0D0, 0.0D0)
            END IF
         END DO
      CASE (6)
         DO I = 1, N
            S(I,I) = DCMPLX(DBLE(I) / DBLE(N), 0.0D0)
            P(I,I) = DCMPLX(1.0D0, 0.0D0)
         END DO
      CASE (7)
         DO I = 1, N
            S(I,I) = DCMPLX(1.0D0, 0.0D0)
            P(I,I) = DCMPLX(DBLE(I) / DBLE(N), 0.0D0)
         END DO
      CASE (8)
         DO I = 1, N
            S(I,I) = DCMPLX(DBLE(I) / DBLE(N), 0.0D0)
            P(I,I) = DCMPLX(DBLE(N - I + 1) / DBLE(N), 0.0D0)
         END DO
      CASE (9)
         DO I = 1, N
            S(I,I) = DCMPLX(BIG * DBLE(I) / DBLE(N), 0.0D0)
            P(I,I) = DCMPLX(SMALL, 0.0D0)
         END DO
      CASE (10)
         DO I = 1, N
            S(I,I) = DCMPLX(SMALL * DBLE(I) / DBLE(N), 0.0D0)
            P(I,I) = DCMPLX(BIG, 0.0D0)
         END DO
      CASE (11)
         DO I = 1, N
            S(I,I) = DCMPLX(BIG, 0.0D0)
            P(I,I) = DCMPLX(SMALL * DBLE(I) / DBLE(N), 0.0D0)
         END DO
      CASE (12)
         DO I = 1, N
            S(I,I) = DCMPLX(SMALL, 0.0D0)
            P(I,I) = DCMPLX(BIG * DBLE(I) / DBLE(N), 0.0D0)
         END DO
      CASE (13)
         DO I = 1, N
            S(I,I) = DCMPLX(BIG * DBLE(I) / DBLE(N), 0.0D0)
            P(I,I) = DCMPLX(BIG, 0.0D0)
         END DO
      CASE (14)
         DO I = 1, N
            S(I,I) = DCMPLX(SMALL * DBLE(I) / DBLE(N), 0.0D0)
            P(I,I) = DCMPLX(SMALL, 0.0D0)
         END DO
      CASE (15)
         DO I = 1, N
            IF (I .EQ. 1 .OR. I .EQ. 2 .OR. I .EQ. N) THEN
               S(I,I) = DCMPLX(0.0D0, 0.0D0)
            ELSE
               S(I,I) = DCMPLX(DBLE(I - 2), 0.0D0)
            END IF

            IF (I .EQ. 1 .OR. I .EQ. N-1 .OR. I .EQ. N) THEN
               P(I,I) = DCMPLX(0.0D0, 0.0D0)
            ELSE
               P(I,I) = DCMPLX(DBLE(N - I), 0.0D0)
            END IF
         END DO
      CASE DEFAULT
         DO J = 1, N
            DO I = 1, J - 1
               CALL RANDOM_NUMBER(RAND_R)
               CALL RANDOM_NUMBER(RAND_I)
               S(I,J) = DCMPLX(RAND_R * 2.0D0 - 1.0D0,
     $                         RAND_I * 2.0D0 - 1.0D0)
               CALL RANDOM_NUMBER(RAND_R)
               CALL RANDOM_NUMBER(RAND_I)
               P(I,J) = DCMPLX(RAND_R * 2.0D0 - 1.0D0,
     $                         RAND_I * 2.0D0 - 1.0D0)
            END DO
            CALL RANDOM_NUMBER(RAND_R)
            CALL RANDOM_NUMBER(RAND_I)
            S(J,J) = DCMPLX(RAND_R * 2.0D0 - 1.0D0,
     $                      RAND_I * 2.0D0 - 1.0D0)

            CALL RANDOM_NUMBER(RAND_R)
            P(J,J) = DCMPLX(ABS(RAND_R * 2.0D0 - 1.0D0) + 0.1D0, 0.0D0)
         END DO
      END SELECT

      DO I = 1, N
         ALPHA(I) = S(I,I)
         BETA(I) = P(I,I)
      END DO
      RETURN
      END SUBROUTINE GENERATE_LAPACK_MATRIX_TYPE

      END PROGRAM TEST_ZTGEVC3