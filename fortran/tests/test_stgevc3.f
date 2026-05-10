      PROGRAM TEST_STGEVC3
      IMPLICIT NONE

      CALL TEST_STGEVC()
      CALL TEST_STGEVC_INFINITE()
      CALL TEST_STGEVC_SCALING()
      CALL TEST_DYNAMIC_SIZE(10)
      CALL TEST_DYNAMIC_SIZE(500)
      CALL TEST_LAPACK_TYPES(10)
      CALL TEST_LAPACK_TYPES(50)

      CONTAINS

      SUBROUTINE TEST_STGEVC()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      REAL, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL S(N,N), P(N,N), ALPHAR(N), ALPHAI(N), BETA(N)
      REAL VR(N,N), VL(N,N), RES3(2), RESL(2)
      LOGICAL SEL(1)

      PRINT *, '--- Testing 4x4 Static Matrix (Both Left/Right) ---'

      S(1,1) = 1.0E0; S(1,2) = 2.0E0; S(1,3) = 3.0E0; S(1,4) = 4.0E0
      S(2,1) = 0.0E0; S(2,2) = 2.0E0; S(2,3) = 5.0E0; S(2,4) = -1.0E0
      S(3,1) = 0.0E0; S(3,2) = -5.0E0; S(3,3) = 2.0E0; S(3,4) = 2.0E0
      S(4,1) = 0.0E0; S(4,2) = 0.0E0; S(4,3) = 0.0E0; S(4,4) = 3.0E0

      P(1,1) = 2.0E0; P(1,2) = 1.0E0; P(1,3) = -1.0E0; P(1,4) = 3.0E0
      P(2,1) = 0.0E0; P(2,2) = 1.0E0; P(2,3) = 0.0E0; P(2,4) = 2.0E0
      P(3,1) = 0.0E0; P(3,2) = 0.0E0; P(3,3) = 1.0E0; P(3,4) = -1.0E0
      P(4,1) = 0.0E0; P(4,2) = 0.0E0; P(4,3) = 0.0E0; P(4,4) = 2.0E0

      ALPHAR(1) = 1.0E0; ALPHAR(2) = 2.0E0
      ALPHAR(3) = 2.0E0; ALPHAR(4) = 3.0E0
      ALPHAI(1) = 0.0E0; ALPHAI(2) = 5.0E0
      ALPHAI(3) = -5.0E0; ALPHAI(4) = 0.0E0
      BETA(1) = 2.0E0; BETA(2) = 1.0E0
      BETA(3) = 1.0E0; BETA(4) = 2.0E0

      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 +
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      ALLOCATE(WORK_LAPACK(6*N))
      SEL(1) = .FALSE.

      ! Test stgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
         END DO
         VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
      END DO
      CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $             BETA, VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $            BETA, VL, N, VR, N, RES3)

      ! Test LAPACK stgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
         END DO
         VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
      END DO
      CALL STGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, INFO)
      CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $            BETA, VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PE14.6)') 
     $   '[stgevc3] sget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[stgevc3] sget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] sget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] sget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(WORK, WORK_LAPACK)
      END SUBROUTINE TEST_STGEVC

      SUBROUTINE TEST_STGEVC_INFINITE()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT
      REAL, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL S(N,N), P(N,N), ALPHAR(N), ALPHAI(N), BETA(N)
      REAL VR(N,N), VL(N,N), RES3(2), RESL(2)
      LOGICAL SEL(1)

      PRINT *, '--- Testing 4x4 Static Matrix (Infinite Eval) ---'

      S(1,1) = 1.0E0; S(1,2) = 2.0E0; S(1,3) = 3.0E0; S(1,4) = 4.0E0
      S(2,1) = 0.0E0; S(2,2) = 2.0E0; S(2,3) = 5.0E0; S(2,4) = -1.0E0
      S(3,1) = 0.0E0; S(3,2) = -5.0E0; S(3,3) = 2.0E0; S(3,4) = 2.0E0
      S(4,1) = 0.0E0; S(4,2) = 0.0E0; S(4,3) = 0.0E0; S(4,4) = 3.0E0

      P(1,1) = 2.0E0; P(1,2) = 1.0E0; P(1,3) = -1.0E0; P(1,4) = 3.0E0
      P(2,1) = 0.0E0; P(2,2) = 1.0E0; P(2,3) = 0.0E0; P(2,4) = 2.0E0
      P(3,1) = 0.0E0; P(3,2) = 0.0E0; P(3,3) = 1.0E0; P(3,4) = -1.0E0
      P(4,1) = 0.0E0; P(4,2) = 0.0E0; P(4,3) = 0.0E0; P(4,4) = 0.0E0

      ALPHAR(1) = 1.0E0; ALPHAR(2) = 2.0E0
      ALPHAR(3) = 2.0E0; ALPHAR(4) = 3.0E0
      ALPHAI(1) = 0.0E0; ALPHAI(2) = 5.0E0
      ALPHAI(3) = -5.0E0; ALPHAI(4) = 0.0E0
      BETA(1) = 2.0E0; BETA(2) = 1.0E0
      BETA(3) = 1.0E0; BETA(4) = 0.0E0

      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 +
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      ALLOCATE(WORK_LAPACK(6*N))
      SEL(1) = .FALSE.

      ! Test stgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
         END DO
         VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
      END DO
      CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $             BETA, VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE, INFO)
      CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $            BETA, VL, N, VR, N, RES3)

      ! Test LAPACK stgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
         END DO
         VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
      END DO
      CALL STGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, INFO)
      CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $            BETA, VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PE14.6)') 
     $   '[stgevc3] sget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[stgevc3] sget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] sget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] sget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(WORK, WORK_LAPACK)
      END SUBROUTINE TEST_STGEVC_INFINITE

      SUBROUTINE TEST_STGEVC_SCALING()
      IMPLICIT NONE
      INTEGER, PARAMETER :: N = 4
      INTEGER INFO, MAX_BSIZE, WORK_SIZE, I, J, M_OUT, TEST_IDX
      REAL S_VAL
      REAL, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL S_BASE(N,N), P_BASE(N,N)
      REAL ALPHAR_BASE(N), ALPHAI_BASE(N), BETA_BASE(N)
      REAL S(N,N), P(N,N), ALPHAR(N), ALPHAI(N), BETA(N)
      REAL VR(N,N), VL(N,N), RES3(2), RESL(2)
      REAL SCALES(2)
      LOGICAL SEL(1)

      S_BASE(1,1) = 1.0E0; S_BASE(1,2) = 2.0E0
      S_BASE(1,3) = 3.0E0; S_BASE(1,4) = 4.0E0
      S_BASE(2,1) = 0.0E0; S_BASE(2,2) = 2.0E0
      S_BASE(2,3) = 5.0E0; S_BASE(2,4) = -1.0E0
      S_BASE(3,1) = 0.0E0; S_BASE(3,2) = -5.0E0
      S_BASE(3,3) = 2.0E0; S_BASE(3,4) = 2.0E0
      S_BASE(4,1) = 0.0E0; S_BASE(4,2) = 0.0E0
      S_BASE(4,3) = 0.0E0; S_BASE(4,4) = 3.0E0

      P_BASE(1,1) = 2.0E0; P_BASE(1,2) = 1.0E0
      P_BASE(1,3) = -1.0E0; P_BASE(1,4) = 3.0E0
      P_BASE(2,1) = 0.0E0; P_BASE(2,2) = 1.0E0
      P_BASE(2,3) = 0.0E0; P_BASE(2,4) = 2.0E0
      P_BASE(3,1) = 0.0E0; P_BASE(3,2) = 0.0E0
      P_BASE(3,3) = 1.0E0; P_BASE(3,4) = -1.0E0
      P_BASE(4,1) = 0.0E0; P_BASE(4,2) = 0.0E0
      P_BASE(4,3) = 0.0E0; P_BASE(4,4) = 2.0E0

      ALPHAR_BASE(1) = 1.0E0; ALPHAR_BASE(2) = 2.0E0
      ALPHAR_BASE(3) = 2.0E0; ALPHAR_BASE(4) = 3.0E0
      ALPHAI_BASE(1) = 0.0E0; ALPHAI_BASE(2) = 5.0E0
      ALPHAI_BASE(3) = -5.0E0; ALPHAI_BASE(4) = 0.0E0
      BETA_BASE(1) = 2.0E0; BETA_BASE(2) = 1.0E0
      BETA_BASE(3) = 1.0E0; BETA_BASE(4) = 2.0E0

      SCALES(1) = 1.0E30
      SCALES(2) = 1.0E-30

      MAX_BSIZE = 32
      WORK_SIZE = 2*N*(MAX_BSIZE+1) + 4*(MAX_BSIZE+1)**2 +
     $            2*(MAX_BSIZE+1)
      ALLOCATE(WORK(WORK_SIZE))
      ALLOCATE(WORK_LAPACK(6*N))
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
               S(I,J) = S_BASE(I,J) * S_VAL
               P(I,J) = P_BASE(I,J) * S_VAL
            END DO
         END DO

         DO I = 1, N
            ALPHAR(I) = ALPHAR_BASE(I) * S_VAL
            ALPHAI(I) = ALPHAI_BASE(I) * S_VAL
            BETA(I) = BETA_BASE(I) * S_VAL
         END DO

         ! Test stgevc3
         DO J = 1, N
            DO I = 1, N
               VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
            END DO
            VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
         END DO
         CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $                BETA, VL, N, VR, N, N, M_OUT, WORK, WORK_SIZE,
     $                INFO)
         CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $               BETA, VL, N, VR, N, RES3)

         ! Test LAPACK stgevc
         DO J = 1, N
            DO I = 1, N
               VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
            END DO
            VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
         END DO
         CALL STGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $               M_OUT, WORK_LAPACK, INFO)
         CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $               BETA, VL, N, VR, N, RESL)

         WRITE(*, '(A, 1PE14.6)') 
     $      '[stgevc3] sget52 Max Right Error: ', RES3(1)
         WRITE(*, '(A, 1PE14.6)') 
     $      '[stgevc3] sget52 Max Left Error:  ', RES3(2)
         WRITE(*, '(A, 1PE14.6)') 
     $      '[lapack ] sget52 Max Right Error: ', RESL(1)
         WRITE(*, '(A, 1PE14.6)') 
     $      '[lapack ] sget52 Max Left Error:  ', RESL(2)
         PRINT *
      END DO

      DEALLOCATE(WORK, WORK_LAPACK)
      END SUBROUTINE TEST_STGEVC_SCALING

      SUBROUTINE TEST_DYNAMIC_SIZE(N)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      REAL, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      REAL, ALLOCATABLE :: ALPHAR(:), ALPHAI(:), BETA(:)
      REAL, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL DUMMY(1), RES3(2), RESL(2), T1, T2
      INTEGER INFO, LWORK, I, J, M_OUT
      LOGICAL SEL(1)

      PRINT *, '--- Testing Dynamic Size Matrix ---', N

      ALLOCATE(S(N,N), P(N,N), VR(N,N), VL(N,N))
      ALLOCATE(ALPHAR(N), ALPHAI(N), BETA(N))
      ALLOCATE(WORK_LAPACK(6*N))
      SEL(1) = .FALSE.

      CALL GENERATE_GENERALIZED_QUASI_TRIANGULAR(N, S, N, P, N,
     $     ALPHAR, ALPHAI, BETA)

      CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $             BETA, VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)

      LWORK = INT(DUMMY(1))
      IF (LWORK .LE. 0) THEN
         LWORK = 2*N*(32+1) + 4*(32+1)**2 + 2*(32+1)
      END IF
      ALLOCATE(WORK(LWORK))

      ! Test stgevc3
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
         END DO
         VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
      END DO
      CALL CPU_TIME(T1)
      CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $             BETA, VL, N, VR, N, N, M_OUT, WORK, LWORK, INFO)
      CALL CPU_TIME(T2)
      PRINT *, 'stgevc3 (''B'') took ', T2-T1, ' seconds'
      CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $            BETA, VL, N, VR, N, RES3)

      ! Test LAPACK stgevc
      DO J = 1, N
         DO I = 1, N
            VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
         END DO
         VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
      END DO
      CALL CPU_TIME(T1)
      CALL STGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $            M_OUT, WORK_LAPACK, INFO)
      CALL CPU_TIME(T2)
      PRINT *, 'LAPACK stgevc (''B'') took ', T2-T1, ' seconds'
      CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $            BETA, VL, N, VR, N, RESL)

      WRITE(*, '(A, 1PE14.6)') 
     $   '[stgevc3] sget52 Max Right Error: ', RES3(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[stgevc3] sget52 Max Left Error:  ', RES3(2)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] sget52 Max Right Error: ', RESL(1)
      WRITE(*, '(A, 1PE14.6)') 
     $   '[lapack ] sget52 Max Left Error:  ', RESL(2)
      PRINT *

      DEALLOCATE(S, P, VR, VL, ALPHAR, ALPHAI, BETA)
      DEALLOCATE(WORK, WORK_LAPACK)
      END SUBROUTINE TEST_DYNAMIC_SIZE

      SUBROUTINE TEST_LAPACK_TYPES(N)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      REAL, ALLOCATABLE :: S(:,:), P(:,:), VR(:,:), VL(:,:)
      REAL, ALLOCATABLE :: ALPHAR(:), ALPHAI(:), BETA(:)
      REAL, ALLOCATABLE :: WORK(:), WORK_LAPACK(:)
      REAL DUMMY(1), RES3(2), RESL(2)
      INTEGER INFO, LWORK, I, J, TYPE, M_OUT
      LOGICAL SEL(1)

      PRINT *, '--- Testing ', N, 'x', N, ' LAPACK Matrix Types ---'

      ALLOCATE(S(N,N), P(N,N), VR(N,N), VL(N,N))
      ALLOCATE(ALPHAR(N), ALPHAI(N), BETA(N))
      ALLOCATE(WORK_LAPACK(6*N))
      SEL(1) = .FALSE.

      DO TYPE = 1, 26
         CALL GENERATE_LAPACK_MATRIX_TYPE(TYPE, N, S, N, P, N,
     $        ALPHAR, ALPHAI, BETA)

         CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $                BETA, VL, N, VR, N, N, M_OUT, DUMMY, -1, INFO)

         LWORK = INT(DUMMY(1))
         IF (LWORK .LE. 0) THEN
            LWORK = 2*N*(32+1) + 4*(32+1)**2 + 2*(32+1)
         END IF
         ALLOCATE(WORK(LWORK))

         DO J = 1, N
            DO I = 1, N
               VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
            END DO
            VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
         END DO
         CALL STGEVC3('B', 'B', SEL, N, S, N, P, N, ALPHAR, ALPHAI,
     $                BETA, VL, N, VR, N, N, M_OUT, WORK, LWORK, INFO)
         CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $               BETA, VL, N, VR, N, RES3)

         DO J = 1, N
            DO I = 1, N
               VR(I,J) = 0.0E0; VL(I,J) = 0.0E0
            END DO
            VR(J,J) = 1.0E0; VL(J,J) = 1.0E0
         END DO
         CALL STGEVC('B', 'B', SEL, N, S, N, P, N, VL, N, VR, N, N,
     $               M_OUT, WORK_LAPACK, INFO)
         CALL SGET52(.TRUE., .TRUE., N, S, N, P, N, ALPHAR, ALPHAI,
     $               BETA, VL, N, VR, N, RESL)

         WRITE(*, '(A, I2, A)') 'Type ', TYPE, ':'
         WRITE(*, '(A, 1PE14.6, A, 1PE14.6)') 
     $      '  [stgevc3] Max R = ', RES3(1), ' L = ', RES3(2)
         WRITE(*, '(A, 1PE14.6, A, 1PE14.6)') 
     $      '  [lapack ] Max R = ', RESL(1), ' L = ', RESL(2)

         DEALLOCATE(WORK)
      END DO
      PRINT *

      DEALLOCATE(S, P, VR, VL, ALPHAR, ALPHAI, BETA, WORK_LAPACK)
      END SUBROUTINE TEST_LAPACK_TYPES

      SUBROUTINE SGET52(COMP_L, COMP_R, N, S, LDS, P, LDP,
     $                  ALPHAR, ALPHAI, BETA, VL, LDVL, VR, LDVR, RES)
      IMPLICIT NONE
      LOGICAL COMP_L, COMP_R
      INTEGER N, LDS, LDP, LDVL, LDVR
      REAL S(LDS,*), P(LDP,*), ALPHAR(*), ALPHAI(*), BETA(*)
      REAL VL(LDVL,*), VR(LDVR,*), RES(2)
      REAL NORMS, NORMP, CSS, CSP, A, B, AR, AI, NORM_R, NORM_V
      REAL VAL, RR_VAL, RI_VAL, V_R, V_I, S_VAL, P_VAL, DEN, A_NORM
      REAL ULP, SAFMIN, MAX_ERR_R, MAX_ERR_L
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
         C = 1
         DO WHILE (C .LE. N)
            IF (ALPHAI(C) .EQ. 0.0E0) THEN
               A = ALPHAR(C)
               B = BETA(C)
               NORM_R = 0.0E0
               NORM_V = 0.0E0

               DO I = 1, N
                  VAL = 0.0E0
                  DO J = 1, N
                     VAL = VAL + B * S(I,J) * VR(J,C) -
     $                           A * P(I,J) * VR(J,C)
                  END DO
                  NORM_R = NORM_R + ABS(VAL)
                  NORM_V = NORM_V + ABS(VR(I,C))
               END DO

               DEN = (ABS(B) * NORMS + ABS(A) * NORMP) * NORM_V
               DEN = MAX(DEN, SAFMIN)
               MAX_ERR_R = MAX(MAX_ERR_R, NORM_R / (DEN * ULP))
               C = C + 1
            ELSE
               AR = ALPHAR(C)
               AI = ALPHAI(C)
               B = BETA(C)
               NORM_R = 0.0E0
               NORM_V = 0.0E0

               DO I = 1, N
                  RR_VAL = 0.0E0
                  RI_VAL = 0.0E0
                  DO J = 1, N
                     V_R = VR(J,C)
                     V_I = VR(J,C+1)
                     S_VAL = S(I,J)
                     P_VAL = P(I,J)

                     RR_VAL = RR_VAL + B * S_VAL * V_R -
     $                        AR * P_VAL * V_R + AI * P_VAL * V_I
                     RI_VAL = RI_VAL + B * S_VAL * V_I -
     $                        AR * P_VAL * V_I - AI * P_VAL * V_R
                  END DO
                  NORM_R = NORM_R + ABS(RR_VAL) + ABS(RI_VAL)
                  NORM_V = NORM_V + ABS(VR(I,C)) + ABS(VR(I,C+1))
               END DO

               A_NORM = ABS(AR) + ABS(AI)
               DEN = (ABS(B) * NORMS + A_NORM * NORMP) * NORM_V
               DEN = MAX(DEN, SAFMIN)
               MAX_ERR_R = MAX(MAX_ERR_R, NORM_R / (DEN * ULP))
               C = C + 2
            END IF
         END DO
      END IF

      IF (COMP_L) THEN
         C = 1
         DO WHILE (C .LE. N)
            IF (ALPHAI(C) .EQ. 0.0E0) THEN
               A = ALPHAR(C)
               B = BETA(C)
               NORM_R = 0.0E0
               NORM_V = 0.0E0

               DO J = 1, N
                  VAL = 0.0E0
                  DO I = 1, N
                     VAL = VAL + B * S(I,J) * VL(I,C) -
     $                           A * P(I,J) * VL(I,C)
                  END DO
                  NORM_R = NORM_R + ABS(VAL)
               END DO
               DO I = 1, N
                  NORM_V = NORM_V + ABS(VL(I,C))
               END DO

               DEN = (ABS(B) * NORMS + ABS(A) * NORMP) * NORM_V
               DEN = MAX(DEN, SAFMIN)
               MAX_ERR_L = MAX(MAX_ERR_L, NORM_R / (DEN * ULP))
               C = C + 1
            ELSE
               AR = ALPHAR(C)
               AI = ALPHAI(C)
               B = BETA(C)
               NORM_R = 0.0E0
               NORM_V = 0.0E0

               DO J = 1, N
                  RR_VAL = 0.0E0
                  RI_VAL = 0.0E0
                  DO I = 1, N
                     V_R = VL(I,C)
                     V_I = VL(I,C+1)
                     S_VAL = S(I,J)
                     P_VAL = P(I,J)

                     RR_VAL = RR_VAL + B * S_VAL * V_R -
     $                        AR * P_VAL * V_R - AI * P_VAL * V_I
                     RI_VAL = RI_VAL + B * S_VAL * V_I -
     $                        AR * P_VAL * V_I + AI * P_VAL * V_R
                  END DO
                  NORM_R = NORM_R + ABS(RR_VAL) + ABS(RI_VAL)
               END DO
               DO I = 1, N
                  NORM_V = NORM_V + ABS(VL(I,C)) + ABS(VL(I,C+1))
               END DO

               A_NORM = ABS(AR) + ABS(AI)
               DEN = (ABS(B) * NORMS + A_NORM * NORMP) * NORM_V
               DEN = MAX(DEN, SAFMIN)
               MAX_ERR_L = MAX(MAX_ERR_L, NORM_R / (DEN * ULP))
               C = C + 2
            END IF
         END DO
      END IF

      RES(1) = MAX_ERR_R
      RES(2) = MAX_ERR_L
      RETURN
      END SUBROUTINE SGET52

      SUBROUTINE GENERATE_GENERALIZED_QUASI_TRIANGULAR(N, S, LDS, P,
     $   LDP, ALPHAR, ALPHAI, BETA)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N, LDS, LDP
      REAL, INTENT(OUT) :: S(LDS,N), P(LDP,N)
      REAL, INTENT(OUT) :: ALPHAR(N), ALPHAI(N), BETA(N)
      INTEGER C, R, K
      REAL A, B, RAND_VAL

      DO C = 1, N
         DO R = 1, N
            S(R,C) = 0.0E0
            P(R,C) = 0.0E0
         END DO
      END DO

      DO C = 1, N
         DO R = 1, C
            CALL RANDOM_NUMBER(RAND_VAL)
            S(R,C) = RAND_VAL * 2.0E0 - 1.0E0
            CALL RANDOM_NUMBER(RAND_VAL)
            P(R,C) = RAND_VAL * 2.0E0 - 1.0E0
         END DO
         IF (P(C,C) .GE. 0.0E0) THEN
            P(C,C) = P(C,C) + 1.0E0
         ELSE
            P(C,C) = P(C,C) - 1.0E0
         END IF
      END DO

      K = 1
      DO WHILE (K .LE. N)
         CALL RANDOM_NUMBER(RAND_VAL)
         IF (K .LT. N .AND. RAND_VAL .LT. 0.4E0) THEN
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

      SUBROUTINE GENERATE_LAPACK_MATRIX_TYPE(TYPE, N, S, LDS, P, LDP,
     $                                       ALPHAR, ALPHAI, BETA)
      IMPLICIT NONE
      INTEGER TYPE, N, LDS, LDP
      REAL S(LDS,*), P(LDP,*)
      REAL ALPHAR(*), ALPHAI(*), BETA(*)

      INTEGER I, J, K
      REAL BIG, SMALL, A, B, P_VAL, RAND_VAL, RAND_VAL2

      BIG = 1.0E10
      SMALL = 1.0E-10

      DO J = 1, N
         DO I = 1, N
            S(I,J) = 0.0E0
            P(I,J) = 0.0E0
         END DO
         ALPHAI(J) = 0.0E0
      END DO

      SELECT CASE (TYPE)
      CASE (1)
         ! Zero matrix
      CASE (2)
         DO I = 1, N
            S(I,I) = 1.0E0
         END DO
      CASE (3)
         DO I = 1, N
            P(I,I) = 1.0E0
         END DO
      CASE (4)
         DO I = 1, N
            S(I,I) = 1.0E0
            P(I,I) = 1.0E0
         END DO
      CASE (5)
         DO I = 1, N
            S(I,I) = 1.0E0
            P(I,I) = 1.0E0
            IF (I .LT. N) THEN
               S(I,I+1) = 1.0E0
               P(I,I+1) = 1.0E0
            END IF
         END DO
      CASE (6)
         DO I = 1, N
            S(I,I) = REAL(I) / REAL(N)
            P(I,I) = 1.0E0
         END DO
      CASE (7)
         DO I = 1, N
            S(I,I) = 1.0E0
            P(I,I) = REAL(I) / REAL(N)
         END DO
      CASE (8)
         DO I = 1, N
            S(I,I) = REAL(I) / REAL(N)
            P(I,I) = REAL(N - I + 1) / REAL(N)
         END DO
      CASE (9)
         DO I = 1, N
            S(I,I) = BIG * REAL(I) / REAL(N)
            P(I,I) = SMALL
         END DO
      CASE (10)
         DO I = 1, N
            S(I,I) = SMALL * REAL(I) / REAL(N)
            P(I,I) = BIG
         END DO
      CASE (11)
         DO I = 1, N
            S(I,I) = BIG
            P(I,I) = SMALL * REAL(I) / REAL(N)
         END DO
      CASE (12)
         DO I = 1, N
            S(I,I) = SMALL
            P(I,I) = BIG * REAL(I) / REAL(N)
         END DO
      CASE (13)
         DO I = 1, N
            S(I,I) = BIG * REAL(I) / REAL(N)
            P(I,I) = BIG
         END DO
      CASE (14)
         DO I = 1, N
            S(I,I) = SMALL * REAL(I) / REAL(N)
            P(I,I) = SMALL
         END DO
      CASE (15)
         DO I = 1, N
            IF (I .EQ. 1 .OR. I .EQ. 2 .OR. I .EQ. N) THEN
               S(I,I) = 0.0E0
            ELSE
               S(I,I) = REAL(I - 2)
            END IF

            IF (I .EQ. 1 .OR. I .EQ. N-1 .OR. I .EQ. N) THEN
               P(I,I) = 0.0E0
            ELSE
               P(I,I) = REAL(N - I)
            END IF
         END DO
      CASE DEFAULT
         DO J = 1, N
            DO I = 1, J
               CALL RANDOM_NUMBER(RAND_VAL)
               S(I,J) = RAND_VAL * 2.0E0 - 1.0E0
               CALL RANDOM_NUMBER(RAND_VAL)
               P(I,J) = RAND_VAL * 2.0E0 - 1.0E0
            END DO
         END DO

         IF (MOD(TYPE, 2) .EQ. 0) THEN
            I = 1
            DO WHILE (I .LT. N)
               CALL RANDOM_NUMBER(RAND_VAL)
               IF (RAND_VAL .GT. 0.5E0) THEN
                  CALL RANDOM_NUMBER(A)
                  A = A * 2.0E0 - 1.0E0
                  CALL RANDOM_NUMBER(B)
                  B = B * 2.0E0 - 1.0E0
                  IF (B .EQ. 0.0E0) B = 1.0E0
                  CALL RANDOM_NUMBER(RAND_VAL2)
                  P_VAL = ABS(RAND_VAL2 * 2.0E0 - 1.0E0) + 0.1E0

                  S(I,I) = A
                  S(I+1,I+1) = A
                  S(I,I+1) = B
                  S(I+1,I) = -B

                  P(I,I) = P_VAL
                  P(I+1,I+1) = P_VAL
                  P(I,I+1) = 0.0E0
                  P(I+1,I) = 0.0E0
               END IF
               I = I + 2
            END DO
         END IF
      END SELECT

      K = 1
      DO WHILE (K .LE. N)
         IF (K .LT. N) THEN
            IF (S(K+1,K) .NE. 0.0E0) THEN
               ALPHAR(K) = S(K,K)
               ALPHAR(K+1) = S(K+1,K+1)
               ALPHAI(K) = ABS(S(K,K+1))
               ALPHAI(K+1) = -ALPHAI(K)
               BETA(K) = P(K,K)
               BETA(K+1) = P(K+1,K+1)
               K = K + 2
               CYCLE
            END IF
         END IF
         ALPHAR(K) = S(K,K)
         ALPHAI(K) = 0.0E0
         BETA(K) = P(K,K)
         K = K + 1
      END DO
      RETURN
      END SUBROUTINE GENERATE_LAPACK_MATRIX_TYPE

      END PROGRAM TEST_STGEVC3