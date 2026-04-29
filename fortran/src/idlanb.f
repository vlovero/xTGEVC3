      INTEGER FUNCTION IDLANB(S, N, LDS, CURR, BSIZE)
      IMPLICIT NONE
      INTEGER N, LDS, CURR, BSIZE
      DOUBLE PRECISION S(LDS, *)
      INTEGER IDX

      IDX = MIN(N, CURR + BSIZE - 1)

*     Check if the boundary splits a 2x2 block.
*     If the subdiagonal element at the boundary is non-zero, increment
*     the index to include the rest of the 2x2 block.
      IF (IDX .LT. N) THEN
         IF (S(IDX + 1, IDX) .NE. 0.0D0) THEN
            IDX = IDX + 1
         END IF
      END IF

      IDLANB = IDX
      RETURN
      END
