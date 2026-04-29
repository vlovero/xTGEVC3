      INTEGER FUNCTION ISLAPB(S, LDS, CURR, BSIZE)
      IMPLICIT NONE
      INTEGER LDS, CURR, BSIZE
      REAL S(LDS, *)
      INTEGER IDX

      IDX = MAX(1, CURR - BSIZE + 1)

*     Check if the proposed boundary splits a 2x2 block.
*     If the subdiagonal element is non-zero, decrement IDX
*     to include the rest of the 2x2 block.
      IF (IDX .GT. 1) THEN
         IF (S(IDX, IDX - 1) .NE. 0.0E0) THEN
            IDX = IDX - 1
         END IF
      END IF

      ISLAPB = IDX
      RETURN
      END
