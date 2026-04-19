#pragma once

#include <complex>

#ifndef RP
#define RP(T) T *__restrict
#endif

extern "C" {
    void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, const int *ldc);
    void dgemv_(const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy);
    double dnrm2_(const int *n, const double *x, const int *incx);
    void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const double *alpha, const double *a, const int *lda, double *b, const int *ldb);
    void dtgevc_(const char *side, const char *howmny, const int *select, const int *n, const double *s, const int *lds, const double *p, const int *ldp, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, int *info);
    void dgghd3_(const char *compq, const char *compz, const int *n, const int *ilo, const int *ihi, RP(double) A, const int *ldA, RP(double) B, const int *ldB, RP(double) Q, const int *ldQ, RP(double) Z, const int *ldZ, RP(double) work, const int *lwork, int *info);
    void dtrevc3_(const char *side, const char *howmny, const int *select, const int *n, double *t, const int *ldt, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, const int *lwork, int *info);

    void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha, const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c, const int *ldc);
    void sgemv_(const char *trans, const int *m, const int *n, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy);
    float snrm2_(const int *n, const float *x, const int *incx);
    void strsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const float *alpha, const float *a, const int *lda, float *b, const int *ldb);
    void stgevc_(const char *side, const char *howmny, const int *select, const int *n, const float *s, const int *lds, const float *p, const int *ldp, float *vl, const int *ldvl, float *vr, const int *ldvr, const int *mm, int *m, float *work, int *info);

    void zgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const std::complex<double> *alpha, const std::complex<double> *a, const int *lda, const std::complex<double> *b, const int *ldb, const std::complex<double> *beta, std::complex<double> *c, const int *ldc);
    void ztrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const std::complex<double> *alpha, const std::complex<double> *a, const int *lda, std::complex<double> *b, const int *ldb);
    void zgemv_(const char *trans, const int *m, const int *n, const std::complex<double> *alpha, const std::complex<double> *a, const int *lda, const std::complex<double> *x, const int *incx, const std::complex<double> *beta, std::complex<double> *y, const int *incy);
    double dznrm2_(const int *n, const std::complex<double> *x, const int *incx);

    void cgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const std::complex<float> *alpha, const std::complex<float> *a, const int *lda, const std::complex<float> *b, const int *ldb, const std::complex<float> *beta, std::complex<float> *c, const int *ldc);
    void ctrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const std::complex<float> *alpha, const std::complex<float> *a, const int *lda, std::complex<float> *b, const int *ldb);
    void cgemv_(const char *trans, const int *m, const int *n, const std::complex<float> *alpha, const std::complex<float> *a, const int *lda, const std::complex<float> *x, const int *incx, const std::complex<float> *beta, std::complex<float> *y, const int *incy);
    float scnrm2_(const int *n, const std::complex<float> *x, const int *incx);
}
