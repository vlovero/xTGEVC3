#pragma once

#include <complex>

#ifndef RP
#define RP(T) T *__restrict
#endif


void dtgevc3(char side, char howmny, const int *select, int n, const double *S, int lds, const double *P, int ldp, const double *alphar, const double *alphai, const double *beta, RP(double) VL, int ldvl, RP(double) VR, int ldvr, int mm, int *m, RP(double) work, int lwork, int *info);
void stgevc3(char side, char howmny, const int *select, int n, const float *S, int lds, const float *P, int ldp, const float *alphar, const float *alphai, const float *beta, RP(float) VL, int ldvl, RP(float) VR, int ldvr, int mm, int *m, RP(float) work, int lwork, int *info);
void ztgevc3(char side, char howmny, const int *select, int n, const std::complex<double> *S, int lds, const std::complex<double> *P, int ldp, const std::complex<double> *alpha, const std::complex<double> *beta, RP(std::complex<double>) VL, int ldvl, RP(std::complex<double>) VR, int ldvr, int mm, int *m, RP(std::complex<double>) work, int lwork, int *info);
void ctgevc3(char side, char howmny, const int *select, int n, const std::complex<float> *S, int lds, const std::complex<float> *P, int ldp, const std::complex<float> *alpha, const std::complex<float> *beta, RP(std::complex<float>) VL, int ldvl, RP(std::complex<float>) VR, int ldvr, int mm, int *m, RP(std::complex<float>) work, int lwork, int *info);