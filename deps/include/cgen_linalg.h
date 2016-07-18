/*
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <math.h>

void cgen_cblas_dgemm(bool tA, bool tB, int m, int n, int k, double* A, int lda, double* B, int ldb, double* C, int ldc)
{

    // printf("tA %d tB %d m %d n %d k %d lda %d ldb %d ldc %d\n",tA,tB,m,n,k,lda,ldb,ldc);

#define Amat(I,J) A[(I) + (J)*(lda)]
#define Bmat(I,J) B[(I) + (J)*(ldb)]
#define Cmat(I,J) C[(I) + (J)*(ldc)]

    int i,j,l;
    double tmp;

    if (!tA) {
        if(!tB){
            for(i=0; i<m; i++) {
                for(j=0; j<n; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(i,l)* Bmat(l,j);
                    Cmat(i,j) = tmp;
                }
            }
        }
        else{
            for(i=0; i<m; i++) {
                for(j=0; j<n; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(i,l)* Bmat(j,l);
                    Cmat(i,j) = tmp;
                }
            }
        }
    }
    else {
        if(!tB){
            for(i=0; i<m; i++) {
                for(j=0; j<n; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(l,i)* Bmat(l,j);
                    Cmat(i,j) = tmp;
                }
            }
        }
        else{
            for(i=0; i<m; i++) {
                for(j=0; j<n; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(l,i)* Bmat(j,l);
                    Cmat(i,j) = tmp;
                }
            }
        }
    }

}

void cgen_cblas_sgemm(bool tA, bool tB, int m, int n, int k, float* A, int lda, float* B, int ldb, float* C, int ldc)
{

    // printf("tA %d tB %d m %d n %d k %d lda %d ldb %d ldc %d\n",tA,tB,m,n,k,lda,ldb,ldc);

#define Amat(I,J) A[(I) + (J)*(lda)]
#define Bmat(I,J) B[(I) + (J)*(ldb)]
#define Cmat(I,J) C[(I) + (J)*(ldc)]

    int i,j,l;
    float tmp;

    if (!tA) {
        if(!tB){
            for(i=0; i<m; i++) {
                for(j=0; j<n; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(i,l)* Bmat(l,j);
                    Cmat(i,j) = tmp;
                }
            }
        }
        else{
            for(i=0; i<m; i++) {
                for(j=0; j<n; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(i,l)* Bmat(j,l);
                    Cmat(i,j) = tmp;
                }
            }
        }
    }
    else {
        if(!tB){
            for(i=0; i<m; i++) {
                for(j=0; j<n; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(l,i)* Bmat(l,j);
                    Cmat(i,j) = tmp;
                }
            }
        }
        else{
            for(i=0; i<m; i++) {
                for(j=0; j<n; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(l,i)* Bmat(j,l);
                    Cmat(i,j) = tmp;
                }
            }
        }
    }

}



void cgen_cblas_dgemv(bool tA, int m, int n, double* A, int lda, double* y, double* x)
{
    double tmp;
    int i,j;
#define Amat(I,J) A[(I) + (J)*(lda)]

    if(!tA) {
        for(i=0; i<m; i++) {
            tmp = 0.0;
            for(j=0; j<n; j++) {
                tmp += Amat(i,j)*x[j];
            }
            y[i] = tmp;
        }
    } else{
        for(i=0; i<n; i++) {
            tmp = 0.0;
            for(j=0; j<m; j++) {
                tmp += Amat(j,i)*x[j];
            }
            y[i] = tmp;
        }
    }
}


void cgen_cblas_sgemv(bool tA, int m, int n, float* A, int lda, float* y, float* x)
{
    float tmp;
    int i,j;
#define Amat(I,J) A[(I) + (J)*(lda)]

    if(!tA) {
        for(i=0; i<m; i++) {
            tmp = 0.0;
            for(j=0; j<n; j++) {
                tmp += Amat(i,j)*x[j];
            }
            y[i] = tmp;
        }
    } else{
        for(i=0; i<n; i++) {
            tmp = 0.0;
            for(j=0; j<m; j++) {
                tmp += Amat(j,i)*x[j];
            }
            y[i] = tmp;
        }
    }
}



double cgen_cblas_dasum(int n, double* y)
{
    double tmp = 0.0;
    for(int i=0; i<n; i++)
        tmp += y[i];
    return tmp;
}

float cgen_cblas_sasum(int n, float* y)
{
    float tmp = 0.0;
    for(int i=0; i<n; i++)
        tmp += y[i];
    return tmp;
}

double cgen_cblas_dnrm2(int n, double* y)
{
    double tmp = 0.0;
    for(int i=0; i<n; i++)
        tmp += y[i]*y[i];
    return sqrt(tmp);
}

float cgen_cblas_snrm2(int n, float* y)
{
    float tmp = 0.0;
    for(int i=0; i<n; i++)
        tmp += y[i]*y[i];
    return sqrt(tmp);
}

void cgen_somatcopy(int m, int n, float *A, int lda, float *B, int ldb)
{
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            Bmat(j, i) = Amat(i, j);
        }
    }
}

void cgen_domatcopy(int m, int n, double *A, int lda, double *B, int ldb)
{
    for(int i=0; i<m; i++) {
        for(int j=0; j<n; j++) {
            Bmat(j, i) = Amat(i, j);
        }
    }
}

