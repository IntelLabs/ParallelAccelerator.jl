
void cgen_cblas_dgemm(bool tA, bool tB, int m, int n, int k, double* A, int lda, double* B, int ldb, double* C, int ldc)
{

#define Amat(I,J) A[(I) + (J)*(lda)]
#define Bmat(I,J) B[(I) + (J)*(ldb)]
#define Cmat(I,J) C[(I) + (J)*(ldc)]

    int i,j,l;
    double tmp;

    if (!tA) {
        if(!tB){
            for(i=0; i<n; i++) {
                for(j=0; j<m; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(i,l)* Bmat(l,j);
                    Cmat(i,j) = tmp;
                }
            }
        }
        else{
            for(i=0; i<n; i++) {
                for(j=0; j<m; j++) {
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
            for(i=0; i<n; i++) {
                for(j=0; j<m; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(l,i)* Bmat(l,j);
                    Cmat(i,j) = tmp;
                }
            }
        }
        else{
            for(i=0; i<n; i++) {
                for(j=0; j<m; j++) {
                    Cmat(i,j) = tmp = 0.0;
                    for(l=0; l<k; l++)
                        tmp += Amat(l,i)* Bmat(j,l);
                    Cmat(i,j) = tmp;
                }
            }
        }
    }

}

