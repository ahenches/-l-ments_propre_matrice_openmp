#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_eigen.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define ITERATION_MAX 1500
#define PRECISION_SOUHAITEE 0.000001 
void algo_PRR(gsl_matrix *a , double *x, int n, int m);
