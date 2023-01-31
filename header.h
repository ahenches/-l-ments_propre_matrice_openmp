#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_eigen.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define M_SIZE_CALCULUS 2//(2 * (1 + ceil(log10(n))))
