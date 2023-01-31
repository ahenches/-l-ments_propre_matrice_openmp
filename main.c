#include "src/inverter.h"
#include "src/input_creation_and_iterative_methods.h"
#include "src/algo_prr.h"
#include "src/load_matrix.h"
#include "header.h"

// https://en.wikipedia.org/wiki/Singular_value_decomposition
// https://www.gnu.org/software/gsl/doc/html/eigen.html#c.gsl_eigen_symm_workspace

int main(int argc, char *argv[]){
	if (argc == 1)
	{
		printf("Argument nom de fichier manquant\n");
		return 0;
	}

	// Initialisation et lecture de la matrice A depuis un fichier
	gsl_matrix *a;
	a = remplir_matrice(argv[1]);
	// print_matrix_contents(a);

	//Initialisation des tailles n et m (espace de depart et sous espace)
	int n = a->size1;
	int m = M_SIZE_CALCULUS;

	if (m <= 1)
		m = 2;

	//Initialisation du vecteur x de taille n (1,0,0,...)
	double X[n];
	for(int i = 1; i < n; i++)
		X[i] = 0;
	X[0] = 1;

	omp_set_num_threads(1);
	//Lancement de l algorithme sur la matrice lue et le vecteur x
	algo_PRR(a, X, n, m);

	//Liberation de la memoire
	gsl_matrix_free(a);

	return 0;
}
