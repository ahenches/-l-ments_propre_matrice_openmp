#include "inverter.h"
#include "input_creation_and_iterative_methods.h"
#include "algo_prr.h"

/**
 * @brief Implementation de l'algorithme de Padé-Rayleigh-Ritz (PRR)
 * N. Emad. The Padé-Rayleigh-Ritz Method for Solving Large Symmetric Eigenproblem. Numerical Algorithms, 11(1-4):159{179, 1996.
 * 
 * @param A Matrice de grande taille (n) ou l on recherche les valeurs et vecteurs propres
 * @param x Vecteur de taille n pour l'initialisation de l'algorithme
 * @param n taile de la matrice A
 * @param m taille du sous espace pour les calculs des valeurs et vecteurs propores
 */
void algo_PRR(gsl_matrix *A , double *x, int n, int m)
{
//// Initialisation des constantes, des vecteurs et des matrices

	int est_precis; //booleen pour la precision de l iteration
	int iteration;//Nombre d iteration 
	double xi;//Variable utillisée dans le calcul de du nouveau vecteur x
	double epsilon_i, epsilon_max; //precision des resultats de l iteration
	
	//Vecteurs utilisés pour le calcul du tableau c
	gsl_vector *y_k = gsl_vector_alloc(n);
	gsl_vector *y_k_moins_un = gsl_vector_alloc(n);

	//Matrice contenant les vecteur y_k
	gsl_matrix *v_m = gsl_matrix_alloc(n,m);
	
	// tableau de flottants contenant les resultats des produits scalaires sur les vecteurs (y_k et y_k-1) ou (y_k et y_k)
	double c[2*m+1];

	//Matrice contenant les vecteurs resultant du produit de Vm avec les vecteurs propres de Fm
	gsl_matrix *q = gsl_matrix_alloc(n,m);

	//matrices contruites a partir du tableau c 
	gsl_matrix *b_m = gsl_matrix_alloc(m,m);
	gsl_matrix *b_m_moins_un = gsl_matrix_alloc(m,m);

	//produit des matrices b_m et e_m
	gsl_matrix *f_m = gsl_matrix_alloc(m,m);
	//inverse de la matrice b_m-1
	gsl_matrix *e_m = gsl_matrix_alloc(m,m);

	//matrice contenant les vecteurs propres et vecteur contenant les valeurs propres
	gsl_vector * valeurs_propres = gsl_vector_alloc(m);

	//vecteur contentant un vecteur propre
	gsl_vector * vecteur_propre = gsl_vector_alloc(m);

	//Vecteurs temporaire pour le calcul de la precision
	gsl_vector * vecteur_A_Qi = gsl_vector_alloc(n);
	gsl_vector * vecteur_lambda_Q = gsl_vector_alloc(n);

	// Vecteur temporaire pour recuperer les valeurs propres complexes
	gsl_vector_complex *valeurs_propres_complexes = gsl_vector_complex_alloc(m);
	gsl_vector_complex *vecteur_propre_complex = gsl_vector_complex_alloc(m);
	// Matrice temporaire pour recuperer les vecteurs propres complexes
	gsl_matrix_complex *vecteurs_propres_complexes = gsl_matrix_complex_alloc(m, m);
	

	//vecteur temporaire pour le calcul de x
	gsl_vector * tmp_vector = gsl_vector_alloc(m);



//// Etape 1 
//1. Calculer C2k−1 = (yk, yk−1), C2k = (yk, yk) où yk = Ayk−1, for k = 1, m.
	iteration = 0;
	est_precis = 0;
	//On s arrete lorsque la precision est correcte ou qu on a depasse le nombre d iteration fixe
	while(!est_precis && iteration < ITERATION_MAX)
	{
		//Initialisation du vecteur y0 avec le vecteur x
		normalize_vector(y_k_moins_un, x, n);

//printf("//print_vector_contents(y_k_moins_un);\n");
//print_vector_contents(y_k_moins_un);
		//Calcul des produit scalaire du tableau c
		c[0] = scalar_product(y_k_moins_un, y_k_moins_un);
		for(int k = 1 ; k <= m ; k++)
		{
			gsl_matrix_set_col(v_m, k-1, y_k_moins_un);
			y_k = matrix_vector_product(A, y_k_moins_un);
			c[2*k-1] = scalar_product(y_k, y_k_moins_un);
			c[2*k] = scalar_product(y_k, y_k);
			gsl_vector_free(y_k_moins_un);
			y_k_moins_un = y_k;
		}

		//Constitution des matrices b_m-1, b_m 
		for(int ligne = 0 ; ligne < m ; ligne++)
		{
			for(int colonne = 0; colonne < m; colonne++)
			{
				gsl_matrix_set(b_m_moins_un, ligne, colonne, c[colonne + ligne]);
				gsl_matrix_set(b_m, ligne , colonne, c[colonne + ligne + 1]);
			}
		}
//printf("Bm\n");
//print_matrix_contents(b_m);
//printf("\nBm-1\n");
//print_matrix_contents(b_m_moins_un);
//printf("\nVm\n");
//print_matrix_contents(v_m);
	////Etape 2
	//2. Calculer Em = B−1 m−1, Fm = Em * Bm

		e_m = invert_a_matrix(b_m_moins_un, m);
//printf("//print_matrix_contents(e_m);\n");
//print_matrix_contents(e_m);
//printf("//print_matrix_contents(b_m);\n");
//print_matrix_contents(b_m);
		gsl_blas_dsymm(CblasLeft, CblasUpper, 1, e_m, b_m, 0, f_m);
//printf("//print_matrix_contents(f_m);\n");
//print_matrix_contents(f_m);
		// Calcul valeurs et vecteurs propres de f_m
		gsl_eigen_nonsymmv_workspace *my_workspace;
		my_workspace = gsl_eigen_nonsymmv_alloc(m);
		gsl_eigen_nonsymmv_params(1, my_workspace); // valeur possiblement modifiable
		gsl_eigen_nonsymmv(f_m, valeurs_propres_complexes, vecteurs_propres_complexes, my_workspace);
		gsl_eigen_nonsymmv_sort(valeurs_propres_complexes, vecteurs_propres_complexes, GSL_EIGEN_SORT_VAL_DESC);
		*valeurs_propres = (gsl_vector_complex_real(valeurs_propres_complexes).vector);
		//vecteurs_propres = gsl_matrix_complex_real(V);
		gsl_eigen_nonsymmv_free(my_workspace);

	////Etape 3
	//3. Calculer qi = Vm × ui pour i = 1, . . . m
		
//printf("/////// ETAPE 3 ///////////\n");
//printf("//print_vector_contents(valeurs_propres);\n");
//print_vector_contents(valeurs_propres);

		for(int i = 0; i < m; i++)
		{
			gsl_matrix_complex_get_col(vecteur_propre_complex, vecteurs_propres_complexes, i);
			*vecteur_propre = gsl_vector_complex_real(vecteur_propre_complex).vector;
//printf("//print_matrix_contents(vecteur_propre[i]);\n ::: %d de 0 à %d ", i, m);
//print_vector_contents(vecteur_propre);
			gsl_matrix_set_col(q, i, matrix_vector_product(v_m, vecteur_propre));
		}

//printf("//print_matrix_contents(q);\n");
//print_matrix_contents(q);
	////Etape 4
	//4. Si maxi=1,m k(Aqi − λiqi)k ε, alors avec un nouveau vecteur x aller `a l’´etape 1.
		// est_precis = 1;
		epsilon_max = 0;
		for (int i = 0; i < m ; i++ )
		{
			gsl_matrix_get_col(vecteur_lambda_Q, q, i);
//printf("//print_vector_contents(vecteur_Q);\n");
//print_vector_contents(vecteur_lambda_Q);
			vecteur_A_Qi = matrix_vector_product(A, vecteur_lambda_Q);
//printf("//print_vector_contents(vecteur_A_Qi);\n");
//print_vector_contents(vecteur_A_Qi);
//printf("Calcul lambda(i) * Qi i:%d\n",i);
//printf("lambda(i) :%f\n", gsl_vector_get(valeurs_propres, i));
			gsl_vector_scale(vecteur_lambda_Q, gsl_vector_get(valeurs_propres, i));
//printf("//print_vector_contents(vecteur_lambda_Q);\n");
//print_vector_contents(vecteur_lambda_Q);
			gsl_vector_sub(vecteur_A_Qi, vecteur_lambda_Q);
			epsilon_i = get_norm_gsl(vecteur_A_Qi); 
			
			if(epsilon_i > epsilon_max)
			{
				// est_precis = 0;
				epsilon_max = epsilon_i;
			}
		}
		//Verification de la precision des calculs
		if(epsilon_max < PRECISION_SOUHAITEE)
		{
			est_precis = 1;
		}
		//re-calcul de x
		for(int i = 0; i < n; i++)
		{
			gsl_matrix_get_row(tmp_vector, q, i);
			xi = 0;
			for (int j =0; j<m ; j++)
			{
				xi += gsl_vector_get(tmp_vector, j);
			}
			x[i] = xi;
		}
		iteration++;
//printf("epsilon max : %lf\n", epsilon_max);
//printf("est precis : %d\n", est_precis);
//printf("iteration : %d\n", iteration);
	}

	//Liberation de la memoire
	gsl_vector_free(y_k);
	gsl_vector_free(valeurs_propres);
	gsl_vector_free(vecteur_propre);
	gsl_vector_free(vecteur_A_Qi);
	gsl_vector_free(vecteur_lambda_Q);
	gsl_vector_free(tmp_vector);

	gsl_matrix_free(v_m);
	gsl_matrix_free(q);
	gsl_matrix_free(b_m);
	gsl_matrix_free(b_m_moins_un);
	gsl_matrix_free(f_m);
	gsl_matrix_free(e_m);
	
}
