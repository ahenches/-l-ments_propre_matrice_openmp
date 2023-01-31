#gsl_include=-I/usr/include/
#gsl_lib=-L/usr/lib/x86_64-linux-gnu/
gsl_flags=-lgsl -lgslcblas -lm
FLAGS=-Wall -fopenmp -g -DHAVE_INLINE
cc=gcc

programme: main.o algo.o load_matrix.o inverter.o input_andCo.o
	$(cc) $(FLAGS) -o $@ $(gsl_lib) $^ $(gsl_flags) 

calcul_perf: calcul_perf.o algo.o load_matrix.o inverter.o input_andCo.o
	$(cc) $(FLAGS) -o $@ $(gsl_lib) $^ $(gsl_flags) 

calcul_perf.o: calcul_perf.c src/inverter.c
	$(cc) $(FLAGS) $(gsl_include) -c $^

main.o: main.c src/inverter.c
	$(cc) $(FLAGS) $(gsl_include) -c $^

algo.o: src/algo_prr.c
	$(cc) $(FLAGS) -o $@ $(gsl_include) -c $^

load_matrix.o: src/load_matrix.c
	$(cc) $(FLAGS) -o $@ $(gsl_include) -c $^

inverter.o: src/inverter.c
	$(cc) $(FLAGS) -o $@ $(gsl_include) -c $^

input_andCo.o: src/input_creation_and_iterative_methods.c
	$(cc) $(FLAGS) -o $@ $(gsl_include) -c $^



test: test.o input_andCo.o load_matrix.o
	$(cc) $(FLAGS) -o $@ $(gsl_lib) $^ $(gsl_flags)

test.o: test.c  
	$(cc) $(FLAGS) -o $@ $(gsl_include) -c $^

clean: 
	rm *.o




