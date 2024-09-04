TESTS_DIR=test_cases
EXECS = ./0_serial ./1_serial ./2_serial ./3_serial ./4_serial ./5_serial ./6_serial ./1_parallel_for ./2_parallel_task ./3_parallel_sections ./4_parallel_sections_splittedsort ./5_parallel_doublesign ./6_parallel_radixsort
NTHREADS=1 2 3 4 5 6 7 8
all:
	gcc -D NDEBUG -Wall -g3 -fopenmp 1_main_for_parallelism.c -g -o 1_parallel_for -lm -std=c11
	gcc -D NDEBUG -Wall -g3 -fopenmp 2_main_task.c -g -o 2_parallel_task -lm -std=c11
	gcc -D NDEBUG -Wall -g3 -fopenmp 3_main_sections.c -g -o 3_parallel_sections -lm -std=c11
	gcc -D NDEBUG -Wall -g3 -fopenmp 4_main_sections_splittedsort.c -g -o 4_parallel_sections_splittedsort -lm -std=c11
	gcc -D NDEBUG -Wall -g3 -fopenmp 5_main_doublesign.c -g -o 5_parallel_doublesign -lm -std=c11
	gcc -D NDEBUG -Wall -g3 -fopenmp 6_main_parallelradixsort.c -g -o 6_parallel_radixsort -lm -std=c11
	gcc -D NDEBUG -Wall -g3 0_main_serial.c -g -o 0_serial -lm -std=c11
	gcc -D NDEBUG -Wall -g3 1_main_for_parallelism.c -g -o 1_serial -lm -std=c11
	gcc -D NDEBUG -Wall -g3 2_main_task.c -g -o 2_serial -lm -std=c11
	gcc -D NDEBUG -Wall -g3 3_main_sections.c -g -o 3_serial -lm -std=c11
	gcc -D NDEBUG -Wall -g3 4_main_sections_splittedsort.c -g -o 4_serial -lm -std=c11
	gcc -D NDEBUG -Wall -g3 5_main_doublesign.c -g -o 5_serial -lm -std=c11
	gcc -D NDEBUG -Wall -g3 6_main_parallelradixsort.c -g -o 6_serial -lm -std=c11


debug:
	gcc -Wall -g3 -pg -fopenmp 1_main_for_parallelism.c -g -o 1_parallel_for -lm -std=c11
	gcc -Wall -g3 -pg -fopenmp 2_main_task.c -g -o 2_parallel_task -lm -std=c11
	gcc -Wall -g3 -pg -fopenmp 3_main_sections.c -g -o 3_parallel_sections -lm -std=c11
	gcc -Wall -g3 -pg -fopenmp 4_main_sections_splittedsort.c -g -o 4_parallel_sections_splittedsort -lm -std=c11
	gcc -Wall -g3 -pg -fopenmp 5_main_doublesign.c -g -o 5_parallel_doublesign -lm -std=c11
	gcc -Wall -g3 -pg -fopenmp 6_main_parallelradixsort.c -g -o 6_parallel_radixsort -lm -std=c11
	gcc -Wall -g3 -pg 0_main_serial.c -g -o 0_serial -lm -std=c11
	gcc -Wall -g3 -pg 1_main_for_parallelism.c -g -o 1_serial -lm -std=c11
	gcc -Wall -g3 -pg 2_main_task.c -g -o 2_serial -lm -std=c11
	gcc -Wall -g3 -pg 3_main_sections.c -g -o 3_serial -lm -std=c11
	gcc -Wall -g3 -pg 4_main_sections_splittedsort.c -g -o 4_serial -lm -std=c11
	gcc -Wall -g3 -pg 5_main_doublesign.c -g -o 5_serial -lm -std=c11
	gcc -Wall -g3 -pg 6_main_parallelradixsort.c -g -o 6_serial -lm -std=c11

clean:
	rm -f [0-6]_serial gmon.out [0-6]_parallel_*


test:
	@for THREADS in $(NTHREADS); do \
		for EXEC in $(EXECS); do \
			for test_file in $(wildcard $(TESTS_DIR)/*.txt); do \
		        args=$$(basename $$test_file .txt); \
		        n=$$(echo $$args | cut -d_ -f1);\
		        m=$$(echo $$args | cut -d_ -f2);\
		        l=$$(echo $$args | cut -d_ -f3);\
		        command="$$EXEC $$n $$m $$THREADS $$l 1" ;\
		        output=$$($$EXEC $$n $$m $$THREADS $$l 1); \
		        expected_output=$$(cat $$test_file); \
		        res=$$(if test "$$output" != "$$expected_output"; then echo '\033[0;31m' Failed test $$command '\n\n'OUTPUT'\n' $$output'\n\n'EXPECTED '\n' $$expected_output '\033[0m' ;  else echo '\033[0;32m' Test $$command OK '\033[0m'; fi;);\
		    	echo "$$res";\
		    done \
		done \
	done


ist1B:
	/usr/bin/time -f "Finished in: %e" ./3_parallel_sections 2000000 30 3 0 0

ist2B:
	/usr/bin/time -f "Finished in: %e" ./3_parallel_sections 5000 20 3 1 0
	
ist3B:
	/usr/bin/time -f "Finished in: %e" ./3_parallel_sections 2000000 30 3 2 0
	
ist1S:
	/usr/bin/time -f "Finished in: %e" ./3_parallel_sections 20000000 2 3 0 0

ist2S:
	/usr/bin/time -f "Finished in: %e" ./5_parallel_doublesign 15000 2 4 1 0
	
ist3S:
	/usr/bin/time -f "Finished in: %e" ./4_parallel_sections_splittedsort 20000000 2 4 2 0
	
	