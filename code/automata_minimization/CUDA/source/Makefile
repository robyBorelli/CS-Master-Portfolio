all:
	nvcc -rdc=true -Xcompiler "-fopenmp" --default-stream per-thread automaton.cu common.cu counting_sort.cu  cuda.cu moore.cu prefix_sum.cu producer_consumer.cu radix_sort.cu -o cuda

ist1B:
	./cuda --verbose -I best-case 2000000 30 -A minimize vertical-b 128 4 -O pass

ist2B:
	./cuda --verbose -I worst-case 5000 20 -A minimize vertical-a 128 4 -O pass

ist3B:
	./cuda --verbose -I random 2000000 30 -A minimize vertical-c 128 4 -O pass

ist1S:
	./cuda --verbose -I best-case 20000000 2 -A minimize horizontal-b 128 4 -O pass

ist2S:
	./cuda --verbose -I worst-case 15000 2 -A minimize horizontal-a 128 4 -O pass

ist3S:
	./cuda --verbose -I random 20000000 2 -A minimize horizontal-c 128 4 -O pass

clean:
	rm cuda