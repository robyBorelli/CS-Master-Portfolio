#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "automaton.cuh"
#include "common.cuh"
#include "moore.cuh"
#include "radix_sort.cuh"

void print_help(cudaDeviceProp prop)
{
	int B_SM = maxBlkPerSM(prop);
	int T_SM = maxThdPerSM(prop);
	int T_blk = maxThdPerBlk(prop);
	int t = T_SM / B_SM;
	if (t > T_blk) {
		t = T_blk;
	}

	fprintf(stderr, "Usage:\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --help                    Prints this message\n");
	fprintf(stderr, "  --verbose                 Prints information about the execution\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "* Input selection\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  -I decode                 Decodes an input DFA (default)\n");
	fprintf(stderr, "  -I best-case  <n> <m>     Generates a best-case input DFA of 3<n> states and <m> symbols\n");
	fprintf(stderr, "  -I worst-case <n> <m>     Generates a worst-case input DFA of 3<n> states and <m> symbols\n");
	fprintf(stderr, "  -I random     <n> <m>     Generates a random input DFA of <n> states and <m> symbols\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "* Action selection\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  -A pass                   Does nothing (default)\n");
	fprintf(stderr, "  -A minimize <m> <t> <r>   Performs Moore's algorithm on the input DFA with <t> threads and <r> strides per block\n");
	fprintf(stderr, "                            Note: <t> must be a power of 2 between 64 and %d\n", 1 << lowerLog2(T_blk));
	fprintf(stderr, "                            Note: suggested value for current device is <t>=%d\n", 1 << lowerLog2(t));
	fprintf(stderr, "    <m>=horizontal-a        Counting-sort, phase 2 horizontal variant, phases 1-3 solution a (for cycle)\n");
	fprintf(stderr, "    <m>=horizontal-b        Counting-sort, phase 2 horizontal variant, phases 1-3 solution b (extraction)\n");
	fprintf(stderr, "    <m>=horizontal-c        Counting-sort, phase 2 horizontal variant, phases 1-3 solution c (atomics)\n");
	fprintf(stderr, "    <m>=vertical-a          Counting-sort, phase 2 vertical variant, phases 1-3 solution a (for-cycle)\n");
	fprintf(stderr, "    <m>=vertical-b          Counting-sort, phase 2 vertical variant, phases 1-3 solution b (extraction)\n");
	fprintf(stderr, "    <m>=vertical-c          Counting-sort, phase 2 vertical variant, phases 1-3 solution c (atomics)\n");
	fprintf(stderr, "    <m>=decompose           Bottom-up decomposition\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "* Output selection\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  -O encode                 Encodes the output DFA (default)\n");
	fprintf(stderr, "  -O pretty-print           Pretty-prints the output DFA\n");
	fprintf(stderr, "  -O pass                   Does nothing\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Input DFA encoding:\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  n=<n>                     Number of states\n");
	fprintf(stderr, "  m=<m>                     Number of symbols\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  <p_1> <s_1> <q_1>\n");
	fprintf(stderr, "  <p_2> <s_2> <q_2>\n");
	fprintf(stderr, "  ...\n");
	fprintf(stderr, "  <p_t> <s_t> <q_t>         Set of transitions Delta={(<p_1>,<s_1>,<q_1>),...,(<p_t>,<s_t>,<q_t>)}\n");
	fprintf(stderr, "                            Note: unspecified transitions are assumed to be of the form (<q>,<s>,<q>)\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  initial <q_0>             Initial state <q_0>\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  final <q_1>\n");
	fprintf(stderr, "  final <q_2>\n");
	fprintf(stderr, "  ...\n");
	fprintf(stderr, "  final <q_f>               Set of final states F={<q_1>,...,<q_f>}\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "                            Note: states must be of the form <q>=0,1,2,...\n");
	fprintf(stderr, "                            Note: symbols must be of the form <s>=a,b,c,...\n");
	fprintf(stderr, "                            Note: press CTRL+D to terminate input\n");
}

enum i_mode_t {
	I_MODE_WRST = DFA_MODE_WRST,
	I_MODE_BEST = DFA_MODE_BEST,
	I_MODE_RAND,
	I_MODE_DECD
};

enum a_mode_t {
	A_MODE_MINM,
	A_MODE_PASS
};

enum o_mode_t {
	O_MODE_PRTY,
	O_MODE_ENCD,
	O_MODE_PASS
};

int main(int argc, char *argv[])
{
	i_mode_t i_mode = I_MODE_DECD;
	a_mode_t a_mode = A_MODE_PASS;
	o_mode_t o_mode = O_MODE_ENCD;
	int n, m;
	int t, r;
	int sort_flags;

	int i = 1;
	while (i < argc) {
		if (strcmp(argv[i], "--help") == 0) {
			_global_help = true; // declared in common.cuh
		} else if (strcmp(argv[i], "--verbose") == 0) {
			_global_verb = true; // declared in common.cuh
		} else if (strcmp(argv[i], "-I") == 0) {
			if (i + 1 >= argc) {
				handle_error("Error: Too few arguments in input selection\n");
			}
			i++;
			if (strcmp(argv[i], "decode") == 0) {
				i_mode = I_MODE_DECD;
			} else {
				if (strcmp(argv[i], "best-case") == 0) {
					i_mode = I_MODE_BEST;
				} else if (strcmp(argv[i], "worst-case") == 0) {
					i_mode = I_MODE_WRST;
				} else if (strcmp(argv[i], "random") == 0) {
					i_mode = I_MODE_RAND;
				} else {
					handle_error("Error: Input mode not recognized: %s\n", argv[i]);
				}
				if (i + 2 >= argc) {
					handle_error("Error: Too few arguments in input selection\n");
				}
				n = (int)strtol(argv[++i], NULL, 10);
				m = (int)strtol(argv[++i], NULL, 10);
				if (n <= 0) {
					handle_error("Error: Wrong number of states in input selection\n");
				}
				if (m <= 0) {
					handle_error("Error: Wrong number of symbols in input selection\n");
				}
			}
		} else if (strcmp(argv[i], "-A") == 0) {
			if (i + 1 >= argc) {
				handle_error("Error: Too few arguments in action selection\n");
			}
			i++;
			if (strcmp(argv[i], "pass") == 0) {
				a_mode = A_MODE_PASS;
			} else {
				if (strcmp(argv[i], "minimize") == 0) {
					a_mode = A_MODE_MINM;
				} else {
					handle_error("Error: Action mode not recognized: %s\n", argv[i]);
				}
				if (i + 3 >= argc) {
					handle_error("Error: Too few arguments in action selection\n");
				}
				i++;
				sort_flags = 0;
				if (strcmp(argv[i], "horizontal-a") == 0) {
					sort_flags = ALGORITHM_CNT | PHASE_2_HOR | PHASE_13_A;
				} else if (strcmp(argv[i], "horizontal-b") == 0) {
					sort_flags = ALGORITHM_CNT | PHASE_2_HOR | PHASE_13_B;
				} else if (strcmp(argv[i], "horizontal-c") == 0) {
					sort_flags = ALGORITHM_CNT | PHASE_2_HOR | PHASE_13_C;
				} else if (strcmp(argv[i], "vertical-a") == 0) {
					sort_flags = ALGORITHM_CNT | PHASE_2_VER | PHASE_13_A;
				} else if (strcmp(argv[i], "vertical-b") == 0) {
					sort_flags = ALGORITHM_CNT | PHASE_2_VER | PHASE_13_B;
				} else if (strcmp(argv[i], "vertical-c") == 0) {
					sort_flags = ALGORITHM_CNT | PHASE_2_VER | PHASE_13_C;
				} else if (strcmp(argv[i], "decompose") == 0) {
					sort_flags = ALGORITHM_DEC;
				} else {
					handle_error("Error: Sort mode not recognized\n");
				}
				t = (int)strtol(argv[++i], NULL, 10);
				r = (int)strtol(argv[++i], NULL, 10);
				if (t < 64 || t > 1024 || lowerLog2(t) != upperLog2(t)) {
					handle_error("Error: Wrong number of threads per block in action selection\n");
				}
				if (r <= 0) {
					handle_error("Error: Wrong number of strides per block in action selection\n");
				}
			}
		} else if (strcmp(argv[i], "-O") == 0) {
			if (i + 1 >= argc) {
				handle_error("Error: Too few arguments in output selection\n");
			}
			i++;
			if (strcmp(argv[i], "encode") == 0) {
				o_mode = O_MODE_ENCD;
			} else if (strcmp(argv[i], "pretty-print") == 0) {
				o_mode = O_MODE_PRTY;
			} else if (strcmp(argv[i], "pass") == 0) {
				o_mode = O_MODE_PASS;
			} else {
				handle_error("Error: Output mode not recognized: %s\n", argv[i]);
			}
		} else {
			handle_error("Error: Option not recognized: %s\n", argv[i]);
		}
		i++;
	}

	int device;
	cudaDeviceProp prop;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	if (_global_help) {
		print_help(prop);
		return 0;
	}

	automaton_t A_i;
	automaton_t A_o;

	switch (i_mode) {
	case I_MODE_DECD:
		DFA_decode(&A_i);
		break;
	case I_MODE_BEST:
	case I_MODE_WRST:
		DFA_init_mode(&A_i, n, m, i_mode);
		break;
	case I_MODE_RAND:
		DFA_init_rand(&A_i, n, m);
		break;
	}

	switch (a_mode) {
	case A_MODE_PASS:
		A_o = A_i;
		break;
	case A_MODE_MINM:
		int N = A_i.n + 1;
		int M = upperDiv(N, t);
		int b = upperDiv(M, r);
		int B = maxBlkPerGrd(prop);
		if (b > B) {
			b = B;
		}
		int R = upperDiv(M, b);
		if (r < R) {
			r = R;
		}
		if (_global_verb) {
			fprintf(stderr, "Using ");
			switch (sort_flags & ALGORITHM) {
			case ALGORITHM_CNT: fprintf(stderr, "counting-sort, ");
				switch (sort_flags & PHASE_2) {
				case PHASE_2_HOR: fprintf(stderr, "phase 2 horizontal variant, "); break;
				case PHASE_2_VER: fprintf(stderr, "phase 2 vertical variant, "); break;
				}
				switch (sort_flags & PHASE_13) {
				case PHASE_13_A: fprintf(stderr, "phase 1-3 solution a (for cycle)\n"); break;
				case PHASE_13_B: fprintf(stderr, "phase 1-3 solution b (extraction)\n"); break;
				case PHASE_13_C: fprintf(stderr, "phase 1-3 solution c (atomics)\n"); break;
				}
				break;
			case ALGORITHM_DEC: fprintf(stderr, "bottom-up decomposition\n"); break;
			}
			fprintf(stderr, "Number of blocks: %d\n", b);
			fprintf(stderr, "Number of threads per block: %d\n", t);
			fprintf(stderr, "Number of strides per block: %d\n", r);
			fprintf(stderr, "Note: these numbers may differ for some subroutines\n");
		}

		float time;
		cudaEvent_t e_s, e_e;
		CATCH(cudaEventCreate(&e_s));
		CATCH(cudaEventCreate(&e_e));
		CATCH(cudaEventRecord(e_s));

		moore(&A_i, &A_o, t, r, sort_flags);

		CATCH(cudaEventRecord(e_e));
		CATCH(cudaDeviceSynchronize());
		CATCH(cudaEventElapsedTime(&time, e_s, e_e));
		if (_global_verb) {
			fprintf(stderr, "Elapsed time: %.2f s\n", time / 1000);
		}

		CATCH(cudaEventDestroy(e_s));
		CATCH(cudaEventDestroy(e_e));

		DFA_term(&A_i);
		break;
	}

	switch (o_mode) {
	case O_MODE_ENCD:
		DFA_encode(&A_o);
		break;
	case O_MODE_PRTY:
		DFA_print(&A_o);
		break;
	}

	DFA_term(&A_o);

	return 0;
}
