#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"
#include "automaton.cuh"
#include "prefix_sum.cuh"
#include "radix_sort.cuh"

struct produce_args_t {
	automaton_t *A;
	int k;
	int s;
	int *Delta;
	int *Tilde;
};

int radix_sort_produce(int *X, int n, int t, int r, void *args)
{
	produce_args_t *p_args = (produce_args_t *)args;

	if (p_args->s >= 0) {
		p_args->s--;

		automaton_t *A = p_args->A;
		int k = p_args->k;
		int s = p_args->s;
		int *Delta = p_args->Delta;
		int *Tilde = p_args->Tilde;

		constant KERNEL_ARGS2(upperDiv(n, t * r), t) (X, n, k);

		int *Delta_h;
		if (s >= 0) {
			Delta_h = A->Delta[s];
		} else {
			Delta_h = A->Q;
		}
		CATCH(cudaMemcpyAsync(Delta, Delta_h, sizeof(int) * A->n, cudaMemcpyHostToDevice));

		permute_src KERNEL_ARGS2(upperDiv(A->n, t * r), t) (Tilde, X, Delta, A->n); // apply tilde
		return upperLog2(k + 1);
	}

	return 0;
}

int moore_refine(automaton_t *A, int *Delta, int *Tilde, int n, int k, int *T, int *M, int *P, int t, int r, rdx_args_t args)
{
	produce_args_t produce_args;
	produce_args.A = A;
	produce_args.k = k;
	produce_args.s = A->m;
	produce_args.Delta = Delta;
	produce_args.Tilde = Tilde;

	void *p_args = (void *)&produce_args;

	return radix_sort(&radix_sort_produce, p_args, T, M, P, n, t, r, args) - 1;
}

__global__ void moore_extract(int *T, int *M, int *P, int *Q, int n)
{
	for (int q = blockDim.x * blockIdx.x + threadIdx.x; q < n; q += blockDim.x * gridDim.x) {
		if (M[q] == 1) {
			Q[P[T[q]]] = T[q];
		}
	}
}

void moore(automaton_t *A_i, automaton_t *A_o, int t, int r, int sort_flags)
{
	int n_i = A_i->n;
	int m_i = A_i->m;

	int n = upperDiv(n_i + 1, t) * t; // make room for fake class

	// phase 1, minimization
	int *Delta;
	int *Tilde;
	int *T, *M, *P;
	CATCH(cudaMalloc((void **)&Delta, sizeof(int) * n_i));
	CATCH(cudaMalloc((void **)&Tilde, sizeof(int) * n_i));
	CATCH(cudaMalloc((void **)&T, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&M, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&P, sizeof(int) * n));

	radix_sort_args_t args;
	radix_sort_init(&args, n, t, r, sort_flags);

	int k1 = 0;
	int k2 = 2; // n_class
	int it = 1;
	CATCH(cudaMemcpy(Tilde, A_i->F, sizeof(int) * n_i, cudaMemcpyHostToDevice));
	while (k1 < k2) {
		k1 = k2;
		k2 = moore_refine(A_i, Delta, Tilde, n, k1, T, M, P, t, r, args);
		CATCH(cudaMemcpy(Tilde, P, sizeof(int) * n_i, cudaMemcpyDeviceToDevice));
		if (_global_verb) {
			fprintf(stderr, "Iteration %d: %d classes found\n", it, k2);
		}
		it++;
	}

	radix_sort_term(&args);

	// phase 2, construction
	const int one = 1;

	int *Q;
	CATCH(cudaMalloc((void **)&Q, sizeof(int) * (k2 + 1))); // make room for fake class

	CATCH(cudaMemcpy(M, &one, sizeof(int), cudaMemcpyHostToDevice));
	moore_extract KERNEL_ARGS2(upperDiv(n, t * r), t) (T, M, P, Q, n); // representatives for each class

	CATCH(cudaFree(M));
	CATCH(cudaFree(P));
	CATCH(cudaFree(T));

	int n_o = k2;
	int m_o = m_i;

	int *Tmp_i, *Tmp_o;
	CATCH(cudaMalloc((void **)&Tmp_i, sizeof(int) * n_i));
	CATCH(cudaMalloc((void **)&Tmp_o, sizeof(int) * n_o));

	DFA_init(A_o, n_o, m_o);

	int b = upperDiv(n_o, t * r);

	for (int s = 0; s < m_o; s++) {
		CATCH(cudaMemcpy(Tmp_i, A_i->Delta[s], sizeof(int) * n_i, cudaMemcpyHostToDevice));
		permute_src KERNEL_ARGS2(b, t) (Tmp_i, Delta, Q, n_o);
		permute_src KERNEL_ARGS2(b, t) (Tilde, Tmp_o, Delta, n_o); // apply tilde
		CATCH(cudaMemcpy(A_o->Delta[s], Tmp_o, sizeof(int) * n_o, cudaMemcpyDeviceToHost));
	}

	CATCH(cudaMemcpy(&A_o->q0, Tilde, sizeof(int), cudaMemcpyDeviceToHost));

	CATCH(cudaMemcpy(Tmp_i, A_i->F, sizeof(int) * n_i, cudaMemcpyHostToDevice));
	permute_src KERNEL_ARGS2(b, t) (Tmp_i, Tmp_o, Q, n_o);
	CATCH(cudaMemcpy(A_o->F, Tmp_o, sizeof(int) * n_o, cudaMemcpyDeviceToHost));

	CATCH(cudaFree(Q));
	CATCH(cudaFree(Tmp_i));
	CATCH(cudaFree(Tmp_o));
	CATCH(cudaFree(Delta));
	CATCH(cudaFree(Tilde));
}
