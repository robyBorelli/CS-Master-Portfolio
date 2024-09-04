#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"
#include "counting_sort.cuh"
#include "prefix_sum.cuh"
#include "producer_consumer.cuh"
#include "radix_sort.cuh"

#define SORT_CNT counting_sort
#define SORT_CNT_INIT counting_sort_init
#define SORT_CNT_TERM counting_sort_term
#define SORT_DEC decompose_bottom_up
#define SORT_DEC_INIT decompose_bottom_up_init
#define SORT_DEC_TERM decompose_bottom_up_term

struct consume_args_t {
	int sort_flags;
	int *S; // n + 1
	int *T; // n
	int *M; // n
	int *P; // n
	int *Q; // n
	int *Y;
	cnt_args_t cnt_args;
	dec_args_t dec_args;
};

__global__ void update_mask(int *X, int *P, int *S, int *M, int m)
{
	int p = threadIdx.x;
	int t = blockDim.x;

	int s0 = t * blockIdx.x;
	int s1 = t * gridDim.x;

	S += s0;
	M += s0;
	for (int j = blockIdx.x; j < m; j += gridDim.x) {
		int x0 = S[p - 0];
		int x1 = S[p - 1];
		M[p] = (int)(
			X[x0] != X[x1] || // curr iteration
			P[x0] != P[x1]);  // prev iterations
		S += s1;
		M += s1;
	}
}

int radix_sort_consume(int *X, int n, int t, int r, void *args, int h)
{
	int m = n / t;
	int b = upperDiv(m, r);

	consume_args_t *c_args = (consume_args_t *)args;

	int *S = c_args->S;
	int *T = c_args->T;
	int *M = c_args->M;
	int *P = c_args->P;
	int *Q = c_args->Q;

	switch (c_args->sort_flags & ALGORITHM) {
	case ALGORITHM_CNT: SORT_CNT(X, S + 1, T, n, t, r, h, c_args->cnt_args); break;
	case ALGORITHM_DEC: SORT_DEC(X, S + 1, T, n, t, r, h, c_args->dec_args); break;
	}

	CATCH(cudaMemcpyAsync(S + 1, T, sizeof(int) * n, cudaMemcpyDeviceToDevice));
	CATCH(cudaMemcpyAsync(S, S + 1, sizeof(int) * 1, cudaMemcpyDeviceToDevice));

	update_mask KERNEL_ARGS2(b, t) (X, P, S + 1, M, m);
	prefix_sum_element_inclusive(M, Q, n, t / 2, r, c_args->Y);
	permute_dst KERNEL_ARGS2(b, t) (Q, P, S + 1, n);

	return 0;
}

void radix_sort_init(rdx_args_t *args, int n, int t, int r, int flags)
{
	CATCH(cudaStreamCreate(&args->s));
	CATCH(cudaMalloc((void **)&args->buf, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&args->p_X, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&args->c_X, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&args->S, sizeof(int) * (n + 1)));
	CATCH(cudaMalloc((void **)&args->Q, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&args->Y, sizeof(int) * n));

	args->sort_flags = flags;
	switch (args->sort_flags & ALGORITHM) {
	case ALGORITHM_CNT: SORT_CNT_INIT(&args->cnt_args, n, t, r, flags); break;
	case ALGORITHM_DEC: SORT_DEC_INIT(&args->dec_args, n, t, r); break;
	}
}

void radix_sort_term(rdx_args_t *args)
{
	CATCH(cudaStreamDestroy(args->s));
	CATCH(cudaFree(args->buf));
	CATCH(cudaFree(args->p_X));
	CATCH(cudaFree(args->c_X));
	CATCH(cudaFree(args->S));
	CATCH(cudaFree(args->Q));
	CATCH(cudaFree(args->Y));

	switch (args->sort_flags & ALGORITHM) {
	case ALGORITHM_CNT: SORT_CNT_TERM(&args->cnt_args); break;
	case ALGORITHM_DEC: SORT_DEC_TERM(&args->dec_args); break;
	}
}

int radix_sort(producer_t produce, void *p_args, int *T, int *M, int *P, int n, int t, int r, rdx_args_t args)
{
	consumer_t consume = &radix_sort_consume;

	volatile int offt = 0;
	volatile int done = 0;

	int k;

#pragma omp parallel sections num_threads(2) firstprivate(k) lastprivate(k) shared(offt, done)
	{
#pragma omp section
		{
			producer(args.s, args.buf, args.p_X, &offt, &done, produce, p_args, n, t, r);
		}
#pragma omp section
		{
			consume_args_t consume_args;
			consume_args.sort_flags = args.sort_flags;
			consume_args.S = args.S;
			consume_args.T = T;
			consume_args.M = M;
			consume_args.P = P;
			consume_args.Q = args.Q;
			consume_args.Y = args.Y;
			consume_args.cnt_args = args.cnt_args;
			consume_args.dec_args = args.dec_args;

			identity KERNEL_ARGS2(upperDiv(n, t * r), t) (args.S + 1, n);
			CATCH(cudaMemsetAsync(M, 0, sizeof(int) * n));
			CATCH(cudaMemsetAsync(P, 0, sizeof(int) * n));

			void *c_args = (void *)&consume_args;

			int h;
			switch (args.sort_flags & ALGORITHM) {
			case ALGORITHM_CNT: h = lowerLog2(t); break;
			case ALGORITHM_DEC: h = BUF_SIZE; break;
			}

			consumer(args.s, args.buf, args.c_X, &offt, &done, consume, c_args, n, t, r, h);

			CATCH(cudaMemcpy(&k, args.Q + n - 1, sizeof(int), cudaMemcpyDeviceToHost)); // no async!
		}
	}

	return k + 1;
}
