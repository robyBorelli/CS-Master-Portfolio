#include "counting_sort.cuh"
#include "producer_consumer.cuh"

#define ALGORITHM 8    // 1000

#ifndef _RADIX_SORT_H_
#define _RADIX_SORT_H_
enum radix_sort_algorithm_t {
	ALGORITHM_CNT = 0, // 0000, counting-sort
	ALGORITHM_DEC = 8  // 1000, bottom-up decomposition
};

typedef struct radix_sort_args_t {
	int sort_flags;
	cudaStream_t s;
	int *buf; // n
	int *p_X; // n
	int *c_X; // n
	int *S;   // n + 1
	int *Q;   // n
	int *Y;   // n
	cnt_args_t cnt_args;
	dec_args_t dec_args;
} rdx_args_t;
#endif


void radix_sort_init(rdx_args_t *args, int n, int t, int r, int flags);
void radix_sort_term(rdx_args_t *args);
int radix_sort(producer_t produce, void *p_args, int *T, int *M, int *P, int n, int t, int r, rdx_args_t args);
