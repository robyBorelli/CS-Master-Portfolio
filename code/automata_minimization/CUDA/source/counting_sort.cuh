#include "prefix_sum.cuh"

#define PHASE_2  4   // 0100
#define PHASE_13 3   // 0011

#ifndef _COUNTING_SORT_H_
#define _COUNTING_SORT_H_
enum counting_sort_phase2__t {
	PHASE_2_HOR = 0, // 0000, horizontal
	PHASE_2_VER = 4  // 0100, vertical
};

enum counting_sort_phase13_t {
	PHASE_13_A = 0,  // 0000, for cycle
	PHASE_13_B = 1,  // 0001, extraction
	PHASE_13_C = 2   // 0010, atomics
};

typedef struct counting_sort_args_t {
	int sort_flags;
	int H;
	int *Bh;     // t * b
	int *Ch;     // t * b
	// horizontal
	int *Bh_glb; // t
	int *Ch_glb; // t
	int *Q;      // t * b
	// vertical
	int *Bv;     // t * b
	int *Cv;     // t * b
	int *Y;      // t * b
} cnt_args_t;

typedef struct decompose_bottom_up_args_t {
	int *S;      // n
	int *M;      // n
	int *P;      // n
	int *Y;      // n
} dec_args_t;
#endif

void counting_sort_init(cnt_args_t *args, int n, int t, int r, int flags);
void counting_sort_term(cnt_args_t *args);
void counting_sort(int *A, int *S, int *T, int n, int t, int r, int h, cnt_args_t args);

void decompose_bottom_up_init(dec_args_t *args, int n, int t, int r);
void decompose_bottom_up_term(dec_args_t *args);
void decompose_bottom_up(int *A, int *S, int *T, int n, int t, int r, int h, dec_args_t args);
