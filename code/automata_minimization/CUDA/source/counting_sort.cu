#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"
#include "counting_sort.cuh"
#include "prefix_sum.cuh"

__global__ void counting_sort_phase1_a(int *A, int *S, int m, int *B, int *B_glb, int r, int H, int h)
{
	int p = threadIdx.x;
	int t = blockDim.x;
	int s = blockDim.x * blockIdx.x;

	S += s * r;
	B += s;

	__shared__ extern int S1[]; // |S1| = blockDim.x

	int b = 0;

	int offt = r * blockIdx.x;
	if (r > m - offt) {
		r = m - offt;
	}
	for (int j = 0; j < r; j++) {
		int y = S[p];
		S1[p] = A[y];
		cuda_SYNCTHREADS();
		for (int q = 0; q < t; q++) {
			if (p == S1[q]) {
				b++;
			}
		}
		cuda_SYNCTHREADS();
		S += t;
	}

	B[p] = b;
	if (B_glb != NULL) {
		cuda_ATOMIC_ADD(&B_glb[p], b);
	}
}

__global__ void counting_sort_phase1_b(int *A, int *S, int m, int *B, int *B_glb, int r, int H, int h)
{
	int p = threadIdx.x;
	int t = blockDim.x;
	int s = blockDim.x * blockIdx.x;

	S += s * r;
	B += s;

	int b = 0;

	int offt = r * blockIdx.x;
	if (r > m - offt) {
		r = m - offt;
	}
	int k = 1 << h;
	for (int j = 0; j < r; j++) {
		int y = S[p];
		int a = A[y];
		for (int i = 0; i < k; i++) {
			if (cuda_SYNCTHREADS_AND(a < i)) { // heuristic
				break;
			}
			int count = cuda_SYNCTHREADS_COUNT(a == i);
			if (p == i) {
				b += count;
			}
		}
		S += t;
	}

	B[p] = b;
	if (B_glb != NULL) {
		cuda_ATOMIC_ADD(&B_glb[p], b);
	}
}

__global__ void counting_sort_phase1_c(int *A, int *S, int m, int *B, int *B_glb, int r, int H, int h)
{
	int p = threadIdx.x;
	int t = blockDim.x;
	int s = blockDim.x * blockIdx.x;

	S += r * blockIdx.x << h;
	B += s;

	__shared__ extern int S1[]; // |S1| = blockDim.x

	S1[p] = 0;
	cuda_SYNCTHREADS();

	int p_quot = p >> h;            // p / (1 << h)
	int p_remd = p & ~(MASK1 << h); // p % (1 << h)
	int x = p_quot * (m << h) + p_remd;

	int offt = r * blockIdx.x;
	if (r > m - offt) {
		r = m - offt;
	}
	int k = 1 << h;
	for (int j = 0; j < r; j++) {
		int y = S[x];
		int a = A[y] << H - h | p_quot;
		cuda_ATOMIC_ADD(&S1[a], 1);
		S += k;
	}
	cuda_SYNCTHREADS();

	B[p] = S1[p];
	if (B_glb != NULL) {
		cuda_ATOMIC_ADD(&B_glb[p], S1[p]);
	}
}

__global__ void counting_sort_phase2_hor1(int *B_glb, int *C_glb)
{
	int p = threadIdx.x;
	int t = blockDim.x;

	__shared__ extern int S2[]; // |S2| = blockDim.x

	S2[p] = B_glb[p];

	cuda_SYNCTHREADS();
	prefix_sum_exclusive(S2, t);

	C_glb[p] = S2[p];
}

__global__ void counting_sort_phase2_hor2(int *C, int *C_glb)
{
	int p = threadIdx.x;
	int s = blockDim.x * blockIdx.x;

	C += s;
	C[p] += C_glb[p];
}

__global__ void counting_sort_phase2_ver1(int *Bh, int *Bv)
{
	int p = threadIdx.x;
	int s = blockDim.x * blockIdx.x;

	Bh += s;
	Bv[blockIdx.x + gridDim.x * p] = Bh[p];
}

__global__ void counting_sort_phase2_ver2(int *Cv, int *Ch)
{
	int p = threadIdx.x;
	int s = blockDim.x * blockIdx.x;

	Ch += s;
	Ch[p] = Cv[blockIdx.x + gridDim.x * p];
}

__global__ void counting_sort_phase3_a(int *A, int *S, int *T, int m, int *C, int r, int H, int h)
{
	int p = threadIdx.x;
	int t = blockDim.x;
	int s = blockDim.x * blockIdx.x;

	S += s * r;
	C += s;

	__shared__ extern int S3[]; // |S3| = 2 * blockDim.x

	int *a = S3 + 0 * t;
	int *y = S3 + 1 * t;
	int d = C[p];
	cuda_SYNCTHREADS();

	int offt = r * blockIdx.x;
	if (r > m - offt) {
		r = m - offt;
	}
	for (int j = 0; j < r; j++) {
		y[p] = S[p];
		a[p] = A[y[p]];
		cuda_SYNCTHREADS();
		for (int q = 0; q < t; q++) {
			if (p == a[q]) {
				T[d++] = y[q];
			}
		}
		cuda_SYNCTHREADS();
		S += t;
	}
}

__global__ void counting_sort_phase3_b(int *A, int *S, int *T, int m, int *C, int r, int H, int h)
{
	int p = threadIdx.x;
	int t = blockDim.x;
	int s = blockDim.x * blockIdx.x;

	S += s * r;
	C += s;

	__shared__ extern int S3[]; // |S3| = 2 * blockDim.x

	int *e = S3 + 0 * t;
	int *d = S3 + 1 * t;
	d[p] = C[p];
	cuda_SYNCTHREADS();

	int offt = r * blockIdx.x;
	if (r > m - offt) {
		r = m - offt;
	}
	int k = 1 << h;
	for (int j = 0; j < r; j++) {
		int y = S[p];
		int a = A[y];
		for (int i = 0; i < k; i++) {
			if (cuda_SYNCTHREADS_AND(a < i)) { // heuristic
				break;
			}
			e[p] = (int)(a == i);
			int count = cuda_SYNCTHREADS_COUNT(a == i);
			if (count > 0) {
				prefix_sum_exclusive(e, t);
				if (a == i) {
					T[d[i] + e[p]] = y;
				}
				cuda_SYNCTHREADS();
				if (p == i) {
					d[i] += count;
				}
			}
		}
		S += t;
	}
}

__global__ void counting_sort_phase3_c(int *A, int *S, int *T, int m, int *C, int r, int H, int h)
{
	int p = threadIdx.x;
	int t = blockDim.x;
	int s = blockDim.x * blockIdx.x;

	S += r * blockIdx.x << h;
	C += s;

	__shared__ extern int S3[]; // |S3| = 2 * blockDim.x

	int *q = S3 + 0 * t;
	int *d = S3 + 1 * t;
	q[p] = t;
	d[p] = C[p];
	cuda_SYNCTHREADS();

	int p_quot = p >> h;            // p / (1 << h)
	int p_remd = p & ~(MASK1 << h); // p % (1 << h)
	int x = p_quot * (m << h) + p_remd;

	int offt = r * blockIdx.x;
	if (r > m - offt) {
		r = m - offt;
	}
	int k = 1 << h;
	for (int j = 0; j < r; j++) {
		int y = S[x];
		int a = A[y] << H - h | p_quot;
		bool done = false;
		bool done_all = false;
		while (!done_all) {
			if (!done) {
				cuda_ATOMIC_MIN(&q[a], p);
			}
			cuda_SYNCTHREADS();
			if (q[a] == p) {
				q[a] = t;
				T[d[a]++] = y;
				done = true;
			}
			done_all = cuda_SYNCTHREADS_AND(done);
		}
		S += k;
	}
}

void counting_sort_init(cnt_args_t *args, int n, int t, int r, int flags)
{
	int m = n / t;
	int b = upperDiv(m, r);
	args->H = upperLog2(t);

	CATCH(cudaMalloc((void **)&args->Bh, sizeof(int) * t * b));
	CATCH(cudaMalloc((void **)&args->Ch, sizeof(int) * t * b));
	args->Bh_glb = NULL;
	args->sort_flags = flags;
	switch (args->sort_flags & PHASE_2) {
	case PHASE_2_HOR:
		CATCH(cudaMalloc((void **)&args->Bh_glb, sizeof(int) * t));
		CATCH(cudaMalloc((void **)&args->Ch_glb, sizeof(int) * t));
		CATCH(cudaMalloc((void **)&args->Q, sizeof(int) * t * b));
		break;
	case PHASE_2_VER:
		CATCH(cudaMalloc((void **)&args->Bv, sizeof(int) * t * b));
		CATCH(cudaMalloc((void **)&args->Cv, sizeof(int) * t * b));
		CATCH(cudaMalloc((void **)&args->Y, sizeof(int) * t * b));
		break;
	}
}

void counting_sort_term(cnt_args_t *args)
{
	CATCH(cudaFree(args->Bh));
	CATCH(cudaFree(args->Ch));
	switch (args->sort_flags & PHASE_2) {
	case PHASE_2_HOR:
		CATCH(cudaFree(args->Bh_glb));
		CATCH(cudaFree(args->Ch_glb));
		CATCH(cudaFree(args->Q));
		break;
	case PHASE_2_VER:
		CATCH(cudaFree(args->Bv));
		CATCH(cudaFree(args->Cv));
		CATCH(cudaFree(args->Y));
		break;
	}
}

void counting_sort(int *A, int *S, int *T, int n, int t, int r, int h, cnt_args_t args)
{
	int m = n / t;
	int b = upperDiv(m, r);
	int H = args.H;

	size_t s = sizeof(int) * t;

	switch (args.sort_flags & PHASE_2) {
	case PHASE_2_HOR: CATCH(cudaMemsetAsync(args.Bh_glb, 0, s)); break;
	}

	switch (args.sort_flags & PHASE_13) {
	case PHASE_13_A: counting_sort_phase1_a KERNEL_ARGS3(b, t, s) (A, S, m, args.Bh, args.Bh_glb, r, H, h); break;
	case PHASE_13_B: counting_sort_phase1_b KERNEL_ARGS3(b, t, 0) (A, S, m, args.Bh, args.Bh_glb, r, H, h); break;
	case PHASE_13_C: counting_sort_phase1_c KERNEL_ARGS3(b, t, s) (A, S, m, args.Bh, args.Bh_glb, r, H, h); break;
	}

	switch (args.sort_flags & PHASE_2) {
	case PHASE_2_HOR:
		counting_sort_phase2_hor1 KERNEL_ARGS3(1, t, s) (args.Bh_glb, args.Ch_glb);
		prefix_sum_segment_exclusive(args.Bh, args.Ch, t * b, t, args.Q);
		counting_sort_phase2_hor2 KERNEL_ARGS2(b, t) (args.Ch, args.Ch_glb);
		break;
	case PHASE_2_VER:
		counting_sort_phase2_ver1 KERNEL_ARGS2(b, t) (args.Bh, args.Bv);
		prefix_sum_element_exclusive(args.Bv, args.Cv, t * b, t / 2, r, args.Y);
		counting_sort_phase2_ver2 KERNEL_ARGS2(b, t) (args.Cv, args.Ch);
		break;
	}

	switch (args.sort_flags & PHASE_13) {
	case PHASE_13_A: counting_sort_phase3_a KERNEL_ARGS3(b, t, 2 * s) (A, S, T, m, args.Ch, r, H, h); break;
	case PHASE_13_B: counting_sort_phase3_b KERNEL_ARGS3(b, t, 2 * s) (A, S, T, m, args.Ch, r, H, h); break;
	case PHASE_13_C: counting_sort_phase3_c KERNEL_ARGS3(b, t, 2 * s) (A, S, T, m, args.Ch, r, H, h); break;
	}
}

/// <summary>
/// Computes the bitmap <paramref name="M"/> representing the matching elements of
/// <paramref name="X"/>.
/// <para>Assume <paramref name="X"/> has length <paramref name="n"/></para>
/// <para>Assume <paramref name="M"/> has length <paramref name="n"/></para>
/// </summary>
/// <param name="n">Length of <paramref name="X"/></param>
__global__ void decompose_bottom_up_step1(int *X, int *S, int *M, int n, int mask)
{
	int s0 = blockDim.x * blockIdx.x;
	int s1 = blockDim.x * gridDim.x;

	for (int id = s0 + threadIdx.x; id < n; id += s1) {
		if ((X[S[id]] & mask) == 0) {
			M[id] = +1;
		} else {
			M[id] = -1;
		}
	}
}

/// <summary>
/// Separates matching <paramref name="S"/>-mapped positions from non-matching ones into
/// <paramref name="T"/> based on the prefix-sum <paramref name="P"/> of the bitmap
/// representing the matches.
/// <para>Assume <paramref name="S"/> and <paramref name="T"/> are disjoint</para>
/// <para>Assume <paramref name="T"/> and <paramref name="P"/> are disjoint</para>
/// <para>Assume <paramref name="S"/> has length <paramref name="n"/>      </para>
/// <para>Assume <paramref name="T"/> has length <paramref name="n"/>      </para>
/// <para>Assume <paramref name="P"/> has length <paramref name="n"/>      </para>
/// </summary>
/// <param name="n">Length of <paramref name="S"/> and <paramref name="T"/></param>
__global__ void decompose_bottom_up_step3(int *S, int *T, int n, int *M, int *P)
{
	int y_pos = 0;
	int y_neg = n + P[n - 1] >> 1;

	int s0 = blockDim.x * blockIdx.x;
	int s1 = blockDim.x * gridDim.x;

	for (int id = s0 + threadIdx.x; id < n; id += s1) {
		if (M[id] == 1) {
			T[y_pos + (id + 1 + P[id] >> 1) - 1] = S[id];
		} else {
			T[y_neg + (id + 1 - P[id] >> 1) - 1] = S[id];
		}
	}
}

void decompose_bottom_up_init(dec_args_t *args, int n, int t, int r)
{
	CATCH(cudaMalloc((void **)&args->S, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&args->M, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&args->P, sizeof(int) * n));
	CATCH(cudaMalloc((void **)&args->Y, sizeof(int) * n));
}

void decompose_bottom_up_term(dec_args_t *args)
{
	CATCH(cudaFree(args->S));
	CATCH(cudaFree(args->M));
	CATCH(cudaFree(args->P));
	CATCH(cudaFree(args->Y));
}

void decompose_bottom_up(int *A, int *S, int *T, int n, int t, int r, int h, dec_args_t args)
{
	int m = n / t;
	int b = upperDiv(m, r);

	int *S_tmp = args.S;
	int *T_tmp = T;

	if (h % 2 == 0) {
		swap(&S_tmp, &T_tmp);
	}

	CATCH(cudaMemcpyAsync(S_tmp, S, sizeof(int) * n, cudaMemcpyDeviceToDevice));

	int mask = 1;
	for (int i = 0; i < h; i++) {
		decompose_bottom_up_step1 KERNEL_ARGS2(b, t) (A, S_tmp, args.M, n, mask);
		prefix_sum_element_inclusive(args.M, args.P, n, t / 2, r, args.Y);
		decompose_bottom_up_step3 KERNEL_ARGS2(b, t) (S_tmp, T_tmp, n, args.M, args.P);
		mask <<= 1;

		swap(&S_tmp, &T_tmp);
	}
}
