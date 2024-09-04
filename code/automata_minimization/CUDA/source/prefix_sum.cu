#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"
#include "prefix_sum.cuh"

/// <summary>
/// Computes the element-wise exclusive prefix-sum of <paramref name="X"/> in-place, based on
/// the two-pass scheme.
/// <para>Assume <paramref name="n"/> is a power of 2 less than or equal to 2 * blockDim.x</para>
/// </summary>
/// <param name="n">Length of <paramref name="X"/></param>
__device__ void prefix_sum_exclusive(int *X, int n)
{
	int p1 = (threadIdx.x << 1) + 1;
	int p2 = (threadIdx.x << 1) + 2;

	// first pass
	int l = 0;
	for (int m = n >> 1; m > 0; m >>= 1) {
		if (threadIdx.x < m) {
			int k1 = (p1 << l) - 1;
			int k2 = (p2 << l) - 1;
			X[k2] += X[k1];
		}
		l++;
		cuda_SYNCTHREADS();
	}

	// second pass
	if (threadIdx.x == 0) {
		X[n - 1] = 0;
	}

	for (int m = 1; m < n; m <<= 1) {
		l--;
		if (threadIdx.x < m) {
			int k1 = (p1 << l) - 1;
			int k2 = (p2 << l) - 1;
			int t = X[k2];
			X[k2] = X[k1] + t;
			X[k1] = t;
		}
		cuda_SYNCTHREADS();
	}
}

__global__ void prefix_sum_element_exclusive_pass1(int *X, int *Y, int m)
{
	int t = blockDim.x << 1;

	__shared__ extern int S[]; // |S| = t

	int s0 = t * blockIdx.x;
	int s1 = t * gridDim.x;

	X += s0;
	int p0 = threadIdx.x + 0 * blockDim.x;
	int p1 = threadIdx.x + 1 * blockDim.x;
	for (int j = blockIdx.x; j < m; j += gridDim.x) {
		S[p0] = X[p0];
		S[p1] = X[p1];

		cuda_SYNCTHREADS();
		prefix_sum_exclusive(S, t);

		if (threadIdx.x == blockDim.x - 1 && Y != NULL) {
			Y[j] = S[t - 1] + X[t - 1];
		}
		X[p0] = S[p0];
		X[p1] = S[p1];

		X += s1;
	}
}

__global__ void prefix_sum_element_exclusive_pass2(int *X, int *Y, int m)
{
	int t = blockDim.x << 1;

	int s0 = t * blockIdx.x;
	int s1 = t * gridDim.x;

	X += s0;
	int p0 = 0 * blockDim.x + threadIdx.x;
	int p1 = 1 * blockDim.x + threadIdx.x;
	for (int j = blockIdx.x; j < m; j += gridDim.x) {
		X[p0] += Y[j];
		X[p1] += Y[j];
		X += s1;
	}
}

void prefix_sum_element_exclusive_rec(int *X, int offt, int m, int t, int r, int *Y);

/// <summary>
/// Computes the element-wise exclusive prefix-sum <paramref name="P"/> of <paramref name="X"/>
/// based on the two-pass scheme.
/// <para>Assume <paramref name="P"/> has length <paramref name="n"/>            </para>
/// <para>Assume <paramref name="X"/> and <paramref name="P"/> are disjoint      </para>
/// <para>Assume <paramref name="n"/> is a multiple of 2 * <paramref name="t"/>  </para>
/// <para>Assume <paramref name="t"/> is a power of 2 greater than or equal to 32</para>
/// </summary>
/// <param name="n">Length of <paramref name="X"/></param>
/// <param name="t">Number of threads per block   </param>
/// <param name="r">Number of strides per block   </param>
void prefix_sum_element_exclusive(int *X, int *P, int n, int t, int r, int *Y)
{
	CATCH(cudaMemcpyAsync(P, X, sizeof(int) * n, cudaMemcpyDeviceToDevice));
	prefix_sum_element_exclusive_rec(P, 0, n / (2 * t), t, r, Y);
}

void prefix_sum_element_exclusive_rec(int *X, int offt, int m, int t, int r, int *Y)
{
	int b = upperDiv(m, r);
	int n = upperDiv(m, 2 * t);
	size_t s = sizeof(int) * 2 * t;

	prefix_sum_element_exclusive_pass1 KERNEL_ARGS3(b, t, s) (X, Y + offt, m);
	if (m > 1) {
		prefix_sum_element_exclusive_rec(Y + offt, offt + 2 * t * n, n, t, r, Y);
		prefix_sum_element_exclusive_pass2 KERNEL_ARGS2(b, t) (X, Y + offt, m);
	}
}

/// <summary>
/// Computes the element-wise inclusive prefix-sum <paramref name="P"/> of <paramref name="X"/>
/// based on the two-pass scheme.
/// <para>Assume <paramref name="P"/> has length <paramref name="n"/>            </para>
/// <para>Assume <paramref name="X"/> and <paramref name="P"/> are disjoint      </para>
/// <para>Assume <paramref name="n"/> is a multiple of 2 * <paramref name="t"/>  </para>
/// <para>Assume <paramref name="t"/> is a power of 2 greater than or equal to 32</para>
/// </summary>
/// <param name="n">Length of <paramref name="X"/></param>
/// <param name="t">Number of threads per block   </param>
/// <param name="r">Number of strides per block   </param>
void prefix_sum_element_inclusive(int *X, int *P, int n, int t, int r, int *Y)
{
	prefix_sum_element_exclusive(X, P, n, t, r, Y);
	add KERNEL_ARGS2(upperDiv(n, 2 * t * r), 2 * t) (X, P, n);
}

__global__ void prefix_sum_segment_inclusive_pass(int *P, int *Q, int s)
{
	int t = blockDim.x;
	int *P1 = P + t * blockIdx.x;
	int *P2 = P + t * (blockIdx.x + s);
	int *Q2 = Q + t * (blockIdx.x + s);
	Q2[threadIdx.x] = P1[threadIdx.x] + P2[threadIdx.x];
}

/// <summary>
/// Computes the segment-wise inclusive prefix-sum <paramref name="P"/> of <paramref name="X"/>
/// based on the naive scheme. Segments are assumed to be of size <paramref name="t"/>.
/// <para>Assume <paramref name="P"/> has length <paramref name="n"/>      </para>
/// <para>Assume <paramref name="X"/> and <paramref name="P"/> are disjoint</para>
/// <para>Assume <paramref name="n"/> is a multiple of <paramref name="t"/></para>
/// </summary>
/// <param name="n">Length of <paramref name="X"/></param>
/// <param name="t">Number of threads per block   </param>
void prefix_sum_segment_inclusive(int *X, int *P, int n, int t, int *Q)
{
	int m = n / t;

	CATCH(cudaMemcpyAsync(P, X, sizeof(int) * n, cudaMemcpyDeviceToDevice));
	CATCH(cudaMemcpyAsync(Q, X, sizeof(int) * n, cudaMemcpyDeviceToDevice));

	int l = upperLog2(m);
	if (l % 2 == 1) {
		swap(&P, &Q);
	}

	for (int s = 1 << l - 1; s > 0; s >>= 1) {
		prefix_sum_segment_inclusive_pass KERNEL_ARGS2(m - s, t) (P, Q, s);
		swap(&P, &Q);
	}
}

void prefix_sum_segment_exclusive(int *X, int *P, int n, int t, int *Q)
{
	CATCH(cudaMemsetAsync(P, 0, sizeof(int) * t));
	prefix_sum_segment_inclusive(X, P + t, n - t, t, Q);
}
