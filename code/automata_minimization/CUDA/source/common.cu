#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"

int _global_help = false;
int _global_verb = false;


int maxThdPerBlk(cudaDeviceProp prop)
{
	return prop.maxThreadsDim[0];
}

int maxBlkPerGrd(cudaDeviceProp prop)
{
	return prop.maxGridSize[0];
}

int maxThdPerSM(cudaDeviceProp prop)
{
	return prop.maxThreadsPerMultiProcessor;
}

int maxBlkPerSM(cudaDeviceProp prop)
{
	// see CUDA programming guide
	if (prop.major >= 9) {
		return 32; // not specified for cc > 9.0, assume 32
	}
	if (prop.major == 8) {
		if (prop.minor >= 9) {
			return 24;
		}
		if (prop.minor >= 6) {
			return 16;
		}
		return 32;
	}
	if (prop.major == 7) {
		if (prop.minor >= 5) {
			return 16;
		}
		return 32;
	}
	if (prop.major >= 5) {
		return 32;
	}
	if (prop.major >= 3) {
		return 16;
	}
	return 8;
}


void handle_error(const char *msg, ...)
{
	va_list ap;
	va_start(ap, msg);
	vfprintf(stderr, msg, ap);
	va_end(ap);
	exit(EXIT_FAILURE);
}

void handle_error(cudaError_t e, const char *file, int line)
{
	if (e != cudaSuccess) {
		fprintf(stderr, "Error: %s in %s at line %d\n", cudaGetErrorString(e), file, line);
		exit(EXIT_FAILURE);
	}
}


__host__ __device__ int upperDiv(int a, int b)
{
	return (a - 1) / b + 1;
}

__host__ __device__ int lowerLog2(int a)
{
	int l = -1;
	for (int b = a; b > 0; b >>= 1) {
		l++;
	}
	return l;
}

__host__ __device__ int upperLog2(int a)
{
	return lowerLog2(a - 1) + 1;
}

void swap(int **A, int **B)
{
	int *A_ = *A;
	int *B_ = *B;
	*A = B_;
	*B = A_;
}

__global__ void constant(int *S, int n, int k)
{
	int s0 = blockDim.x * blockIdx.x;
	int s1 = blockDim.x * gridDim.x;

	for (int id = s0 + threadIdx.x; id < n; id += s1) {
		S[id] = k;
	}
}

__global__ void identity(int *S, int n)
{
	int s0 = blockDim.x * blockIdx.x;
	int s1 = blockDim.x * gridDim.x;

	for (int id = s0 + threadIdx.x; id < n; id += s1) {
		S[id] = id;
	}
}


__global__ void add(int *X, int *Y, int n)
{
	int s0 = blockDim.x * blockIdx.x;
	int s1 = blockDim.x * gridDim.x;

	for (int id = s0 + threadIdx.x; id < n; id += s1) {
		Y[id] += X[id];
	}
}

__global__ void permute_src(int *X, int *Y, int *S, int n)
{
	int s0 = blockDim.x * blockIdx.x;
	int s1 = blockDim.x * gridDim.x;

	for (int id = s0 + threadIdx.x; id < n; id += s1) {
		Y[id] = X[S[id]];
	}
}

__global__ void permute_dst(int *X, int *Y, int *S, int n)
{
	int s0 = blockDim.x * blockIdx.x;
	int s1 = blockDim.x * gridDim.x;

	for (int id = s0 + threadIdx.x; id < n; id += s1) {
		Y[S[id]] = X[id];
	}
}
