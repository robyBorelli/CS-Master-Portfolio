#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.cuh"
#include "producer_consumer.cuh"

__global__ void chunk(int *src, int *dst, int n, int offt, int size, int h)
{
	int s0 = blockDim.x * blockIdx.x;
	int s1 = blockDim.x * gridDim.x;

	int mask = ~(MASK1 << size);
	for (int id = s0 + threadIdx.x; id < n; id += s1) {
		dst[id] |= (src[id] & mask) << offt;
		src[id] >>= size;
	}
}

void producer(cudaStream_t s, int *buf, int *X, volatile int *offt, volatile int *done, producer_t produce, void *p_args, int n, int t, int r)
{
	cudaEvent_t e_i, e_o;
	CATCH(cudaEventCreate(&e_o));
#pragma omp critical (producer_consumer)
	{
		CATCH(cudaEventRecord(e_o, s));
		CATCH(cudaMemsetAsync(buf, 0, sizeof(int) * n, s));
	}

	int m = n / t;
	int b = upperDiv(m, r);

	int temp_offt;
	while (true) {
		CATCH(cudaEventCreate(&e_i));
		CATCH(cudaStreamWaitEvent(0, e_o, 0));
		CATCH(cudaMemsetAsync(X, 0, sizeof(int) * n));
		// produce next chunk
		int h_p = produce(X, n, t, r, p_args);
		CATCH(cudaEventRecord(e_i, 0));
		CATCH(cudaEventDestroy(e_o));
		if (h_p == 0) {
#pragma omp critical (producer_consumer)
			{
				*done = 1;
			}
			CATCH(cudaEventDestroy(e_i));
			break;
		}
		CATCH(cudaEventCreate(&e_o));
		while (h_p > 0) {
#pragma omp critical (producer_consumer)
			{
				temp_offt = *offt;
			}
			if (temp_offt < BUF_SIZE) {
				int size = h_p;
#pragma omp critical (producer_consumer)
				{
					if (size > BUF_SIZE - *offt) {
						size = BUF_SIZE - *offt;
					}
					CATCH(cudaStreamWaitEvent(s, e_i, 0));
					// write buffer
					chunk KERNEL_ARGS4(b, t, 0, s) (X, buf, n, *offt, size, h_p);
					CATCH(cudaEventRecord(e_o, s));
					*offt += size;
				}
				h_p -= size;
			}
		}
		CATCH(cudaEventDestroy(e_i));
	}
}

void consumer(cudaStream_t s, int *buf, int *X, volatile int *offt, volatile int *done, consumer_t consume, void *c_args, int n, int t, int r, int h)
{
	cudaEvent_t e_i, e_o;
	CATCH(cudaEventCreate(&e_o));
	CATCH(cudaEventRecord(e_o, 0));

	int m = n / t;
	int b = upperDiv(m, r);

	int temp_done;
	int temp_offt;
	while (true) {
		CATCH(cudaEventCreate(&e_i));
		int h_c = 0;
		while (h_c < h) {
#pragma omp critical (producer_consumer)
			{
				temp_done = *done;
				temp_offt = *offt;
			}
			if (temp_done > 0 && temp_offt == 0) {
				break;
			}
			if (temp_offt > 0) {
				int size = h - h_c;
#pragma omp critical (producer_consumer)
				{
					if (size > *offt) {
						size = *offt;
					}
					CATCH(cudaStreamWaitEvent(s, e_o, 0));
					// read buffer
					chunk KERNEL_ARGS4(b, t, 0, s) (buf, X, n, h_c, size, *offt);
					CATCH(cudaEventRecord(e_i, s));
					*offt -= size;
				}
				h_c += size;
			}
		}
		CATCH(cudaEventDestroy(e_o));
		if (h_c == 0) {
			CATCH(cudaEventDestroy(e_i));
			break;
		}
		CATCH(cudaEventCreate(&e_o));
		CATCH(cudaStreamWaitEvent(0, e_i, 0));
		// consume next chunk
		consume(X, n, t, r, c_args, h_c);
		CATCH(cudaMemsetAsync(X, 0, sizeof(int) * n));
		CATCH(cudaEventRecord(e_o, 0));
		CATCH(cudaEventDestroy(e_i));
	}
}
