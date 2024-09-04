
// trick to work around Visual Studio's Intellisense
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#define cuda_SYNCTHREADS() __syncthreads()
#define cuda_SYNCTHREADS_AND(expr) __syncthreads_and(expr)
#define cuda_SYNCTHREADS_COUNT(expr) __syncthreads_count(expr)
#define cuda_ATOMIC_ADD(addr, val) atomicAdd(addr, val)
#define cuda_ATOMIC_MIN(addr, val) atomicMin(addr, val)
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#define cuda_SYNCTHREADS()
#define cuda_SYNCTHREADS_AND(expr) false
#define cuda_SYNCTHREADS_COUNT(expr) 0
#define cuda_ATOMIC_ADD(addr, val) 0
#define cuda_ATOMIC_MIN(addr, val) 0
#endif


extern int _global_help;
extern int _global_verb;


int maxThdPerBlk(cudaDeviceProp prop);
int maxBlkPerGrd(cudaDeviceProp prop);
int maxThdPerSM(cudaDeviceProp prop);
int maxBlkPerSM(cudaDeviceProp prop);

#define KERNEL_ARGS2_1(N, block) KERNEL_ARGS2(upperDiv(N, block), block)
#define KERNEL_ARGS3_1(N, block, size) KERNEL_ARGS3(upperDiv(N, block), block, block * size)

void handle_error(const char *msg, ...);
void handle_error(cudaError_t e, const char *file, int line);

#define CATCH(e) handle_error(e, __FILE__, __LINE__)


#define MASK1 ~0

__host__ __device__ int lowerLog2(int a);
__host__ __device__ int upperLog2(int a);
__host__ __device__ int upperDiv(int a, int b);

void swap(int **A, int **B);


__global__ void constant(int *S, int n, int k);
__global__ void identity(int *S, int n);

__global__ void add(int *X, int *Y, int n);
__global__ void permute_src(int *X, int *Y, int *S, int n);
__global__ void permute_dst(int *X, int *Y, int *S, int n);
