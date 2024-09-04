
// assume two's complement representation for bit-shift operations
#define BUF_SIZE sizeof(int) * 8 - 1

typedef int(*producer_t)(int *X, int n, int t, int r, void *args);
typedef int(*consumer_t)(int *X, int n, int t, int r, void *args, int h);

void producer(cudaStream_t s, int *buf, int *X, volatile int *offt, volatile int *done, producer_t produce, void *p_args, int n, int t, int r);
void consumer(cudaStream_t s, int *buf, int *X, volatile int *offt, volatile int *done, consumer_t consume, void *c_args, int n, int t, int r, int h);
