
void swap(int **A, int **B);

__device__ void prefix_sum_exclusive(int *X, int n);

void prefix_sum_element_exclusive(int *X, int *P, int n, int t, int r, int *Y);
void prefix_sum_element_inclusive(int *X, int *P, int n, int t, int r, int *Y);
void prefix_sum_segment_exclusive(int *X, int *P, int n, int t, int *Q);
void prefix_sum_segment_inclusive(int *X, int *P, int n, int t, int *Q);
