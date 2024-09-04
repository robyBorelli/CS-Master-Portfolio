#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <limits.h>
#include "dfa.h"


// multithreads constants
int THREADS = 1;
int printIterations=1;

// comparison constants
#define GT 1
#define EQ 0
#define LT -1

#ifndef _OPENMP
    int omp_get_thread_num(){
        return 0;
    }
    #pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

int compare (int* s1, int*s2, int d){
    if(s1 == s2) return EQ;
    for(int j = d-1; j >= 0; j--){
        if(s1[j] < s2[j]){
            return LT;
        }else if(s1[j] > s2[j]) {
            return GT;
        }
    }
    return EQ;
}

// O(N + n/N + k)
void parallelCountingSort(int **arr, int *in, int*out, int**count, int* globalCount, int n, int dig, int k){
    int N = THREADS;
    assert(N > 0);
    int chunkSize = n%N == 0 ? n/N : (n/N) + 1;
    memset(globalCount, 0, sizeof(int)*(k+1));
    for(int p = 0; p < N; p++){
        memset(count[p], 0, sizeof(int)*(k+1));
    }

    #pragma omp parallel num_threads(N) firstprivate(chunkSize)
    {
        // count: O(n/N)
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static, chunkSize)
        for (int i = 0; i < n; i++) {
            
            count[tid][arr[in[i]][dig]]++;
        }

        // sum: O(k/N * N) = O(k)
        #pragma omp for schedule(static)
        for(int j = 0; j <= k; j++){
            for(int p = 0; p < N; p++){
                globalCount[j] += count[p][j];
            }
        }

        #pragma omp single
        {
            // prefixSum: O(k) = O(k)
            for (int j = 1; j <= k; j++){ 
                globalCount[j] += globalCount[j - 1];
            }
        }

        // end: O(k/N * N) = O(k)
        #pragma omp for schedule(static)
        for(int j = 0; j <= k; j++){
            int end = globalCount[j];
            for(int p = N-1; p >= 0; p--){
                int cpj = count[p][j];
                count[p][j] = end;
                end -= cpj;
            } 
        }

        // sorting: O(n/N)
        int realChunkSize = chunkSize;
        if(chunkSize*(tid+1) > n){
            int calc = chunkSize*(tid+1)-n;
            chunkSize = calc>chunkSize ? 0 : chunkSize-calc;
        }
        #pragma omp barrier
        #pragma omp for schedule(static, realChunkSize) 
        for (int i = 0; i < n; i++) {
            assert(chunkSize!=0);
            int local_i; // from 0 to chunckSize-1
            local_i = i - (realChunkSize*tid);
            int inverse_i = (chunkSize-1)-local_i; // from cuncksize-1 to 0
            int ind = inverse_i + (realChunkSize*tid);
            int j = arr[in[ind]][dig];
            int outIndex = --count[tid][j];
            out[outIndex] = in[ind];
        }
    }
}

// O(N + n/N + k)*O(m)
int* parallelRadixSort(int **arr, int *input, int* output, int**count, int* globalCount, int n, int m, int k){
    int *in = input; int *out = output;
    for(int j = 0; j<m; j++){
        parallelCountingSort(arr, in, out, count, globalCount, n, j, k);

        int *temp = in;
        in = out;
        out = temp;
    }

    if(m%2==1){
        return output;
    }else{
        return input;
    }
}


dfa* minimize(dfa* A){

    // memory allcoation
	int iter = 0,n_classes = 2,old_n_classes=0;
	int *classes = malloc(sizeof(int)*A->n);
    int *representative = malloc(sizeof(int)*A->n);
    int *posArray = malloc(sizeof(int)*A->n);
    int *comparison = malloc(sizeof(int)*A->n);
    int** signatures = (int **)malloc(sizeof(int*)*A->n);
    int *output = (int*)malloc(sizeof(int)*A->n);
    int *globalCount = (int*)malloc(sizeof(int)*(A->n+1));
    int **count = (int**)malloc(sizeof(int*)*THREADS);
    for(int p = 0; p < THREADS; p++){
        count[p] = (int*)malloc(sizeof(int)*(A->n+1));
    }

    // initialization
    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < A->n; i++) {
            classes[i] = A->F[i];
            representative[A->F[i]] = i;
            signatures[i] = (int*)malloc(sizeof(int)*((A->m)+1));
        }                
    }


    // main minimization loop
    while (old_n_classes != n_classes) {
        old_n_classes = n_classes;
        int k = n_classes; // max digit we can find in signatures[i]
        {
            #pragma omp parallel num_threads(THREADS)
            {

                #pragma omp for schedule(static) nowait
                for (int i = 0; i < A->n; i++) {
                    signatures[i][0] = classes[i];
                    posArray[i] = i;
                }

                #pragma omp for schedule(static) collapse(2)
                for (int i = 0; i < A->n; i++) {
                    for(int j = 1; j < A->m+1; j++){
                        signatures[i][j] = classes[ A->Delta[i][j-1] ];
                    }
                }

            }

            int* sort = parallelRadixSort(signatures, posArray, output, count, globalCount, A->n, A->m+1, k);

            #pragma omp for schedule(static)
            for(int i = 1; i < A->n; i++){
                int idx = sort[i];
                int idx_ = sort[i-1];
                comparison[i] = compare(signatures[idx], signatures[idx_], A->m+1);
                assert(idx < A->n && idx >= 0 && idx_ < A->n && idx_ >= 0 &&
                   i < A->n && i >= 0 && i - 1 < A->n && i - 1 >= 0);
            } // implicit barrier


            int newN = 0;
            classes[ sort[0] ] = newN;
            for (int i = 1; i < A->n; i++) {
                int idx = sort[i];

                if (comparison[i] != EQ) { 
                    newN++; 
                }
                classes[idx] = newN;
                representative[newN] = idx;
            }

            n_classes = newN + 1;
            iter++;
        }
    }

    if(printIterations) printf("Iterations: %d\n", iter);


    // creating quotient dfa
    dfa *B = (dfa *) malloc(sizeof(dfa));
    B->n = n_classes;
    B->m = A->m;
    B->Sigma = malloc(sizeof(char) * B->m);
    B->Delta = (int **) malloc(sizeof(int *) * n_classes);
    B->F = (int *) malloc(sizeof(int) * n_classes);
    B->q0 = classes[A->q0];
    memset(B->F, 0, sizeof(int) * n_classes);

    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < B->m; i++) {
            B->Sigma[i] = A->Sigma[i];
        }
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n_classes; i++) {
            B->Delta[i] = (int *) malloc(sizeof(int) * B->m);
            int q = representative[i];
            assert(q >= 0 && q < A->n);
            B->F[i] = A->F[q];
            for (int j = 0; j < B->m; j++) {
                B->Delta[i][j] = classes[A->Delta[q][j]];
            }
        }

        // memory deallocation
        #pragma omp for schedule(static) nowait
        for(int i = 0; i < A->n; i++){
            free(signatures[i]);
        }
    }

    // memory deallocation
    free(representative);
    free(classes);
    free(signatures);
    free(posArray);
    free(comparison);
    for(int i = 0; i < THREADS; i++){free(count[i]);}
    free(count);
    free(globalCount);
    free(output);
	return B;
}

int main(int argc, char const *argv[])
{
    if (argc <= 4) {printf("error. correct invocation %s <n> <m> <threads> <language> [<output>]\n", argv[0]);return -1;}
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    THREADS = atoi(argv[3]);
    #ifndef _OPENMP
        THREADS=1;
    #endif
    int mode = atoi(argv[4]);
    int outputMode = 2;
    if(argc == 6){
        outputMode=atoi(argv[5]) %3;
    }
    dfa* A;
    if(mode >=2){
        A = buildRandomDfa(n,m);
    }else{
        A = buildDfA_abStar_n(n,m,mode);
    }
    printIterations = (outputMode>=1 ? 1 : 0);
    if(outputMode >= 2) {printDfa(A); printf("\n\n");}
    dfa *B = minimize(A);
    if(outputMode >= 1) printDfa(B);

    freeDfa(A);
    freeDfa(B);
    return 0;
}