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

void radixSort(int**sig, int*input, int**count, int n, int m){
    int *output = malloc(sizeof(int)*n);
    int *in = input, *out = output;
    int i, outIndex;

    for (int dig = 0; dig< m+1; dig++)
    {
        for (i = n - 1; i >= 0; i--) {
            int index = sig[in[i]][dig];
            outIndex = --count[dig][index];
            out[outIndex] = in[i];
        }

        // scambio input e output
        int *temp = in;
        in = out;
        out = temp;
    }
    if((m+1)%2==1){
        for (i = 0; i < n; i++){
            input[i] = output[i];
        }
    }

    free(output);
}

dfa* minimize(dfa* A){

    // memory allcoation
	int iter = 0,n_classes = 2,old_n_classes=0;
	int *classes = malloc(sizeof(int)*A->n);
    int *representative = malloc(sizeof(int)*A->n);
    int *posArray = malloc(sizeof(int)*A->n);
    int *comparison = malloc(sizeof(int)*A->n);
    int **count = malloc(sizeof(int*)*((A->m)+1));
    int** signatures = (int **)malloc(sizeof(int*)*A->n);

    // initialization
    #pragma omp parallel num_threads(THREADS)
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < A->n; i++) {
            classes[i] = A->F[i];
            representative[A->F[i]] = i;
            signatures[i] = (int*)malloc(sizeof(int)*((A->m)+1));
        }
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < (A->m)+1; i++) {
            count[i] = malloc(sizeof(int)*A->n);
        }
    }


    // main minimization loop
    while (old_n_classes != n_classes) {
        old_n_classes = n_classes;
        int k = n_classes; // max digit we can find in signatures[i]
        {
            #pragma omp parallel num_threads(THREADS)
            {
                #pragma omp for schedule(static)
                for (int i = 0; i < (A->m)+1; i++) {
                    memset(count[i], 0, sizeof(int)*k);
                }

                #pragma omp for schedule(static) nowait
                for (int i = 0; i < A->n; i++) {
                    int index = (signatures[i][0] = classes[i]);
                    #pragma omp atomic update 
                    count[0][index]++;
                    posArray[i] = i;
                }

                #pragma omp for schedule(static) collapse(2)
                for (int i = 0; i < A->n; i++) {
                    // fillSignature
                    for(int j = 0; j < A-> m; j++){
                        int index = (signatures[i][j+1] = classes[ A->Delta[i][j] ]);
                        #pragma omp atomic update 
                        count[j+1][index]++;
                    }
                }

                // prefix sums
                #pragma omp for schedule(static) nowait
                for (int dig = 0; dig < (A->m)+1; dig++){
                    for (int i = 1; i < k; i++) {
                        count[dig][i] += count[dig][i - 1];
                    }
                }
            }

            radixSort(signatures, posArray, count, A->n, A->m);


            #pragma omp parallel for schedule(static)
            for(int i = 1; i < A->n; i++){
                int idx = posArray[i];
                int idx_ = posArray[i-1];
                comparison[i] = compare(signatures[idx], signatures[idx_], A->m+1);
                assert(idx < A->n && idx >= 0 && idx_ < A->n && idx_ >= 0 &&
                   i < A->n && i >= 0 && i - 1 < A->n && i - 1 >= 0);
            } // implicit barrier


            int newN = 0;
            classes[ posArray[0] ] = newN;
            for (int i = 1; i < A->n; i++) {
                int idx = posArray[i];

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

            // memory deallocation
            free(count[i]); 
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
    free(count[A->m]);
    free(count);
    free(comparison);
	return B;
}

int main(int argc, char const *argv[])
{
    if (argc <= 4) {printf("error. correct invocation %s <n> <m> <threads> <language> [<output>]\n", argv[0]);return -1;}
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    THREADS = atoi(argv[3]);
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