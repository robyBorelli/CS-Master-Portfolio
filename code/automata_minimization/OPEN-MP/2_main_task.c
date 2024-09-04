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
#define grainSize 1

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

dfa* minimize(dfa* A){

    // memory allcoation
	int iter = 0,n_classes = 2,old_n_classes=0;
	int *classes = malloc(sizeof(int)*A->n);
    int *representative = malloc(sizeof(int)*A->n);
    int *posArray = malloc(sizeof(int)*A->n);
    int *comparison = malloc(sizeof(int)*A->n);
    int *auxiliary = malloc(sizeof(int)*A->n);
    int **count = malloc(sizeof(int*)*((A->m)+1));
    int** signatures = (int **)malloc(sizeof(int*)*A->n);

    // dependencies
    int *depSig = malloc(sizeof(int)*(A->m+1));
    int *depCount = malloc(sizeof(int)*(A->m+1));
    int *depSort = malloc(sizeof(int)*(A->m+1));
    int *depSum = malloc(sizeof(int)*(A->m+1));

    // global variables
    int *in, *out, k;
    dfa *B;
    
    #pragma omp parallel num_threads(THREADS)
    {
        // 1) initialization
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < A->n; i++) {
            classes[i] = A->F[i];
            representative[A->F[i]] = i;
            signatures[i] = (int*)malloc(sizeof(int)*((A->m)+1));
        }

        #pragma omp master
        {
            for (int i = 0; i < (A->m)+1; i++) {
                count[i] = malloc(sizeof(int)*A->n);
            }
        }
    }

    // 2) main minimization loop
    while (old_n_classes != n_classes) {
        #pragma omp parallel num_threads(THREADS)
        {
            // one unique thread defines tasks
            #pragma omp single
            {
                old_n_classes = n_classes;
                k = n_classes; // max digit we can find in signatures[i]
                in = posArray; out = auxiliary;
                memset(auxiliary, 0, sizeof(int)*A->n);
                memset(depSig, 0, sizeof(int)*(A->m+1));
                memset(depSum, 0, sizeof(int)*(A->m+1));
                memset(depCount, 0, sizeof(int)*(A->m+1));
                memset(depSort, 0, sizeof(int)*(A->m+1));


                // --DEFINING TASKS FOR FIRST DIGIT--

                // a0) signature creation task (0-th digit)
                #pragma omp task depend(out:depSig[0])
                {
                    assert(depSig[0] == 0);
                    memset(count[0], 0, sizeof(int)*k);
                    for (int i = 0; i < A->n; i++) {
                        signatures[i][0] = classes[i];
                        in[i] = i;
                    }
                    depSig[0] = 1;

                }

                // b0 joint c0) counting and summing
                #pragma omp task depend(in:depSig[0]) depend(out:depSum[0])
                {
                    assert(depSum[0] == 0);
                    assert(depSig[0] == 1);
                    // b0) counting task (0-th digit)
                    // depend(in:depSig[0]) depend(out:depCount[0])
                    for(int i = 0; i < A->n; i++){
                        int index = signatures[i][0];
                        count[0][index]++;
                    }

                    // c0) prefix-summing task (0-th digit)
                    // depend(in:depCount[0]) depend(out:depSum[0])
                    for (int i = 1; i < k; i++) {
                        count[0][i] += count[0][i - 1];
                    }
                    depSum[0] = 1;
                }


                // d0) sorting task (0-th digit)
                #pragma omp task depend(in:depSum[0]) depend(out:depSort[0])
                {
                    assert(depSort[0] == 0);
                    assert(depSum[0] == 1);
                    for (int i = A->n - 1; i >= 0; i--) {
                        int index = signatures[in[i]][0];
                        int outIndex = --count[0][index];
                        assert(outIndex >= 0 && outIndex < A->n);
                        out[outIndex] = in[i];
                    }
                    
                    // scambio input e output
                    int *temp = in;
                    in = out;
                    out = temp;
                    depSort[0] = 1;
                }

                // --DEFINING TASKS FOR j-TH DIGIT (j>1)--
                int depJ = 1;
                for(int j = 1; j < (A->m)+1; j+=grainSize){
                    // aj) signature creation task (j-th digit)
                    #pragma omp task firstprivate(j, depJ) depend(out:depSig[depJ])
                    {
                        assert(depSig[depJ] == 0);
                        for(int l = 0; l < grainSize; l++){
                            if(j+l >= (A->m)+1){ break; }
                            memset(count[j+l], 0, sizeof(int)*k);
                            for(int i = 0; i < A->n; i++){
                                signatures[i][j+l] = classes[ A->Delta[i][j+l-1] ];
                            }        
                        }
                        depSig[depJ] = 1;

                    }

                    // bj joint cj) counting and summing
                    #pragma omp task firstprivate(j, depJ) depend(in:depSig[depJ]) depend(out:depSum[depJ])
                    {
                        assert(depSum[depJ] == 0);
                        assert(depSig[depJ] == 1);
                        for(int l = 0; l < grainSize; l++){
                            if(j+l >= (A->m)+1){ break; }
                            // bj) counting task (j-th digit)
                            // depend(in:depSig[depJ]) depend(out:depCount[depJ])
                            for(int i = 0; i < A->n; i++){
                                int index = signatures[i][j+l];
                                count[j+l][index]++;
                            }
                            // cj) prefix-summing task (j-th digit)
                            // depend(in:depCount[depJ]) depend(out:depSum[depJ])
                            for (int i = 1; i < k; i++) {
                                count[j+l][i] += count[j+l][i - 1];
                            }  
                        }
                        depSum[depJ] = 1;
                        
                    }

                    // dieci undici duemilaventitre, sofia è qui con me! TI AMO AMORE
                    // dj) sorting task (j-th digit)
                    #pragma omp task firstprivate(j, depJ) depend(in:depSum[depJ]) depend(in:depSort[depJ-1]) depend(out:depSort[depJ])
                    {
                        assert(depSort[depJ] == 0);
                        assert(depSort[depJ-1] == 1);
                        assert(depSum[depJ] == 1);
                        for(int l = 0; l < grainSize; l++){
                            if(j+l >= (A->m)+1){ break; }
                            for (int i = A->n - 1; i >= 0; i--) {
                                int index = signatures[in[i]][j+l];
                                int outIndex = --count[j+l][index];
                                assert(outIndex >= 0 && outIndex < A->n);
                                out[outIndex] = in[i];
                            }

                            // scambio input e output
                            int *temp = in;
                            in = out;
                            out = temp;
                        }
                        depSort[depJ] = 1;
                    }
                    depJ++;
                }

                // e) swap task
                #pragma omp task firstprivate(depJ) depend(in:depSort[depJ-1])
                {
                    if(depJ==1){
                        int *temp = in;
                        in = out;
                        out = temp;   
                    }
                }

            } // end of omp single
              // ... there is an implicit barrier which causes tasks to be executed 
              // tasks are executed in parallel!


            #pragma omp for schedule(static)
            for(int i = 1; i < A->n; i++){
                int idx = in[i];
                int idx_ = in[i-1];
                comparison[i] = compare(signatures[idx], signatures[idx_], A->m+1);
                assert(idx < A->n && idx >= 0 && idx_ < A->n && idx_ >= 0 &&
                   i < A->n && i >= 0 && i - 1 < A->n && i - 1 >= 0);
            }

            #pragma omp master
            {
                int newN = 0;
                classes[ in[0] ] = newN;
                for (int i = 1; i < A->n; i++) {
                    int idx = in[i];

                    if (comparison[i] != EQ) { 
                        newN++; 
                    }
                    classes[idx] = newN;
                    representative[newN] = idx;
                }

                n_classes = newN + 1;
                iter++;


            } // end of omp master 

        } // end of omp parallel (implicit barrier)

    } // end of while

    #pragma omp parallel num_threads(THREADS)
    {
        
        #pragma omp single
        {
            // debug information
            if(printIterations) printf("Iterations: %d\n",iter);

            // 3) creating quotient dfa
            B = (dfa *) malloc(sizeof(dfa));
            B->n = n_classes;
            B->m = A->m;
            B->Sigma = malloc(sizeof(char) * B->m);
            B->Delta = (int **) malloc(sizeof(int *) * n_classes);
            B->F = (int *) malloc(sizeof(int) * n_classes);
            B->q0 = classes[A->q0];
            memset(B->F, 0, sizeof(int) * n_classes);

        } // end of omp single (implicit barrier)


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

    } // end of omp parallel

    // memory deallocation
    free(representative);
    free(classes);
    free(signatures);
    free(posArray);
    free(count[A->m]);
    free(count);
    free(auxiliary);
    free(comparison);
    free(depSig);
    free(depCount);
    free(depSort);
    free(depSum);
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