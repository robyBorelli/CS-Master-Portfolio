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
#define UNDEF 0
#define LT 1
#define EQ 2
#define GT 3

#ifndef _OPENMP
    #pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

int abs(int x){ return (x>=0?x:-x); }

int compare (int** sig, int s1 , int s2, int d){
    if(s1 == s2) return EQ;
    for(int j = d-1; j >= 0; j--){
        if(sig[j][s1] < sig[j][s2]){
            return LT;
        }else if(sig[j][s1] > sig[j][s2]) {
            return GT;
        }
    }
    return EQ;
}

void doCount(int *count, int *signature, int n){
    for(int i = 0; i < n; i++){
        int index = signature[i];
        count[index]++;
    }
}

void prefixSum(int *arr, int n){
    for (int i = 1; i < n; i++) {
        arr[i] += arr[i - 1];
    }  
}

void createFirstDigitSignature(int *signature, int* classes, int n){
    for (int i = 0; i < n; i++) {
        signature[i] = classes[i];
    }
}

void createSignature(int *signature, int* classes, int digit, dfa* A){
    for (int i = 0; i < A->n; i++) {
        signature[i] = classes[ A->Delta[i][digit] ];
    } 
}

void initializeWithSequence(int *arr, int n){
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }  
}


void doSort(int * out, int *in, int* signature, int* count, int n){
    for (int i = n-1; i >= 0; i--) {
        int index = signature[in[i]];
        int outIndex = --count[index];
        assert(outIndex >= 0 && outIndex < n);
        out[outIndex] = in[i];
    }
}

dfa* minimize(dfa* A){

    // memory allcoation
	int iter = 0,n_classes = 2,old_n_classes=0;
	int *classes = malloc(sizeof(int)*A->n);
    int *representative = malloc(sizeof(int)*A->n);
    int *comparison = malloc(sizeof(int)*A->n);
    int **count = malloc(sizeof(int*)*((A->m)+1));
    int **sort = malloc(sizeof(int*)*((A->m)+2));
    int **signature = malloc(sizeof(int*)*((A->m)+1));

    // global variables
    int *posArray = sort[(A->m)+1], k;
    dfa *B;
    
    #pragma omp parallel num_threads(THREADS)
    {
        // 1) initialization
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < A->n; i++) {
            classes[i] = A->F[i];
            representative[A->F[i]] = i;
        }

        #pragma omp master
        {
            sort[(A->m)+1] = malloc(sizeof(int)*A->n);
            for (int i = 0; i < (A->m)+1; i++) {
                count[i] = malloc(sizeof(int)*A->n);
                sort[i] = malloc(sizeof(int)*A->n);
                signature[i] = malloc(sizeof(int)*A->n);
            }
        }
    }

    // 2) main minimization loop
    while (old_n_classes != n_classes) {
        
        #pragma omp parallel num_threads(THREADS)
        {
            // A(0)
            #pragma omp single
            {
                old_n_classes = n_classes;
                k = n_classes; // max digit we can find in signature[][i]  
                createFirstDigitSignature(signature[0], classes, A->n);
            
            } // implicit barrier

            #pragma omp sections
            {
                // A(1)
                #pragma omp section
                {
                    memset(count[1], 0, sizeof(int)*k);
                    initializeWithSequence(sort[0], A->n);
                    createSignature(signature[1], classes, 0, A);                
                }

                // B(0)
                #pragma omp section
                {  
                    memset(count[0], 0, sizeof(int)*k);
                    doCount(count[0], signature[0], A->n);
                    prefixSum(count[0], k);
                }
            
            } // implicit barrier

            int j;
            for(j = 1; j <= (A->m)-1; j++){
                #pragma omp sections
                {
                    // A(j+1) signature creation
                    #pragma omp section
                    {
                        memset(count[j+1], 0, sizeof(int)*k);
                        createSignature(signature[j+1], classes, j-1+1, A);
                    }

                    // B(j) counting and summing
                    #pragma omp section
                    {
                        doCount(count[j], signature[j], A->n);
                        prefixSum(count[j], k);  
                    }

                    #pragma omp section
                    {
                        doSort(sort[j], sort[j-1], signature[j-1], count[j-1], A->n);
                    }
                
                } // implicit barrier
            } // end for

            #pragma omp sections
            {
                //B(A->m)
                #pragma omp section
                {
                    doCount(count[A->m], signature[A->m], A->n);
                    prefixSum(count[A->m], k);
                }

                #pragma omp section
                {
                    doSort(sort[(A->m)], sort[(A->m)-1], signature[(A->m)-1], count[(A->m)-1], A->n);
                }

            } // implicit barrier

            #pragma omp single
            {
                doSort(sort[(A->m)+1], sort[(A->m)], signature[(A->m)], count[(A->m)], A->n);
                posArray = sort[(A->m)+1];
            
            }// implicit barrier

            #pragma omp for schedule(static)
            for(int i = 1; i < A->n; i++){
                int idx = posArray[i];
                int idx_ = posArray[i-1];
                comparison[i] = compare(signature, idx, idx_, (A->m)+1);
                assert(idx < A->n && idx >= 0 && idx_ < A->n && idx_ >= 0 &&
                   i < A->n && i >= 0 && i - 1 < A->n && i - 1 >= 0);
            } // implicit barrier

            #pragma omp master
            {
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
            free(sort[i]); 
            free(signature[i]);
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
    } // end of omp parallel

    // memory deallocation
    free(count[A->m]);
    free(signature[A->m]);
    free(sort[A->m]);
    free(sort[A->m+1]);
    free(representative);
    free(classes);
    free(signature);
    free(count);
    free(comparison);
    free(sort);
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