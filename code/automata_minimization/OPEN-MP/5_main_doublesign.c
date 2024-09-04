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

void chooseThreshold(int *count, int* th, int j, int k, int n){
    th[j] = (n % 2 == 0 ? n/2 -1 : n/2);
    for(int i = 1; i < k; i++){
        if(count[i] < (n/2)){
            continue;
        }else{
            int d1 = abs(n-count[i]);
            int d2 = abs(n-count[i-1]);
            th[j] = (d1 < d2 ? i : i-1);
            return;
        }
    }

}

void createZeroLevelSignature(int *signature, int* classes, int n){
    for (int i = 0; i < n; i++) {
        signature[i] = classes[i];
    }
}

void createFirstLevelSignature(int *signature, int* classes, int digit, dfa* A){
    for (int i = 0; i < A->n; i++) {
        signature[i] = classes[ A->Delta[i][digit] ];
    }
}


void createSecondLevelSignature(int *signature, int* classes, int digit, dfa* A){
    int ind = digit;
    assert(ind >= 0 && ind <= A->m*A->m -1);

    int a = ind / A->m;
    int b = ind % A->m;
    assert(a >= 0 && a < A->m && b >= 0 && b < A->m);
    for (int i = 0; i < A->n; i++) {
        int state = A->Delta[i][a];
        signature[i] = classes[ A->Delta[state][b] ];
    }
}

void createSignature(int *signature, int* classes, int digit, dfa* A, int firstLevel){
    if(firstLevel == 1){
        createFirstLevelSignature(signature, classes, digit, A);
    }else{
        createSecondLevelSignature(signature, classes, digit, A);
    }
}

void initializeWithSequence(int *arr, int n){
    for (int i = 0; i < n; i++) {
        arr[i] = i;
    }  
}


void doSortHalf2(int * out, int *in, int* signature, int* count, int th, int n){
    for (int i = n-1; i >= 0; i--) {
        int index = signature[in[i]];
        if(index <= th){ continue; }
        int outIndex = --count[index];
        assert(outIndex >= 0 && outIndex < n);
        out[outIndex] = in[i];
    }
}

void doSortHalf1(int * out, int *in, int* signature, int* count, int th, int n){
    for (int i = n-1; i >= 0; i--) {
        int index = signature[in[i]];
        if(index > th){ continue; }
        int outIndex = --count[index];
        assert(outIndex >= 0 && outIndex < n);
        out[outIndex] = in[i];
    }
}


dfa* minimize(dfa* A){

    int DIG;
    int firstLevelDig = 1 + A->m;
    int secondLevelDig = 1 + A->m*A->m;
    if(A->m == 2){ DIG = secondLevelDig; }
    else{ DIG = firstLevelDig; }

    // memory allocation
	int iter = 0,n_classes = 2,old_n_classes=0;
	int *classes = malloc(sizeof(int)*A->n);
    int *representative = malloc(sizeof(int)*A->n);
    int *comparison = malloc(sizeof(int)*A->n);
    int **count = malloc(sizeof(int*)*(DIG));
    int **sort = malloc(sizeof(int*)*(DIG+1));
    int **signature = malloc(sizeof(int*)*(DIG));
    int *th = malloc(sizeof(int)*(DIG));

    // global variables
    int *posArray = sort[DIG], k;
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
            sort[DIG] = malloc(sizeof(int)*A->n);
            for (int i = 0; i < DIG; i++) {
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
                if(iter == 0 && A->m == 2){ DIG = firstLevelDig; }
                else if(A->m == 2){ DIG = secondLevelDig; }

                old_n_classes = n_classes;
                k = n_classes; // max digit we can find in signature[][i]  
                createZeroLevelSignature(signature[0], classes, A->n);
            
            } // implicit barrier

            #pragma omp sections
            {
                // A(1)
                #pragma omp section
                {
                    memset(count[1], 0, sizeof(int)*k);
                    initializeWithSequence(sort[0], A->n);
                    createSignature(signature[1], classes, 0, A, DIG == firstLevelDig);                
                }

                // B(0)
                #pragma omp section
                {  
                    memset(count[0], 0, sizeof(int)*k);
                    doCount(count[0], signature[0], A->n);
                    prefixSum(count[0], k);
                    chooseThreshold(count[0], th, 0, k, A->n);
                }
            
            } // implicit barrier

            int j;
            for(j = 1; j <= DIG-2; j++){
                #pragma omp sections
                {
                    // A(j+1) signature creation
                    #pragma omp section
                    {
                        memset(count[j+1], 0, sizeof(int)*k);
                        createSignature(signature[j+1], classes, j-1+1, A, DIG == firstLevelDig);
                    }

                    // B(j) counting and summing
                    #pragma omp section
                    {
                        doCount(count[j], signature[j], A->n);
                        prefixSum(count[j], k);
                        chooseThreshold(count[j], th, j, k, A->n);  
                    }

                    #pragma omp section
                    {
                        assert(th[j-1] >= 0 && th[j-1] < A->n-1);
                        doSortHalf1(sort[j], sort[j-1], signature[j-1], count[j-1], th[j-1], A->n);
                    }

                    #pragma omp section
                    {
                        assert(th[j-1] >= 0 && th[j-1] < A->n-1);
                        doSortHalf2(sort[j], sort[j-1], signature[j-1], count[j-1], th[j-1], A->n);
                    }
                
                } // implicit barrier
            } // end for

            #pragma omp sections
            {
                //B(DIG-1)
                #pragma omp section
                {
                    doCount(count[DIG-1], signature[DIG-1], A->n);
                    prefixSum(count[DIG-1], k);
                    chooseThreshold(count[DIG-1], th, DIG-1, k, A->n);
                }

                #pragma omp section
                {
                    doSortHalf1(sort[DIG-1], sort[DIG-2], signature[DIG-2], count[DIG-2], th[DIG-2], A->n);
                }

                #pragma omp section
                {
                    doSortHalf2(sort[DIG-1], sort[DIG-2], signature[DIG-2], count[DIG-2], th[DIG-2], A->n);
                }
            
            } // implicit barrier

            #pragma omp sections
            {
                #pragma omp section
                {
                    doSortHalf1(sort[DIG], sort[DIG-1], signature[DIG-1], count[DIG-1], th[DIG-1], A->n);
                }

                #pragma omp section
                {
                    doSortHalf2(sort[DIG], sort[DIG-1], signature[DIG-1], count[DIG-1], th[DIG-1], A->n);
                    posArray = sort[DIG];
                }
            
            }// implicit barrier

            #pragma omp for schedule(static)
            for(int i = 1; i < A->n; i++){
                int idx = posArray[i];
                int idx_ = posArray[i-1];
                comparison[i] = compare(signature, idx, idx_, DIG);
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
    for(int i = B->m; i < DIG; i++){
        free(count[i]);
        free(signature[i]);
        free(sort[i]);   
    }
    free(sort[DIG]);
    free(representative);
    free(classes);
    free(signature);
    free(count);
    free(comparison);
    free(sort);
    free(th);
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