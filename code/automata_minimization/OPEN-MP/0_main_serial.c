#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "dfa.h"

// comparison constants
#define GT 1
#define EQ 0
#define LT -1

int printIterations=1;

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

void countingSort(int **sig, int *pos, int n, int dig, int max){
    int** output = malloc(sizeof(int*)*n);
    int* outputPos=malloc(sizeof(int)*n);
    int k = max+1;
    int i, outIndex;
    int* count = malloc(sizeof(int)*k);
    memset(count, 0, sizeof(int)*k);

    for (i = 0; i < n; i++){
        int index = sig[i][dig];
        count[index]++;
    }

    for (i = 1; i < k; i++) {
        count[i] += count[i - 1];
    }

    for (i = n - 1; i >= 0; i--) {
        int index = sig[i][dig];
        outIndex = --count[index];
        output[outIndex] = sig[i];
        outputPos[outIndex] = pos[i];

    }

    for (i = 0; i < n; i++){
        sig[i] = output[i];
        pos[i] = outputPos[i];
    }

    free(count);
    free(output);
    free(outputPos);
}

void radixSort(int **sig, int*pos, int n, int m, int max){
    for (int dig = 0; dig<= m; dig++){
        countingSort(sig, pos, n, dig, max);
    }
}


int* createSignature(int q, int *classes, dfa *A, int n_classes){
    assert(A != NULL && classes != NULL);
    int* res = (int*)malloc(sizeof(int)*((A->m)+1));
    res[0] = classes[q];
    for(int j = 0; j < A-> m; j++){
        res[j+1] = classes[ A->Delta[q][j] ];
    }
    return res;
}

int refineClasses(int *classes, int* representative, int n_classes, dfa* A){
	int** signatures = malloc(sizeof(int*)*A->n);
	int* posArray = malloc(sizeof(int)*A->n);

	for(int i = 0; i < A->n; i++){
		signatures[i] = createSignature(i,classes, A, n_classes);
        posArray[i] = i;
	}

	radixSort(signatures,posArray,A->n, A->m, n_classes);

    int newN = 0, i, idx = posArray[0];
    classes[idx] = newN;
    for(i = 1; i < A->n; i++){
        idx = posArray[i];
        assert( idx < A->n && idx >= 0 &&
            i < A->n && i >= 0 && i-1 < A->n && i-1 >= 0);
        if(compare(signatures[i], signatures[i-1], A->m+1) != EQ ){newN++;}
        classes[idx] = newN;
        representative[newN] = idx;
        free(signatures[i-1]);
    }
    free(signatures[i-1]);
    free(signatures);
    free(posArray);
	return newN+1;
}

dfa* minimize(dfa* A){
	int iter = 0;
	int n_classes = 2;
	int old_n_classes=0;
	int *classes = malloc(sizeof(int)*A->n);
    int *representative = malloc(sizeof(int)*A->n);
	for(int i = 0; i < A->n; i++){classes[i] = A->F[i]; representative[A->F[i]]=i;}


	while(old_n_classes != n_classes) {
		old_n_classes = n_classes;
		n_classes = refineClasses(classes,representative,n_classes,A);
        iter++;
	}

    if(printIterations) printf("Iterations: %d\n",iter);


	dfa* B = (dfa*)malloc(sizeof(dfa));
	B->n = n_classes;
	B->m = A->m;
    B->Sigma = malloc(sizeof(char)*B->m);
	B->Delta = (int**)malloc(sizeof(int*)*n_classes);
    B->F = (int *)malloc(sizeof(int)*n_classes);
    memset(B->F, 0, sizeof(int)*n_classes);
    for(int i = 0; i < B->m; i++){
        B->Sigma[i] = A->Sigma[i];
    }
    B->q0 = classes[A->q0];
	for(int i = 0 ; i < n_classes; i++){
        B->Delta[i] = (int *)malloc(sizeof(int)*B->m);
        int q = representative[i];
		assert(q >= 0 && q<A->n);
        B->F[i] = A->F[q];
		for(int j = 0; j < B->m; j++){
			B->Delta[i][j] = classes[A->Delta[q][j]];
		}
	}

    free(representative);
    free(classes);
	return B;
}

int main(int argc, char const *argv[])
{
    if (argc <= 4) {printf("error. correct invocation %s <n> <m> <threads> <language> [<output>]\n", argv[0]);return -1;}
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
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