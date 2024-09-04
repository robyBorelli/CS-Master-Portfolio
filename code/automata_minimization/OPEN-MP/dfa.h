#ifndef _OPENMP
    #define rand_r(r) rand()
#endif

// A = (Q, Sigma, F, Delta)
// |Q| = n
// |Sigma| = m
// Delta: matrix n x m
// Delta: Q x Sigma -> Q
struct dfa_s
{
    int n,m;
    int q0;
    char *Sigma;
    int **Delta;
    int *F;     // bitmap representation
    
}; typedef struct dfa_s dfa;

// mode = 0 -> (ab)*
// mode = 1 -> (ab^n)*
dfa *buildDfA_abStar_n(int n, int k, int mode){
    assert(n > 0 && (mode == 0 || mode == 1) && k>=2);
    dfa *A = malloc(sizeof(dfa));
    A->n = 3*n; A->m = k;
    A->q0 = 0;
    A->Sigma = malloc(sizeof(char)*A->m);
    for(int i = 0; i < k; i++){ A->Sigma[i] = 'a'+i; }
    A->F = malloc(sizeof(int)*A->n);
    memset(A->F, 0, A->n*sizeof(int));
    A->Delta = malloc(sizeof(int*)*A->n);
    for(int i = 0; i < A->n; i++){ A->Delta[i] = malloc(sizeof(int)*A->m);}
    A->F[0] = 1;
    for(int i = 0; i <= 2*n-2; i+=2){
        if(mode == 0){
            A->F[i] = 1; // se 1 riconosco (ab)*
            // se commentato riconosco (ab^n)*
        }
        A->Delta[i][0] = i+1;
        A->Delta[i][1] = 2*n;

        for(int hi = 2; hi<k; hi++){A->Delta[i][hi] = 2*n;}
    }
    for(int i = 1; i <= 2*n-3; i+=2){
        A->Delta[i][0] = 2*n;
        A->Delta[i][1] = i+1;

        for(int hi = 2; hi<k; hi++){A->Delta[i][hi] = 2*n;}
    }
    A->Delta[2*n-1][0] = 2*n;
    A->Delta[2*n-1][1] = 0;

    for(int hi = 2; hi<k; hi++){A->Delta[2*n-1][hi] = 2*n;}


    for(int i = 2*n; i <= 3*n-2; i++){
        A->Delta[i][0] = i;
        A->Delta[i][1] = i+1;

        for(int hi = 2; hi<k; hi++){A->Delta[i][hi] = i+1;}
    }

    A->Delta[3*n-1][0] = 3*n-1;
    A->Delta[3*n-1][1] = 3*n-1;

    for(int hi = 2; hi<k; hi++){A->Delta[3*n-1][hi] = 3*n-1;}

    return A;
}

dfa *buildRandomDfa(int n, int k){
    unsigned int seed = 1;
    #ifndef _OPENMP
    // prevents the warning of unused seed
    seed = seed *seed;
    #endif
    dfa *A = malloc(sizeof(dfa));
    A->n = n; A->m = k; A->q0 = 0;
    A->Sigma = malloc(sizeof(char)*A->m);
    for(int i = 0; i < k; i++){ A->Sigma[i] = 'a'+i; }
    A->F = malloc(sizeof(int)*A->n);
    memset(A->F, 0, A->n*sizeof(int));
    A->Delta = malloc(sizeof(int*)*A->n);
    for(int i = 0; i < A->n; i++){ 
        A->Delta[i] = malloc(sizeof(int)*A->m);
        for(int j = 0; j<A->m; j++){
            A->Delta[i][j] = -1;
        }
    }
    
    for(int i = 0; i < A->n; i++){
        A->F[i] = rand_r(&seed) % 2;
    }
    
    // random walk
    int i = 0, ind = 0;
    while(ind<3*A->n){
        int c = rand_r(&seed) % A->m;
        int j = 0;
        while (A->Delta[i][c] != -1 && j < 3*A->m){
            c = rand_r(&seed) % A->m;
            j++;
        }
        int next = (rand_r(&seed) % (A->n));
        A->Delta[i][c] =  next;
        i = next;
        ind++;
    }


    for(int i = 0; i < A->n; i++){
        for(int j = 0; j < A->m; j++){
            if (A->Delta[i][j] == -1){
                A->Delta[i][j] = rand_r(&seed) % (A->n);
            }
        }
    }

    return A;
}

void freeDfa(dfa* A){
    free(A->Sigma);
    free(A->F);
    for(int i = 0; i < A->n; i++){ free(A->Delta[i]);}
    free(A->Delta);
    free(A);
}

void printDfa(dfa* A){
    printf("Q: ");
    for(int i = 0; i < A->n; i++) {
        if (A->q0 == i) {
            printf(" ->");
        }
        if (A->F[i] == 1) {
            printf("[%d] ", i);
        } else {
            printf("%d ", i);
        }
    }
    printf("\nSigma: ");
    for(int i = 0; i < A->m; i++){
        printf("%c ", A->Sigma[i]);
    }
    printf("\nDelta: \n   ");
    for(int i = 0; i < A->m; i++){
        printf("%c ", A->Sigma[i]);
    }
    printf("\n");
    for(int i = 0; i < A->n; i++){
        printf("%d  ", i);
        for(int j = 0; j < A->m; j++){
            printf("%d ", A->Delta[i][j]);
        }
        printf("\n");
    }
}