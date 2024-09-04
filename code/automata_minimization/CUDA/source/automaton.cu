#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"

#include "common.cuh"
#include "automaton.cuh"

void DFA_init(automaton_t *A, int n, int m)
{
	A->n = n;
	A->m = m;
	int *mem;
	CATCH(cudaMallocHost((void **)&mem, sizeof(int) * n * (m + 2)));
	A->Delta = (int **)malloc(sizeof(int *) * m);
	for (int s = 0; s < m; s++) {
		A->Delta[s] = mem + n * s;
	}
	A->Q = mem + n * m;
	A->F = mem + n * (m + 1);
	for (int q = 0; q < n; q++) {
		A->Q[q] = q;
	}
}

// mode = 0 -> ((ab)^n)*
// mode = 1 -> (ab)*
void DFA_init_mode(automaton_t *A, int n, int m, int mode)
{
	DFA_init(A, 3 * n, m);

	A->q0 = 0;

	// top component
	for (int q = 0; q < 2 * n - 1; q += 2) {
		A->Delta[0][q] = q + 1;
		A->Delta[1][q] = 2 * n;
		for (int s = 2; s < m; s++) { A->Delta[s][q] = 2 * n; }

		A->F[q] = mode;
	}
	A->F[0] = 1;

	for (int q = 1; q < 2 * n - 1; q += 2) {
		A->Delta[0][q] = 2 * n;
		A->Delta[1][q] = q + 1;
		for (int s = 2; s < m; s++) { A->Delta[s][q] = 2 * n; }

		A->F[q] = 0;
	}

	A->Delta[0][2 * n - 1] = 2 * n;
	A->Delta[1][2 * n - 1] = 0;
	for (int s = 2; s < m; s++) { A->Delta[s][2 * n - 1] = 2 * n; }

	A->F[2 * n - 1] = 0;

	// bot component
	for (int q = 2 * n; q < 3 * n - 1; q++) {
		A->Delta[0][q] = q;
		A->Delta[1][q] = q + 1;
		for (int s = 2; s < m; s++) { A->Delta[s][q] = q + 1; }

		A->F[q] = 0;
	}

	for (int s = 0; s < m; s++) { A->Delta[s][3 * n - 1] = 3 * n - 1; }

	A->F[3 * n - 1] = 0;
}

int my_random()
{
	int mask = ~(MASK1 << 15); // RAND_MAX >= 2^15 - 1
	int a = rand() & mask;
	int b = rand() & mask;
	return a << 15 | b;
}

void fisher_yates(int *P, int n)
{
	for (int i = 0; i < n; i++) {
		P[i] = i;
	}
	for (int i = 0; i < n - 1; i++) {
		int j = i + my_random() % (n - i);
		int t = P[i];
		P[i] = P[j];
		P[j] = t;
	}
}

void DFA_init_rand(automaton_t *A, int n, int m)
{
	DFA_init(A, n, m);

	srand((unsigned int)time(NULL));

	// the DFA must be accessible
	// 1. clear all the edges
	for (int q = 0; q < n; q++) {
		for (int s = 0; s < m; s++) {
			A->Delta[s][q] = -1;
		}
	}

	// 2. generate a spanning tree at random
	int *P = (int *)malloc(sizeof(int) * n);
	fisher_yates(P, n);

	A->q0 = P[0];
	for (int q = 1; q < n; q++) {
		int p = my_random() % q;
		int s = my_random() % m;
		A->Delta[s][P[p]] = P[q];
	}

	free(P);

	// 3. generate all the other edges at random
	for (int q = 0; q < n; q++) {
		for (int s = 0; s < m; s++) {
			if (A->Delta[s][q] < 0) {
				A->Delta[s][q] = my_random() % n;
			}
		}
	}

	for (int q = 0; q < n; q++) {
		A->F[q] = my_random() % 2;
	}
}

void DFA_term(automaton_t *A)
{
	CATCH(cudaFreeHost(A->Delta[0]));
	free(A->Delta);
}

void DFA_encode(automaton_t *A)
{
	int L = (int)log10(A->n) + 1;

	printf("n=%d\n", A->n);
	printf("m=%d\n", A->m);

	int F = 0;
	for (int q = 0; q < A->n; q++) {
		if (A->F[q] == 1) {
			F++;
		}
	}

	for (int q = 0; q < A->n; q++) {
		for (int s = 0; s < A->m; s++) {
			printf("%*d %c %*d\n", L, q, 'a' + s, L, A->Delta[s][q]);
		}
	}

	printf("initial %d\n", A->q0);

	for (int q = 0; q < A->n; q++) {
		if (A->F[q] == 1) {
			printf("final %d\n", q);
		}
	}
}

void DFA_decode(automaton_t *A)
{
	int n, m;

	scanf("n=%d\n", &n);
	scanf("m=%d\n", &m);

	DFA_init(A, n, m);

	for (int q = 0; q < n; q++) {
		A->F[q] = 0;
		for (int s = 0; s < m; s++) {
			A->Delta[s][q] = q;
		}
	}
	{
		int p, q;
		char c;

		while (scanf("%d %c %d\n", &p, &c, &q) == 3) {
			A->Delta[c - 'a'][p] = q;
		}

		scanf("initial %d\n", &A->q0);

		while (scanf("final %d\n", &q) == 1) {
			A->F[q] = 1;
		}
	}
}

void DFA_print(automaton_t *A)
{
	int L = (int)log10(A->n) + 1;

	for (int l = 0; l <= L + 1; l++) { putchar(' '); }

	for (int s = 0; s < A->m; s++) {
		printf("%*c", L + 1, 'a' + s);
	}
	putchar('\n');

	for (int l = 0; l <= L + 1; l++) { putchar(' '); }
	for (int l = 0; l < (L + 1) * A->m; l++) { putchar('-'); }
	putchar('\n');

	for (int q = 0; q < A->n; q++) {
		printf("%*d |", L, q);
		for (int s = 0; s < A->m; s++) {
			printf("%*d", L + 1, A->Delta[s][q]);
		}
		putchar(' ');
		putchar(' ');
		if (A->q0 == q) {
			printf("initial ");
		}
		if (A->F[q] == 1) {
			printf("final");
		}
		putchar('\n');
	}
}
