
#ifndef _AUTOMATON_H_
#define _AUTOMATON_H_
typedef struct automaton_t {
	int n;       // number of states
	int m;       // number of symbols
	int *Q;      // states (identity, for convenience)
	int q0;      // initial state
	int **Delta; // transition function
	int *F;      // final states
} automaton_t;
#endif


void DFA_init(automaton_t *A, int n, int m);
void DFA_term(automaton_t *A);

#define DFA_MODE_WRST 0
#define DFA_MODE_BEST 1

void DFA_init_mode(automaton_t *A, int n, int m, int mode);
void DFA_init_rand(automaton_t *A, int n, int m);

void DFA_encode(automaton_t *A);
void DFA_decode(automaton_t *A);

void DFA_print(automaton_t *A);
