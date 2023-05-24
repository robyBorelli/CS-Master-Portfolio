



/* esempio per compilare il file in autonomia
#define n 3
int giorni[n]={355, 334, 315};
int lunghezze[n]={15, 35, 15};
int distanze[n]={72, 54, 382};
int budget = 30000;
int costo_benzina = 190;
int km_litro = 12;
*/

#include <stdio.h>

#define max_lunghezza 50
#define max_giorno 366
#define max_distanza 1000

long soglia_distanza;

int maratona = 42;
int mezza = 21;

int elaborato[366] = {};

int max(int *arr, int giorno){
	int m = 0;
	for(int i = 0; i < n; i++){
		if(giorno == giorni[i] && arr[i] >= m){
			m = arr[i];
		}
	}
	return m;
}

int main()
{
	soglia_distanza = ((long)budget * (long)100 * (long)km_litro) / costo_benzina;
	int ubct = 0;
	int ubdt = 0;
	for(int i = 0; i < n; i++){
		int giorno = giorni[i];
		if(elaborato[giorno-1] == 0){
			ubct += max(lunghezze, giorno);
			ubdt += max(distanze, giorno);
		}
		elaborato[giorno-1]= 1;
	}

	printf("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n");
	printf("\\\\ Atoms:\n");
	printf("\\\\");
	for(int i = 1; i <= n; i++){printf("partecipa(%d,%d,%d) ", giorni[i-1], lunghezze[i-1], distanze[i-1]);}
	printf("\n\\\\soglia_distanza(%ld)", soglia_distanza);
	printf("\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n\n");

	printf("Maximize\ncorsa_totale: ");
	for(int i = 1; i <= n; i++){
		printf("%d x_%d",lunghezze[i-1], i);
		if(i != n) printf(" + ");
	}

	printf("\n\nSubject to\n");


	// [BUDGET]
	printf("\n\n");
	for(int i = 1; i <= n; i++){
		printf("%d x_%d",distanze[i-1]*2, i);
		if(i != n) printf(" + ");
	}
	printf(" <= %ld\n\n",soglia_distanza);


	// [UNICITY]
	for(int i = 1; i <= n; i++){
		for(int j = i+1; j <= n; j++){
			if(giorni[i-1] == giorni[j-1]){
				printf("x_%d + x_%d <= 1\n", i, j);
			}
		}
	}


	// [MAR-MZ]
	printf("\n\n");
	for(int i = 1; i <= n; i++){
		for(int j = 1; j <= n; j++){
			if(i!=j && lunghezze[i-1] == maratona 
				&& lunghezze[j-1] == mezza 
				&& giorni[j-1] <= giorni[i-1]+28 && giorni[j-1] > giorni[i-1]){
				printf("x_%d + x_%d <= 1\n", i, j);
			}
		}
	}


	// [MZ-MZ]
	printf("\n\n");
	for(int i = 1; i <= n; i++){
		for(int j = 1; j <= n; j++){
			if(i!=j && lunghezze[i-1] == mezza 
				&& lunghezze[j-1] == mezza 
				&& giorni[j-1] <= giorni[i-1]+21 && giorni[j-1] > giorni[i-1]){
				printf("x_%d + x_%d <= 1\n", i, j);
			}
		}
	}


	// [MAR-MAR]
	printf("\n\n");
	for(int i = 1; i <= n; i++){
		for(int j = 1; j <= n; j++){
			if(i!=j && lunghezze[i-1] == maratona 
				&& lunghezze[j-1] == maratona 
				&& giorni[j-1] <= giorni[i-1]+42 && giorni[j-1] > giorni[i-1]){
				printf("x_%d + x_%d <= 1\n", i, j);
			}
		}
	}

	// [15KM]
	printf("\n\n");
	for(int i = 1; i <= n; i++){
		for(int j = 1; j <= n; j++){
			if(i!=j && lunghezze[i-1] >=15
				&& lunghezze[j-1] >= 10 
				&& giorni[j-1] <= giorni[i-1]+10 && giorni[j-1] > giorni[i-1]){
				printf("x_%d + x_%d <= 1\n", i, j);
			}
		}
	}

	// [UBCT]
	printf("\n\n");
	for(int i = 1; i <= n; i++){
		printf("%d x_%d",lunghezze[i-1], i);
		if(i != n) printf(" + ");
	}
	printf(" <= %d\n",ubct);

	// [UBCT]
	printf("\n\n");
	for(int i = 1; i <= n; i++){
		printf("%d x_%d",distanze[i-1]*2, i);
		if(i != n) printf(" + ");
	}
	printf(" <= %d\n",ubdt*2);


	printf("\n\nBinary\n");
	for(int i = 1; i <= n; i++){
		printf("x_%d\n", i);
	}

	printf("\n\nEnd\n");
	return 0;
}