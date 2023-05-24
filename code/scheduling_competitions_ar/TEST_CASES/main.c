#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define nLunghezze 20


struct lista{
	int giorno; int lunghezza; int distanza;
	struct lista * next;
}; typedef struct lista lista;

int lunghezze[nLunghezze] = {5, 10, 15, 21, 42, 30, 20, 2, 1, 35, 17, 18, 19, 4, 7, 33, 27, 50, 47, 29};
int arrQ[100];
lista * gare = NULL;
int qVal(){ return arrQ[rand()%100]; }
int min(int i1, int i2){return (i1<i2?i1:i2);}

void inserisci(int giorno, int lunghezza, int distanza){
	lista *nodo = malloc(sizeof(lista));
	nodo->giorno = giorno;
	nodo->lunghezza = lunghezza;
	nodo->distanza = distanza;
	nodo->next = gare;
	gare = nodo;
}

void distruggiLista(lista *l){
	if(l!=NULL){
		distruggiLista(l->next);
		free(l);
	}
}

int sumDist(lista *l){
	int r = 0;
	while(l != NULL){
		r += l->distanza;
		l = l->next;
	}
	return r;
}

int maxDistGiorno(lista *l, int g){
	int m = 0;
	while(l != NULL){
		if(l->giorno == g && l->distanza>m){
			m = l->distanza;
		}
		l = l->next;
	}
	return m;
}

int maxDistUpperBound(lista *l){
	int d = 0;
	for(int i = 1; i <= 366; i++){
		d = d + maxDistGiorno(l, i);
	}
	return d;
}

void stampaAsp(FILE* f, int n, lista* l, int budget, int costo_benzina, int km_litro, int printId){
	int i = 1;
	while(l != NULL){
		if(printId == 1){
			fprintf(f, "gara(%d, %d, %d, %d).\n", i, l->giorno, l->lunghezza, l->distanza);
		}else{
			fprintf(f, "gara(%d, %d, %d).\n", l->giorno, l->lunghezza, l->distanza);
		}
		i++;
		l = l->next;
	}

	fprintf(f, "budget(%d). \ncosto_benzina(%d). \nkm_litro(%d). \n", 
		budget, costo_benzina, km_litro);
	fprintf(f, "\n\n");
}

void stampaMzn(FILE* f, int n, lista* l, int budget, int costo_benzina, int km_litro){
	fprintf(f, "\n\nn=%d;\n",n);
	lista * root = l;

	fprintf(f, "giorni=[");
	while(l!=NULL){
		fprintf(f, "%d", l->giorno);
		if(l->next != NULL) fprintf(f, ", ");
		l = l->next;
	} fprintf(f, "];\n"); l=root;

	fprintf(f, "lunghezze=[");
	while(l!=NULL){
		fprintf(f, "%d", l->lunghezza);
		if(l->next != NULL) fprintf(f, ", ");
		l = l->next;
	} fprintf(f, "];\n");l=root;

	fprintf(f, "distanze=[");
	while(l!=NULL){
		fprintf(f, "%d", l->distanza);
		if(l->next != NULL) fprintf(f, ", ");
		l = l->next;
	} fprintf(f, "];\n");
	fprintf(f, "budget = %d;\ncosto_benzina = %d;\nkm_litro = %d;\n", 
		budget, costo_benzina, km_litro);
	fprintf(f, "\n\n");
}

void stampaIlp(FILE* f, int n, lista* l, int budget, int costo_benzina, int km_litro){
	fprintf(f, "\n\n#define n %d\n",n);
	lista * root = l;

	fprintf(f, "int giorni[n]={");
	while(l!=NULL){
		fprintf(f, "%d", l->giorno);
		if(l->next != NULL) fprintf(f, ", ");
		l = l->next;
	} fprintf(f, "};\n"); l=root;

	fprintf(f, "int lunghezze[n]={");
	while(l!=NULL){
		fprintf(f, "%d", l->lunghezza);
		if(l->next != NULL) fprintf(f, ", ");
		l = l->next;
	} fprintf(f, "};\n");l=root;

	fprintf(f, "int distanze[n]={");
	while(l!=NULL){
		fprintf(f, "%d", l->distanza);
		if(l->next != NULL) fprintf(f, ", ");
		l = l->next;
	} fprintf(f, "};\n");

	fprintf(f, "int budget = %d;\nint costo_benzina = %d;\nint km_litro = %d;\n", 
		budget, costo_benzina, km_litro);
	fprintf(f, "\n\n");
}

void stampaDescrizione(FILE *f, int nGare, int budget, int costo_benzina, int km_litro, int q, int p_soglia_distanza, int distanza_totale, int soglia_distanza, int ubdt, char*cartella){
	fprintf(f, 
		"\n\nnGare=%d\nbudget=%d\ncosto_benzina=%d\nkm_litro=%d\nq=%d\np_soglia_distanza=%d\ndistanza_totale=%d\nsoglia_distanza=%d\nubdt=%d\ncartella=%s\n\n",
		nGare,budget,costo_benzina,km_litro,q, p_soglia_distanza,distanza_totale,soglia_distanza,ubdt,cartella);
}

void generaIstanza(int p_soglia_distanza, int nGare, int q, char* cartella){
	int maxGare = 20;     // massime gare in un giorno
	int minGare = 1;      // minime gare in un giorno
	int maxDist = 1000;
	int minDist = 5;
	int costo_benzina = 183;
	int km_litro = 12;

	distruggiLista(gare); gare =NULL;

	int l;
	for(l=0; l< q; l++){arrQ[l] = 1;}
	for(; l< 100; l++){arrQ[l] = 0;}


	int i=0;
	while(i < nGare){
		int j = (qVal()*((rand()%(maxGare-minGare)) + 1) + minGare);
		int giorno = (rand()%366)+1;
		for (int k = 0; k < j && i < nGare; k++){
			inserisci(giorno, lunghezze[rand()%nLunghezze], min(rand()%(maxDist+1) + minDist, maxDist));
			i++;
		} 
	}

	int distanza_totale = sumDist(gare)*2;
	int ubdt = maxDistUpperBound(gare)*2;
	int soglia_distanza = (ubdt * p_soglia_distanza)/100;
	int budget = (soglia_distanza * costo_benzina)/(100*km_litro);


	char nameAsp[50]; strncpy(nameAsp,cartella, 50); strcat(nameAsp, "asp.txt");
	FILE* f_asp = fopen(nameAsp, "w");
	stampaAsp(f_asp, nGare, gare, budget, costo_benzina, km_litro, 0);
	fclose(f_asp);

	char nameAspId[50]; strncpy(nameAspId,cartella, 50); strcat(nameAspId, "asp_id.txt");
	FILE* f_aspId = fopen(nameAspId, "w");
	stampaAsp(f_aspId, nGare, gare, budget, costo_benzina, km_litro, 1);
	fclose(f_aspId);


	char nameMinizinc[50]; strncpy(nameMinizinc,cartella, 50); strcat(nameMinizinc, "minizinc.dzn");
	FILE* f_mzn = fopen(nameMinizinc, "w");
	stampaMzn(f_mzn, nGare, gare, budget, costo_benzina, km_litro);
	fclose(f_mzn);

	char nameIlp[50]; strncpy(nameIlp,cartella, 50); strcat(nameIlp, "ilp.txt");
	FILE* f_ilp = fopen(nameIlp, "w");
	stampaIlp(f_ilp, nGare, gare, budget, costo_benzina, km_litro);
	fclose(f_ilp);

	char nameDesc[50]; strncpy(nameDesc,cartella, 50); strcat(nameDesc, "descrizione.txt");
	FILE* f_desc = fopen(nameDesc, "w");
	stampaDescrizione(f_desc, nGare, budget, costo_benzina, km_litro, q, p_soglia_distanza, distanza_totale, soglia_distanza, ubdt, cartella);
	fclose(f_desc);


  // DEBUG
/*	stampaAsp(stdout, nGare, gare, budget, costo_benzina, km_litro);
	stampaMzn(stdout, nGare, gare, budget, costo_benzina, km_litro);
	stampaIlp(stdout, nGare, gare, budget, costo_benzina, km_litro);*/
	stampaDescrizione(stdout, nGare, budget, costo_benzina, km_litro, q, p_soglia_distanza, distanza_totale, soglia_distanza, ubdt, cartella);

}


int main()
{
	srand(time(NULL)); char temp1[10];char temp2[50];
	int budget = 10000;
	char cartella[50]="TEST_INSTANCES";

	int nGare[30] = {5, 10, 15, 20, 25,
					 30, 32, 35, 40, 42,
					 45, 50, 52, 55, 60,
					 62, 65, 70, 72, 75,
					 80, 82, 85, 90, 95,
					 150, 250, 350, 450, 550};

	// percentuale piu di una gara in un giorno con gare
	int q[30] = {10, 20, 30, 40, 50,
                 10, 20, 30, 40, 50,
                 10, 20, 30, 40, 50,
                 10, 20, 30, 40, 50,
                 10, 20, 30, 40, 50,
                 10, 20, 30, 40, 50}; 

	// percentuale della distanza ammissibile
	int p_soglia_distanza[30] = {90, 80, 70, 60, 40,
	                             90, 80, 70, 60, 40,
	                             90, 80, 70, 60, 40,
	                             90, 80, 70, 60, 40,
	                             90, 80, 70, 60, 40,
	                             90, 80, 70, 60, 40}; 

	for(int i = 0; i < 30; i++){
		sprintf(temp1,"%d",i+1);
		strncpy(temp2, cartella, 50);
		strcat(temp2,"/");
		strcat(temp2,temp1);
		strcat(temp2,"/");
		generaIstanza(p_soglia_distanza[i], nGare[i], q[i], temp2);	
	}


	return 0;
}