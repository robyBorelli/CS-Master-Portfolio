CONTENUTO CARTELLA: ======================================
main.c    programma per generare un modello in formato CPLEX lp
ESEGUI    compila il programma, lo esegue, salva il modello
          in un file, esegue glpk e stampa la soluzione in un formato
          human readable

ESEGUIRE SU LINUX:  ======================================
./ESEGUI <data_folder>

esempi:
./ESEGUI ../TEST_CASES/SAMPLE_TEST/

ESEGUIRE SU ALTRO S.O.: ==================================
- Concatenare un file di input che contiene le dfinizioni di n,giorni e lunghezze,(...) al file main.c
- compilare il file main.c
- eseguire il file main.c e salvare il risultato in un nuovo file: model.lp
- invocare glpk (o un qualsiasi altro solver ILP) con input model.lp

Esempio definizioni file di input:
#define n 3
int giorni[n]={355, 334, 315};
int lunghezze[n]={15, 35, 15};
int distanze[n]={72, 54, 382};
int budget = 30000;
int costo_bennzina = 190;
int km_litro = 12;


ESEGUIRE ISTANZE TEST SU LINUX:  =========================
comando per eseguire le istanze test in ../TEST_CASES/TEST_INSTANCES
python ESEGUI_TEST.py
