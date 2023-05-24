CONTENUTO CARTELLA: ======================================
main_cop.mzn modello con constraint globale
main_ilp.mzn modello simil-ilp

ESEGUIRE SU LINUX:  ======================================
./ESEGUI_COP <data_folder>
./ESEGUI_ILP <data_folder>

esempi:
./ESEGUI_COP ../TEST_CASES/SAMPLE_TEST/
./ESEGUI_ILP ../TEST_CASES/SAMPLE_TEST/

ESEGUIRE SU ALTRO S.O.: ==================================
minizinc --solver Gecode main_cop.mzn <data_file>
minizinc --solver Gecode main_ilp.mzn <data_file>

esempi:
minizinc --solver Gecode main_cop.mzn ../TEST_CASES/SAMPLE_TEST/minizinc.dzn
minizinc --solver Gecode main_ilp.mzn ../TEST_CASES/SAMPLE_TEST/minizinc.dzn

Esempio definizioni file di input:
budget = 30000;
costo_benzina = 190;
km_litro = 12;
n=3;
giorni=[355, 334, 315];
lunghezze=[15, 35, 15];
distanze=[72, 54, 382];


ESEGUIRE ISTANZE TEST SU LINUX:  =========================
comando per eseguire le istanze test in ../TEST_CASES/TEST_INSTANCES
python ESEGUI_TEST.py