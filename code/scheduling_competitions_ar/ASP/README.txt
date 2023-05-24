CONTENUTO CARTELLA: ======================================
main.lp   prmodello in ASP
ESEGUI    

ESEGUIRE SU LINUX:  ======================================
./ESEGUI <data_folder>

esempi:
./ESEGUI ../TEST_CASES/SAMPLE_TEST/

ESEGUIRE SU ALTRO S.O.: ==================================
- Concatenare un file di input che contiene le dfinizioni di n,giorni e lunghezze al file main.lp
- eseguire clingo sul file ottenuto

Esempio definizioni file di input:
gara(1, 355, 15, 72).
gara(2, 334, 35, 54).
gara(3, 315, 15, 382).
budget(30000).
costo_benzina(190).
km_litro(12).


ESEGUIRE ISTANZE TEST SU LINUX:  =========================
comando per eseguire le istanze test in ../TEST_CASES/TEST_INSTANCES
python ESEGUI_TEST.py