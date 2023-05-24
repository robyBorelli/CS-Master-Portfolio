par int: budget;
par int: costo_benzina;
par int: km_litro;
par int: soglia_distanza = floor((budget * 100 * km_litro) / costo_benzina);

par int: maratona = 42;
par int: mezza = 21;
par int: max_giorno = 366;
par int: max_lunghezza = 50;
par int: max_distanza = 1000;

par int: n;
array [1..n] of par 1..max_giorno: giorni;
array [1..n] of par 1..max_lunghezza: lunghezze;
array [1..n] of par 1..max_distanza: distanze;

array [1..n] of var 0..1: partecipa;


% [UNICITY]
constraint forall(i in 1..n, j in i+1..n)
	(if (giorni[i] == giorni[j]) then (partecipa[i] + partecipa[j]<= 1) endif); 
	

% [MAR-MZ]
constraint forall(i,j in 1..n where i!=j)
	(forall(l in 1..max_giorno)
		(if(lunghezze[i] == maratona /\ lunghezze[j] == mezza /\ 
			  giorni[j] <= giorni[i]+28 /\ giorni[j] > giorni[i]) then 
				(partecipa[i] + partecipa[j] <= 1)
		endif)
	);

% [MZ-MZ]
constraint forall(i,j in 1..n where i!=j)
	(forall(l in 1..max_giorno)
		(if(lunghezze[i] == mezza /\ lunghezze[j] == mezza /\ 
			  giorni[j] <= giorni[i]+21 /\ giorni[j] > giorni[i]) then 
				(partecipa[i] + partecipa[j] <= 1)
		endif)
	);

% [MAR-MAR]
constraint forall(i,j in 1..n where i!=j)
	(forall(l in 1..max_giorno)
		(if(lunghezze[i] == maratona /\ lunghezze[j] == maratona /\ 
			  giorni[j] <= giorni[i]+42 /\ giorni[j] > giorni[i]) then 
				(partecipa[i] + partecipa[j] <= 1)
		endif)
	);

% [15KM]
constraint forall(i,j in 1..n where i!=j)
	(forall(l in 1..max_giorno)
		(if(lunghezze[i] >=15 /\ lunghezze[j] >= 10 /\ 
			  giorni[j] <= giorni[i]+10 /\ giorni[j] > giorni[i]) then 
				(partecipa[i] + partecipa[j] <= 1)
		endif)
	);

% calcolo distanza totale corsa a piedi
% e distanza percorsa in macchina
% nota: per la distanza in macchina consideriamo andata e ritorno
par int: ubct = sum(g in 1..max_giorno)(max([0]++[lunghezze[i] | i in 1..n where giorni[i] == g])); 
par int: ubdt =  2 * (sum(g in 1..max_giorno)(max([0]++[distanze[i] | i in 1..n where giorni[i] == g]))); 

var 0..ubct: corsa_totale = sum (i in 1..n)(partecipa[i] * lunghezze[i]);
var 0..ubdt: distanza_totale = sum (i in 1..n)(partecipa[i] * distanze[i] * 2);


% [BUDGET]
constraint distanza_totale <= soglia_distanza;

solve maximize corsa_totale;

output[
  (if (fix(partecipa[i]) == true) then 
  	("partecipa("++show(giorni[i])++","++show(lunghezze[i])++","++show(distanze[i])++") ")
  else ""
  endif)++
  (if (i == n) then 
  	("corsa_totale("++show(corsa_totale)++") "++
  	"distanza_totale("++show(distanza_totale)++") "++
  	"soglia_distanza("++show(soglia_distanza)++")")
  else ""
  endif) | i in 1..n];