include "globals.mzn"; 

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

array [1..n] of var bool: partecipa;


% [UNICITY]
constraint alldifferent_except_0([partecipa[j]*giorni[j] | j in 1..n]);	

% [MAR-MZ]
constraint forall(i in 1..n)
	(partecipa[i] == true /\ lunghezze[i] == maratona ->
		(sum (j in 1..n where giorni[j] > giorni[i] /\ giorni[j] <= giorni[i] + 28 /\ lunghezze[j] == mezza) 
				 (partecipa[j])) == 0
	);

% [MZ-MZ]
constraint forall(i in 1..n)
	(partecipa[i] == true /\ lunghezze[i] == mezza ->
		(sum (j in 1..n where giorni[j] > giorni[i] /\ giorni[j] <= giorni[i] + 21 /\ lunghezze[j] == mezza) 
				 (partecipa[j])) == 0
	);

% [MAR-MAR]
constraint forall(i in 1..n)
	(partecipa[i] == true /\ lunghezze[i] == maratona ->
		(sum (j in 1..n where giorni[j] > giorni[i] /\ giorni[j] <= giorni[i] + 42 /\ lunghezze[j] == maratona) 
				 (partecipa[j])) == 0
	);

% [15KM]
constraint forall(i in 1..n)
	(partecipa[i] == true /\ lunghezze[i] >= 15 ->
		(sum (j in 1..n where giorni[j] > giorni[i] /\ giorni[j] <= giorni[i] + 10 /\ lunghezze[j]>=10) 
				 (partecipa[j])) == 0
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



% strategia base
solve maximize corsa_totale;

% strategia S1
%solve 
%      :: bool_search(partecipa,occurrence,indomain_random)
%      :: restart_linear(2*n)
%maximize corsa_totale;


% strategia S2
%solve 
%      :: bool_search(partecipa,dom_w_deg,indomain_random)
%      :: restart_linear(2*n)
%maximize corsa_totale;

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
