gara(1, 97, 35, 408).
gara(2, 97, 2, 566).
gara(3, 97, 30, 745).
gara(4, 97, 21, 407).
gara(5, 97, 47, 249).
gara(6, 97, 10, 453).
gara(7, 97, 15, 860).
gara(8, 97, 21, 472).
gara(9, 97, 42, 396).
gara(10, 71, 42, 659).
gara(11, 140, 27, 156).
gara(12, 221, 27, 208).
gara(13, 271, 29, 475).
gara(14, 259, 5, 308).
gara(15, 189, 27, 968).
budget(775). 
costo_benzina(183). 
km_litro(12). 




giorno(1..366).    % enumerazione dei giorni
lunghezza(1..50).  % lunghezza della gara
maratona(42).      % lunghezza di una maratona
mezza(21).         % lunghezza di una mezza maratona
distanza(1..1000). % distanza da casa

% importante che la divisione venga valutata come ultima operazione
soglia_distanza(S) :- S=((B * 100 * K) / C), km_litro(K), costo_benzina(C), budget(B).

% [UNICITY]
0{partecipa(I) : gara(I,G,L,D)}1 :- giorno(G).

% [MAR-MZ]
:- partecipa(I2), gara(I2,G2,L2,D2), mezza(L2),
partecipa(I1), gara(I1,G1,L1,D1), maratona(L1), G2 > G1, G2 <= G1 + 28.

% [MZ-MZ]
:- partecipa(I2), gara(I2,G2,L2,D2), mezza(L2),
partecipa(I1), gara(I1,G1,L1,D1), mezza(L1), G2 > G1, G2 <= G1 + 21.


% [MAR-MAR]
:- partecipa(I2), gara(I2,G2,L2,D2), maratona(L2),
partecipa(I1), gara(I1,G1,L1,D1), maratona(L1), G2 > G1, G2 <= G1 + 42.

% [15KM]
:- partecipa(I2), gara(I2,G2,L2,D2), L2>=10,
partecipa(I1), gara(I1,G1,L1,D1), L1 >= 15, G2 > G1, G2 <= G1 + 10.

% calcolo distanza totale corsa a piedi
% e distanza percorsa in macchina
% nota: per la distanza in macchina consideriamo andata e ritorno
corsa_totale(T)    :- T = #sum{L,I : partecipa(I), gara(I,G,L,D)}.
distanza_totale(T) :- P = #sum{D,I : partecipa(I), gara(I,G,L,D)}, T = 2*P.

% [BUDGET]
:- distanza_totale(T), soglia_distanza(T1), T > T1.


% [UBCT] e [UBDT]
max_d_giorno(G,MD) :- MD = #max{0; D : gara(I,G,L,D)}, giorno(G).
max_c_giorno(G,MC) :- MC = #max{0; L : gara(I,G,L,D)}, giorno(G).

ubct(T) :- T = #sum{MC,G : max_c_giorno(G, MC)}.
ubdt(T) :- P = #sum{MD,G : max_d_giorno(G, MD)}, T = 2*P.

:- ubdt(T1), distanza_totale(T2), T2 > T1.
:- ubct(C1), corsa_totale(C2), C2 > C1.

% output
partecipa(G,L,D) :- partecipa(I), gara(I,G,L,D).
#maximize{T : corsa_totale(T)}.
#show partecipa/1.
#show partecipa/3.
#show corsa_totale/1.
#show distanza_totale/1.
#show soglia_distanza/1.
#show ubct/1.
#show ubdt/1.