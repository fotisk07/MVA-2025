# Progress Notes


## First Session

### Preparatory Notes

Paper identifies player roles and formations using a 2 step approach based on change point detection.

Pipeline : 
1) Identify role positions using another Paper
2)
	a) Form CPD in each half -> Gives segments of same formation
	b) Role CPD in each segment -> Gives player positions (detecting changes)

Suggested Method : 
1) Each one investigates more in detail one method, while vaguely reading and understanding the other one
2) Next week start deciding on experiments to run and testing them


### Meeting Notes 19/11

- Timeline : 
	- 1semaine, 1semaine et demi deep dive sur (a,b) . Fotis fait a, Adonis fait b : 
		- Comprendre comment ça marche
		- Ecrire le rapport (partie methode)
		- Reimplementer certaines parties, montrer des exemples, analyser les limites
	- Suite : 
		- Utiliser la libraire donnée pour run des experiences (cf Effet Messi)
		- Distinguer formation defense de formation attack (+ feature en plus)



Messi Effet : 
- Analyser les matchs sans et avec messi pour montrer la diff de formation pour le barca et l'équipe adverse
- Montrer la stationairité des joueurs (exemple messi centre droit mais en effet vient souvent au centre)

### Meeting Notes 30/11 

Présentation des résultats

Timeline
	-  Ecrire partie methode rapport avant Lundi 8/12
	-  Peaufiner les experiences la dernière semaine avant de rendre le rapport



 

Pertinence SoccerCPD
Stationnarité des joueurs
Defensive/offensive formation
Explore pseudo trajectories from Statsbomb

Mettre de cote Messi effect.



#### CODE A RENDRE ####
Structure du Jupyter: 
I. Methodes
1. FormCPD
2. RoleCPD
II. Data
1. Preprocessing
III. Experiments
1. SoccerCPD
2. Attack/Defense formations
3. Stationarity of players
4. Pseudo trajectories from Statsbomb

#### RAPPORT A RENDRE ####
Introduction: un peu trop longue, raccourcir. Changer experiments, source codes.
Data: nouvelles données, où elles sont, comment on les traite. SoccerCPD (R) pénible à utiliser.
Results: à rédiger. Suivre experiments.
Conclusion: mini paragraphe à rédiger.

D'ici 1h 
finaliser ma partie (Adonis).
finaliser soccerCPD (Fotis).

Ce soir:
Jupyter complet. (Fotis)
Experiments (Fotis)
Introduction + Conclusion + Resultats (Adonis)

Slides
F : Slide 2 à rendre pous jolie + à merge qques elements de la 3 + slide 3 : OK
F : 4 : OK
A : 5 + 6 à condenscer en une slide
A : Slie 7 transformer citatinos en 1,2,3 + ajouter qu'ona utilisé leurs données + enlenve LastRow
F : Slide 9 texte à adapter
A : 11 à rendre plus jilie

