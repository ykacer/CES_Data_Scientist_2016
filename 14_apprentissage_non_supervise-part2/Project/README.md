# Projet apprentissage non-supervisée
## Clustering de documents numérisées

### utils.py 
Ce fichier contient les fonctions pour le découpage de l'image en patch et le post-processing, la dernière utilise **scikit-image**

### functions.py
Ce fichier contient les fonctions permettant l'extraction des features. Ces fonctions utilisent **OpenCV** ainsi que **scikit-learn**

### run.py
Ce fichier est le fichier principal. Il télécharge le dataset, le decompresse et lance le processus de clusterisation. Il affiche aussi les performances en précision/rappel à partir des images de vérité terrain. Il contient aussi les différents paramètres comme la taille de voisinage, la nature des descripteurs...

 * python run.py

### test.py
Ce fichier permet de tester n'importe quel image représentant un document numérisé pour en réaliser le clustering en trois classes, selon la méthode décrite dans ce document. A noter que les paramètres comme le redimensionnement de l'image, la taille de fenêtre, les descripteurs, etc.. sont à modifier au tout début du fichier. Le fichier resultat est enregisté dans le même répertoire. Depuis un terminal linux :

 * python test.py nom_image.ext
