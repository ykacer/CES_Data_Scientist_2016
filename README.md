# CES_Data_Scientist_2016


## Motivation
Shows the content of the 25-days Data Scientist certification from Telecom-ParisTech.

One can find courses of the 12 different thematics and especially, all the work i had to provide.

Credit to Telecom-Paristech, See [CES Data Scientist-Telecom ParisTech](https://telecom-paristech.fr/fileadmin/documents/pdf/formation_continue/CES/Data-Scientist.pdf) for more details.

## Content

### Introduction à l'apprentissage statistique

#### part1 :
 * cours-ces-1-intro.pdf (credit to [*Florence d'Alché*](http://perso.telecom-paristech.fr/~fdalche/Site/index.html)).

keywords : features classification, cost function, empirical risk, bayesian classifier, bias/variance compromise.

 * Papers/

Some introductive papers to Machine Learning.

 * Exercises/ 
- TP_intro_python_ml_digits.v3.ipynb : K-neighrest-neighboors on MNIST digits dataset using **scikit-learn**, cross validation, confusion matrix.

#### part2 :
 * COURS2.pdf (credit to [*Stephan Clemençon*](http://perso.telecom-paristech.fr/~clemenco/Home.html))

keywords : basic algorithms for classification (Logistic Regression, Linear Discriminant Analysis, Perceptron, Partitionning algorithms, Decision Tree), model complexity, model selection (penalization, cross validation, bootstrap).

 * Papers/

Some introductive papers to Machine Learning.

 * Exercises/
- TP_intro_Pandas_lin_model.ipynb : Linear regression, Ridge Regression on auto-mpg data using **scikit-learn**, polymial complexity, cross-validation.

### Données numériques et structurées

#### part1 :
 * TelecomParisTechCours1-NoSQL.pdf (credit to [*Raja Chiky*](http://perso.isep.fr/rchiky/nosql/))

 * TelecomParisTechCours2-NoSQL.pdf

 * TelecomParisTechCours3-streaming.pdf

 * TP_MongoDB/
- TPMongoDB.pdf : exercises to get familiar with document-oriented NoSQL **MongoDB**.
- test.js : short javascript to familiarize with MongoDB.
- lapins.js : create a collection and run basic queries.
- earthquakes.js : use earthquakes_big.geojson database, re-formate data and run queries (regex,2dsphere).
- populate_phones.js : create huge collection of phone number.
- earthquakes_big.geojson : earthquakes database.
- run.sh : bash file to import data, run above javascripts files and test master/slave replication and sharding database with MongoDB.

#### part2 :
 * Project/
- Big Data Project with NoSQL DB.pdf : exercises for MongoDB project.
- mongodb_1_data_integration.js : javascript to import .csv data to be reformated in **MongoDB** collections.
- mongodb_2_data_queries.js : run queries using aggregate method of MongoDB.
- run_mongo.sh : bash file to run javascripts for MongoDB.
- mysql_1_data_integration.sql : SQL file to integrate .csv data to be reformated in **MySQL** tables.
- mysql_2_data_queries.sql : run queries using SQL jointure and group by.
- run_mysql.sh : bash file to run SQL files for MySQL.
- report.pdf : report that resume results and compare time execution between MongoDB and MySQL.

### Données textuelles et web

#### part1 :

 * [Motivation and Knowledge Representation](https://suchanek.name/work/teaching/ces2016/01_motknow/0.svg) (credit to [*Fabian Suchanek*](https://suchanek.name/))

 * [Named Entity Recognition](https://suchanek.name/work/teaching/athens2015/03_ner/0.svg)

#### part2 :

 * [Disambiguation](https://suchanek.name/work/teaching/athens2015/05_disambiguation/0.svg)

 * [Fact Extraction](https://suchanek.name/work/teaching/athens2015/07_unstructured/0.svg)
	
 * [Semantic Web](https://suchanek.name/work/teaching/athens2015/10_semanticWeb/0.svg)

 * Lab/ ([Tasks](https://suchanek.name/work/teaching/ces2016/lab/index.html))
- wikifirst.txt : 150000 articles taken from the Simple English Wikipedia.
- ie_dates.py : python script to extract dates from the 150000 articles.
- ie_dates_evaluation.txt : containing the first 20 dates found and the manually measured precision on the first 20 dates found.
- ie_types.py : python script to extract the type of each article
- ie_types_evaluation.txt : containing the first 20 types found and the manually measured precision on the first 20 types found.

## Donnees multimedia

#### part 1 :

 * audio/
- audio-analysis-lecture_2016.pdf (credit to [*Slim Essid*](http://perso.telecom-paristech.fr/~essid/)) : keywords : audio content analysis, short-term analysis, spectral analysis (DFT,DCT) and spectrogram, MFCC features extraction, temporal integration.
- test.py : small python script to get hands on **scipy.fftpack** module analyzing small wav audio files
- wav/ : short audio samples taken from [*www.spirit-science.fr*](http://www.spirit-science.fr/doc_musique/Sensation-sonore.html)

 * image/ (credit to [*Michel Roux*](http://perso.telecom-paristech.fr/~mroux/))
- bcontours-ds.pdf 
- Donnees Multimedia - Images et Video.pdf
- introtdi-ds [Mode de compatibilité].pdf
- sift [Mode de compatibilité].pdf
- calibrage-ds.pdf
- formes-ds.pdf 
- notebooks/ : some python notebook files for advanced tutorials on optimization and algebra operations using scipy (credit to [*Alexandre Chamfort*](http://alexandre.gramfort.net/index.fr.html) and [*Slim Essid*](http://perso.telecom-paristech.fr/~essid/))

#### part 2 :

 * Project/
- run.sh: download a mp4 video 06-11-22.mp4 of a debate from swiss TV, and a related annotation file 06-11-22.trs. Then, the video is cut into jpeg images, each frame is labelled using annotation file, see labels.csv created (take time...xmlstarlet is needed)
 
## Apprentissage supervisé

#### part 1:

 * apprentissage_classique.pdf (credit to [*Joseph Salmon*](http://www.josephsalmon.eu/))

keywords : Linear Discriminant Analysis, Quadratic Discriminant Analysis, Naive Bayesian classifier, Logistic Regression, K-Neighrest Neighboors.

 * tp_classifiers/
- ClassifieurNaif.ipynb : LDA, QDA, Naive Gaussian Bayes, Logistic regression, KNN on IRIS dataset using **scikit learn**
- module5_source.py : hand crafted module imported for plotting
- module5_source.pyc : compiled module

#### part 2:

 * cours_arbres_selection_modele.pdf (credit to [*Aurélien Bellet*](http://researchers.lille.inria.fr/abellet/))

keywords : decision tree, CART algorithm, entropy, regression tree, regularization, random forest

 * tp_learning_curve/
- tp_learning_curve.pdf : exercises dealing with decision trees and hyperparameters, random forests, selection model, regularization parameters on digits dataset using **scikit-learn**
- tp_learning_curve.ipynb : python code for exercises
- learning_curve.py : function using module learning_curve from scikit-learn
- learning_curve.pyc : compiled module


## introduction à l'apprentissage profond

#### part 1:

 * 1_cours_nnet_basics.pdf (credit to [*Alexandre Allauzen*](https://perso.limsi.fr/allauzen/webpages/pmwiki.php))

keywords : feed-forward neural networks, gradient back propagation, activation function, loss function

 * 2_cours_nnet_basics.pdf (credit to [*Alexandre Allauzen*](https://perso.limsi.fr/allauzen/webpages/pmwiki.php))

keywords : Regularization and dropout, vanishing gradient and Rectified Linear Unit
 * lab_mnist/ (credti to [*Gaetan Marceau-Caron*](http://gmarceaucaron.github.io/))
- telecom.odp : lab queries
- train_mnist.lua : **Lua** file that runs a 400x400x10 neural network on 60000 28x28 handwritten mnist images using **Torch**. Apply 'nn' module for a sequential network, and 'optim' module for gradient back propagation solver. Apply 'nn.Dropout' function for regularization.

#### part 2:

