#TP MapReduce/Spark


## Creation d'un index inversé à partir de crawling d'un repertoire Wikipedia minimale, accessible en localhost via un serveur Apache.

Bonjour Pierre,

Je vous prie de trouver en pièce jointe le dossier compressé de mes fichiers répondant au TP MapReduce/Spark (module Calcul Distribué du CES Data Scientist 2016)

Vous trouverez les fichiers suivants :
## Description

    * stop_words.txt : liste des mots pour le stemming

### partie 1: HBASE->HDFS->HADOOP MapReduce->HDFS->HBASE

    * TP_MapReduce/1_wikiFromHBaseToHdfs.py : transforme la table HBASE 'simplewiki' en un fichier HDFS 'simplewikiFromHbase'
    * TP_MapReduce/2_wikiIndexMapReduce.py : lance les fonctions Hadoop map/reduce du fichier WordCount.py sur le fichier HDFS 'simplewikiFromHbase' afin de créer un fichier HDFS 'indexwikiFromMapReduce' contenant l'index inversé.
    * TP_MapReduce/3_wikiIndexMapReduceToHBase.py : transforme le fichier HDFS 'indexwikiFromMapReduce' en une table HBASE 'indexwikiFromMapReduce'


### partie 2: HBASE->HDFS->SPARK map/reduce->HDFS->HBASE

    * TP_Spark/1_wikiFromHBaseToHdfs.py : transforme la table HBASE 'simplewiki' en un fichier HDFS 'simplewikiFromHbase'
    * TP_Spark/2_wikiSpark.py : lance les fonctions Spark map/reduce sur le fichier HDFS 'simplewikiFromHbase' afin de créer un fichier HDFS 'indexwikiFromSpark' contenant l'index inversé
    * TP_Spark/3_wikiIndexSparkToHBase.py : transforme le fichier HDFS 'indexwikiFromSpark' en une table HBASE 'indexwikiFromSpark'


### partie 3: HBASE->SPARK map/reduce->HBASE

    * TP_Spark_Hbase/1_wikiFromHBaseToSpark.py : recupère directement la table Hbase 'simplewiki', lance les fonctions Spark map/reduce pour créer l'index inversé, puis enregistre cet index directement dans une table Hbase 'indexwikiFromSpark2'


Par avance, merci.

Cordialement,

Youcef KACER
