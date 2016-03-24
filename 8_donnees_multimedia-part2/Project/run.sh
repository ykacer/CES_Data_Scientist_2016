#!/bin/bash


if [ ! -d data ]
then
	mkdir data
	wget http://perso.telecom-paristech.fr/~essid/ces_ds/06-11-22.mp4 -P data
	wget http://perso.telecom-paristech.fr/~essid/ces_ds/annotations.zip -P data
	unzip data/annotations.zip -d data
fi


