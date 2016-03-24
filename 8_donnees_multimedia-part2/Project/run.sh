#!/bin/bash

# time between frame
delta_t = 1

# get video and annotation .trs file
if [ ! -d data ]
then
	mkdir data
	#wget http://perso.telecom-paristech.fr/~essid/ces_ds/06-11-22.mp4 -P data
	wget http://perso.telecom-paristech.fr/~essid/ces_ds/annotations.zip -P data
	unzip data/annotations.zip -d data
fi

# transform .trs file into speakers.csv where each row reveals who's on the screen ("speaker") between startTime and endTime
xmlstarlet sel -T -t -m /Trans/Episode/Section/Turn -v "concat(@speaker,';',@startTime,';',@endTime)" -n data/06-11-22.trs > data/speakers.csv

# loop over jpeg images, and label it with its label :
# 2 for speaker M on the screen
# 3 	"	A	"
# 4 	"	B	"
# 5 	"	C	"
# 6	"	D	"
# 7	"	MULTI	"	(means between 2 and 4 people)
# 8 	"	ALL	"	(means all 5 people)
# 9 	"	INTRO	"	(means inroductive documentary)
# 10 	"	CREDITS	"	(means ending credits)

# loop over jpeg images, and label it with its binary_label 
# 0 		M|A|B|C|D	(means one person)
# 1		MULTI|ALL	(means at least 2 people)
# NaN		INTRO|CREDITS

startTime = 
let "current_time = 0"
for image in `ls data/*jpg`
do
	echo $image
	echo "${current_time}s"

	let "current_time += 1"
done

