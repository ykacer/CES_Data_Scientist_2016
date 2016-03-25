#!/bin/bash

# time between frame (in seconds)
delta_time=1

# get video and annotation .trs file
if [ ! -d data ]
then
	mkdir data
	#wget http://perso.telecom-paristech.fr/~essid/ces_ds/06-11-22.mp4 -P data
	wget http://perso.telecom-paristech.fr/~essid/ces_ds/annotations.zip -P data
	unzip data/annotations.zip -d data
fi

# transform .trs file into speakers.csv. Each row reveals who's on the screen (field "speaker") between "startIme" end "endTime" in seconds :
if [ -e data/speakers.csv ]
then
	rm data/speakers.csv
fi
xmlstarlet sel -T -t -m /Trans/Episode/Section/Turn -v "concat(@speaker,';',@startTime,';',@endTime)" -n data/06-11-22.trs > data/speakers.csv
sed -i "1ispeaker;startTime;endTime" data/speakers.csv

# loop over jpeg images, and mark them with their label. Possible label values are :
# 2 for speaker M on the screen
# 3 	"	A	"
# 4 	"	B	"
# 5 	"	C	"
# 6	"	D	"
# 7	"	MULTI	"	(means between 2 and 4 people)
# 8 	"	ALL	"	(means all 5 people)
# 9 	"	INTRO	"	(means inroductive documentary)
# 10 	"	CREDITS	"	(means ending credits)

# loop over jpeg images, and mark them with their binary_label. Possible binary_label values are :
# 0 when M|A|B|C|D (means one person)
# 1 when MULTI|ALL (means at least 2 people)
# NaN when INTRO|CREDITS (means intro or cedits)

all_labels=( 00 01 02 03 04 05 06 07 08 09 10 NaN )
for i in "${all_labels[@]}" 
do
	if [ -d data/${i}_class ]
	then
		rm -rf data/${i}_class
	fi
	mkdir data/${i}_class # create directory for each label and binary_label possible values to store corresponding images
done

if [ -e data/labels.csv ]
then
	rm data/labels.csv
fi

let "current_time=0"
for image in `ls data/*jpg`
do
	echo -e "\n*** image : $image"
	echo "*** current time : ${current_time}s"
	for laps in `cat data/speakers.csv | tail -n+2` # skip first line (containing name fields)
	do
        	SPEAKER=`echo $laps | awk -F";" '{ print $1 }'`
       		START_TIME=`echo $laps | awk -F";" '{ print $2 }'`
        	END_TIME=`echo $laps | awk -F";" '{ print $3 }'`
		if [ `echo "$current_time>=$START_TIME" | bc` = 1 ] && [ `echo "$current_time<$END_TIME" | bc` = 1 ]
		then
			case $SPEAKER in
				"M")
					label=2
					binary_label=0
					cp $image data/02_class
					cp $image data/00_class
					;;
				"A")
					label=3
					binary_label=0
					cp $image data/03_class
					cp $image data/00_class
					;;
				"B")
					label=4
					binary_label=0
					cp $image data/04_class
					cp $image data/00_class
					;;
				"C")
					label=5
					binary_label=0
					cp $image data/05_class
					cp $image data/00_class
					;;
				"D")
					label=6
					binary_label=0
					cp $image data/06_class
					cp $image data/00_class
					;;
				"MULTI")
					label=7
					binary_label=1
					cp $image data/07_class
					cp $image data/01_class
					;;
				"ALL")
					label=8
					binary_label=1
					cp $image data/08_class
					cp $image data/01_class
					;;
				"INTRO")
					label=9
					binary_label="NaN"
					cp $image data/09_class
					cp $image data/NaN_class
					;;
				"CREDITS")
					label=10
					binary_label="NaN"
					cp $image data/10_class
					cp $image data/NaN_class
					;;
				*)
					echo "Warning : unknown speaker"
					;;
			esac
			echo "$image;$label;$binary_label" >> data/labels.csv
			echo "*** results : $image - $label - $binary_label" 
			
			break	
		fi 
	done
	let "current_time+=delta_time"
done

