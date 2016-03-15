#!/bin/bash


# connect to localhost mongodb server
xterm -e "mongod --dbpath data/db" &

# get enernoc data
if [ ! -d data ]
then
	wget https://open-enernoc-data.s3.amazonaws.com/anon/all-data.tar.gz
       	mkdir data
	tar -xzvf all-data.tar.gz -C data

	# convert csv site file into Unix format
	sed -i -e 's/\r/\n/g' data/meta/all_sites.csv

	# convert csv conso files into Unix format
	for conso_csv in `ls data/csv/*csv`
	do
		sed -i -e 's/\r/\n/g' $conso_csv
	done
fi

# import site data into mongodb collection 
if [ ! -d data/db ]
then
	mkdir data/db
	mongoimport --type csv -d enernoc -c sites --file data/meta/all_sites.csv --headerline
	# import conso data into mongodb collections
	for conso_csv in `ls data/csv/*.csv`
	do
		site_number=$(basename ${conso_csv%%.*})
		echo  "currently importing site $site_number ..."
		mongoimport --type csv -d enernoc --file $conso_csv --headerline
		mongo mongodb_1_data_integration.js --eval "var site_id = $site_number"
	done
fi
	
# 

mongo mongodb_2_data_queries.js;
