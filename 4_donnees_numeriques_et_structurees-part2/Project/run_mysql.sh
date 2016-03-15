#!/bin/bash

# put user name for password user for sql server connection
current_user=''
current_user_password=''

# get enernoc data from website
if [ ! -d data ]
then
	echo "*** currently downloading enernoc data from website"

	wget https://open-enernoc-data.s3.amazonaws.com/anon/all-data.tar.gz
       	mkdir data
	tar -xzvf all-data.tar.gz -C data
	
	echo "*** currently converting .csv files to Unix format"

	# convert csv site file into Unix format
	sed -i -e 's/\r/\n/g' data/meta/all_sites.csv

	# convert csv conso files into Unix format
	for conso_csv in `ls data/csv/*csv`
	do
		sed -i -e 's/\r/\n/g' $conso_csv
	done
fi

# put user name and user password into .cnf file to avoid password connection to mysql server every time
echo "[client]" >> ~/.my.cnf
echo "user=$current_user" >> ~/.my.cnf
echo "password=$current_user_password" >> ~/.my.cnf
echo "host=localhost" >> ~/.my.cnf
chmod 600 ~/.my.cnf

# import site data into mysql tables
echo "*** currently create enernoc sql database and grant privileges on it to current user"

mysql -e "CREATE DATABASE IF NOT EXISTS enernoc; GRANT ALL PRIVILEGES ON enernoc.* TO $current_user@localhost IDENTIFIED BY $current_user_password" -u root -p

echo "*** currenlty importing sites metadata to SITES table"

mysql -e "SOURCE mysql_1_data_integration.sql" -u "$current_user"

echo "*** currently importing conso data to CONSO table"

for conso_csv in `ls data/csv/*.csv`
do
	site_number=$(basename ${conso_csv%%.*})
	echo  "*** currently importing site $site_number ..."
	mysql -e "USE enernoc; LOAD DATA LOCAL INFILE '"$conso_csv"' INTO TABLE CONSO FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\n' IGNORE 1 LINES (timestamp,dttm_utc,value,estimated,anomaly) SET site='$site_number';" -u "$current_user"
done

echo "*** currently removing measure with anomaly"

mysql -e "USE enernoc; DELETE FROM CONSO WHERE anomaly NOT IN ('');" -u "$current_user" # delete measure with anomaly 

echo "*** currently computing Load Curve statistics"
mysql -e "USE enernoc; SOURCE mysql_2_data_queries.sql;" -u "$current_user" 

