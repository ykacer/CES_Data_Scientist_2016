#/!bin/bash

rm *.log*



echo "\n********************************"
echo   "*** EXERCISES ELEVAGE LAPINS ***"
echo   "********************************"

if [ ! -d data/ ]
then
	mkdir data
fi

if [ -d data/db ]
then
	rm -rf data/db
fi
mkdir data/db

# connect to mongodb server
echo -e "\n*** connect to localhost server"
xterm -e "mongod --dbpath data/db --verbose --logpath mongod_tp.log" &

# deal with elevage database
echo -e "\n*** launch mongo client for elevage database"
sleep 1
mongo lapins.js # --shell

echo "\n***************************************"
echo   "*** EXERCISES EARTHQUAKES JSON FILE ***"
echo   "***************************************"

# deal with earthquakes database
echo -e "\n*** import earthquakes database"
mongoimport -d geodb --type json -c earthquakes --file earthquakes.json
echo -e "\n*** launch mongo client for earthquakes database"
mongo earthquakes.js

echo -e "\n*** kill all mongod process"
killall mongod

# replication 1 master - various slaves
echo -e "\n*******************************************"
echo -e   "* REPLICATION ONE MASTER - VARIOUS SLAVES *"
echo -e   "*******************************************"
echo -e "\n*** create database folder for master and slaves localhost servers"

if [ -d data/master1/ ]
then
	rm -rf data/master1
fi
mkdir data/master1

if [ -d data/slave1/ ]
then
	rm -rf data/slave1
fi
mkdir data/slave1

if [ -d data/slave2/ ]
then
	rm -rf data/slave2
fi
mkdir data/slave2

echo -e "\n*** launch localhost master1"
xterm -e "mongod --master --dbpath data/master1 --verbose --logpath mongod_master1.log" &

echo -e "\n*** launch localhost slave1 connected to localhost master1"
xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave1 --port 27018 --verbose --logpath mongod_slave1.log" &

echo -e "\n*** launch localhost slave2 connected to localhost master1"
xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave2 --port 27019 --verbose --logpath mongod_slave2.log" &

echo -e "\n*** launch 10.136.38.4 slave3 connected to localhost master1"
xterm -e "mongod --slave --source localhost:27017 --dbpath /home/ethiy/Documents/mnogodb/data/slave --bind_ip 10.136.38.4 --port 27020 --verbose --logpath mongod_slave3.log" &

sleep 3 # wait for replication

echo -e "\n*** launch client connected to localhost master1"
mongo --port 27017 --eval "
			print('--- database list');
			db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
			db = db.getSiblingDB('local'); 
			print('--- collection list in local database');
			db.getCollectionNames().forEach( function(doc) {print(tojson(doc))} );
			print('--- me collection in local database')
			db.me.find().forEach( function(doc) {print(tojson(doc))})" 

echo -e "\n*** launch client connected to localhost slave1"
mongo --port 27018 --eval "
			print('--- set read slave OK');
			db.getMongo().setSlaveOk();
			print('--- database list');
			db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
			db = db.getSiblingDB('local');
			print('--- sources collection in local database');
			db.sources.find().forEach( function(doc) {print(tojson(doc))} );"

echo -e "\n*** launch client connected to localhost slave2"
mongo --port 27019 --eval "
			print('--- set read slave OK');
			db.getMongo().setSlaveOk();
			print('--- database list');
			db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
			db = db.getSiblingDB('local');
			print('--- sources collection in local database');
			db.sources.find().forEach( function(doc) {print(tojson(doc))} );"

echo -e "\n*** launch client connected to localhost master1"
mongo --eval "
		db = db.getSiblingDB('test');
		db.dropDatabase();
		db.testcollection.insert({name:'Tim',surname:'Hawkins'});
		print('--- database list');
		db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
		print('--- collection list in test database');
		db.getCollectionNames().forEach( function(doc) {print(tojson(doc))} );
		print('--- testcollection collection in test database')
		db.testcollection.find().forEach( function(doc) {print(tojson(doc))} );"

sleep 2 # wait for replication

echo -e "\n*** launch client connected to localhost slave1"
mongo --port 27018 --eval  "
		print('--- set read slave OK');
		db.getMongo().setSlaveOk();
		print('--- database list');
		db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
		db = db.getSiblingDB('test');
		print('--- testcollection collection in test database (verify replication)');
		db.testcollection.find().forEach( function(doc) {print(tojson(doc))})"

echo -e "\n*** launch client connected to localhost slave2"
mongo --port 27019 --eval  "
		print('--- set read slave OK');
		db.getMongo().setSlaveOk();
		print('--- database list');
		db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
		db = db.getSiblingDB('test');
		print('--- testcollection collection in test database (verify replication)');
		db.testcollection.find().forEach( function(doc) {print(tojson(doc))})"

echo -e "\n*** kill all mongod process"
killall mongod
sleep 3 # wait for ending all mongod process

echo -e "\n*******************************************"
echo -e   "* REPLICATION VARIOUS MASTERS - ONE SLAVE *"
echo -e   "*******************************************"

echo -e "\n*** launch localhost master1" 
if [ -d data/master1 ]
then
	rm -rf data/master1
fi
mkdir data/master1

xterm -e "mongod --master --dbpath data/master1 --port 27017 --verbose --logpath mongod_master1.log" &

echo -e "\n*** launch localhost master2" 
if [ -d data/master2 ]
then
	rm -rf data/master2
fi
mkdir data/master2

xterm -e "mongod --master --dbpath data/master2 --port 27022 --verbose --logpath mongod_master2.log" &

#echo -e "\n*** launch localhost slave1 connected to localhost master1 "
#xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave1 --port 27023 --verbose --logpath mongod_master1_to_slave1.log" &

#echo -e "\n*** launch localhost slave1 connected to localhost master2"
#xterm -e "mongod --slave --source localhost:27022 --dbpath data/slave1 --port 27024 --verbose --logpath mongod_master2_to_slave1.log" &

xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave1 --port 27023" &

sleep 5 # wait for master 1, master2 and slave1 connections

echo -e "\n*** launch client connected to localhost master1"
mongo --port 27017 --eval "
	db = db.getSiblingDB('foo');
	db.dropDatabase();
	db.foocollection.insert({foodata:'our second document'});
	print('--- database list');
	db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
	print('--- collection list in foo database');
	db.getCollectionNames().forEach( function(doc) {print(tojson(doc))} );
	print('--- collection foocollection in database foo')
	db.foocollection.find().forEach( function(doc) {print(tojson(doc))} );"

echo -e "\n*** launch client connected to localhost master2"
mongo --port 27022 --eval "
	db = db.getSiblingDB('bar');
	db.dropDatabase();
	db.barcollection.insert({bardata:'our first document'});
	print('--- database list');
	db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
	print('--- collection list in bar database');
	db.getCollectionNames().forEach( function(doc) {print(tojson(doc))} );
	print('--- collection barcollection in database bar')
	db.barcollection.find().forEach( function(doc) {print(tojson(doc))} );"

#echo -e "\n*** launch localhost slave1 connected to localhost master1 "
#xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave1 --port 27023 --verbose --logpath mongod_master1_to_slave1.log" &

#echo -e "\n*** launch localhost slave1 connected to localhost master2"
#xterm -e "mongod --slave --source localhost:27022 --dbpath data/slave1 --port 27024 --verbose --logpath mongod_master2_to_slave1.log" &

#xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave1 --port 27023" &

sleep 5 # wait for replications from master 1 and master 2 to slave1

echo -e "\n*** launch client connected to localhost slave1"
mongo --port 27023 --eval "
	print('--- set read slave Ok');
	db.getMongo().setSlaveOk();
	print('--- database list');
	db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
	db = db.getSiblingDB('bar');
	print('barcollection collection in bar database');
	db.barcollection.find().forEach( function(doc) {print(tojson(doc))} );
	db = db.getSiblingDB('foo');
	print('foocollection collection in foo database');
	db.foocollection.find().forEach( function(doc) {print(tojson(doc))} );"
