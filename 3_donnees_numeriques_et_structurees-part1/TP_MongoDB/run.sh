#/!bin/bash

rm *.log*

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
echo -e "\n*** launch localhost server"
xterm -e "mongod --dbpath data/db --verbose --logpath mongod_tp.log" &

sleep 1 # wait for server connection

echo "\n********************************"
echo   "*** EXERCISES ELEVAGE LAPINS ***"
echo   "********************************"

# deal with elevage database
echo -e "\n*** launch mongo client for elevage database"
mongo lapins.js # --shell


echo "\n***************************************"
echo   "*** EXERCISES EARTHQUAKES JSON FILE ***"
echo   "***************************************"

# deal with earthquakes database
echo -e "\n*** import earthquakes database"
mongoimport -d geodb --type json -c earthquakes --file earthquakes_big.geojson

sleep 1 # wait for importing json file

echo -e "\n*** launch mongo client for earthquakes database"
mongo earthquakes.js

echo -e "\n*** kill all mongod process"
killall mongod

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

sleep 3 # wait for test database replication from master1

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

echo -e "\n*** create database folder for master2 localhost server"
if [ -d data/master2 ]
then
	rm -rf data/master2
fi
mkdir data/master2

echo -e "\n*** launch localhost master1" 
xterm -e "mongod --master --dbpath data/master1 --port 27017 --verbose --logpath mongod_master1.log" &

echo -e "\n*** launch localhost master2" 
xterm -e "mongod --master --dbpath data/master2 --port 27022 --verbose --logpath mongod_master2.log" &

sleep 3 # wait for master1, master2 connections

echo -e "\n*** launch client connected to localhost master1"
mongo --port 27017 --eval "
	db = db.getSiblingDB('foo');
	db.dropDatabase();
	db.foocollection.insert({foodata:'our first foo document'});
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
	db.barcollection.insert({bardata:'our first bar document'});
	print('--- database list');
	db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
	print('--- collection list in bar database');
	db.getCollectionNames().forEach( function(doc) {print(tojson(doc))} );
	print('--- collection barcollection in database bar')
	db.barcollection.find().forEach( function(doc) {print(tojson(doc))} );"

echo -e "\n*** launch localhost slave1"
xterm -e "mongod --slave --dbpath data/slave1 --port 27023 --verbose --logpath mongod_slave1.log" &

sleep 3 # wait for slave1 connection

echo -e "\n*** launch client connected to localhost slave1"
mongo --port 27023 --eval "
	print('--- set read slave Ok');
	db.getMongo().setSlaveOk();
	print('--- database list');
	db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
	db = db.getSiblingDB('local');
	/*print('--- add master1');*/
	/*db.sources.insert({host:'localhost:27017'});*/
	print('--- add master2 to sources collection');
	db.sources.insert({host:'localhost:27022',only:'bar'});
	print('--- sources collection in local database');
	db.sources.find().forEach( function(doc) {print(tojson(doc))} );"

sleep 3 # wait for database foo and bar replication from master1 and master2

echo -e "*** launch client connected to localhost slave1"
mongo --port 27023 --eval "
	print('--- set read slave Ok');
	db.getMongo().setSlaveOk();
	print('--- database list');
	db.getMongo().getDBNames().forEach( function(doc) {print(tojson(doc))} );
	db = db.getSiblingDB('bar');
	print('--- barcollection collection in bar database (verify replication)');
	db.barcollection.find().forEach( function(doc) {print(tojson(doc))} );
	db = db.getSiblingDB('foo');
	print('--- foocollection collection in foo database (verify replication)');
	db.foocollection.find().forEach( function(doc) {print(tojson(doc))} );"

echo -e "\n*** kill all mongod process"
killall mongod
sleep 3 # wait for ending all mongod process


echo -e "\n*********************"
echo -e  "*** PARTITIONNING ***"
echo -e  "*********************"

if [ -d data/config ]
then
	rm -rf data/config
fi
mkdir data/config

echo "*** launch config server"
xterm -e "mongod --port 27025 --dbpath data/config -configsvr" &

sleep 1 # wait for config server connection

echo "*** launch controller"
xterm -e "mongos --configdb localhost:27025 --port 27024 --chunkSize 1" &

if [ -d data/shard0 ]
then
	rm -rf data/shard0
fi
mkdir data/shard0
xterm -e "mongod --port 27026 --dbpath data/shard0 --shardsvr" &

if [ -d data/shard1 ]
then
	rm -rf data/shard1
fi
mkdir data/shard1
xterm -e "mongod --port 27027 --dbpath data/shard1 --shardsvr" &

sleep 6

echo "*** launch client connected to controller"
mongo --port 27024 --eval "
	print('--- create database admin');
	db = db.getSiblingDB('admin');
	print('--- add shard0 server to controller');
	db.runCommand({addshard : 'localhost:27026',allowLocal:true});
	print('--- add shard1 server to controller');
	db.runCommand({addshard : 'localhost:27027',allowLocal:true});
	print('--- add phones database');
	phones = db.getSisterDB('phones');
	db.runCommand({enablesharding:'phones'});
	db.runCommand({shardcollection:'phones.testcollection',key:{_id:1}});
	db = db.getSiblingDB('phones');
	load('populate_phones.js');
	populatePhones(800,5550000,5750000)
	print('--- number of documents in testcollection of phones database');
	print(db.testcollection.find().count());
	print('--- two first documents of testcollection');
	db.testcollection.find().limit(2).forEach( function(doc) {print(tojson(doc)) } );"

echo -e "\n*** launch client connected to controller"
mongo --port 27024 --eval "
	db = db.getSiblingDB('phones');
	print('--- number of documents in testcollection collection in phones database');
	print(db.testcollection.find().count());"

echo -e "\n*** launch client connected to first shard server"
mongo --port 27026 --eval "
	db = db.getSiblingDB('phones');
	print('--- number of documents in testcollection collection in phones database');
	print(db.testcollection.find().count());"

echo -e "\n*** launch client connected to second shard server"
mongo --port 27027 --eval "
	db = db.getSiblingDB('phones');
	print('--- number of documents in testcollection collection in phones database');
	print(db.testcollection.find().count());"
