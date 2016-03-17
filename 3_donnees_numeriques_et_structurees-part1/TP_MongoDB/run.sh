#/!bin/bash

rm *.log*

if [ ! -d data/ ]
then
	mkdir data
fi

if [ ! -d data/db ]
then
	mkdir data/db
fi

# connect to mongodb server
echo -e "\n*** connect to localhost server"
xterm -e "mongod --dbpath data/db --verbose --logpath mongod_tp.log" &

# deal with elevage database
echo -e "\n*** launch mongo client for elevage database"
sleep 1
mongo lapins.js # --shell

# deal with earthquakes database
echo -e "\n*** import earthquakes database"
mongoimport -d geodb --type json -c earthquakes --file earthquakes.json
echo -e "\n*** launch mongo client for earthquakes database"
mongo earthquakes.js

echo -e "\n*** kill all mongod process"
killall mongod

# replication master(s) slave(s)
echo -e "\n*** create database folder for master and slaves localhost servers"
if [ ! -d data/master/ ]
then
	mkdir data/master
fi

if [ ! -d data/slave/ ]
then
	mkdir data/slave
fi

if [ ! -d data/slave2/ ]
then
	mkdir data/slave2
fi

echo -e "\n*** launch mongo master localhost server"
xterm -e "mongod --master --dbpath data/master --verbose --logpath mongod_master.log" &

echo -e "\n*** launch mongo slave localhost server"
xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave --port 27018" &

echo -e "\n*** launch mongo second slave localhost server"
xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave2 --port 27019" &

# xterm -e "mongod --slave --source localhost:27017 --dbpath /home/ethiy/Documents/mnogodb/data/slave --bind_ip 192:168:181:2--port 27020 -- verbose --logpath mongod_slave.log" &


echo -e "\n*** launch mongo client to master localhost server and look at 'local' database"
sleep 3
mongo --port 27017 --eval "
			db.getMongo().getDBNames()
			db = db.getSiblingDB('local'); 
			db.getCollectionNames();
			db.me.find().forEach( function(doc) {print(tojson(doc))})" 


echo -e "\n*** launch mongo client to slave localhost server and look at 'local' database"
mongo --port 27018 --eval "
			db.getMongo().setSlaveOk();
			db.getMongo().getDBNames()
			db = db.getSiblingDB('local');
			db.getCollectionNames();"

echo -e "\n*** launch mongo client to second slave localhost server and look at 'local' database"
mongo --port 27019 --eval "
			db.getMongo().setSlaveOk();
			db.getMongo().getDBNames()
			db = db.getSiblingDB('local');
			db.getCollectionNames();"

echo -e "\n*** launch mongo client to master localhost server and create 'testdb' database"
mongo --eval "
		db = db.getSiblingDB('testdb');
		db.testcollection.remove({});
		db.testcollection.insert({name:'Tim',surname:'Hawkins'})
		db.testcollection.find().forEach( function(doc) {print(tojson(doc))})"


echo -e "\n*** launch mongo client to slave localhost server and look at 'testdb' database replication"
mongo --port 27018 --eval  "
		db.getMongo().setSlaveOk();
		db = db.getSiblingDB('testdb');
		db.getCollectionNames();
		db.testcollection.find().forEach( function(doc) {print(tojson(doc))})"

echo -e "\n*** launch mongo client to second slave localhost server and look at 'testdb' database replication"
mongo --port 27019 --eval  "
		db.getMongo().setSlaveOk();
		db = db.getSiblingDB('testdb');
		db.getCollectionNames();
		db.testcollection.find().forEach( function(doc) {print(tojson(doc))})"
