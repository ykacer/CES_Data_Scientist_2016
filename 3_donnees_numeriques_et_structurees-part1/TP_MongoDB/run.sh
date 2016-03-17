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
xterm -e "mongod --dbpath data/db --verbose --logpath mongod_tp.log" &

# deal with elevage database
mongo lapins.js

# deal with earthquakes database
mongoimport -d geodb --type json -c earthquakes --file earthquakes.json
mongo earthquakes.js

killall mongod

# replication master(s) slave(s)
if [ ! -d data/master/ ]
then
	mkdir data/master
fi
xterm -e "mongod --master --dbpath data/master --verbose --logpath mongod_master.log" &

if [ ! -d data/slave/ ]
then
	mkdir data/slave
fi
xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave --port 27018" &

if [ ! -d data/slave2/ ]
then
	mkdir data/slave2
fi
xterm -e "mongod --slave --source localhost:27017 --dbpath data/slave2 --port 27019" &

xterm -e "mongod --slave --source localhost:27017 --dbpath /home/ethiy/Documents/mnogodb/data/slave --bind_ip 192:168:181:2--port 27020 -- verbose --logpath mongod_slave.log" &


