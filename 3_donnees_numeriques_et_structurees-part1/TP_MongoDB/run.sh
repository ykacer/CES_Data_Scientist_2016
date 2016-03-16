#/!bin/bash

if [ ! -d data/ ]
then
	mkdir data
fi
if [ ! -d data/db ]
	mkdir data/db
fi

# connect to mongodb server
xterm -e "mongod --dbpath data/db" &
