#!/bin/bash

file=$1
header=`head -n 1 $1`

sed '1 d' $1 | \
sed 's/\,/\t/g' | \
awk 'BEGIN {OFS=FS="\t"} {gsub(/\./,",",$24)}1' | \
sort -t $'\t' -k 2n -k 3n -k 24g | \
awk 'BEGIN {OFS=FS="\t"} {gsub(/\,/,".",$24)}1' | \
sed 's/\t/\,/g' | \
sort -t',' -k 2,3 -u > ${file%%.*}_clean.csv
sed -i "1i${header}" ${file%%.*}_clean.csv


