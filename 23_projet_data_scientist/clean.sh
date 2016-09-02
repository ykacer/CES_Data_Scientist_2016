#!/bin/bash

header=`head -n 1 $1`
echo 1i"$header"
sed '1 d' $1 | \
sed 's/\,/\t/g' | \
awk 'BEGIN {OFS=FS="\t"} {gsub(/\./,",",$24)}1' | \
sort -t $'\t' -k 2n -k 3n -k 24g | \
awk 'BEGIN {OFS=FS="\t"} {gsub(/\,/,".",$24)}1' | \
sed 's/\t/\,/g' | \
sort -t',' -k 2,3 -u > "$1"_clean.csv
sed -i "1i${header}" "$1"_clean.csv


