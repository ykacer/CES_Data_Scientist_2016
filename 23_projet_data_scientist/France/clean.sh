#!/bin/bash

sed 's/\,/\t/g' LANDSAT_8_134311.csv | \
awk 'BEGIN {OFS=FS="\t"} {gsub(/\./,",",$24)}1' | \
sort -t $'\t' -k 2n -k 3n -k 24g | \
awk 'BEGIN {OFS=FS="\t"} {gsub(/\,/,".",$24)}1' | \
sed 's/\t/\,/g' \
sort -t',' -k 2,3 -u \
> LANDSAT_8_134311_clean.csv
