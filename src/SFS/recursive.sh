#!/bin/bash
imp=0
touch Narrowed.csv
cp Startrange.csv Oldrange.csv
while [ "$(echo "$imp < 0.95"| bc -l)"  -eq 1 ]
do
	snakemake -q --jobs 6
	imp=$(cut -f4  -d ","  Newrange.csv |sort -n | head -n 1)
	mv Newrange.csv Oldrange.csv
done
mv Oldrange.csv Finalrange.csv