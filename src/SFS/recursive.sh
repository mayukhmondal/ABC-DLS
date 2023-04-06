#!/bin/bash
imp=0 #starting improvement is 0 so that it can at least go through one cycle
touch Narrowed.csv All.csv #To reuse already ran simulations. All.csv has all the run. Narrowed only has Newrange parameters
cp Startrange.csv Oldrange.csv #hardrange or Starting Range as you like to call it. move it the Oldrange to initialise the loop
while [ "$(echo "$imp < 0.95"| bc -l)"  -eq 1 ] #if no improvement is less than 95 stop the loop. we reached convergence
do
	echo "Params,Lower,Upper,imp" | cat - Oldrange.csv | tr "," "\t"  #this will print the current state in the terminal
	snakemake -q --jobs 6    #snakemake running quietly
	imp=$(cut -f4  -d ","  Newrange.csv |sort -n | head -g 1) #calculating the improvement we got for one run
	mv Newrange.csv Oldrange.csv # to start it again for another round
done
mv Oldrange.csv Finalrange.csv #done and create the final result