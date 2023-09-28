#!/bin/bash

# navigate to the folder containing the files
cd /Users/mikensubuga/Downloads/lastset

# loop through all files in the folder
for file in *.snps.txt; do
    # remove the ".snps.txt" extension and store the result in a variable
    new_name=$(echo $file | sed 's/.snps.txt//')
    # rename the file
    mv $file $new_name
done
