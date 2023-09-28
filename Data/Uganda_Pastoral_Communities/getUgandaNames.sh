#!/bin/bash

# Set the input file name
input_file="sampleids.txt"

# Set the output file name
output_file="samples.csv"

# Create the header row of the CSV file
echo "sampleName" > $output_file

# Extract the sample names from the input file, remove the '_trim1.fastq.gz' and '_trim2.fastq.gz' parts, and sort them
samples=$(sed 's/_trim[12]\.fastq\.gz//' $input_file | sort | uniq)

# Loop through the unique sample names and add them to the CSV file
for sample in $samples
do
    echo $sample >> $output_file
done
