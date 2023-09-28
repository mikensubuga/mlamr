#A script to run quality control on the raw data using fastp tool on the paired end data
#!/bin/bash

for i in $(cat $1)
do
    echo "Processing sample $i"
    fastp -i $i\_1.fastq -I $i\_2.fastq -o $i\_1.fastq.gz -O $i\_2.fastq.gz -h $i\_fastp.html -j $i\_fastp.json
done

#copy the html, json and gz files to the results folder
mkdir -p Results/01-quality

for i in $(cat $1)
do
    cp $i\_fastp.html Results/01-quality
    cp $i\_fastp.json Results/01-quality
    cp $i\_1.fastq.gz Results/01-quality
    cp $i\_2.fastq.gz Results/01-quality
done
