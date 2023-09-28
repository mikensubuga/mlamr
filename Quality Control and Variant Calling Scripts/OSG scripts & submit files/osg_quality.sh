#A script to run quality control on the raw data using fastp tool on the paired end data
#!/bin/bash

#set -e

#Extracting tools from Conda and setting up the environment
ENVNAME=snp-calling

ENVDIR=$ENVNAME

export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

echo "Environment setup completed";
#Quality
cd data 

echo "Switched to Data folder";
ls

for i in $(cat $1);
do
    echo "Processing sample $i";
    fastp -i $i\_1.fastq -I $i\_2.fastq -o $i\_1.fastq.gz -O $i\_2.fastq.gz -h $i\_fastp.html -j $i\_fastp.json;

    echo "Completed processing for sample $i";
done

pwd
ls 

echo "$(pwd)"

#Step added to compress the outputs after analysis from OSG
mkdir my_output
mv *.gz *.html *json my_output/
tar -czf my_job.output.tar.gz my_output/

cp my_job.output.tar.gz ../my_job.output.tar.gz

