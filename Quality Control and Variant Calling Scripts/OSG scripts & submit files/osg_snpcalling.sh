ENVNAME=snp-calling

ENVDIR=$ENVNAME

export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

echo "Environment setup completed";

cd bash_results

echo "switched to results folder";
ls

#Step 1: Index the reference genome
bwa index -a bwtsw $1

echo "Sample Indexed to the reference genome $1"

mkdir -p sam bam raw_vcf flt_vcf snps
echo "Results directories created"

for i in $(cat $2)
do
    echo "Working with file $i"

    sam=sam/${i}.sam
    bam=bam/${i}.bam
    sorted_bam=bam/${i}.sorted.bam
    raw_vcf=raw_vcf/${i}.raw.vcf
    flt_vcf=flt_vcf/${i}.flt.vcf
    snps=snps/${i}.snps.txt

    echo "Aligning sample $i to reference genome started: $(date)"
    bwa mem -t 4 $1 $i\_1.fastq.gz $i\_2.fastq.gz > $sam
    echo "Aligning sample $i to reference genome finished: $(date)"
    echo "Converting sample $i to BAM started: $(date)"
    samtools view -bS $sam > $bam
    echo "Converting sample $i to BAM finished: $(date)"
    echo "Sorting sample $i started: $(date)"
    samtools sort $bam -o $sorted_bam
    echo "Sorting sample $i finished: $(date)"
    echo "Indexing sample $i started: $(date)"
    samtools index $sorted_bam
    echo "Indexing sample $i finished: $(date)"
    echo "Calling SNPs for sample $i started: $(date)"
    samtools mpileup -uf $1 $sorted_bam | bcftools call -cv - > $raw_vcf
    echo "Calling SNPs for sample $i finished: $(date)"
    echo "Filtering SNPs for sample $i started: $(date)" 
    bcftools view $raw_vcf | vcfutils.pl varFilter > $flt_vcf
    echo "Filtering SNPs for sample $i finished: $(date)"
    echo "Extracting SNPs for sample $i started: $(date)"
    grep -v "#" $flt_vcf | cut -f1,2,4,5 > $snps
    echo "Extracting SNPs for sample $i finished: $(date)"


done

mkdir my_output
tar -czf my_output.tar.gz snps/

cp my_output.tar.gz ../my_job.output.tar.gz