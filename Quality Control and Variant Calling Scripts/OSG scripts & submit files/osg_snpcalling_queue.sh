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
bwa index -a bwtsw k12.fasta

echo "Sample Indexed to the reference genome"

mkdir -p sam bam raw_vcf flt_vcf $1
echo "Results directories created"

    echo "Working with file $1"

    sam=sam/${1}.sam
    bam=bam/${1}.bam
    sorted_bam=bam/${1}.sorted.bam
    raw_vcf=raw_vcf/${1}.raw.vcf
    flt_vcf=flt_vcf/${1}.flt.vcf
    snps=$1/${1}.snps.txt

    echo "Aligning sample $1 to reference genome started: $(date)"
    bwa mem -t 4 k12.fasta $1\_1.fastq.gz $1\_2.fastq.gz > $sam
    echo "Al1gning sample $1 to reference genome finished: $(date)"
    echo "Converting sample $1 to BAM started: $(date)"
    samtools view -bS $sam > $bam
    echo "Converting sample $1 to BAM finished: $(date)"
    echo "Sorting sample $1 started: $(date)"
    samtools sort $bam -o $sorted_bam
    echo "Sorting sample $1 finished: $(date)"
    echo "Indexing sample $1 started: $(date)"
    samtools index $sorted_bam
    echo "Indexing sample $1 finished: $(date)"
    echo "Calling SNPs for sample $1 started: $(date)"
    samtools mpileup -uf k12.fasta $sorted_bam | bcftools call -cv - > $raw_vcf
    echo "Calling SNPs for sample $1 finished: $(date)"
    echo "Filtering SNPs for sample $1 started: $(date)" 
    bcftools view $raw_vcf | vcfutils.pl varFilter > $flt_vcf
    echo "Filtering SNPs for sample $1 finished: $(date)"
    echo "Extracting SNPs for sample $1 started: $(date)"
    grep -v "#" $flt_vcf | cut -f1,2,4,5 > $snps
    echo "Extracting SNPs for sample $1 finished: $(date)"

tar -czf $1.tar.gz $1

cp $1.tar.gz ../snps.tar.gz

pwd
ls 
#cat snps/ERR434259.snps.txt