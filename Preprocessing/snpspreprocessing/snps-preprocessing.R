#An R program to preprocess the SNP data by the reference allele, variant allele and position based on the position of the reference alleles.
#The program takes the following arguments:
#1. A file containing the SNP data with the Chrom, reference allele, variant allele and position. It then merges the files based on the position of the reference allele.
library(dplyr)
library(data.table)

#Read the file containing the the phenotype data

list <- read.table("samplelist.txt", stringsAsFactors = FALSE)

View(list)
#path = "snp-analysis-results/"

list_vec = list$V1

#construct the file path for the first file
file_path <- file.path("edited-snp-samples", list_vec[1])
merge_data = fread(file_path)


View(merge_data)
#rename the columns
names(merge_data) = c("V1", "REF",list_vec[1])
for (i in seq_along(list_vec[2:1529])){
  
  #construct file path
  file_path <- file.path("edited-snp-samples", list_vec[i])
  new_data = fread(file_path)
  
  merge_data = merge(merge_data, new_data, by="V1", all=T)
  
  names(merge_data)[names(merge_data)=="V2"] = "ref_temp"
  names(merge_data)[names(merge_data)=="V3"] = "ALT"
  merge_data[which(is.na(merge_data$REF)),][,2]=merge_data[which(is.na(merge_data$REF)),][,"ref_temp"]
  merge_data = merge_data[,-"ref_temp"]
  names(merge_data)[names(merge_data)=="ALT"] = list_vec[2:1529][[i]]
}

View(merge_data)
merge_data[is.na(merge_data)] = "N"

encoded_m_data <- merge_data
View(encoded_m_data)
#Label encoding the reference allele and variant allele  with A=1,G=2,C=3,T=4, N=0
encoded_m_data[merge_data=="A"] = 1
encoded_m_data[merge_data=="G"] = 2
encoded_m_data[merge_data=="C"] = 3 
encoded_m_data[merge_data=="T"] = 4
encoded_m_data[merge_data=="N"] = 0

names(encoded_m_data)[names(encoded_m_data)=="V1"] = "Position"

encoded_m_data = as.data.frame(encoded_m_data)
merge_data2 = as.data.frame(lapply(encoded_m_data, as.numeric))
merge_data2
#colnames(merge_data2) = colnames(merge_data)

View(merge_data2)
tinput <- as.data.frame(t(merge_data2))

#Calculate the number of 0 for each column
f <- function(x)
{
  sum(x==0)
}
num_0 <- t(as.data.frame(apply(tinput, 2, f)))

tinput2 <- as.data.frame(t(rbind(num_0, tinput)))
colnames(tinput2)[1] = "Num_0"

## Get those with total number of zeros less than 200
tinput3 <- subset(tinput2, Num_0 < 500)
tinput4 <- as.data.frame(t(tinput3))

View(tinput4)

#Delete the first row
tinput5 <- tinput4[-1,]

#Remove row two called "REF"
tinput5 <- tinput5[-2,]

View(tinput5)

write.csv(tinput5, "snps.csv", quote = FALSE)

#End of workflow -> proceed to merge in separate script after manual cleaning

