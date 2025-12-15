#!/bin/bash

VCF="clinvar.vcf.gz"
OUTDIR="gene_tsv"

mkdir -p "$OUTDIR"

while read -r GENE; do
    echo "Processing $GENE"

    bcftools view -i "INFO/GENEINFO~\"$GENE\"" "$VCF" \
    | bcftools query -f '%ID\t%INFO/CLNSIG\t%INFO/CLNREVSTAT\n' \
    > "$OUTDIR/${GENE}.tsv"

done < unique_genes.txt

