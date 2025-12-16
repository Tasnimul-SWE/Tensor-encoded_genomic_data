#!/bin/bash

VCF="clinvar.vcf.gz"
OUTDIR="gene_tsv_new"

mkdir -p "$OUTDIR"

while read -r GENE; do
    echo "Processing $GENE"

    # Header
    echo -e "Chromosome\tStart_Position\tEnd_Position\tReference_Allele\tTumor_Seq_Allele2\trsID\tGene\tClinical_Significance\tReview_Status\tMolecular_Consequence" \
        > "$OUTDIR/${GENE}.tsv"

    bcftools view -i "INFO/GENEINFO~\"$GENE\"" "$VCF" \
    | bcftools query -f '%CHROM\t%POS\t%POS\t%REF\t%ALT\t%INFO/RS\t%INFO/GENEINFO\t%INFO/CLNSIG\t%INFO/CLNREVSTAT\t%INFO/MC\n' \
    >> "$OUTDIR/${GENE}.tsv"

done < unique_genes.txt
