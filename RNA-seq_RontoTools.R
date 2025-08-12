# Load data
df <- read.csv("GSE104131_counts_genes_only.csv", check.names = FALSE)


# Remove duplicate gene_symbol entries, keeping the first occurrence
df_no_dupes <- df[!duplicated(df$gene_symbol), ]


# ---- Build count matrix from GSM* columns only ----
stopifnot("gene_symbol" %in% colnames(df_no_dupes))
rownames(df_no_dupes) <- df_no_dupes$gene_symbol
gsm_cols <- grepl("^GSM", colnames(df_no_dupes))
stopifnot(any(gsm_cols))  # ensure GSM columns exist

count_mat <- df_no_dupes[, gsm_cols, drop = FALSE]

# ensure integer matrix (round if needed)
count_mat[] <- lapply(count_mat, function(x) as.integer(round(as.numeric(x))))
count_mat <- as.matrix(count_mat)

meta_df <- read.csv("GSE104131_filtered_metadata.csv", check.names = FALSE)

# Set rownames of metadata to GSM IDs (assumed to be the 1st column)
rownames(meta_df) <- as.character(meta_df[[1]])


# Keep only samples present in BOTH count_mat and meta_df (and same order)
common <- intersect(colnames(count_mat), rownames(meta_df))
count_mat <- count_mat[, common, drop = FALSE]
meta_df   <- meta_df[common, , drop = FALSE]

# 4) Use 'tissue' as the group (normal vs tumor)
meta_df$tissue <- factor(meta_df$tissue)
if ("normal prostate" %in% levels(meta_df$tissue)) {
  meta_df$tissue <- relevel(meta_df$tissue, ref = "normal prostate")
}

table(meta_df$tissue)


# 5) Create DESeq2 object and filter low-count genes
dds <- DESeqDataSetFromMatrix(countData = count_mat,
                              colData   = meta_df,
                              design    = ~ tissue)
dds <- dds[rowSums(counts(dds)) > 10, ] 

dds <- DESeq(dds)

# 7) Get results (FDR = 0.05), ordered by adjusted p-value
res <- results(dds, alpha = 0.05)

res <- res[order(res$padj), ]
res_df <- as.data.frame(res)

# 8) Look at the top differentially expressed genes
head(res_df)

write.csv(res_df, "deseq2_results_simple.csv", row.names = TRUE)

# Load results
res_df <- read.csv("deseq2_results_simple.csv", check.names = FALSE)

# Filter DEGs based on criteria
deg_df <- subset(res_df, padj < 0.05 & (log2FoldChange > 1 | log2FoldChange < -1))


# Move rownames into a column named GENE_SYMBOL
deg_df <- cbind(GENE_SYMBOL = rownames(deg_df), deg_df)

# Remove row names
rownames(deg_df) <- NULL

deg_df <- deg_df[, -ncol(deg_df)]

write.table(deg_df, file = "DEG_filtered.tsv", sep = "\t", quote = FALSE, row.names = FALSE)

nrow(deg_df)

res_df <- read.csv("deseq2_results_simple.csv", check.names = FALSE)

write.table(res_df, file = "deseq2_results_simple.tsv", sep = "\t", quote = FALSE, row.names = FALSE)




#pathway

library(ROntoTools)
library(org.Hs.eg.db)
library(graph)

deg_df <- read.table("deseq2_results_simple.tsv", header = TRUE, sep = "\t", stringsAsFactors = FALSE)


# Step 2: Map SYMBOL to Entrez IDs
deg_df$entrez_id <- mapIds(
  org.Hs.eg.db,
  keys = deg_df$GENE_SYMBOL,
  column = "ENTREZID",
  keytype = "SYMBOL",
  multiVals = "first"
)

deg_df <- deg_df[!is.na(deg_df$entrez_id), ]
deg_df$entrez_id <- paste0("hsa:", deg_df$entrez_id)  # KEGG format

# --- Keep only rows with Entrez IDs and deduplicate by Entrez ---
deg_df <- deg_df[!is.na(deg_df$entrez_id), ]
deg_df <- deg_df[!duplicated(deg_df$entrez_id), ]
cat("After dedup by Entrez:", nrow(deg_df), "rows\n")

# step 3: Filter DEGs by your cutoff
deg_df_DEGs <- subset(
  deg_df,
  (log2FoldChange > 1 | log2FoldChange < -1) & padj < 0.05
)
cat("Number of DEGs:", nrow(deg_df_DEGs), "\n")

# Step 3: Define vectors for PE input
fc <- deg_df$log2FoldChange
names(fc) <- deg_df$entrez_id
fc <- setNames(deg_df_DEGs$log2FoldChange, deg_df_DEGs$entrez_id)

pv <- deg_df$pvalue
names(pv) <- deg_df$entrez_id
pv <- setNames(deg_df_DEGs$pvalue, deg_df_DEGs$entrez_id)

ref <- unique(deg_df$entrez_id)


# Step 4: Get KEGG pathway graphs for human
kpg <- keggPathwayGraphs("hsa", verbose = TRUE)



# Step 5: Assign edge weights based on interaction type
kpg <- setEdgeWeights(kpg)


# Step 6: Assign node weights using significance of p-values
kpg <- setNodeWeights(kpg, weights = alpha1MR(pv), defaultWeight = 1)

# Try fewer permutations for exploration
pe_res_new <- pe(x = fc, graphs = kpg, ref = ref, nboot = 500, verbose = TRUE)


# Explore raw combined p-values
pe_summary <- Summary(pe_res_new)

# Add pathway names to ALL results
kpn <- keggPathwayNames("hsa")
pe_summary$PathwayName <- kpn[rownames(pe_summary)]

# Step 10: Save all pathway results
write.csv(pe_summary, "all_pathways_results_ontotools_new.csv", row.names = TRUE)


# Apply multiple testing correction on pComb
pe_summary$pComb_Bonferroni <- p.adjust(pe_summary$pComb, method = "bonferroni")
pe_summary$pComb_FDR <- p.adjust(pe_summary$pComb, method = "fdr")

# Identify significant pathways (Bonferroni < 0.05 OR FDR < 0.05)
sig_pe_bonf <- pe_summary[pe_summary$pComb_Bonferroni < 0.05, ]
sig_pe_fdr  <- pe_summary[pe_summary$pComb_FDR < 0.05, ]


# Save into two separate CSV files
write.csv(sig_pe_bonf, "bonferroni_significant_pathways_raw.csv", row.names = TRUE)
write.csv(sig_pe_fdr, "fdr_significant_pathways_raw.csv", row.names = TRUE)

