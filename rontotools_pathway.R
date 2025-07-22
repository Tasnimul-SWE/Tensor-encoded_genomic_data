
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# Now install SPIA and biomaRt from Bioconductor
BiocManager::install("SPIA")
BiocManager::install("biomaRt")


library(SPIA)
library(org.Hs.eg.db)
library(KEGGREST)

deg_df_1 <- read.table("differential_expression.tsv", header = TRUE, sep = "\t", stringsAsFactors = FALSE)


# 3. Map gene symbols to Entrez IDs
symbol_to_entrez_1 <- mapIds(
  org.Hs.eg.db,
  keys = deg_df_1$gene,
  column = "ENTREZID",
  keytype = "SYMBOL",
  multiVals = "first"
)

# 4. Add Entrez IDs to dataframe
deg_df_1$entrez_id <- symbol_to_entrez_1
deg_df_1 <- deg_df_1[!is.na(deg_df_1$entrez_id), ]


# 5. Create named log2FC vector
fc_vector_1 <- deg_df_1$log2_fold_change
names(fc_vector_1) <- as.character(deg_df_1$entrez_id)


# 6. Define background Entrez IDs
background_entrez_1 <- unique(as.character(na.omit(deg_df_1$entrez_id)))


# 9. Run SPIA
spia_res_1 <- spia(
  de = fc_vector_1,
  all = background_entrez_1,
  organism = "hsa",
  nB = 2000
)

# 10. View result
head(spia_res_1)

significant_pathways_1 <- spia_res_1[spia_res_1$pGFdr < 0.05, ]
print(significant_pathways_1)


significant_pathways_1
# Plot using SPIA's plotP function
plotP(spia_res_1, threshold = 0.05)

BiocManager::install("ROntoTools")

library(ROntoTools)

# Download KEGG pathways for human
kpg <- keggPathwayGraphs("hsa")

spia_ronto_res <- runSPIA(
  de = fc_vector_1,
  all = background_entrez_1,
  pathways = kpg,
  beta = 0.05,
  verbose = TRUE
)

# 10. View result
head(spia_ronto_res)
