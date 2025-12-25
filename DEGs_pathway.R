#if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
#BiocManager::install(c("org.Hs.eg.db", "KEGGREST", "ROntoTools", "graph"))



library(org.Hs.eg.db)

library(KEGGREST)

library(ROntoTools)


library(graph)


deg_df <- read.table("differential_expression.tsv", header = TRUE, sep = "\t", stringsAsFactors = FALSE)

# Filter DEGs
deg_df <- deg_df[!is.na(deg_df$gene), ]
# Filter DEGs using FDR < 0.05 and |log2FC| > 1
degs <- deg_df[deg_df$FDR < 0.05 & abs(deg_df$log2_fold_change) > 1, "gene"]

deg_full <- deg_df[deg_df$FDR < 0.05 & abs(deg_df$log2_fold_change) > 1, ]

write.csv(deg_full, file = "filtered_DEGs_with_info.csv", row.names = FALSE)

sig_pathways <- c("path:hsa04060", "path:hsa04740", "path:hsa05160", "path:hsa05200")

deg_entrez <- mapIds(org.Hs.eg.db,
                     keys = degs,
                     column = "ENTREZID",
                     keytype = "SYMBOL",
                     multiVals = "first")
deg_entrez <- deg_entrez[!is.na(deg_entrez)]

kpg <- keggPathwayGraphs("hsa", verbose = FALSE)

pathway_genes <- unique(unlist(lapply(sig_pathways, function(p) {
  if (p %in% names(kpg)) {
    nodes(kpg[[p]])
  } else {
    character(0)
  }
})))

# Remove "hsa:" prefix
pathway_genes <- gsub("hsa:", "", pathway_genes)

pathway_genes 

deg_genes_in_pathways <- intersect(deg_entrez, pathway_genes)

# Optional: get gene symbols
deg_genes_in_pathways_symbols <- mapIds(org.Hs.eg.db,
                                        keys = deg_genes_in_pathways,
                                        column = "SYMBOL",
                                        keytype = "ENTREZID",
                                        multiVals = "first")

print(deg_genes_in_pathways_symbols)
