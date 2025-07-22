# Load necessary libraries
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(c("ROntoTools", "org.Hs.eg.db", "graph"))

library(ROntoTools)
library(org.Hs.eg.db)
library(graph)

deg_df <- read.table("differential_expression.tsv", header = TRUE, sep = "\t", stringsAsFactors = FALSE)


# Step 2: Map SYMBOL to Entrez IDs
deg_df$entrez_id <- mapIds(
  org.Hs.eg.db,
  keys = deg_df$gene,
  column = "ENTREZID",
  keytype = "SYMBOL",
  multiVals = "first"
)

deg_df <- deg_df[!is.na(deg_df$entrez_id), ]
deg_df$entrez_id <- paste0("hsa:", deg_df$entrez_id)  # KEGG format

# Step 3: Define vectors for PE input
fc <- deg_df$log2_fold_change
names(fc) <- deg_df$entrez_id

pv <- deg_df$p_value
names(pv) <- deg_df$entrez_id

ref <- unique(deg_df$entrez_id)

# Step 4: Get KEGG pathway graphs for human
kpg <- keggPathwayGraphs("hsa", verbose = TRUE)


# Step 5: Assign edge weights based on interaction type
kpg <- setEdgeWeights(
  kpg,
  edgeTypeAttr = "subtype",
  edgeWeightByType = list(
    activation = 1,
    inhibition = -1,
    expression = 1,
    repression = -1
  ),
  defaultWeight = 0
)

# Step 6: Assign node weights using significance of p-values
kpg <- setNodeWeights(kpg, weights = alpha1MR(pv), defaultWeight = 1)

# Try fewer permutations for exploration
pe_res <- pe(x = fc, graphs = kpg, ref = ref, nboot = 500, verbose = TRUE)

# Explore raw combined p-values
pe_summary <- Summary(pe_res)
head(pe_summary[order(pe_summary$pComb), ])


significant_pe <- pe_summary[pe_summary$pPert.fdr < 0.05, ]

significant_pe <- pe_summary[
  pe_summary$pPert.fdr < 0.05 & pe_summary$pAcc.fdr < 0.05, 
]

kpn <- keggPathwayNames("hsa")
significant_pe$PathwayName <- kpn[rownames(significant_pe)]



