'''
Install r tools

1) install.packages("httr2")
2) Sys.which("sh")
3) Sys.which("make")
4) BiocManager::install("pasilla", dependency = TRUE, ask = FALSE)

'''

# Load necessary packages
if (!require("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")  # Install BiocManager if not already installed
}

# Install required Bioconductor packages
BiocManager::install("DESeq2")  # For differential expression analysis
BiocManager::install("pasilla")   # For example datasets
BiocManager::install("ggplot2")   # For data visualization
BiocManager::install("ggrepel")   # For better labeling in plots

# Load the libraries into the R session
library("DESeq2")
library("pasilla")
library("ggplot2")
library("ggrepel")

# Load count data and annotation data from the pasilla package
count_csv <- system.file("extdata", "pasilla_gene_counts.tsv", package = "pasilla", mustWork = TRUE)
annotation_csv <- system.file("extdata", "pasilla_sample_annotation.csv", package = "pasilla", mustWork = TRUE)

# Read the count data into a data frame and convert it to a matrix
count_data <- read.csv(count_csv, sep = "\t", row.names = "gene_id")
count_matrix <- as.matrix(count_data)

# Read the sample annotation data into a data frame
annotation_data <- read.csv(annotation_csv, row.names = 1)
annotation_data$condition <- factor(annotation_data$condition)  # Convert condition to a factor

# Clean up row names by removing 'fb' prefix
rownames(annotation_data) <- sub("fb", "", rownames(annotation_data))

# Check if all sample names in annotation match those in count matrix
all(rownames(annotation_data) %in% colnames(count_matrix))  # Should return TRUE

# Ensure the order of columns in count_matrix matches the row names in annotation_data
all(rownames(annotation_data) == colnames(count_matrix))  # Should return TRUE

# Reorder count matrix to match the order of samples in annotation data
count_matrix <- count_matrix[, rownames(annotation_data)]

# Verify that the reordering was successful
all(rownames(annotation_data) == colnames(count_matrix))  # Should return TRUE

# Create a DESeqDataSet object for differential expression analysis
deseq <- DESeqDataSetFromMatrix(countData = count_matrix, colData = annotation_data, design = ~ condition)

# Perform differential expression analysis using DESeq2
diff_exp_analysis <- DESeq(deseq)

# Extract results from the differential expression analysis
diff_exp_result <- results(diff_exp_analysis)

# Write results to a CSV file for further inspection (corrected file name)
write.csv(as.data.frame(diff_exp_result), file = "./DESeqAnalysis.csv", row.names = TRUE)

# Read the results back into R for visualization and further analysis
dataframe <- read.csv("./DESeqAnalysis.csv", header = TRUE)

# Initialize a new column to classify gene expression status
dataframe$expressed <- "NO"  # Default value

# Classify genes as upregulated or downregulated based on log2 fold change and p-value thresholds
dataframe$expressed[dataframe$log2FoldChange > 0.1 & dataframe$pvalue < 0.05] <- "UP"
dataframe$expressed[dataframe$log2FoldChange < -0.1 & dataframe$pvalue < 0.05] <- "DOWN"

# Extract lists of upregulated and downregulated genes based on classification
upregulated_genes <- rownames(dataframe[dataframe$expressed == "UP", ])
downregulated_genes <- rownames(dataframe[dataframe$expressed == "DOWN", ])

# Write lists of upregulated and downregulated genes to text files for further analysis or reporting
write(upregulated_genes, file = "upregulated_genes.txt")
write(downregulated_genes, file = "downregulated_genes.txt")

# Create a volcano plot to visualize differential expression results
ggplot(data = dataframe, aes(x = log2FoldChange, y = -log10(pvalue), col = expressed)) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed") +  # Vertical lines at -1 and 1 for significance threshold
  geom_hline(yintercept = -log10(0.05), linetype = "dashed") +  # Horizontal line at p-value threshold of 0.05
  geom_point(size = 2) +  # Points representing genes in the plot
  theme_minimal() + 
  scale_color_manual(values = c("turquoise", "grey", "pink"), labels = c("Downregulated", "Not Significant", "Upregulated")) + 
  labs(x = "log2 Fold Change", y = "-log10 P-value", color = "Expression") + 
  ggtitle("Volcano Plot of Differential Expression Analysis")  # Add title to the plot

