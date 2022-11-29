library(tidyverse)
library(dplyr)
library(Biostrings)
library(seqinr)

V1 <- NULL
V2 <- NULL
KIN_ACC_ID <- NULL
GENE <- NULL
. <- NULL
kinase <- NULL
Family <- NULL
V1 <- NULL
kinase_seq <- NULL
kdseq <- NULL

calculate_alignment_dists <- function(aa_string_set = NULL, names, cs = NULL, alin = NULL) {
    names <- unname(unlist(names))
    if (is.null(cs)) {
            message("Computing Alignment Distance Matrix ----")
            seqs <- as.character(aa_string_set)
            
            ## Compute all combinations of sequences (not considering sequence and itself to be a comb)
            message("gathering all combinations of sequences...")
            cs <- as.data.frame(t(combn(seqs, 2)))
        }

    if (is.null(alin)) {
        ## Compute all pairwise alignments (vectorized)
        message("computing all pairwise alignments...")
        pa <-
            pairwiseAlignment(
                cs$V1,
                cs$V2,
                scoreOnly = T,
                type = "local",
                substitutionMatrix = "BLOSUM62"
            )
        # Organize back into distance matrix
        
    } else {
        pa <- alin
    }
    dist_df <- cbind(cs, pa) %>%
            mutate(V1 = as.factor(V1), V2 = as.factor(V2))
    
    
    if (!is.null(aa_string_set)) {
        # Create distance matrix and fill in the diagonal w/zeroes
        dist_matr <-
            diag(pairwiseAlignment(aa_string_set, aa_string_set, scoreOnly = T, type = "local", substitutionMatrix = "BLOSUM62"), nrow = length(names), ncol = length(names))
        dist_matr[t(upper.tri(dist_matr))] <-
            dist_df$pa # Fill in upper and lower triangulars
        dist_matr <- t(dist_matr)
        dist_matr[lower.tri(dist_matr)] <- dist_df$pa
    }
    else{
        dist_matr <- matrix(pa, nrow = length(names), ncol = length(names)) # Create distance matrix and fill in the diagonal w/zeroes
    }
    
    
    # Name rows and columns, convert to dist object
    rownames(dist_matr) <- names
    colnames(dist_matr) <- names
    return(dist_matr)
}

f <- read.csv('../raw_data/kinase_seq_396.txt', sep = '\t')
names <- f$gene_name
aass <- f$kinase_seq

dm <- calculate_alignment_dists(aass, names)