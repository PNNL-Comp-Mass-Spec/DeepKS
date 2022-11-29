loading_fn <- function(libnames, is_in_cran) {
  for (i in seq_along(libnames)) {
    if (suppressPackageStartupMessages(!require(libnames[i], character.only = TRUE, quietly = TRUE))) {
      if (is_in_cran[i]) {
        install.packages(libnames[i])
      }
      else{
        if (suppressPackageStartupMessages(!require("BiocManager", character.only = TRUE, quietly = TRUE))) {
          install.packages("BiocManager")
        }
        BiocManager::install(libnames[i])
      }
    }
  }
}

loading_fn(c("dplyr", "tidyverse", "readxl", "tibble", "Biostrings"), c(T, T, T, T, F))

# suppress editor warnings by showing bindings
library(dplyr)
library(readxl)
library(tibble)
library(Biostrings)

read_data <-
  function(n,
           data_path = NULL,
           include_uniprot_id = TRUE,
           disct = TRUE) {
    
    excel.data <- KIN_ORGANISM <- GENE <- SUB_GENE <- SUB_ORGANISM <- SUB_MOD_RSD <- KIN_ACC_ID <- `SITE_+/-7_AA` <- site <- flank_seq <- kinase <- uniprot_id <- NULL
    
    if (is.null(data_path)) {
      x0  <- excel.data
    }
    else{
      if (include_uniprot_id) {
        x0 <- read_xlsx(data_path) %>%
          transmute(
            kinase = GENE,
            site = paste0(SUB_GENE, "-", SUB_MOD_RSD),
            flank_seq = gsub("_", "X", toupper(`SITE_+/-7_AA`)),
            uniprot_id = KIN_ACC_ID
          )
      }
      
      else{
        x0 <- read_xlsx(data_path) %>%
          dplyr::filter(KIN_ORGANISM == SUB_ORGANISM, # remove cases of autophosphorylation
                        KIN_ORGANISM == 'human') %>%
          transmute(
            kinase = GENE,
            site = paste0(SUB_GENE, "-", SUB_MOD_RSD),
            flank_seq = gsub("_", "X", toupper(`SITE_+/-7_AA`))
          )
      }
    }
    
    
    ## Group x0 by kinase; only choose distinct site, flank_seq ----
    if (disct) {
      x0 <- x0 %>%
        distinct(site, flank_seq, .keep_all = T) %>%
        group_by(kinase, uniprot_id)
    } else {
      x0 <- x0 %>%
        group_by(kinase, uniprot_id)
    }
    
    ## Get top n kinases and sequences ----
    kin_inds <-
      data.frame(
        kinase = toupper(paste(group_keys(x0)$kinase, "|", group_keys(x0)$uniprot_id, sep = "")),
        indices = group_rows(x0),
        num_sites = group_size(x0)
      ) %>%
      arrange(desc(num_sites))
    
    num_sites <- kin_inds$num_sites
    
    names(num_sites) <- kin_inds$kinase
    
  
    top_n_x0 <- x0 %>%
      mutate(num_sites = num_sites[toupper(paste(kinase, "|", uniprot_id, sep = ""))]) %>%
      arrange(desc(num_sites)) %>%
      ungroup()
    
    aa_top_n <- top_n_x0 %>%
      distinct(site, flank_seq) %>%
      deframe() %>%
      AAStringSet()
    
    return(list(aa_top_n, top_n_x0))
  }