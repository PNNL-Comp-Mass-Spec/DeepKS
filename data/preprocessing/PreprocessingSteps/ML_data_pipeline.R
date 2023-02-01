loading_fn <- function(libnames) {
  tempfile <- tempfile()

  cran_fn <- function(pkg) {
    if (suppressPackageStartupMessages(!require(pkg, character.only = TRUE, quietly = TRUE))) {
      cat(sprintf("Attempting to install `%s` through CRAN...", pkg))
      sink(file = tempfile, type = "output")
      install.packages(libnames[i], repos = "http://cran.us.r-project.org", quiet = T)

      find.package(pkg)
      sink()
      return(TRUE)
    }
  }

  github_fn <- function(pkg, gh_prefix = "") {
    if (suppressPackageStartupMessages(!require(pkg, character.only = TRUE, quietly = TRUE))) {
      cat(sprintf("Attempting to install `%s` through Github (prefix `%s`)...", pkg, gh_prefix))
      sink(file = tempfile, type = "output")
      remotes::install_github(file.path(gh_prefix, pkg), quiet = T)

      find.package(pkg)
      sink()
      return(TRUE)
    }
  }

  bioc_fn <- function(pkg) {
    if (suppressPackageStartupMessages(!require("BiocManager", character.only = TRUE, quietly = TRUE))) {
      cat("Installing BiocManager...")
      install.packages("BiocManager", repos = "http://cran.us.r-project.org")
      library(BiocManager)
    }
    cat(sprintf("Attempting to install `%s` through BiocManager...", libnames[i]))
    sink(file = tempfile, type = "output")
    suppressMessages(BiocManager::install(pkg, quiet = T))

    find.package(pkg)
    sink()
    return(TRUE)
  }

  errors <- c()
  err_fn <- function(x) {
    sink()
    return(F)
  }
  for (i in seq_along(libnames)) {
    outcome <- T
    if (suppressPackageStartupMessages(!require(libnames[i], character.only = TRUE, quietly = TRUE))) {
      outcome <- tryCatch(cran_fn(libnames[i]), error = err_fn)
      msg <- ifelse(outcome, " Success!\n", " Failed.\n")
      cat(msg)
    }

    if (!outcome) {
      outcome <- tryCatch(bioc_fn(libnames[i]), error = err_fn)
      msg <- ifelse(outcome, " Success!\n", " Failed.\n")
      cat(msg)
    }

    if (!outcome) {
      outcome <- tryCatch(github_fn(libnames[i], "PNNL-Comp-Mass-Spec"), error = function(e) {
        errors[length(errors) + 1] <<- libnames[i]
        sink()
        return(F)
      })
      msg <- ifelse(outcome, " Success!\n", " Failed.\n")
      cat(msg)
    }
  }

  if (length(errors) != 0) {
    cat("\n\nErrors:\n")
    print(errors)
    cat("\n")
  } else {
    cat("Completed requested installation with no problems!\n")
  }

  return(errors)
}





errs <- loading_fn(c("dplyr", "tidyverse", "readr", "seqinr", "DECIPHER", "this.path", "stringr"))

# suppress editor warnings by showing bindings
library(dplyr)
library(readr)
library(this.path)
library(stringr)

set_wd <- function() {
  cur_file_loc <- this.path()
  dir <- str_split(cur_file_loc, "/")
  path <- do.call(file.path, dir[[1]][1:(length(dir[[1]]) - 1)] %>% as.list())
  setwd(path)
}

set_wd()

source("Kinase_Inference_Testing.R")
cat("[R] Reading in data...\n")
mode <- colnames(read.delim("../../../config/mode.cfg"))

n <- 1000000 # How many kinases to pull from PhosphositePlusDB (max number of kinases is  so use `1000000` to get all)

dat <- read_data(n, data_path = "../../raw_data/PSP_script_download.xlsx", disct = F)
top_n_x0 <- dat[[2]]

at_least_n_sites <- 1

top_n_x0 <- top_n_x0 %>%
  dplyr::rename(lab = kinase, seq = flank_seq) %>%
  mutate(lab = toupper(lab)) %>% 
  filter(num_sites >= at_least_n_sites) %>% #  & !grepl("/", lab)
  na.omit() %>%
  arrange(desc(num_sites), lab) %>%
  mutate(class = 1)

large_fn <- sprintf("../../raw_data/raw_data_%d.csv", nrow(top_n_x0))
write.csv(top_n_x0, file = large_fn, row.names = FALSE, quote = FALSE)


obtain_associated_fastas <- function(df, outfile) {
  uniprot_id <- . <- NULL
  lab_list <- df %>%
    distinct(uniprot_id) %>%
    as.list() %>%
    unlist(.$uniprot_id) %>%
    unname()
  partial_res <- ""
  front_idx <- 1
  max_indices <- 199
  breakout <- 0
  while (front_idx <= length(lab_list) && breakout < 20) {
    back_idx <- min(front_idx + max_indices, length(lab_list))
    partial_url <- paste("(accession:", lab_list[front_idx:back_idx], ")", sep = "", collapse = "+OR+")
    first_part <- "https://rest.uniprot.org/uniprotkb/stream?format=fasta&includeIsoform=true&query="
    url <- paste(first_part, partial_url, sep = "")
    cat(sprintf("[R] Downloading proteins %d to %d...\n", front_idx, back_idx))
    if (curl::curl_fetch_memory(url)$status != 200) {
      tf <- tempfile()
      curl::curl_fetch_disk(url, tf)
      errors <- readLines(tf, warn = F)
      print("There was a URL problem:")
      print(errors)
      stop("Bad Request.")
    }
    con <- curl::curl(url)
    temp <- readLines(con)
    close(con)
    temp_lines <- paste(temp, collapse = "\n")
    partial_res <- paste(partial_res, temp_lines, sep = "\n")
    front_idx <- back_idx + 1
    breakout <- breakout + 1
  }
  seq_splits <- stringr::str_split(partial_res, "\n>")[[1]]
  names(seq_splits) <- sub("^[^\\|]+\\|([^\\|]+)\\|.*", "\\1", seq_splits)
  seq_splits <- seq_splits[lab_list]
  partial_res <- paste(seq_splits, collapse = "\n>")
  partial_res <- paste(">", partial_res, sep = "")
  write_file(paste(partial_res, "\n", sep = ""), outfile)
}

tempfile_ <- "../../raw_data/cache/UNIPROT_kinase_sequences.fasta"

if (!file.exists(tempfile_)) {
  obtain_associated_fastas(top_n_x0, tempfile_)
}

cat("[R] Assembing sequences...\n")
rf <- read.fasta(tempfile_, whole.header = T)
aass_accid <- lapply(names(rf) %>% as.list(), function(x) {
  return(str_split(x, "\\|")[[1]][2])
})
aass_gn <- str_match(lapply(names(rf) %>% as.list(), function(x) {
  return(str_split(x, "\\|")[[1]][3])
}), ".*GN=(.*?)($|\\sPE=.*)")[, 2]
aass_values <- AAStringSet(unlist(unname(lapply(rf, function(x) {
  return(toupper(paste(x, collapse = "")))
}))) %>% as.vector())

fi_name <- NULL
if (mode == "alin") {
  cat("[R] Computing alignments...\n")
  alignments <- AlignSeqs(aass_values, iterations = 3, refinements = 2)
  input_aligned_df <- data.frame(kinase = as.character(aass_accid), kinase_seq = as.vector(alignments), gene_name = aass_gn)
  fi_name <- sprintf("../../raw_data/kinase_seq_alin_%d.csv", nrow(input_aligned_df))
  write.table(input_aligned_df, file = fi_name, quote = FALSE, row.names = FALSE, sep = ",")
} else {
  input_df <- data.frame(kinase = toupper(as.character(aass_accid)), kinase_seq = toupper(as.vector(aass_values)), gene_name = toupper(aass_gn))
  fi_name <- sprintf("../../raw_data/kinase_seq_%d.csv", nrow(input_df))
  write.table(input_df, file = fi_name, quote = FALSE, row.names = FALSE, sep = ",")
}
cat("[R] Completed.\n")
cat("[@python_capture_output]")
cat(sprintf("%s\n", normalizePath(fi_name)))
cat("[@python_capture_output]")
cat(sprintf("%s", normalizePath(large_fn)))
