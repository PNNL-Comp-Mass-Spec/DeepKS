library(dendextend)
library(circlize)
library(tidyverse)
library(viridis)
library(ggplot2)
library(grImport2)
library(rsvg)
library(readr)
library(stringr)
library(broman)
library(numbers)
library(this.path)
library(Cairo)
set.seed(0)

set_wd <- function() {
  cur_file_loc <- this.path()
  dir <- str_split(cur_file_loc, "/")
  path <- do.call(file.path, dir[[1]][1:(length(dir[[1]]) - 1)] %>% as.list())
  setwd(path)
}

set_wd()

is_dark <- function(color) {
  r <- str_sub(color, 2, 3) %>% hex2dec()
  g <- str_sub(color, 4, 5) %>% hex2dec()
  b <- str_sub(color, 6, 7) %>% hex2dec()
  lum <- sqrt(0.2126 * r^2 + 0.587 * g^2 + 0.114 * b^2) / 255
  return(lum < 0.5)
}

kfg <- read.csv("../../data/preprocessing/kin_to_fam_to_grp_826.csv") %>%
  mutate(mtx_names = paste(gsub("[\\(\\)\\*]", "", Kinase), sep = "|", Uniprot)) # %>%

the_subset <- seq_len(nrow(kfg)) # 100:175

dm <- read.csv("../../data/preprocessing/pairwise_mtx_826.csv", row.names = 1) %>%
  filter(rownames(.) %in% kfg$mtx_names)
# Distance matrix
d <- dist(dm)
# Hierarchical clustering dendrogram
hc <- set(as.dendrogram(hclust(d, method = "ward.D2")), "branches_lwd", 0.25)
dend_order <- rownames(dm)[order.dendrogram(hc)]
hc <- dendextend::prune(hc, dend_order[setdiff(seq_len(nrow(dm)), the_subset)], reindex_dend = F)

num_families <- kfg %>%
  filter(mtx_names %in% dend_order[the_subset]) %>%
  select(Family) %>%
  dplyr::distinct() %>%
  nrow(.)

num_groups <- kfg %>%
  filter(mtx_names %in% dend_order[the_subset]) %>%
  select(Group) %>%
  dplyr::distinct() %>%
  nrow(.)

kfg <- kfg %>%
  filter(mtx_names %in% dend_order[the_subset]) %>%
  mutate(Group_index = as.numeric(as.factor(Group)))

dend_order <- rownames(dm)[order.dendrogram(hc)]
ct <- cutree(hc, k = num_families)
cut_heights <- heights_per_k.dendrogram(hc)
cut_height <- cut_heights[num_families]
kin_to_clust <- ct[dend_order]
cluster_order <- kin_to_clust %>% unique()

color_vector <- data.frame(clust = cluster_order) %>%
  mutate(color = ifelse(clust %% 2 == 0, "#001eff", "#f58700")) %>%
  arrange(clust) %>%
  .[["color"]]

hc <- hc %>%
  color_branches(clusters = kin_to_clust, col = color_vector)

par(cex = 0.25)

num_fam_levels <- 10
cmap_groups <- circlize::colorRamp2(c(1, (num_groups + 1) / 2, num_groups), c("#00001c", "#1b5bff", "#e4f2ff"))
cmap_families <- circlize::colorRamp2(c(0, (num_fam_levels - 1) / 2, num_fam_levels - 1), c("#b00000", "#ffff00", "#00bba5"))

vals <- c()
counter <- 1
offset_factor <- 29
stopifnot(coprime(num_families, offset_factor))
while (length(vals) < num_families) {
  vals <- append(vals, ((counter * offset_factor) %% num_families) + 1)
  counter <- counter + 1
}
vals_inv <- 1:num_families
count <- 1
for (v in vals) {
  vals_inv[v] <- count
  count <- count + 1
}

fam_color_order <- vals_inv

stopifnot(sort(fam_color_order) == 1:num_families)

fam_anno_data <- kfg

rownames(fam_anno_data) <- paste(gsub("[\\(\\)*]", "", fam_anno_data$Kinase), "|", fam_anno_data$Uniprot, sep = "")

fam_anno_data <- fam_anno_data %>%
  .[dend_order, ] %>%
  mutate(Family_index = as.numeric(
    factor(Family,
      levels = (Family %>% unique())
    )
  ) - 1) %>%
  mutate(Group_color = cmap_groups(Group_index)) %>%
  mutate(Family_color = cmap_families(Family_index %% num_fam_levels))

key_group <- fam_anno_data %>%
  distinct(Group, Group_index) %>%
  deframe()

key_family <- fam_anno_data %>%
  distinct(Family, Family_index %% num_fam_levels) %>%
  deframe()

make_main <- function(fin = NULL, cex = 1) {
  if (is.null(fin)) {
    par(cex = cex, family = "FPL Neu", fig = c(0, 1, 0, 1))
  } else {
    par(cex = cex, family = "FPL Neu", fin = fin)
  }
  circos.par(
    cell.padding = rep(0, 4),
    track.margin = c(0.005, 0.005)
  )
  circos.initialize("a", xlim = c(0, length(dend_order))) # only one sector

  circos.track(
    ylim = c(0, 1),
    bg.border = NA,
    track.height = 0.1,
    panel.fun = function(x, y) {
      for (i in seq_len(nrow(fam_anno_data))) {
        circos.rect(i - 1, 0, i, 1, col = fam_anno_data[i, "Group_color"], border = NA)
        circos.text(i - 0.5, 0.5,
          fam_anno_data[i, "Group"],
          facing = "clockwise",
          niceFacing = TRUE,
          cex = 0.3, # 1,
          col = ifelse(is_dark(fam_anno_data[i, "Group_color"]), "white", "black")
        )
      }
    }
  )

  circos.track(
    ylim = c(0, 1),
    bg.border = NA,
    track.height = 0.1,
    panel.fun = function(x, y) {
      for (i in seq_len(nrow(fam_anno_data))) {
        circos.rect(i - 1, 0, i, 1, col = fam_anno_data[i, "Family_color"], border = NA)
        circos.text(i - 0.5, 0.5,
          fam_anno_data[i, "Family"],
          facing = "clockwise",
          niceFacing = TRUE,
          cex = 0.3, # .6,
          col = ifelse(is_dark(fam_anno_data[i, "Family_color"]), "white", "black")
        )
      }
    }
  )


  circos.track(
    ylim = c(0, 1),
    bg.border = NA,
    track.height = 0.04,
    panel.fun = function(x, y) {
      for (i in seq_len(length(dend_order))) {
        circos.text(
          i - 0.5,
          0,
          dend_order[i],
          adj = c(0, 0.5),
          facing = "clockwise",
          niceFacing = TRUE,
          cex = 0.2 # ifelse(str_length(dend_order[i]) > 15, 0.33, 0.65)
        )
      }
    }
  )

  dend <- hc
  dend_height <- attr(dend, "height")
  circos.track(
    ylim = c(0, dend_height),
    bg.border = NA,
    track.height = 0.4, # 0.5
    panel.fun = function(x, y) {
      circos.dendrogram(dend)
      circos.segments(0, dend_height - cut_height, length(dend_order), dend_height - cut_height, lty = "dotted", lwd = 0.5)
    }
  )

  # legend(
  #   x = -1.5,
  #   y = 1,
  #   legend = names(key_group),
  #   fill = cmap_groups(key_group),
  #   title = "Group Color Map",
  #   cex = 1,
  #   border = "#FFFFFF00"
  # )

  # legend(
  #   x = -1.5,
  #   y = -1,
  #   legend = c("Alternating Tree Cut Clusters", "Alternating Tree Cut Clusters", "Cut Height for\n# clusters = # families"),
  #   col = c("#001eff", "#f58700", "black"),
  #   title = "Dendrogram Colors",
  #   cex = 1,
  #   border = "#FFFFFF00",
  #   yjust = 0,
  #   lwd = unit(1, "mm"),
  #   lty = c("solid", "solid", "dotted"),
  #   text.width = 0.5
  # )
}

################################################################################

make_legend <- function() {
  fn <- "legend.pdf"
  cairo_pdf(
    fn,
    family = "FPL Neu",
    width = 2,
    height = 2,
    pointsize = 8
  )
  start <- 133
  circos.par(
    cell.padding = rep(0, 4),
    track.margin = c(0.01, 0.01)
  )
  circos.initialize("a", xlim = c(0, 90)) # only one sector
  par(cex = 1)
  circos.track(
    ylim = c(0, 1.5),
    bg.border = NA,
    track.height = 0.2,
    panel.fun = function(x, y) {
      for (i in start:(start + 10)) {
        print(i)
        circos.rect(i - 1 - 90 / 4 - start,
          0,
          i - 90 / 4 - start,
          1,
          col = fam_anno_data[i, "Group_color"],
          border = NA
        )
      }
      circos.text(-10, 0.5, "Kinase Group\nClassification", cex = 0.33)
    }
  )

  circos.track(
    ylim = c(0, 1.5),
    bg.border = NA,
    track.height = 0.2,
    panel.fun = function(x, y) {
      for (i in start:(start + 10)) {
        circos.rect(i - 1 - 90 / 4 - start,
          0,
          i - 90 / 4 - start,
          1,
          col = fam_anno_data[i, "Family_color"],
          border = NA
        )

        circos.text(i - 0.5 - 90 / 4 - start,
          0.5,
          fam_anno_data[i, "Family"],
          facing = "clockwise",
          niceFacing = F,
          cex = 0.2
        )
      }
      circos.text(-9, 0.5, "Kinase Family\nClassification", cex = 0.33)
    }
  )
  dev.off()
  knitr::plot_crop(fn)
  system("pdftocairo -svg legend.pdf legend.svg")
  svg_txt <- read_file("legend.svg")
  svg_txt <- str_remove(svg_txt, "<rect.*fill:rgb\\(100%,100%,100%\\);fill-opacity:1;stroke:none;.*")
  write(svg_txt, file = "legend.svg")
}

CairoPDF(file = "phylo_families_Cairo.pdf", family = "FPL Neu", width = 10, height = 10)
main <- make_main(c(10, 10), cex = 0.5)
dev.off()
# make_legend()
# rsvg_svg("legend.svg", "legend-B.svg")
# color_legend <- readPicture("legend-B.svg")
# grid.picture(color_legend, width = unit(0.18, "npc"), height = unit(0.18, "npc"), x = unit(.95, "npc"), y = unit(.95, "npc"), just = c("right", "top"))
# par(cex = 0.75, mar = rep(2, 4))
# title("Phylogenetic Kinase Tree Correlation with Group and Family Annotations")
