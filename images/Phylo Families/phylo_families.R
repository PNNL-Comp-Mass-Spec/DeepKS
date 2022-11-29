library(dendextend)
library(circlize)
library(tidyverse)
library(viridis)
library(ggplot2)
library(grImport2)
library(rsvg)




set.seed(0)

dm <- read.csv("../data/raw_data/mtx_396.csv", row.names = 1)
# Distance matrix
d <- dist(dm)
# Hierarchical clustering dendrogram
hc <- as.dendrogram(hclust(d, method = "ward.D2"))
num_families <- read.csv("~/Desktop/Kinase-Substrate-Deep-Learning/data/preprocessing/kin_to_vec.csv") %>%
  select(Family) %>%
    dplyr::distinct() %>% 
      nrow(.)





dend_order <- colnames(dm)[order.dendrogram(hc)]
ct <- cutree(hc, k = num_families)
cut_heights <- heights_per_k.dendrogram(hc)
cut_height <- cut_heights[num_families]
kin_to_clust <- ct[dend_order]
cluster_order <- kin_to_clust %>% unique()


color_vector <- data.frame(clust = cluster_order) %>% 
  mutate(color = ifelse(clust %% 2 == 0, "#001eff", "#f58700")) %>% 
    arrange(clust) %>% 
      .[['color']]



hc <- hc %>%
  color_branches(clusters = kin_to_clust, col = color_vector)

par(cex = 0.25)

# Circular dendrogram
# cd <- circlize_dendrogram(hc,
#                     labels_track_height = NA,
#                     dend_track_height = 0.5)

cmap_groups <- circlize::colorRamp2(c(1, 5, 10), c("#000044", "#ff00ff", "#ffc9ea"))
cmap_families <- circlize::colorRamp2(c(0, 4, 9), c("#660000", "#ffff00","#00aaff"))
# fam_color_order <- sample(1:num_families, num_families)

vals <- c()
counter <- 1
while(length(vals) < 124){
  vals <- append(vals, ((counter*23) %% 124) + 1)
  counter <- counter + 1
}
vals_inv <- 1:124
count <- 1
for(v in vals){
  vals_inv[v] <- count
  count <- count + 1
}

fam_color_order <- vals_inv

stopifnot(sort(fam_color_order) == 1:124)

# fam_anno_maxes <- read.csv("../data/preprocessing/kin_to_fam_to_grp.csv") %>% 
#   group_by(Group) %>% 
#     mutate(fam_factor = as.numeric(as.factor(Family))) %>% 
#       summarise(n_colors = max(fam_factor)) %>% 
#         deframe() %>% 
#           sort()

# cmap_list <- list()
# for(i in seq_along(fam_anno_maxes)){
#   to_app <- circlize::colorRamp2(c(1, fam_anno_maxes[i]/2, fam_anno_maxes[i]), c("#660000", "#ffff00","#00ffff"))
#   cmap_list[[names(fam_anno_maxes)[i]]] <- to_app
# }
# 
# fam_anno_data <- read.csv("../data/preprocessing/kin_to_fam_to_grp.csv") %>%
#   group_by(Group) %>% 
#   mutate(fam_idx_num = as.numeric(as.factor(Family))) %>% 
#   ungroup() %>% 
#   mutate(Group_cmap_factor_num = match(Group, names(fam_anno_maxes))) %>% 
#   mutate(Group_color = cmap_groups(Group_cmap_factor_num)) %>% 
#   add_column(data.frame(Family_color = apply(., 1, function(x){cmap_list[[x[['Group']]]](as.numeric(x[["fam_idx_num"]]))}))) %>% 
#   as.data.frame()

fam_anno_data <- read.csv("../data/preprocessing/kin_to_fam_to_grp.csv") %>%
  mutate(Group_index = as.numeric(as.factor(Group)))

rownames(fam_anno_data) <- fam_anno_data$Kinase

fam_anno_data <- fam_anno_data %>% 
  .[dend_order, ] %>% 
  mutate(Family_index = as.numeric(factor(Family, levels = (Family %>% unique()))) - 1) %>% 
  mutate(Group_color = cmap_groups(Group_index)) %>% 
  mutate(Family_color = cmap_families(Family_index %% 10))  

# rownames(fam_anno_data) <- fam_anno_data$Kinase
# fam_anno_data <- fam_anno_data[dend_order, ]

# apply(., 1, function(x){cmap_list[[x[['Group']]]](as.numeric(x[["fam_idx_num"]]))})

# group_anno_data <- read.csv("../data/preprocessing/kin_to_grp.csv")[, c('Kinase', 'Group')] %>% deframe()
# num_groups <- length(group_anno_data %>% unique())
# colorscale <- circlize::rand_color(num_groups, luminosity = 'bright')
# anno_colors <- colorscale[group_anno_data[dend_order] %>% as.factor() %>% as.numeric()]

key_group <- fam_anno_data %>% 
  distinct(Group, Group_index) %>% 
    deframe()

key_family <- fam_anno_data %>% 
  distinct(Family, Family_index %% 10) %>% 
  deframe()

# par(cex = 0.25)
# colored_bars(anno_colors, circlize_dendrogram(hc,
#                                               labels_track_height = NA,
#                                               dend_track_height = 0.5), rowLabels = c("Kinase Group"))
# par(cex = 0.25)


make_main <- function(fin = NULL) {
  if(is.null(fin)){
    par(cex = 0.4, family = "Palatino", fig = c(0, 1, 0, 1))
  } else {
    par(cex = 0.4, family = "Palatino", fin = fin, )
  }
  circos.par(cell.padding = rep(0, 4),
             track.margin = c(0.005, 0.005))
  circos.initialize("a", xlim = c(0, length(dend_order))) # only one sector
  
  circos.track(
    ylim = c(0, 1),
    bg.border = NA,
    track.height = 0.05,
    panel.fun = function(x, y) {
      for (i in 1:nrow(fam_anno_data)) {
        circos.rect(i - 1, 0, i, 1, col = fam_anno_data[i, 'Group_color'], border = NA)
      }
    }
  )
  
  circos.track(
    ylim = c(0, 1),
    bg.border = NA,
    track.height = 0.05,
    panel.fun = function(x, y) {
      for (i in 1:nrow(fam_anno_data)) {
        circos.rect(i - 1, 0, i, 1, col = fam_anno_data[i, 'Family_color'], border = NA)
        circos.text(i - 0.5, 0.5, 
                    fam_anno_data[i, 'Family'],
                    facing = "clockwise",
                    niceFacing = TRUE,
                    cex = 0.3)
      }
    }
  )
  
  
  circos.track(
    ylim = c(0, 1),
    bg.border = NA,
    track.height = 0.05,
    panel.fun = function(x, y) {
      for (i in seq_along(dend_order)) {
        circos.text(
          i - 0.5,
          0,
          dend_order[i],
          adj = c(0, 0.5),
          facing = "clockwise",
          niceFacing = TRUE,
          cex = 0.5
        )
      }
    }
  )
  
  dend <- hc
  dend_height = attr(dend, "height")
  circos.track(
    ylim = c(0, dend_height),
    bg.border = NA,
    track.height = 0.4,
    panel.fun = function(x, y) {
      circos.dendrogram(dend)
      circos.segments(0, dend_height - cut_height, length(dend_order), dend_height - cut_height, lty = "dotted", lwd = 0.5)
    }
  )
  
  rg <- recordPlot()
  
  # 
  # legend(
  #   x = 1,
  #   y = 0,
  #   legend = names(key_family),
  #   fill = cmap_families(key_family),
  #   title = "Family Color Map",
  #   cex = 0.4,
  #   border = "#FFFFFF00",
  #   ncol = 3
  # )
  
  legend(
    x = -1.5,
    y = 1,
    legend = names(key_group),
    fill = cmap_groups(key_group),
    title = "Group Color Map",
    cex = 1,
    border = "#FFFFFF00"
  )
  
  legend(
    x = -1.5,
    y = -1,
    legend = c("Alternating Tree Cut Clusters", "Alternating Tree Cut Clusters", "Cut Height for\n# clusters = # families"),
    col = c("#001eff", "#f58700", "black"),
    title = "Dendrogram Colors",
    cex = 1,
    border = "#FFFFFF00",
    yjust = 0,
    lwd = unit(1, 'mm'),
    lty = c("solid", "solid", "dotted"),
    text.width = 0.5
  )
}

make_main()

################################################################################
################################################################################
# circos.clear()
# 
# cmap_grp <- circlize::colorRamp2(c(1, 10, 20), c("#660000", "#ffff00","#00ffff"))
# legend(x = "right", fill = lapply(1:20, function(x){cmap_grp(x)}) %>% unlist() %>% substr(1, 7), legend = 1:20)
# legend(x = "left", fill = lapply(1:10, function(x){cmap(x)}) %>% unlist() %>% substr(1, 7), legend = 1:10)

make_legend <- function() {
  fn <- "legend.pdf"
  cairo_pdf(
    fn,
    family = "Palatino",
    width = 2,
    height = 2,
    pointsize = 8
  )
  start <- 133
  circos.par(cell.padding = rep(0, 4),
             track.margin = c(0.01, 0.01))
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
                    col = fam_anno_data[i, 'Group_color'],
                    border = NA)
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
                    col = fam_anno_data[i, 'Family_color'],
                    border = NA)
        
        circos.text(i - 0.5 - 90/4 - start, 
                    0.5, 
                    fam_anno_data[i, 'Family'],
                    facing = "clockwise",
                    niceFacing = F,
                    cex = 0.2)
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

make_main()
# make_legend()
rsvg_svg("legend.svg", "legend-B.svg")
color_legend <- readPicture("legend-B.svg")
grid.picture(color_legend, width = unit(0.18, "npc"), height = unit(0.18, "npc"), x = unit(.95, "npc"), y = unit(.95, "npc"), just = c("right", "top"))
par(cex = 0.75, mar = rep(2, 4))
title("Phylogenetic Kinase Tree Correlation with Group and Family Annotations")
