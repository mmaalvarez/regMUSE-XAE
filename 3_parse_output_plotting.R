library(tidyverse)
library(data.table)
library(parallel)
library(msm) # rtnorm
#install.packages('terra', repos='https://rspatial.r-universe.dev')
library(NMF) # for NMF to defeature_nameine optimal k signatures
#ENMTools::install.extras("NMF")
library(lsa)
library(sigminer)
library(cowplot)
#library(cluster)
library(ggrepel)
#devtools::install_github("nicolash2/ggdendroplot")
library(ggdendroplot)
library(conflicted)
conflict_prefer("filter", "dplyr")
conflict_prefer("rename", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("map", "purrr")
conflict_prefer("extract", "magrittr")
conflict_prefer("Position", "ggplot2")
conflict_prefer("cosine", "lsa")
conflict_prefer("clusterExport", "parallel")
conflict_prefer("clusterEvalQ", "parallel")
conflict_prefer("parLapply", "parallel")
conflict_prefer("parLapplyLB", "parallel")
conflict_prefer("nmf", "NMF")
conflict_prefer("Position", "ggplot2")


##### samples info

K562 = read_tsv("/g/strcombio/fsupek_cancer3/malvarez/WGS_tumors/somatic_variation/cell_lines/marcel_K562/metadata/WGS_clones_info.tsv") %>% 
  rename("Sample" = "sample_id",
         "alteration" = "genotype KOs") %>%
  mutate(dataset = "Marcel",
         Sample = gsub("_REP1", "", Sample),
         # I consider MSH3-/- as MMRwt
         altered_pathway_or_treatment_type = ifelse((str_detect(alteration, "MSH2|MSH6|MLH1|PMS2") | str_detect(`MMR deficiency expected`, "strong|middle")) & str_detect(alteration, "OGG1|MUTYH|NTH1|NEIL2"),
                                                    "MMR & BER",
                                                    ifelse(str_detect(alteration, "MSH2|MSH6|MLH1|PMS2") | str_detect(`MMR deficiency expected`, "strong|middle"),
                                                           "MMR",
                                                           ifelse(str_detect(alteration, "OGG1|MUTYH|NTH1|NEIL2"),
                                                                  "BER",
                                                                  "control"))),
         alteration = ifelse(alteration != "WT",
                             gsub("$", " KO", alteration),
                             alteration)) %>%  
  select(Sample, dataset, alteration, altered_pathway_or_treatment_type)

iPSC = c("/g/strcombio/fsupek_cancer3/malvarez/WGS_tumors/somatic_variation/cell_lines/kucab_2019/processed/sample_treatments.tsv",
         "/g/strcombio/fsupek_cancer3/malvarez/WGS_tumors/somatic_variation/cell_lines/zou_2021/processed/sample_gene_ko.tsv") %>% 
  # only Sample and info* columns are selected
  map_df(~read_tsv(.x)) %>% 
  rename("alteration" = "info1",
         "altered_pathway_or_treatment_type" = "info2",
         "Sample" = "sample_id") %>% 
  mutate(altered_pathway_or_treatment_type = gsub("^[a-k]_", "", altered_pathway_or_treatment_type),
         altered_pathway_or_treatment_type = gsub("Control", "control", altered_pathway_or_treatment_type),
         dataset = ifelse(str_detect(Sample, "MSM0"),
                          "Kucab",
                          ifelse(str_detect(Sample, "MSK0"),
                                 "Zou",
                                 "ERROR: Unexpected sample name")),
         altered_pathway_or_treatment_type = gsub("DNA damage response inhibitors", "DNA damage resp. inh.", altered_pathway_or_treatment_type),
         # "MMRko" sample with very low mut burden, treat as control
         `altered_pathway_or_treatment_type` = ifelse(Sample == "MSK0.123_s1",
                                                      "control",
                                                      `altered_pathway_or_treatment_type`),
         alteration = ifelse(dataset == "Zou",
                             gsub("$", " KO", alteration),
                             alteration)) %>% 
  select(Sample, dataset, alteration, altered_pathway_or_treatment_type)

petljak = read_tsv("/g/strcombio/fsupek_cancer3/malvarez/WGS_tumors/somatic_variation/cell_lines/petljak_2022/info/metadata.tsv") %>% 
  rename("Sample" = "sample_id",
         "alteration" = "info1",
         "cell_line" = "info2") %>%
  mutate(dataset = "Petljak",
         # I consider MSH3-/- as MMRwt
         altered_pathway_or_treatment_type = ifelse(alteration == "WT",
                                                    "control",
                                                    "APOBEC"),
         alteration = gsub("_KO", " KO", alteration)) %>% 
  select(Sample, dataset, alteration, altered_pathway_or_treatment_type)

# merge datasets metadata
samples_info = bind_rows(K562, iPSC) %>% 
  bind_rows(petljak)



coefficient_Resamp = read_tsv(Sys.glob(paste0("../coefficients_resampling/permuted_coefficients_*iters.tsv")))



#####################################################################
#### NMF
#####################################################################


#### evaluate best combination of n variables and k signatures

coefficient_Resamp = coefficient_Resamp %>% 
  ungroup %>% 
  pivot_wider(names_from = feature_name, values_from = resampled_estimate) %>% 
  unite("Sample", Sample, nIter, sep = "__") %>% 
  # NAs to 0
  mutate_all(~ifelse(is.na(.), 0, .)) %>% 
  # remove features with all 0s
  select(where(~any(. != 0)))

# a) keep positive coefficients, and convert negative to zero 
coefficient_Resamp_posmatrix = coefficient_Resamp %>% 
  mutate_if(is.numeric,
            ~if_else(.<=0, 0, .))
# b) convert positive coefficients to zero, and convert negative to positive
coefficient_Resamp_negmatrix = coefficient_Resamp %>% 
  mutate_if(is.numeric,
            ~if_else(.>=0, 0, abs(.)))
# merge converted coefficients into the NMF input
coefficient_Resamp = merge(coefficient_Resamp_posmatrix,
                           coefficient_Resamp_negmatrix,
                           by = "Sample",
                           suffixes = c("_poscoeff", "_negcoeff")) %>%
  mutate(Sample = gsub("__muts", "_muts", Sample)) %>% 
  separate(Sample, into = c("id", "nIter"), sep = "__", remove = F) %>% 
  mutate(Sample = gsub("__", "_nIter", Sample)) %>% 
  column_to_rownames("Sample") %>% 
  select(-id) %>% 
  split(., f = .$nIter) %>% 
  map(.f = list(. %>% data.matrix))



## Run NMF for each resampled coefficients matrix generated in the previous step

## Parameters and initializing of some objects
set.seed(1)
maxK = 12 # max number of signatures to consider, it will go from 2 to maxK -- shouldn't be larger than nº of features + expected_n_cosmic
# resampled NMF input matrices, generated just once and then re-used for any possible # factors/clusters
nmfHmatAllByFact = list()
nmfWmatAllByFact = list()
nCPUs = 8

## "cl" is only used if using parLapply (creates a set of copies of R running in parallel and communicating over sockets, slower?) instead of mclapply (parallel processing):
#cl = makeCluster(nCPUs)
#clusterExport(cl, list("coefficient_Resamp"))
#clusterEvalQ(cl, library(NMF))

## extract nFact factors...
for(nFact in 2:maxK){
  
  cat(sprintf("Running NMF: nFact %d (all iters)\n", nFact))
  
  # ...from each of the 1000 permuted matrices
  nmfOutputByIter = mclapply(coefficient_Resamp,
                             mc.cores = nCPUs,
                             function(x, nFact){
                               # store colnames
                               orig_colnames_order = colnames(x)[-1] %>% 
                                 gsub("\\&|\\-", "\\.", .)
                               # NAs to 0
                               x[is.na(x)] <- 0
                               # store nIter
                               nIter = x[1,1]
                               x = x[,-1]
                               ## cannot handle all-zero columns (such as MSH6-neg, since all coeff. are +...)
                               # store their colnames
                               all_zero_columns = data.frame(x) %>% 
                                 select_if(colSums(.) <= 0) %>% 
                                 colnames
                               # remove them from matrix before NMF
                               x_nozerocols = x[, colSums(x) > 0]
                               ### NMF
                               nmf_run = NMF::nmf(x_nozerocols, 
                                                  rank = nFact, 
                                                  maxIter = 10000, 
                                                  seed = nIter)
                               # re-add the all-zero columns as zeros
                               all_zero_columns_matrix = data.frame(matrix(rep(0, len = length(all_zero_columns)), 
                                                                           ncol = length(all_zero_columns), 
                                                                           nrow = nrow(nmf_run@fit@H))) %>% 
                                 `colnames<-`(all_zero_columns) %>% 
                                 as.matrix
                               full_matrix = cbind(nmf_run@fit@H, all_zero_columns_matrix) %>% 
                                 data.frame
                               # when making it data.frame the '(' '>' ')' '×' symbols become '.', so revert this
                               full_matrix_colnames = colnames(full_matrix) %>% 
                                 str_replace(., 
                                             "([A,C,G,T])\\.([C,T])\\.([A,C,G,T])\\.([A,C,G,T])_", 
                                             "\\1(\\2>\\3)\\4_") %>% 
                                 str_replace(., "X53BP1", "53BP1")
                               colnames(full_matrix) = full_matrix_colnames
                               full_matrix = full_matrix %>% 
                                 # reorder column names to be the same order as before
                                 select(all_of(orig_colnames_order)) %>%
                                 as.matrix
                               nmf_run@fit@H = full_matrix
                               nmf_run},
                             nFact)
  idString = sprintf("nFact=%03d", nFact)
  nmfHmatAllByFact[[idString]] = matrix(nrow = 0, ncol = ncol(nmfOutputByIter[[1]])); # columns = original features...
  nmfWmatAllByFact[[idString]] = matrix(nrow = 0, ncol = nrow(nmfOutputByIter[[1]])); # columns = samples...
  for (nIter in 1:totalNumIters) {
    nmfOutput = nmfOutputByIter[[nIter]]
    # weights (H)
    rownames(nmfOutput@fit@H) = sprintf("run%03d_fact%02d", nIter, 1:nFact)
    # exposures ("basis", W)
    colnames(nmfOutput@fit@W) = sprintf("run%03d_fact%02d", nIter, 1:nFact)
    nmfHmatAllByFact[[idString]] = nmfHmatAllByFact[[idString]] %>% rbind(nmfOutput@fit@H) # same thing that you get using the coef(NMFfit) command in R   -> loadings
    # WARNING: the rbind causes that the colnames become always ..._nIter1, e.g. MSM0.96_nIter2 --> MSM0.96_nIter1; this shouldn't matter as long as all samples are always in the same column i
    nmfWmatAllByFact[[idString]] = nmfWmatAllByFact[[idString]] %>% rbind(t(nmfOutput@fit@W)) # same thing that you get using the basis(NMFfit) command in R  -> transformed data
    gc()
  }
  gc()
}
#stopCluster(cl)
#rm(idString, nmfOutput, nFact)

## Save clustering results
saveRDS(nmfHmatAllByFact, paste0("heatmap_Ks/", "Hmatrix.Rda")) # signatures
#nmfHmatAllByFact = readRDS(paste0("heatmap_Ks/", "Hmatrix.Rda"))
saveRDS(nmfWmatAllByFact, paste0("heatmap_Ks/", "Wmatrix.Rda")) # exposures
#nmfWmatAllByFact = readRDS(paste0("heatmap_Ks/", "Wmatrix.Rda"))


## [results of pos] + -[results of neg] to have a "-" sign when the larger/non-zero value comes from the neg. submatrix
nmfHmatAllByFact_combined_pos_negcoeff = nmfHmatAllByFact %>% 
  map(., ~as_tibble(.x, rownames = NA)) %>%
  map(., ~rownames_to_column(.x, "id")) %>%
  map(., ~pivot_longer(.x,
                       cols = contains("coeff"),
                       names_to = "dna_repair_feature_name",
                       values_to = "Weight")) %>%
  map(., ~mutate(.x, Weight = ifelse(str_detect(dna_repair_feature_name, "_negcoeff"),
                                     -Weight,
                                     Weight))) %>% 
  map(., ~mutate(.x, dna_repair_feature_name = gsub("_...coeff", "", dna_repair_feature_name))) %>% 
  map(., ~group_by(.x, dna_repair_feature_name, id)) %>%
  map(., ~summarise(.x, Weight = .Primitive("+")(Weight[1], Weight[2]))) %>%
  map(., ~ungroup(.x)) %>%
  ## scale weights per signature so they add up to 1
  #map(., ~group_by(.x, id)) %>% 
  #map(., ~mutate(.x, sumWeight = sum(Weight))) %>% 
  #map(., ~group_by(.x, id, dna_repair_feature_name)) %>% 
  #map(., ~summarise(.x, Weight = Weight/sumWeight)) %>% 
  #map(., ~ungroup(.x)) %>% 
  ## go to original format
  map(., ~pivot_wider(.x, names_from = dna_repair_feature_name, values_from = Weight)) %>%
  # arrange iterations (rows)
  map(., ~arrange(.x, id)) %>%
  map(., ~column_to_rownames(.x, 'id')) %>%
  # arrange dna repair feature_name names (columns)
  map(.f = list(. %>% select(coefficient_Resamp[[1]] %>% 
                               data.frame %>% 
                               select(contains("poscoeff")) %>% 
                               names() %>% 
                               # when making it data.frame the '(' '>' ')'symbols become '.', so revert this
                               str_replace(., 
                                           "([A,C,G,T])\\.([C,T])\\.([A,C,G,T])\\.([A,C,G,T])_", 
                                           "\\1(\\2>\\3)\\4_") %>% 
                               str_replace(., "X53BP1", "53BP1") %>% 
                               gsub("_poscoeff", "", .)))) %>%
  map(., ~as.matrix(.x))



## Cluster the NMF factors, and find combination that yields best silhouette (and possibly deviation from random data)
## Also run in parallel on as many CPUs as specified
## Returns a data.table with nFact, k, and the quality measures per nFact*k combination.
#' @param maxNumClusters Global override - holds true for any number of nFact... note that some nFact will have less clusters than that 
#'                       For a given nFact, the max # clusters is 5*nFact.  Meaning: max 10 clusters for 2 factors, max 15 for 3 factors, etc.
#'                       (probably even this many does not make sense to extract...)

# initialize cosine distance function which is used for clustering
cosineDist <- function(x) {
  mat = as.dist(t(1 - cosine(t(x))))
  return(mat)
}

# cosine SIMILARITY will be used later for similarities between signatures' regional features weight profiles
cosineSimil <- function(x) {
  mat = t(cosine(t(x)))
  return(mat)
}

# initialize function for clustering
optimizeClustersPar = function(nmfHmatAllByFact_combined_pos_negcoeff = NULL, maxNumClusters = maxK, threads = nCPUs) {
  # the highest nFact (number of factors) we had checked in this dataset... the lowest is always assumed to be ==2 but this could be inspected as well
  maxNumFact = names(nmfHmatAllByFact_combined_pos_negcoeff) %>% 
    str_match("^nFact=(\\d+)") %>% .[, 2] %>% as.integer %>% max
  
  cl = makeCluster(threads, useXDR = F) # 'makeCluster' refers to computer clusters!
  clusterEvalQ(cl, library(cluster)) # to be able to run "pam" ('cluster' refers to data clustering, not computer clusters!)  
  sigDiscoScores = data.table()
  numNmfIters = -1 # this will be found from the nmfHmatAllByFact_combined_pos_negcoeff
  
  gc()
  
  for (aNmfSettings in names(nmfHmatAllByFact_combined_pos_negcoeff)) {
    
    # aNmfSettings is a string like "nFact=002", which encodes both the "nFact" parameter
    nFact = aNmfSettings %>% str_match("^nFact=(\\d+)") %>% .[, 2] %>% as.integer
    
    # how many iterations of NMF were run for this maxNumFact
    if (numNmfIters == -1) {
      numNmfIters = nmfHmatAllByFact_combined_pos_negcoeff[[aNmfSettings]] %>% rownames %>% str_match("^run(\\d+)_") %>% .[, 2] %>% as.integer %>% max
    } else {
      numNmfIters2 = nmfHmatAllByFact_combined_pos_negcoeff[[aNmfSettings]] %>% rownames %>% str_match("^run(\\d+)_") %>% .[, 2] %>% as.integer %>% max
      if (numNmfIters != numNmfIters2) {
        cat(sprintf("For aNmfSettings=%s, the numNmfIters %d is different than previously, when it was %d. Will continue running normally.\n", aNmfSettings, numNmfIters2, numNmfIters))
      }
    }
    
    ## get matrix of cosine distances between nFact×r factors (nFact = aNmfSettings, r = 1000 matrices)
    # each factor's vector goes from coord. origin to a point whose coordinates are its weights vector (so if 8 chromatin features, it's an intersection point of 8 dimensions)
    nmfReBootstrappedDists = cosineDist(nmfHmatAllByFact_combined_pos_negcoeff[[aNmfSettings]])
    gc()
    
    # parallel code: in each thread, runs the "pam" clustering for a different # clusters, starting from the same matrix of cosine similarities
    # (note: Sanger (Alexandrov et al.) always run this with #clusters = #NMF factors, which may not be ideal)
    clusterExport(cl = cl, list("nmfReBootstrappedDists"), envir = environment()) # , envir=tand)
    pamOutputByIter = parLapplyLB(# load balancing
      cl = cl,
      X = as.list(rev(2:min(maxNumClusters, nFact * 5))), # pam() with different values of k is run in parallel 
      fun = function(x) {
        set.seed(42 + x)
        # partitioning (clustering) of the nFact×r factors into k clusters "around medoids" (a more robust version of K-means), based on the cosine measures matrix
        pam(nmfReBootstrappedDists, k = x, diss = T)
      })
    gc()
    
    for (clusters in pamOutputByIter) {
      k = length(clusters$medoids)
      ## get clustering "stability": mean(clusters$silinfo$clus.avg.widths), i.e. average-across-K-clusters of the average silhouette widths per cluster
      # "silhouette width" of a sample (i.e. a given nFactor×run) i in a cluster is the "dissimilarity between i and the closest sample from an external cluster"
      sigDiscoScores = sigDiscoScores %>% 
        rbind(data.table(nFact = nFact, 
                         k = k,
                         avgClScore = mean(clusters$silinfo$clus.avg.widths), 
                         minClScore = min(clusters$silinfo$clus.avg.widths),
                         secondWorstClScore = clusters$silinfo$clus.avg.widths[order(clusters$silinfo$clus.avg.widths)][2],
                         avgPtScore = mean(silhouette(clusters)[, "sil_width"]), 
                         medianPtScore = median(silhouette(clusters)[, "sil_width"])))
      cat(sprintf("%02d\t%02d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                  nFact, k,
                  mean(clusters$silinfo$clus.avg.widths), 
                  min(clusters$silinfo$clus.avg.widths),
                  clusters$silinfo$clus.avg.widths[order(clusters$silinfo$clus.avg.widths)][2],
                  mean(silhouette(clusters)[, "sil_width"]), 
                  median(silhouette(clusters)[, "sil_width"])))
      gc()
    }
  }
  
  stopCluster(cl)
  #rm(maxNumClusters, threads, cl, pamOutputByIter, k )
  return(sigDiscoScores)
}

# run clustering
sigClusterScores = optimizeClustersPar(nmfHmatAllByFact_combined_pos_negcoeff, 
                                       maxNumClusters = maxK)

## Save clustering results 
saveRDS(sigClusterScores, paste0("heatmap_Ks/", "sigClusterScores.Rda")) # 'quality' of clustering(s), used to defeature_nameine the number of signatures 
#sigClusterScores = readRDS(paste0("heatmap_Ks/", "sigClusterScores.Rda"))


## Visualize the clustering quality scores in heatmaps (faceted by the 'smooth' parameter)
heatmap_clustering = sigClusterScores[k <= maxK] %>% 
  melt(id.vars = c("nFact", "k"),
       measure.vars = "avgClScore") %>%
  rename("Stability" = "value")

heatmap_clustering_plot = ggplot(heatmap_clustering %>% 
                                   # only maxK factors and k
                                   filter(nFact<=maxK & k<=maxK),
                                 aes(nFact, k)) +
  scale_x_continuous(breaks = seq(2, maxK)) +
  scale_y_continuous(breaks = seq(2, maxK)) +
  #coord_fixed() +
  geom_tile(aes(fill = Stability)) +
  scale_fill_gradient2(low = "white", high = "blue", midpoint = 0,
                       breaks = seq(-0.25,1.25,0.25)) +
  geom_label(aes(label = round(Stability, 2)),
             #fontface = "bold",
             label.r = unit(0.01, "lines"),
             label.size = unit(0, "mm"),
             label.padding  = unit(0.01, "lines"),
             size = 8) +
  xlab("N factors") +
  ylab("K clusters") +
  theme_classic() +
  theme(text = element_text(size = 28),
        legend.position = "top",
        legend.text = element_text(angle = 90, vjust = 0.5, hjust = 0.25))

ggsave(paste0("heatmap_Ks/NMF_heatmap_clustering_maxnFact", maxK, "_maxK", maxK, ".jpg"),
       plot = heatmap_clustering_plot,
       device = "jpg",
       width = 12.5,
       height = 7,
       dpi = 600)


####
### Run NMF for optimal signatures

## for plots
dir.create("plots")
# for exposures and weights matrices
dir.create("exposures_weights")
## for QC (the deviance between NMF fitted model and original coefficients matrices in regfeat+SBS96 matrices vs. in SBS96-only matrices)
dir.create("QC")
deviances_list = list()
# this stores deviances after removing the reg feat residuals, for a direct comparison to SBS96only
deviances_NO_REG_FEAT_list = list()
# for significance deltas (regfeat vs SBS96) of pairwise signature cosine similarities 
dir.create("plots/cos_sim_deltas")
real_deltas_vs_bootstrap_hits = list()
# for feature hits that could be somehow related to a cosmic signature
stability_cutoff = 0
cosine_similarity_cutoff = 0
cosmic_similarity_cutoff = 0
max_n_feature_hits = 3
regfeats_cosmic_assoc = list()

trinuc_96 = c("A(C>A)A", "A(C>A)C", "A(C>A)G", "A(C>A)T",
              "C(C>A)A", "C(C>A)C", "C(C>A)G", "C(C>A)T",
              "G(C>A)A", "G(C>A)C", "G(C>A)G", "G(C>A)T",
              "T(C>A)A", "T(C>A)C", "T(C>A)G", "T(C>A)T",
              "A(C>G)A", "A(C>G)C", "A(C>G)G", "A(C>G)T",
              "C(C>G)A", "C(C>G)C", "C(C>G)G", "C(C>G)T",
              "G(C>G)A", "G(C>G)C", "G(C>G)G", "G(C>G)T",
              "T(C>G)A", "T(C>G)C", "T(C>G)G", "T(C>G)T",
              "A(C>T)A", "A(C>T)C", "A(C>T)G", "A(C>T)T",
              "C(C>T)A", "C(C>T)C", "C(C>T)G", "C(C>T)T",
              "G(C>T)A", "G(C>T)C", "G(C>T)G", "G(C>T)T",
              "T(C>T)A", "T(C>T)C", "T(C>T)G", "T(C>T)T",
              "A(T>A)A", "A(T>A)C", "A(T>A)G", "A(T>A)T",
              "C(T>A)A", "C(T>A)C", "C(T>A)G", "C(T>A)T",
              "G(T>A)A", "G(T>A)C", "G(T>A)G", "G(T>A)T",
              "T(T>A)A", "T(T>A)C", "T(T>A)G", "T(T>A)T",
              "A(T>C)A", "A(T>C)C", "A(T>C)G", "A(T>C)T",
              "C(T>C)A", "C(T>C)C", "C(T>C)G", "C(T>C)T",
              "G(T>C)A", "G(T>C)C", "G(T>C)G", "G(T>C)T",
              "T(T>C)A", "T(T>C)C", "T(T>C)G", "T(T>C)T",
              "A(T>G)A", "A(T>G)C", "A(T>G)G", "A(T>G)T",
              "C(T>G)A", "C(T>G)C", "C(T>G)G", "C(T>G)T",
              "G(T>G)A", "G(T>G)C", "G(T>G)G", "G(T>G)T",
              "T(T>G)A", "T(T>G)C", "T(T>G)G", "T(T>G)T")
# reversed because they go in y axis
rev_trinuc_96 = rev(trinuc_96)

sample_pheno_levels = unique(samples_info$altered_pathway_or_treatment_type)
jet_colors = colorRampPalette(c("gray", "red", "yellow", "green", "cyan", "blue", "magenta", "black"))
## if you want a fixed assignment (i.e. that is constant across plots) of a given color to a given sample's source × dMMR status:
fixed_jet_colors = jet_colors(length(sample_pheno_levels))
names(fixed_jet_colors) = sample_pheno_levels


## create plots

# define combinations of nFact and k that we want to plot
range_nFact = seq(3, maxK)
range_k = seq(3, maxK)
optimal_nFact_k_list = expand.grid(range_nFact, # nFact
                                   range_k) %>% # K
  as.matrix %>% t %>% data.frame %>% as.list

for(optimal_nFact_k in optimal_nFact_k_list){
  
  ## extract K signatures (i.e, each cluster's medoid) for the desired combination of the number of NMF factors (nFact) (×1000 resampled matrices) and number of clusters (k)
  nFact = optimal_nFact_k[1]
  optimal_k = optimal_nFact_k[2] # the final number of signatures 
  
  wMatHere = nmfWmatAllByFact[[sprintf("nFact=%03d", nFact)]]
  hMatHere = nmfHmatAllByFact[[sprintf("nFact=%03d", nFact)]] # the NMF w-matrices had been transposed, so rows in the H-matrices and in the W-matrices are the same

  #### for QC: calculate the deviance between each NMF fitted model and its corresponding original (positivized) coefficient matrix (i.e. for each nFact==[2-12] × iter==[1-100] combination); this will be compared to the equivalent when there are only SBS96 coeff, to see whether we actually get more accurate NMF fit when SBS96 coefficients are combined with regional feature coefficients
  if(is.null(deviances_list[[paste0('nFact',nFact)]])){ # only do it if it has not been already done for a previous optimal_k

    deviances_list[[paste0('nFact',nFact)]] = list()
    deviances_NO_REG_FEAT_list[[paste0('nFact',nFact)]] = list()

    for(nIter in 1:totalNumIters){

      original_matrix = coefficient_Resamp[[nIter]]
      # remove first column
      original_matrix = original_matrix[, colnames(original_matrix) != "nIter"]
      # sometimes it can be that e.g. coefficient_Resamp[[6]] contains ..._nIter013 in rownames
      actual_nIter = rownames(original_matrix) %>% gsub(".*_nIter", "", .) %>% unique %>% as.numeric %>% sprintf("%03d", .)

      exposures_this_actual_nIter = wMatHere %>%
        data.frame %>%
        rownames_to_column("nIter_nFact") %>%
        filter(str_detect(nIter_nFact, paste0("run",actual_nIter,"_"))) %>%
        column_to_rownames("nIter_nFact") %>%
        as.matrix %>%
        t

      weights_this_actual_nIter = hMatHere %>%
        data.frame %>%
        rownames_to_column("nIter_nFact") %>%
        filter(str_detect(nIter_nFact, paste0("run",actual_nIter,"_"))) %>%
        column_to_rownames("nIter_nFact") %>%
        as.matrix

      exposures_by_weights = exposures_this_actual_nIter %*% weights_this_actual_nIter

      # mean of the squared differences
      original_vs_nmf_deviance = sum(as.numeric((exposures_by_weights - original_matrix)^2)) / prod(dim(original_matrix))

      deviances_list[[paste0('nFact',nFact)]][[paste0('nIter',nIter)]] = original_vs_nmf_deviance

      # mean of the squared differences without regfeat columns
      original_matrix_NO_REG_FEAT = data.frame(original_matrix) %>%
        select(contains(".")) %>% as.matrix
      exposures_by_weights_NO_REG_FEAT = data.frame(exposures_by_weights) %>%
        select(contains(".")) %>% as.matrix
      original_vs_nmf_deviance_NO_REG_FEAT = sum(as.numeric((exposures_by_weights_NO_REG_FEAT - original_matrix_NO_REG_FEAT)^2)) / prod(dim(original_matrix_NO_REG_FEAT))
      deviances_NO_REG_FEAT_list[[paste0('nFact',nFact)]][[paste0('nIter',nIter)]] = original_vs_nmf_deviance_NO_REG_FEAT

      gc()
    }
  }
  ####
  
  ## continue
  rownames(wMatHere) = gsub("run", "Run ", rownames(wMatHere))
  rownames(wMatHere) = gsub("fact0", "", rownames(wMatHere))
  rownames(wMatHere) = gsub("fact", "", rownames(wMatHere))
  rownames(wMatHere) = gsub("_", "\nFactor ", rownames(wMatHere))
  rownames(hMatHere) = gsub("run", "Run ", rownames(hMatHere))
  rownames(hMatHere) = gsub("fact0", "", rownames(hMatHere))
  rownames(hMatHere) = gsub("fact", "", rownames(hMatHere))
  rownames(hMatHere) = gsub("_", "\nFactor ", rownames(hMatHere))

  # need to run the clustering again...
  set.seed(42 + optimal_k) # this should reproduce the same output as in the "optimizeClustersPar"
  clustHere = pam(cosineDist(hMatHere), k = optimal_k, diss = T)
  # medoid: sample in a cluster that is the least dissimilar on average to all the other samples in that cluster
  signatures = copy(hMatHere[clustHere$medoids,])
  # sort signatures based on their stability
  signature_stabilities = data.frame('Signature' = clustHere$medoids, 'Stability' = round(clustHere$silinfo$clus.avg.widths,4)) %>%
    arrange(Stability)
  signature_stabilities_sorted = as.character(signature_stabilities$Stability)
  signatures_sorted = signature_stabilities$Signature

  # get sample exposures for these cluster medoids (signatures)
  sample_exposures = wMatHere %>%
    as_tibble() %>%
    mutate(signature = rownames(wMatHere)) %>%
    filter(signature %in% signatures_sorted) %>%
    arrange(signature) %>% relocate(signature)

  # and again, [results of pos] + -[results of neg] to have a "-" sign when the larger/non-zero value comes from the neg. submatrix
  signatures_combined_pos_negcoeff = signatures %>%
    as_tibble(rownames = NA) %>%
    rownames_to_column("id") %>%
    pivot_longer(cols = contains("coeff"),
                 names_to = "dna_repair_feature_name",
                 values_to = "Weight") %>%
    mutate(Weight = ifelse(str_detect(dna_repair_feature_name, "_negcoeff"),
                           -Weight,
                           Weight),
           dna_repair_feature_name = gsub("_...coeff", "", dna_repair_feature_name)) %>%
    group_by(dna_repair_feature_name, id) %>%
    summarise(Weight = .Primitive("+")(Weight[1], Weight[2])) %>%
    ungroup() %>%
    ## scale weights per signature so they add up to 1
    #group_by(id) %>%
    #mutate(sumWeight = sum(Weight)) %>%
    #group_by(id, dna_repair_feature_name) %>%
    #summarise(Weight = Weight/sumWeight) %>%
    #ungroup() %>%
    ## go to original format
    pivot_wider(names_from = dna_repair_feature_name, values_from = Weight) %>%
    # arrange iterations (rows)
    arrange(id) %>%
    column_to_rownames('id') %>%
    # arrange dna repair feature_name names (columns)
    select(coefficient_Resamp[[1]] %>%
             as_tibble %>%
             select(contains("poscoeff")) %>%
             names() %>%
             gsub("_poscoeff", "", .)) %>%
    as.matrix()


  ## parse signature weights

  # SBS weights in signatures
  weights = signatures_combined_pos_negcoeff %>%
    as_tibble() %>%
    mutate(Signature = rownames(signatures_combined_pos_negcoeff)) %>%
    left_join(signature_stabilities) %>%
    pivot_longer(cols = !contains("Signature") & !contains('Stability'), names_to = "feature", values_to = "Weight") %>%
    mutate(feature_group = ifelse(str_detect(feature, ">"),
                                  "SBS",
                                  "Regional\nfeature"),
           `SBS group` = ifelse(feature_group == "SBS",
                                gsub("^.\\(|\\).$", "", feature),
                                NA),
           `SBS group` = factor(`SBS group`,
                                levels = c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G"),
                                ordered = T),
           SBS96 = ifelse(feature_group == "SBS",
                          feature,
                          NA),
           SBS96 = factor(SBS96,
                          levels = rev_trinuc_96,
                          ordered = T),
           `Regional\nfeature` = ifelse(feature_group == "Regional\nfeature",
                                         feature,
                                         NA),
           # convert signature's stabilities into a factor to be used in y axis, instead of Signature's name which is already indicated in exposures heatmap's y axis
           Stability = as.character(Stability),
           Stability = factor(Stability,
                              levels = unique(signature_stabilities_sorted))) %>%
    arrange(SBS96, `Regional\nfeature`, Stability)

  ### sigprofiler (for SBS)
  n_top_similar_cosmic = 3
  cosmic_sig_similarity = weights %>%
    filter(feature_group == "SBS") %>%
    select(-c(feature, feature_group, `SBS group`, `Regional\nfeature`, Stability)) %>%
    mutate(SBS96 = gsub("\\(", "\\[", SBS96),
           SBS96 = gsub("\\)", "\\]", SBS96),
           SBS96 = factor(SBS96,
                          levels = gsub("\\)", "\\]", gsub("\\(", "\\[", trinuc_96)),
                          ordered = T),
           # neg. weights to 0
           Weight = ifelse(Weight<0, 0, Weight),
           Signature = gsub("\n", "_", Signature)) %>%
    # each signature's weights have to sum 1
    group_by(Signature) %>%
    mutate(sumWeight = sum(Weight)) %>%
    group_by(Signature, SBS96) %>%
    summarise(Weight = Weight/sumWeight) %>%
    ungroup %>%
    filter(Weight != "NaN") %>%
    pivot_wider(names_from = Signature, values_from = Weight) %>%
    arrange(SBS96) %>%
    column_to_rownames("SBS96") %>%
    as.matrix %>%
    get_sig_similarity(., sig_db="SBS") %>%
    pluck("similarity") %>%
    data.frame %>%
    rownames_to_column("Signature") %>%
    mutate(Signature = gsub("_", "\n", Signature)) %>%
    pivot_longer(cols = !contains("Signature"), names_to = "COSMIC", values_to = "Similarity") %>%
    # keep top similarity cosmic sbs for each signature
    group_by(Signature) %>%
    arrange(desc(Similarity)) %>%
    slice_head(n = n_top_similar_cosmic) %>%
    ungroup %>%
    mutate(Similarity = round(Similarity, 2)) %>%
    unite("Max. sim. COSMIC", COSMIC, Similarity, sep = ": ") %>% 
    aggregate(`Max. sim. COSMIC` ~ Signature, data = ., FUN = paste, collapse = " / ")

  weights = left_join(weights, cosmic_sig_similarity)

  
  #######################################################################################
  #### cosine similarities between the weight profiles of the different signatures
  
  ### 1) SBS96 
  SBS96_sig_similarities = weights %>%
    filter(feature_group == "SBS") %>%
    select(-c(feature_group, `Regional\nfeature`, `SBS group`, SBS96, Stability, `Max. sim. COSMIC`)) %>%
    mutate(Signature = gsub("\n", "_", Signature)) %>%
    pivot_wider(names_from = feature, values_from = Weight) %>%
    arrange(Signature) %>%
    column_to_rownames("Signature") %>%
    as.matrix() %>%
    cosineSimil()
  
  # perform hierarchical clustering (on negative matrix, to trick it since it requires distances, not similarities)
  rowclus = hclust(as.dist(-SBS96_sig_similarities))    # cluster the rows
  colclus = hclust(t(as.dist(-SBS96_sig_similarities))) # cluster the columns
  
  # bring the data.frame into a from easily usable by ggplot
  SBS96_sig_similarities_clustered = ggdendroplot::hmReady(-SBS96_sig_similarities,
                                                              colclus=colclus, rowclus=rowclus) %>%
    rename("cosine similarity" = "value",
           "SigA" = "rowid",
           "SigB" = "variable") %>%
    mutate(`cosine similarity` = -`cosine similarity`,
           feature_type = "SBS96")
  
  heatmap_SBS96_sig_similarities = ggplot(SBS96_sig_similarities_clustered %>%
                                               # make low-right triangle
                                               filter(x >= y),
                                             aes(x = x,
                                                 y = y)) +
    geom_tile(aes(fill = `cosine similarity`)) +
    scale_fill_gradientn(colours=c("white", "blue"),
                         limits = c(min(SBS96_sig_similarities_clustered$`cosine similarity`),
                                    1)) +
    scale_x_continuous(breaks = SBS96_sig_similarities_clustered$x, labels = SBS96_sig_similarities_clustered$SigB) +
    scale_y_continuous(breaks = SBS96_sig_similarities_clustered$y, labels = SBS96_sig_similarities_clustered$SigA) +
    geom_label(aes(label = round(`cosine similarity`, 2)),
               label.r = unit(0.01, "lines"),
               label.size = unit(0, "mm"),
               label.padding  = unit(0.01, "lines"),
               size = 10) +
    xlab("") +
    ylab("") +
    theme_classic() +
    theme(text = element_text(size = 25),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top",
          legend.text = element_text(angle = 90, vjust = 0.5, hjust = 0.25))
  ggsave(paste0("plots/cos_sim_deltas/SBS96_weights_similarities_btw_sigs__nFact", nFact, "_K", optimal_k, ".jpeg"),
         plot = heatmap_SBS96_sig_similarities,
         device = "jpeg",
         width = 25,
         height = 14,
         dpi = 600)
  
  ### 2) regional features
  reg_feat_sig_similarities = weights %>%
    filter(feature_group != "SBS") %>%
    select(-c(feature_group, `Regional\nfeature`, `SBS group`, SBS96, Stability, `Max. sim. COSMIC`)) %>%
    mutate(Signature = gsub("\n", "_", Signature)) %>%
    pivot_wider(names_from = feature, values_from = Weight) %>%
    arrange(Signature) %>%
    column_to_rownames("Signature") %>%
    as.matrix() %>%
    cosineSimil()

  # perform hierarchical clustering (on negative matrix, to trick it since it requires distances, not similarities)
  rowclus = hclust(as.dist(-reg_feat_sig_similarities))    # cluster the rows
  colclus = hclust(t(as.dist(-reg_feat_sig_similarities))) # cluster the columns

  # bring the data.frame into a from easily usable by ggplot
  reg_feat_sig_similarities_clustered = ggdendroplot::hmReady(-reg_feat_sig_similarities,
                                                              colclus=colclus, rowclus=rowclus) %>%
    rename("cosine similarity" = "value",
           "SigA" = "rowid",
           "SigB" = "variable") %>%
    mutate(`cosine similarity` = -`cosine similarity`,
           feature_type = "Regional features")

  heatmap_reg_feat_sig_similarities = ggplot(reg_feat_sig_similarities_clustered %>%
                                               # make low-right triangle
                                               filter(x >= y),
                                             aes(x = x,
                                                 y = y)) +
    geom_tile(aes(fill = `cosine similarity`)) +
    scale_fill_gradientn(colours=c("white", "blue"),
                         limits = c(min(reg_feat_sig_similarities_clustered$`cosine similarity`),
                                    1)) +
    scale_x_continuous(breaks = reg_feat_sig_similarities_clustered$x, labels = reg_feat_sig_similarities_clustered$SigB) +
    scale_y_continuous(breaks = reg_feat_sig_similarities_clustered$y, labels = reg_feat_sig_similarities_clustered$SigA) +
    geom_label(aes(label = round(`cosine similarity`, 2)),
               label.r = unit(0.01, "lines"),
               label.size = unit(0, "mm"),
               label.padding  = unit(0.01, "lines"),
               size = 10) +
    xlab("") +
    ylab("") +
    theme_classic() +
    theme(text = element_text(size = 25),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top",
          legend.text = element_text(angle = 90, vjust = 0.5, hjust = 0.25))
  ggsave(paste0("plots/cos_sim_deltas/RegFeat_weights_similarities_btw_sigs__nFact", nFact, "_K", optimal_k, ".jpeg"),
         plot = heatmap_reg_feat_sig_similarities,
         device = "jpeg",
         width = 25,
         height = 14,
         dpi = 600)

  ### 3) calculate delta
  # "real" because later will be compared to bootstrapped ones
  delta_cos_simil_real = bind_rows(SBS96_sig_similarities_clustered,
                                   reg_feat_sig_similarities_clustered) %>% 
    select(-c(x,y)) %>% 
    unite("Sigpair", SigA, SigB, sep = "__") %>% 
    pivot_wider(names_from = feature_type, values_from = `cosine similarity`) %>% 
    group_by(Sigpair) %>% 
    ## regfeat - SBS96 (the more negative, the more dissimilar the signatures in regfeat compared to how similar they are in SBS96)
    summarise("Δ cosine similarity" = `Regional features` - SBS96) %>% 
    separate(Sigpair, into = c("SigA", "SigB"), sep = "__")
  
  delta_cos_simil = delta_cos_simil_real %>% 
    pivot_wider(names_from = SigB, values_from = `Δ cosine similarity`) %>% 
    column_to_rownames("SigA") %>% 
    as.matrix()
  
  # perform hierarchical clustering
  rowclus = hclust(as.dist(-delta_cos_simil))    # cluster the rows
  colclus = hclust(t(as.dist(-delta_cos_simil))) # cluster the columns
  
  # bring the data.frame into a from easily usable by ggplot
  delta_cos_simil_clustered = ggdendroplot::hmReady(-delta_cos_simil,
                                                    colclus=colclus, rowclus=rowclus) %>%
    rename("Δ cosine similarity" = "value",
           "SigA" = "rowid",
           "SigB" = "variable") %>%
    mutate(`Δ cosine similarity` = -`Δ cosine similarity`)
  
  heatmap_delta_cos_simil = ggplot(delta_cos_simil_clustered %>%
                                     # make low-right triangle
                                     filter(x >= y),
                                   aes(x = x,
                                       y = y)) +
    geom_tile(aes(fill = `Δ cosine similarity`)) +
    scale_fill_gradientn(colours=c("blue", "white"),
                         limits = c(min(delta_cos_simil_clustered$`Δ cosine similarity`),
                                    -0.0000001)) +
    scale_x_continuous(breaks = delta_cos_simil_clustered$x, labels = delta_cos_simil_clustered$SigB) +
    scale_y_continuous(breaks = delta_cos_simil_clustered$y, labels = delta_cos_simil_clustered$SigA) +
    geom_label(aes(label = round(`Δ cosine similarity`, 2)),
               label.r = unit(0.01, "lines"),
               label.size = unit(0, "mm"),
               label.padding  = unit(0.01, "lines"),
               size = 10) +
    xlab("") +
    ylab("") +
    theme_classic() +
    theme(text = element_text(size = 25),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top",
          legend.text = element_text(angle = 90, vjust = 0.5, hjust = 0.25))
  ggsave(paste0("plots/cos_sim_deltas/delta_cos_simil__nFact", nFact, "_K", optimal_k, ".jpeg"),
         plot = heatmap_delta_cos_simil,
         device = "jpeg",
         width = 25,
         height = 14,
         dpi = 600)
  
  ### 4) bootstrap: shuffle cosine similarities matrix of SBS96 to create a null distribution of deltas to which compare the real deltas for significance
  bootstrap_n = 1000
  bootstrapped_deltas = list()

  for(i in seq(1,bootstrap_n)){

    SBS96_sig_similarities_clustered_shuffled = SBS96_sig_similarities_clustered %>%
      mutate(SigB = as.character(SigB)) %>%
      # remove SigA==SigB
      filter(SigA != SigB) %>%
      as_tibble %>%
      # remove rows where row1$SigA == row2$SigB && row1$SigB == row2$SigA
      mutate(pair_id = pmap_chr(list(SigA, SigB), ~paste(sort(c(...)), collapse = "__"))) %>%
      distinct(pair_id, .keep_all = TRUE) %>%
      select(-pair_id) %>%  # remove the auxiliary column
      transform(`cosine similarity` = sample(`cosine similarity`)) %>%
      rename("cosine similarity" = "cosine.similarity")

    reg_feat_sig_similarities_clustered_shuffled = reg_feat_sig_similarities_clustered %>%
      mutate(SigB = as.character(SigB)) %>%
      # remove SigA==SigB here as well
      filter(SigA != SigB) %>%
      as_tibble %>%
      # remove rows where row1$SigA == row2$SigB && row1$SigB == row2$SigA here as well
      mutate(pair_id = pmap_chr(list(SigA, SigB), ~paste(sort(c(...)), collapse = "__"))) %>%
      distinct(pair_id, .keep_all = TRUE) %>%
      select(-pair_id) %>%  # remove the auxiliary column
      # it's not actually necessary to sample this matrix as well, but anyways
      transform(`cosine similarity` = sample(`cosine similarity`)) %>%
      rename("cosine similarity" = "cosine.similarity")

    bootstrap_delta_cos_simil = bind_rows(SBS96_sig_similarities_clustered_shuffled,
                                          reg_feat_sig_similarities_clustered_shuffled) %>%
      select(-c(x,y)) %>%
      unite("Sigpair", SigA, SigB, sep = "__") %>%
      pivot_wider(names_from = feature_type, values_from = `cosine similarity`) %>%
      group_by(Sigpair) %>%
      ## regfeat - SBS96 (the more negative, the more dissimilar the signatures in regfeat compared to how similar they are in SBS96)
      summarise("Δ cosine similarity" = `Regional features` - SBS96) %>%
      separate(Sigpair, into = c("SigA", "SigB"), sep = "__")

    bootstrapped_deltas[[i]] = select(bootstrap_delta_cos_simil,
                                      `Δ cosine similarity`)
  }
  bootstrapped_deltas = bind_rows(bootstrapped_deltas) %>%
    mutate(Sigpair = "Bootstrap")

  # to standardize delta
  mean_delta = mean(bootstrapped_deltas$`Δ cosine similarity`)
  sd_delta = sd(bootstrapped_deltas$`Δ cosine similarity`)

  real_deltas_vs_bootstrap = delta_cos_simil_real %>%
    # remove SigA==SigB
    filter(SigA != SigB) %>%
    # remove rows where row1$SigA == row2$SigB && row1$SigB == row2$SigA
    mutate(pair_id = pmap_chr(list(SigA, SigB), ~paste(sort(c(...)), collapse = "__"))) %>%
    distinct(pair_id, .keep_all = TRUE) %>%
    select(-pair_id) %>%  # remove the auxiliary column
    unite("Sigpair", SigA, SigB, sep = " vs. ") %>%
    bind_rows(bootstrapped_deltas) %>%
    mutate(group = ifelse(Sigpair == "Bootstrap",
                          "Bootstrap",
                          "Real"),
           `Δ cosine similarity (standardized)` = (`Δ cosine similarity`-mean_delta) / sd_delta)

  real_deltas_vs_bootstrap_hits[[paste0("nFact", nFact, "_K", optimal_k)]] = real_deltas_vs_bootstrap %>%
    filter(group == "Real" & `Δ cosine similarity (standardized)` <= -1.96) %>%
    mutate("nFact" = nFact,
           "K" = optimal_k) %>%
    select(-group)

  plot_real_deltas_vs_bootstrap = ggplot(real_deltas_vs_bootstrap,
                                         aes(x = `Δ cosine similarity (standardized)`,
                                             col = group,
                                             fill = group)) +
    scale_x_continuous(breaks = seq(-3, 3, 0.5)) +
    geom_density(data = real_deltas_vs_bootstrap %>%
                          filter(group == "Bootstrap")) +
    geom_bar(data = real_deltas_vs_bootstrap %>%
               filter(group == "Real")) +
    scale_color_manual(values = c("red", "blue")) +
    scale_fill_manual(values = c("red", "blue")) +
    geom_text(data = real_deltas_vs_bootstrap %>%
                filter(group == "Real" & `Δ cosine similarity (standardized)` <= -1.96),
              aes(label = Sigpair,
                  y = 0),
              angle=90,
              vjust = -0.1,
              hjust = -0.1) +
    theme_bw() +
    xlab("Absolute difference of the signature cosine similarities in regional features vs. in SBS96 (standardized)") +
    ylab("Density (bootstrap, red) or counts (real, blue)") +
    theme(legend.title = element_blank())
  ggsave(paste0("plots/cos_sim_deltas/deltas_real_vs_bootstrap", nFact, "_K", optimal_k, ".jpeg"),
         plot = plot_real_deltas_vs_bootstrap,
         device = "jpeg",
         width = 13,
         height = 7,
         dpi = 600)
  
  
  ##### 5) detect relationships between regional feature(s) and cosmic SBS
  cosmic_sim_good = weights %>% 
    filter(as.numeric(Stability) >= stability_cutoff) %>% 
    select(Signature, `Max. sim. COSMIC`) %>% 
    distinct %>% 
    mutate(Signature = gsub("\n", " ", Signature),
           `Max. sim. COSMIC` = gsub(" / .*", "", `Max. sim. COSMIC`)) %>%
    separate(`Max. sim. COSMIC`, into = c("COSMIC", "Max. sim."), sep = ": ") %>% 
    filter(`Max. sim.` >= cosmic_similarity_cutoff)
    
  regfeat_cosmic = bind_rows(SBS96_sig_similarities_clustered,
                             reg_feat_sig_similarities_clustered) %>% 
    select(-c(x,y)) %>% 
    # remove SigA==SigB
    filter(SigA != SigB) %>% 
    as_tibble %>% 
    # remove rows where row1$SigA == row2$SigB && row1$SigB == row2$SigA
    group_by(feature_type) %>% 
    mutate(SigA = gsub("_", " ", SigA),
           SigB = as.character(gsub("_", " ", SigB)),
           pair_id = pmap_chr(list(SigA, SigB), ~paste(sort(c(...)), collapse = "__"))) %>% arrange(pair_id) %>% 
    distinct(pair_id, .keep_all = TRUE) %>%
    filter(`cosine similarity` >= cosine_similarity_cutoff) %>% 
    ungroup
  
  # only continue if it is promising 
  if((select(regfeat_cosmic, pair_id) %>% distinct %>% nrow) < nrow(regfeat_cosmic) & # this indicates that at least 1 same pair of signatures is very similar both within in SBS96 and within reg feat
      any(regfeat_cosmic$SigA %in% cosmic_sim_good$Signature) & any(regfeat_cosmic$SigB %in% cosmic_sim_good$Signature)){ # this indicates that there is at least 1 pair of signatures that are very similar in regfeat_cosmic AND that their SBS components are the most similar to the same cosmic signature
    
    # filter to make effective the conditions above
    regfeat_cosmic_filtered = regfeat_cosmic %>% 
      group_by(pair_id) %>% 
      filter(n() == 2) %>% ungroup %>% 
      filter(SigA %in% cosmic_sim_good$Signature & SigB %in% cosmic_sim_good$Signature) %>% 
      select(pair_id) %>% 
      distinct
    
    # just to make sure, because the 'any(..' condition can pass in some cases in which there are not actually the 2 desired Sig in the same row
    if(nrow(regfeat_cosmic_filtered) >= 1){
      
      regfeat_cosmic_filtered_pairs = regfeat_cosmic_filtered %>% 
        separate(pair_id, into = c("SigA", "SigB"), sep = "__")
      
      for(sigpair in seq(1, nrow(regfeat_cosmic_filtered_pairs))){
        
        SigA = as.character(regfeat_cosmic_filtered_pairs[sigpair,1])
        SigB = as.character(regfeat_cosmic_filtered_pairs[sigpair,2])
        
        weights_sigpair = weights %>% 
          filter(feature_group=="Regional\nfeature") %>% 
          mutate(Signature = gsub("\n", " ", Signature),
                 is.hit = ifelse(Signature %in% c(SigA, SigB),
                                 "hit", "no hit")) %>% 
          select(Signature, feature, Weight, is.hit) %>% 
          group_by(Signature) %>%
          arrange(desc(abs(Weight))) %>% 
          slice_head(n = max_n_feature_hits) %>% 
          ungroup %>% 
          select(Signature, feature, is.hit) %>% 
          distinct
        
        top5_in_hit_signatures = weights_sigpair %>% filter(is.hit=="hit") %>% select(feature) %>% pull
        top5_in_hit_signatures = unique(top5_in_hit_signatures[duplicated(top5_in_hit_signatures)])
        top5_in_nohit_signatures = weights_sigpair %>% filter(is.hit=="no hit") %>% select(feature) %>% pull %>% unique
        
        feature_hits = top5_in_hit_signatures[!top5_in_hit_signatures %in% top5_in_nohit_signatures]
        
        if(length(feature_hits) >=1){
          
          regfeats_cosmic_assoc_table = weights_sigpair %>% 
            filter(feature %in% feature_hits) %>%         
            left_join(mutate(weights, Signature = gsub("\n", " ", Signature))) %>% 
            select(Signature, Stability, feature, Weight, `Max. sim. COSMIC`) %>% 
            mutate("nFact" = nFact,
                   "K" = optimal_k,
                   Sigpair_i = sigpair)
          
          # only record it IF the cosmic signature that is most common to the SBS component is the same for both signatures
          if(length(unique(gsub(": .*", "", regfeats_cosmic_assoc_table$`Max. sim. COSMIC`))) == 1){
          
            regfeats_cosmic_assoc[[paste0("nFact", nFact, "_K", optimal_k)]] = regfeats_cosmic_assoc_table
          }
        }
      }
    }
  }
  
  
  ###################################################################################################
  
  
  #### parse signature exposures in samples
  exposures = sample_exposures %>%
    pivot_longer(cols = !contains("signature") & !(contains("Stability")), names_to = "Sample", values_to = "Exposure") %>%
    mutate(Sample = gsub("_nIter.*$", "", Sample)) %>%
    # add metadata info (e.g. treatments, MSI, HR, smoking...)
    left_join(samples_info) %>%
    rename("Signature" = "signature") %>%
    filter(!is.na(Exposure)) %>%
    mutate(Signature = factor(Signature, levels = signatures_sorted))

  ## write it for random forests
  write_tsv(exposures %>%
              mutate(Signature = gsub("\n", " ", Signature)),
            paste0("exposures_weights/fct", nFact, "_", "k", optimal_k, "_exposures.tsv"))
  write_tsv(weights %>%
              rename("Chromatin feature" = "Regional\nfeature") %>%
              mutate(`Chromatin feature` = gsub("\n", "_", `Chromatin feature`),
                     Signature = gsub("\n", " ", Signature)),
            paste0("exposures_weights/fct", nFact, "_", "k", optimal_k, "_weights.tsv"))


  ##### plotting

  #### exposures plot (heatmap)

  exposures_plot = ggplot(exposures,
                          aes(x = `alteration`,
                              y = Signature)) +
    geom_tile(aes(fill = Exposure)) +
    scale_fill_gradientn(colours = c('white','red')) +
    ## group sample labels in x axis by their altered_pathway_or_treatment_type
    facet_grid(. ~ altered_pathway_or_treatment_type, scales = "free_x", space = "free_x") +
    theme_bw() +
    xlab("Altered pathway OR treatment") +
    ylab("Signature id") +
    theme(axis.text.x = element_text(angle = 45, hjust=1, size = 6),
          axis.text.y = element_text(angle = 90, hjust=0.5, size = 15),
          strip.text.x.top = element_text(size = 10, angle = 90, hjust=0),
          text = element_text(size = 15),
          strip.background = element_blank(),
          legend.key.size = unit(1, "cm"),
          legend.position = "top")


  #### weights plots

  ## regional features

  # to label the regfeat plots with their stability instead of the regional feature name
  stability_labels = weights %>%
    filter(!is.na(`Regional\nfeature`)) %>%
    mutate(desc_Stability = factor(Stability, levels = rev(unique(signature_stabilities_sorted)), ordered = T)) %>%
    select(desc_Stability,Signature) %>%
    distinct %>%
    arrange(desc_Stability) %>%
    mutate(Signature = factor(Signature, levels = .$Signature, ordered = T)) %>%
    select(Signature, desc_Stability) %>%
    deframe()

   weights_plot_regfeat = ggplot(weights %>%
                                  filter(feature_group != "SBS") %>%
                                  select(-c(feature, feature_group, contains("SBS"))) %>%
                                  mutate(desc_Stability = factor(Stability, levels = rev(unique(signature_stabilities_sorted)), ordered = T)),
                                aes(x = Weight,
                                    y = `Regional\nfeature`)) +
    scale_x_continuous(expand = c(0, 0),
                       breaks = seq(-10, 10, 0.1),
                       labels = function(x) round(x,1)) +
    scale_y_discrete(position = "right") +
    geom_col(aes(fill = `Regional\nfeature`)) +
    scale_fill_manual(values = jet_colors(length(unique(weights$`Regional\nfeature`)))) +
    guides(fill = guide_legend(override.aes = list(size=0.5),
                               ncol = 4)) +
    geom_vline(xintercept = 0, linetype = "dotted", color = "white", linewidth = 1) +
    facet_wrap(~desc_Stability, ncol = 1, scales = "free",
               strip.position="right",
               labeller = as_labeller(stability_labels)) +
    xlab("Contribution (regional features)") +
    ylab("Signature stability") +
    theme_classic() +
    theme(axis.ticks.y = element_blank(),
          axis.text.y = element_blank(),
          axis.line.y = element_blank(),
          axis.text.x = element_text(size = 10),
          strip.text.y = element_text(size = 15),
          text = element_text(size = 15),
          strip.background = element_blank(),
          legend.title = element_blank(),
          legend.position = "top",
          legend.text = element_text(size = 7))

  ## SBS

  # to label the SBS plots with their COSMIC similarity instead of their stability (which is already in the MMR+BER)
  max_sim_cosmic = weights %>%
    mutate(desc_Stability = factor(Stability, levels = rev(unique(signature_stabilities_sorted)), ordered = T)) %>%
    select(desc_Stability,`Max. sim. COSMIC`) %>%
    distinct %>%
    arrange(desc_Stability) %>%
    mutate(`Max. sim. COSMIC` = as.character(`Max. sim. COSMIC`)) %>%
    deframe()

  weights_plot_SBS = ggplot(weights %>%
                              filter(feature_group == "SBS") %>%
                              select(-c(feature, feature_group, `Regional\nfeature`)) %>%
                              mutate(desc_Stability = factor(Stability, levels = rev(unique(signature_stabilities_sorted)), ordered = T),
                                     `Max. sim. COSMIC` = factor(`Max. sim. COSMIC`, levels = as.character(max_sim_cosmic), ordered = T)),
                            aes(x = Weight,
                                y = SBS96)) +
    scale_x_continuous(expand = c(0, 0),
                       breaks = seq(-1, 1, 0.1),
                       labels = function(x) round(x, 1)) +
    scale_y_discrete(position = "right") +
    geom_col(aes(fill = `SBS group`)) +
    scale_fill_manual(values = c("#00bfeb", "black", "#f3282f", "#cdc9ca", "#a1cc6b", "#f1c6c5")) +
    guides(fill = guide_legend(override.aes = list(size=1),
                               nrow = 6)) +
    facet_wrap(~desc_Stability, ncol = 1, scales = "free",
               strip.position="right",
               labeller = as_labeller(max_sim_cosmic)) +
    geom_vline(xintercept = 0, linetype = "dotted", color = "white", linewidth = 1) +
    xlab("Contribution (SBS)") +
    ylab("Top similar COSMIC SBS") +
    theme_classic() +
    theme(axis.ticks.y = element_blank(),
          axis.text.y = element_blank(),
          axis.line.y = element_blank(),
          axis.text.x = element_text(size = 10),
          strip.text.y = element_text(size = 15),
          text = element_text(size = 15),
          strip.background = element_blank(),
          ## THIS removes the SBS similarity names and values, since we will add pie charts instead
          strip.text.y.right = element_blank(),
          ## THIS makes the 'top similar cosmic SBS' text flipped
          axis.title.y.right = element_text(angle = 90),
          legend.title = element_blank(),
          legend.position = "top")
  
  # barplot of SBS cosmic top similarities
  barplot_cosmic = ggplot(weights %>%
           filter(feature_group == "SBS") %>%
           select(-c(feature, feature_group, `Regional\nfeature`)) %>%
           mutate(desc_Stability = factor(Stability, levels = rev(unique(signature_stabilities_sorted)), ordered = T),
                  `Max. sim. COSMIC` = factor(`Max. sim. COSMIC`, levels = as.character(max_sim_cosmic), ordered = T)) %>% 
           select(desc_Stability, `Max. sim. COSMIC`) %>% 
           distinct %>% 
           separate(`Max. sim. COSMIC`, into = c("top-1", "top-2", "top-3"), sep = " / ") %>% 
           pivot_longer(cols = contains("top-"),
                        names_to = "Max. sim. COSMIC",
                        values_to = "COSMIC name and similarity") %>% 
           separate(`COSMIC name and similarity`, into = c("COSMIC", "cosine similarity"), sep = ": ") %>% 
           select(-`Max. sim. COSMIC`) %>% 
           mutate(`cosine similarity` = as.numeric(`cosine similarity`)) %>% 
           arrange(desc_Stability, desc(`cosine similarity`)),
         aes(x = COSMIC,
             y = `cosine similarity`)) +
    coord_flip() +
    scale_y_continuous(breaks = c(0, 0.5, 0.75, 1.2)) +
    geom_col() +
    facet_wrap(~desc_Stability, ncol = 1, scales = "free_y") +
    theme_classic() +
    xlab("") +
    ylab("cos similarity") +
    theme(text = element_text(size = 15),
          axis.text.x = element_text(size = 8, angle = 90),
          strip.text = element_blank(),
          strip.background = element_blank())

  combined_plots = plot_grid(NULL,
                             plot_grid(exposures_plot, ncol = 1),
                             NULL,
                             plot_grid(weights_plot_regfeat, ncol = 1),
                             NULL,
                             plot_grid(weights_plot_SBS, ncol = 1),
                             NULL,
                             plot_grid(barplot_cosmic, ncol = 1),
                             NULL,
                             ncol = 9,
                             rel_widths = c(0.02, 1, 0.02, 0.2, 0.01, 0.1, 0.01,0.1, 0.01))
  ggsave(paste0("plots/NMF_exposures_weights_plot__nFact", nFact, "_K", optimal_k, ".jpeg"),
         plot = combined_plots,
         device = "jpeg",
         width = 25,
         height = 14,
         dpi = 600)
}

write_tsv(bind_rows(real_deltas_vs_bootstrap_hits) %>% 
            arrange(`Δ cosine similarity (standardized)`),
          "plots/cos_sim_deltas/real_deltas_vs_bootstrap_hits.tsv")

write_tsv(bind_rows(regfeats_cosmic_assoc) %>% 
            arrange(nFact, K, Sigpair_i),
          "plots/cos_sim_deltas/regfeats_cosmic_assoc.tsv")


write_tsv(bind_rows(deviances_list) %>% 
            as.matrix %>% t %>% data.frame %>% `colnames<-`(paste0("nFact", range_nFact)), 
          "QC/deviances_with_regfeat_coeff.tsv")

write_tsv(bind_rows(deviances_NO_REG_FEAT_list) %>% 
            as.matrix %>% t %>% data.frame %>% `colnames<-`(paste0("nFact", range_nFact)), 
          "QC/deviances_REGFEAT_REMOVED.tsv")
