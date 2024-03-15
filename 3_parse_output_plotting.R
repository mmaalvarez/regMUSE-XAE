library(tidyverse)
library(data.table)
library(lsa)
library(sigminer)
library(cowplot)
library(ggrepel)
#devtools::install_github("nicolash2/ggdendroplot")
library(ggdendroplot)
library(gtools)
library(conflicted)
conflict_prefer("filter", "dplyr")
conflict_prefer("rename", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("map", "purrr")
conflict_prefer("extract", "magrittr")
conflict_prefer("Position", "ggplot2")
conflict_prefer("cosine", "lsa")


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


dir.create("plots")

###########

## plot all parameter configurations' losses info
all_best_model_losses_epoch = read_tsv("res/all_best_model_losses_epoch.tsv") %>% 
  separate(output_folder_name,
           into = c('nFeatures','nSignatures','nEpochs','batchSize','l1Size','validationPerc','normalization','allow_negative_weights','seed'), sep = "__") %>% 
  select(-seed) %>% 
  distinct %>% 
  mutate(across(matches(c('nFeatures','nSignatures','nEpochs','batchSize','l1Size','validationPerc','normalization','allow_negative_weights')),
                ~gsub(".*_", "", .))) %>% 
  mutate(across(matches(c('nFeatures','nSignatures','nEpochs','batchSize','l1Size','validationPerc')),
                ~as.numeric(.))) %>% 
  #filter(nSignatures<=32) %>% 
  mutate(across(matches(c('nFeatures','nSignatures','nEpochs','batchSize','l1Size','validationPerc')),
                ~factor(., levels = sort(unique(.)))),
         `Validation loss` = best_model_validation_loss) %>% 
  pivot_longer(cols = contains('best_model_'),
               names_to = 'model', values_to = 'loss') %>% 
  mutate(model = gsub("best_model_|_loss", "", model)) %>% 
  ## get mean loss, Validation loss, "Validation / training loss", and min_val_loss_epoch for same-parameter (except seed) runs, if they exist
  group_by(nFeatures,nSignatures,nEpochs,batchSize,l1Size,validationPerc,normalization,allow_negative_weights,model) %>% 
  summarise_at(vars(min_val_loss_epoch, `Validation loss`, loss),
               ~mean(.)) %>% 
  ungroup


## split plot for normalization yes/no, as the scale of loss is different since the normalized data has a wider range
for(normalized in c("yes", "no")){
  
  # N_epochs = 6000
  # all_best_model_losses_epoch_plot = ggplot(all_best_model_losses_epoch %>% 
  #                                             filter(normalization==normalized),
  #                                           aes(x = model,
  #                                               y = loss,
  #                                               col = `Validation loss`)) +
  #   geom_col(aes(fill = model),
  #            linewidth = 1) +
  #   scale_y_continuous(labels = scales::label_number(accuracy = 0.1)) +
  #   scale_fill_manual(values = c("black", "lightgray")) +
  #   coord_flip() +
  #   geom_tile() +
  #   scale_color_gradient(low = "blue", high = "red") +
  #   geom_label(data = all_best_model_losses_epoch %>% 
  #                       filter(normalization==normalized) %>% 
  #                       filter(model == "training"),
  #              aes(label = min_val_loss_epoch),
  #              fill = "white",
  #              col = "black",
  #              size = 2,
  #              nudge_y = quantile(all_best_model_losses_epoch %>% 
  #                                   filter(normalization==normalized) %>% 
  #                                   pull(loss))[[1]],
  #              label.padding = unit(0.01, "lines"),
  #              label.size = 0) +
  #   ggh4x::facet_nested(rows = vars(nSignatures, batchSize),
  #                       cols = vars(normalization, validationPerc, l1Size),
  #                       labeller = label_both) +
  #   xlab("") +
  #   ylab(paste0("Best model's training vs. validation losses for ",unique(all_best_model_losses_epoch$nFeatures)," features and ",N_epochs," epochs -- Epoch of min. validation loss indicated")) +
  #   theme_classic() +
  #   theme(text = element_text(size = 7),
  #         axis.text.y = element_blank(),
  #         axis.line.y = element_blank(),
  #         axis.ticks.y = element_blank(),
  #         legend.position = "top")
  # ggsave(paste0("plots/all_best_model_losses_",N_epochs,"epochs_normalization_", normalized, ".jpg"),
  #        plot = all_best_model_losses_epoch_plot,
  #        device = "jpg",
  #        width = 16,
  #        height = 9,
  #        dpi = 600)
  
  ### compare nepochs, nsignatures, and validation %
  nepochs_nsig_valperc = ggplot(all_best_model_losses_epoch %>% 
                                  filter(normalization==normalized &
                                           batchSize == 64 &
                                           l1Size == 128),
                                aes(x = model,
                                    y = loss,
                                    col = `Validation loss`)) +
    geom_col(aes(fill = model),
             linewidth = 1.2) +
    scale_fill_manual(values = c("black", "lightgray")) +
    geom_tile() +
    scale_color_gradient(low = "blue", high = "red") +
    geom_hline(yintercept = all_best_model_losses_epoch %>% 
                 filter(normalization==normalized &
                          batchSize == 64 &
                          l1Size == 128) %>% 
                 pull(`Validation loss`) %>% min, 
               linetype = "dotted", color = "yellow") +
    geom_label(data = all_best_model_losses_epoch %>% 
                 filter(normalization==normalized &
                          batchSize == 64 &
                          l1Size == 128 &
                          model == "training"),
               aes(label = min_val_loss_epoch),
               fill = "white",
               col = "black",
               size = 1,
               nudge_y = -0.13,
               nudge_x = 0.5,
               label.padding = unit(0.05, "lines"),
               label.size = 0) +
    ggh4x::facet_nested(cols = vars(nSignatures, allow_negative_weights),
                        rows = vars(nEpochs, validationPerc),
                        switch = "x") +
    ylab("Loss") +
    scale_y_continuous(sec.axis = sec_axis(~., name = "N epochs\n% samples used for validation", breaks = NULL, labels = NULL)) +
    xlab("Negative weights allowed in decoder\nK signatures") +
    ggtitle(paste0("Epoch of model with lowest validation loss -- ", unique(all_best_model_losses_epoch$nFeatures)," features, input coefficients normalization: '", normalized, "', batch size: ", 64, ", 1st encoder layer size: ", 128, " neurons")) +
    theme_bw() +
    theme(text = element_text(size = 8), 
          axis.text.x = element_blank(),
          axis.text.y = element_text(size = 6),
          panel.grid.minor = element_blank(),
          axis.line.x = element_blank(),
          axis.ticks.x = element_blank(),
          strip.background = element_rect(fill = "white"),
          legend.position = "right",
          plot.title = element_text(hjust = 0.5))
  ggsave(paste0("plots/compare_nepochs_nsig_valperc_decoderweights_normalized_", normalized, ".jpg"),
         plot = nepochs_nsig_valperc,
         device = "jpg",
         width = 16,
         height = 9,
         dpi = 600)
}

## split plot for ALLOW NEGATIVE WEIGHTS yes/no
for(negative_weights_allowed in c("yes", "no")){

  ### compare nepochs, nsignatures, and validation %
  nepochs_nsig_valperc = ggplot(all_best_model_losses_epoch %>% 
                                  filter(allow_negative_weights==negative_weights_allowed &
                                           batchSize == 64 &
                                           l1Size == 128),
                                aes(x = model,
                                    y = loss,
                                    col = `Validation loss`)) +
    geom_col(aes(fill = model),
             linewidth = 1.2) +
    scale_fill_manual(values = c("black", "lightgray")) +
    geom_tile() +
    scale_color_gradient(low = "blue", high = "red") +
    geom_hline(yintercept = all_best_model_losses_epoch %>% 
                 filter(allow_negative_weights==negative_weights_allowed &
                          batchSize == 64 &
                          l1Size == 128) %>% 
                 pull(`Validation loss`) %>% min, 
               linetype = "dotted", color = "yellow") +
    geom_label(data = all_best_model_losses_epoch %>% 
                 filter(allow_negative_weights==negative_weights_allowed &
                          batchSize == 64 &
                          l1Size == 128 &
                          model == "training"),
               aes(label = min_val_loss_epoch),
               fill = "white",
               col = "black",
               size = 1,
               nudge_y = -0.13,
               nudge_x = 0.5,
               label.padding = unit(0.05, "lines"),
               label.size = 0) +
    ggh4x::facet_nested(cols = vars(nSignatures, normalization),
                        rows = vars(nEpochs, validationPerc),
                        switch = "x") +
    ylab("Loss") +
    scale_y_continuous(sec.axis = sec_axis(~., name = "N epochs\n% samples used for validation", breaks = NULL, labels = NULL)) +
    xlab("Input coefficients normalization\nK signatures") +
    ggtitle(paste0("Epoch of model with lowest validation loss -- ", unique(all_best_model_losses_epoch$nFeatures)," features, negative weights allowed in decoder: '", negative_weights_allowed, "', batch size: ", 64, ", 1st encoder layer size: ", 128, " neurons")) +
    theme_bw() +
    theme(text = element_text(size = 8), 
          axis.text.x = element_blank(),
          axis.text.y = element_text(size = 6),
          panel.grid.minor = element_blank(),
          axis.line.x = element_blank(),
          axis.ticks.x = element_blank(),
          strip.background = element_rect(fill = "white"),
          legend.position = "right",
          plot.title = element_text(hjust = 0.5))
  ggsave(paste0("plots/compare_nepochs_nsig_valperc_normalization_negweightsallowed_", negative_weights_allowed, ".jpg"),
         plot = nepochs_nsig_valperc,
         device = "jpg",
         width = 16,
         height = 9,
         dpi = 600)
}


## show signatures for a promising set of parameters
N_features = unique(all_best_model_losses_epoch$nFeatures)
N_epochs = 1000
validation_perc = 10
normalized = "no"
negative_weights_allowed = "yes"
batch_size = 64
l1_size = 128

best_parameters_plot = ggplot(all_best_model_losses_epoch %>% 
                                filter(nFeatures == N_features &
                                         nEpochs == N_epochs &
                                         validationPerc == validation_perc &
                                         normalization == normalized &
                                         allow_negative_weights == negative_weights_allowed &
                                         batchSize == batch_size &
                                         l1Size == l1_size),
                              aes(x = model,
                                  y = loss)) +
  geom_col(aes(fill = model),
           linewidth = 1) +
  scale_fill_manual(values = c("black", "lightgray")) +
  geom_label(data = all_best_model_losses_epoch %>% 
               filter(nFeatures == N_features &
                        nEpochs == N_epochs &
                        validationPerc == validation_perc &
                        normalization==normalized &
                        allow_negative_weights == negative_weights_allowed &
                        batchSize == batch_size &
                        l1Size == l1_size &
                        model == "training"),
             aes(label = min_val_loss_epoch),
             fill = "white",
             col = "black",
             size = 3,
             nudge_y = -0.1,
             nudge_x = 0.5,
             label.padding = unit(0.05, "lines"),
             label.size = 0) +
  facet_wrap(facets = vars(nSignatures),
             nrow = 1,
             strip.position = "bottom") +
  ylab("Loss") +
  xlab("K signatures") +
  ggtitle(paste0("Epoch of model with lowest validation loss -- ", N_features," features, ", N_epochs, " epochs, ", validation_perc, "% samples used for validation, input coefficients normalization: '", normalized, "', allow negative decoder weights: '", negative_weights_allowed, "', batch size: ", batch_size, ", 1st encoder layer size: ", l1_size, " neurons")) +
  theme_classic() +
  theme(text = element_text(size = 10), 
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = "right",
        strip.background = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 10))
ggsave(paste0("plots/best_parameters_", N_epochs, "epochs_negweightsallowed_", negative_weights_allowed, ".jpg"),
       plot = best_parameters_plot,
       device = "jpg",
       width = 16,
       height = 9,
       dpi = 600)



####
# for exposures and weights matrices
dir.create("exposures_weights")
# for significance deltas (regfeat vs SBS96) of pairwise signature cosine similarities 
dir.create("plots/cos_sim_deltas")
real_deltas_vs_bootstrap_hits = list()
# for feature hits that could be somehow related to a cosmic signature
stability_cutoff = 0
cosine_similarity_cutoff = 0
cosmic_similarity_cutoff = 0
max_n_feature_hits = 3
regfeats_cosmic_assoc = list()
n_top_similar_cosmic = 2

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

cosineSimil <- function(x) {
  mat = t(lsa::cosine(t(x)))
  return(mat)
}

sample_pheno_levels = unique(samples_info$altered_pathway_or_treatment_type)
jet_colors = colorRampPalette(c("gray", "red", "yellow", "green", "cyan", "blue", "magenta", "black"))
## if you want a fixed assignment (i.e. that is constant across plots) of a given color to a given sample's source × dMMR status:
fixed_jet_colors = jet_colors(length(sample_pheno_levels))
names(fixed_jet_colors) = sample_pheno_levels


## create plots

# load exposures and weights for given set of parameters (for all values of k)

N_features = unique(all_best_model_losses_epoch$nFeatures)
N_epochs = 1000
validation_perc = 10
normalized = "no"
negative_weights_allowed = "yes"
batch_size = 64
l1_size = 128

exposures_list = lapply(Sys.glob(paste0("res/nFeatures_", N_features, "__nSignatures_*__nEpochs_",N_epochs,"__batchSize_", batch_size, "__l1Size_", l1_size, "__validationPerc_", validation_perc, "__normalization_", normalized, "__allow_negative_weights_", negative_weights_allowed, "__seed_*/signature_exposures.tsv")),
                        read_tsv) %>% 
  setNames(sapply(., function(x) paste0("K",ncol(x) - 1)))
weights_list = lapply(Sys.glob(paste0("res/nFeatures_", N_features, "__nSignatures_*__nEpochs_",N_epochs,"__batchSize_", batch_size, "__l1Size_", l1_size, "__validationPerc_", validation_perc, "__normalization_", normalized, "__allow_negative_weights_", negative_weights_allowed, "__seed_*/signature_weights.tsv")),
                      read_tsv) %>% 
  setNames(sapply(., function(x) paste0("K",ncol(x) - 1)))

# define combinations of nFact and k that we want to plot

range_k = mixedsort(names(exposures_list))
maxK = max(as.numeric(gsub("K", "", range_k)))

for(optimal_k in range_k){
  
  ## parse sample exposures for these k signatures
  exposures = exposures_list[[optimal_k]] %>%
    pivot_longer(cols = contains("ae"), names_to = "Signature", values_to = "Exposure") %>%
    mutate(Signature = factor(Signature, levels = paste0("ae", seq(0, as.numeric(gsub("K", "", optimal_k))-1)))) %>% 
    arrange(Signature) %>% 
    relocate(Signature) %>% 
    # add metadata info (e.g. treatments, MSI, HR, smoking...)
    left_join(samples_info) %>%
    filter(!is.na(Exposure))
  
  ## parse signature weights
  
  # SBS weights in signatures
  weights = weights_list[[optimal_k]] %>%
    pivot_longer(cols = contains("ae"), names_to = "Signature", values_to = "Weight") %>%
    mutate(Signature = factor(Signature, levels = paste0("ae", seq(as.numeric(gsub("K", "", optimal_k))-1, 0)))) %>% 
    arrange(Signature) %>% 
    relocate(Signature) %>% 
    mutate(feature_group = ifelse(str_detect(Feature, ">"),
                                  "SBS",
                                  "Regional\nfeature"),
           `SBS group` = ifelse(feature_group == "SBS",
                                gsub("^.\\(|\\).$", "", Feature),
                                NA),
           `SBS group` = factor(`SBS group`,
                                levels = c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G"),
                                ordered = T),
           SBS96 = ifelse(feature_group == "SBS",
                          Feature,
                          NA),
           SBS96 = factor(SBS96,
                          levels = rev_trinuc_96,
                          ordered = T),
           `Regional\nfeature` = ifelse(feature_group == "Regional\nfeature",
                                        Feature,
                                        NA)) %>%
    arrange(SBS96, `Regional\nfeature`)
  
  ### sigprofiler (for SBS)
  cosmic_sig_similarity = weights %>%
    filter(feature_group == "SBS") %>%
    select(-c(Feature, feature_group, `SBS group`, `Regional\nfeature`)) %>%
    mutate(SBS96 = gsub("\\(", "\\[", SBS96),
           SBS96 = gsub("\\)", "\\]", SBS96),
           SBS96 = factor(SBS96,
                          levels = gsub("\\)", "\\]", gsub("\\(", "\\[", trinuc_96)),
                          ordered = T),
           # neg. weights to 0
           Weight = ifelse(Weight<0, 0, Weight)) %>%
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
    mutate(Signature = factor(Signature, levels = paste0("ae", seq(as.numeric(gsub("K", "", optimal_k))-1, 0)))) %>% 
    pivot_longer(cols = !contains("Signature"), names_to = "COSMIC", values_to = "Similarity") %>%
    # keep top similarity cosmic sbs for each signature
    group_by(Signature) %>%
    arrange(Similarity) %>%
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
    select(-c(feature_group, `Regional\nfeature`, `SBS group`, SBS96, `Max. sim. COSMIC`)) %>%
    pivot_wider(names_from = Feature, values_from = Weight) %>%
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
               size = 6) +
    xlab("") +
    ylab("") +
    theme_classic() +
    theme(text = element_text(size = 25),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top",
          legend.text = element_text(angle = 90, vjust = 0.5, hjust = 0.25))
  ggsave(paste0("plots/cos_sim_deltas/nEpochs",N_epochs,"_SBS96_weights_similarities_btw_sigs__", optimal_k, ".jpeg"),
         plot = heatmap_SBS96_sig_similarities,
         device = "jpeg",
         width = 25,
         height = 14,
         dpi = 600)
  
  ### 2) regional features
  reg_feat_sig_similarities = weights %>%
    filter(feature_group != "SBS") %>%
    select(-c(feature_group, `Regional\nfeature`, `SBS group`, SBS96, `Max. sim. COSMIC`)) %>%
    mutate(Signature = gsub("\n", "_", Signature)) %>%
    pivot_wider(names_from = Feature, values_from = Weight) %>%
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
               size = 6) +
    xlab("") +
    ylab("") +
    theme_classic() +
    theme(text = element_text(size = 25),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top",
          legend.text = element_text(angle = 90, vjust = 0.5, hjust = 0.25))
  ggsave(paste0("plots/cos_sim_deltas/nEpochs",N_epochs,"RegFeat_weights_similarities_btw_sigs__", optimal_k, ".jpeg"),
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
               size = 6) +
    xlab("") +
    ylab("") +
    theme_classic() +
    theme(text = element_text(size = 25),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top",
          legend.text = element_text(angle = 90, vjust = 0.5, hjust = 0.25))
  ggsave(paste0("plots/cos_sim_deltas/nEpochs",N_epochs,"_delta_cos_simil__", optimal_k, ".jpeg"),
         plot = heatmap_delta_cos_simil,
         device = "jpeg",
         width = 25,
         height = 14,
         dpi = 600)
  
  # ### 4) bootstrap: shuffle cosine similarities matrix of SBS96 to create a null distribution of deltas to which compare the real deltas for significance
  # bootstrap_n = 1000
  # bootstrapped_deltas = list()
  # 
  # for(i in seq(1,bootstrap_n)){
  # 
  #   SBS96_sig_similarities_clustered_shuffled = SBS96_sig_similarities_clustered %>%
  #     mutate(SigB = as.character(SigB)) %>%
  #     # remove SigA==SigB
  #     filter(SigA != SigB) %>%
  #     as_tibble %>%
  #     # remove rows where row1$SigA == row2$SigB && row1$SigB == row2$SigA
  #     mutate(pair_id = pmap_chr(list(SigA, SigB), ~paste(sort(c(...)), collapse = "__"))) %>%
  #     distinct(pair_id, .keep_all = TRUE) %>%
  #     select(-pair_id) %>%  # remove the auxiliary column
  #     transform(`cosine similarity` = sample(`cosine similarity`)) %>%
  #     rename("cosine similarity" = "cosine.similarity")
  # 
  #   reg_feat_sig_similarities_clustered_shuffled = reg_feat_sig_similarities_clustered %>%
  #     mutate(SigB = as.character(SigB)) %>%
  #     # remove SigA==SigB here as well
  #     filter(SigA != SigB) %>%
  #     as_tibble %>%
  #     # remove rows where row1$SigA == row2$SigB && row1$SigB == row2$SigA here as well
  #     mutate(pair_id = pmap_chr(list(SigA, SigB), ~paste(sort(c(...)), collapse = "__"))) %>%
  #     distinct(pair_id, .keep_all = TRUE) %>%
  #     select(-pair_id) %>%  # remove the auxiliary column
  #     # it's not actually necessary to sample this matrix as well, but anyways
  #     transform(`cosine similarity` = sample(`cosine similarity`)) %>%
  #     rename("cosine similarity" = "cosine.similarity")
  # 
  #   bootstrap_delta_cos_simil = bind_rows(SBS96_sig_similarities_clustered_shuffled,
  #                                         reg_feat_sig_similarities_clustered_shuffled) %>%
  #     select(-c(x,y)) %>%
  #     unite("Sigpair", SigA, SigB, sep = "__") %>%
  #     pivot_wider(names_from = feature_type, values_from = `cosine similarity`) %>%
  #     group_by(Sigpair) %>%
  #     ## regfeat - SBS96 (the more negative, the more dissimilar the signatures in regfeat compared to how similar they are in SBS96)
  #     summarise("Δ cosine similarity" = `Regional features` - SBS96) %>%
  #     separate(Sigpair, into = c("SigA", "SigB"), sep = "__")
  # 
  #   bootstrapped_deltas[[i]] = select(bootstrap_delta_cos_simil,
  #                                     `Δ cosine similarity`)
  # }
  # bootstrapped_deltas = bind_rows(bootstrapped_deltas) %>%
  #   mutate(Sigpair = "Bootstrap")
  # 
  # # to standardize delta
  # mean_delta = mean(bootstrapped_deltas$`Δ cosine similarity`)
  # sd_delta = sd(bootstrapped_deltas$`Δ cosine similarity`)
  # 
  # real_deltas_vs_bootstrap = delta_cos_simil_real %>%
  #   # remove SigA==SigB
  #   filter(SigA != SigB) %>%
  #   # remove rows where row1$SigA == row2$SigB && row1$SigB == row2$SigA
  #   mutate(pair_id = pmap_chr(list(SigA, SigB), ~paste(sort(c(...)), collapse = "__"))) %>%
  #   distinct(pair_id, .keep_all = TRUE) %>%
  #   select(-pair_id) %>%  # remove the auxiliary column
  #   unite("Sigpair", SigA, SigB, sep = " vs. ") %>%
  #   bind_rows(bootstrapped_deltas) %>%
  #   mutate(group = ifelse(Sigpair == "Bootstrap",
  #                         "Bootstrap",
  #                         "Real"),
  #          `Δ cosine similarity (standardized)` = (`Δ cosine similarity`-mean_delta) / sd_delta)
  # 
  # real_deltas_vs_bootstrap_hits[[paste0("nFact", nFact, "_", optimal_k)]] = real_deltas_vs_bootstrap %>%
  #   filter(group == "Real" & `Δ cosine similarity (standardized)` <= -1.96) %>%
  #   mutate("nFact" = nFact,
  #          "K" = optimal_k) %>%
  #   select(-group)
  # 
  # plot_real_deltas_vs_bootstrap = ggplot(real_deltas_vs_bootstrap,
  #                                        aes(x = `Δ cosine similarity (standardized)`,
  #                                            col = group,
  #                                            fill = group)) +
  #   scale_x_continuous(breaks = seq(-3, 3, 0.5)) +
  #   geom_density(data = real_deltas_vs_bootstrap %>%
  #                         filter(group == "Bootstrap")) +
  #   geom_bar(data = real_deltas_vs_bootstrap %>%
  #              filter(group == "Real")) +
  #   scale_color_manual(values = c("red", "blue")) +
  #   scale_fill_manual(values = c("red", "blue")) +
  #   geom_text(data = real_deltas_vs_bootstrap %>%
  #               filter(group == "Real" & `Δ cosine similarity (standardized)` <= -1.96),
  #             aes(label = Sigpair,
  #                 y = 0),
  #             angle = 90,
  #             vjust = -0.1,
  #             hjust = -0.1) +
  #   theme_bw() +
  #   xlab("Absolute difference of the signature cosine similarities in regional features vs. in SBS96 (standardized)") +
  #   ylab("Density (bootstrap, red) or counts (real, blue)") +
  #   theme(legend.title = element_blank())
  # ggsave(paste0("plots/cos_sim_deltas/nEpochs",N_epochs,"_deltas_real_vs_bootstrap__", optimal_k, ".jpeg"),
  #        plot = plot_real_deltas_vs_bootstrap,
  #        device = "jpeg",
  #        width = 13,
  #        height = 7,
  #        dpi = 600)
  
  
  ##### 5) detect relationships between regional feature(s) and cosmic SBS
  cosmic_sim_good = weights %>% 
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
          select(Signature, Feature, Weight, is.hit) %>% 
          group_by(Signature) %>%
          arrange(desc(abs(Weight))) %>% 
          slice_head(n = max_n_feature_hits) %>% 
          ungroup %>% 
          select(Signature, Feature, is.hit) %>% 
          distinct
        
        top5_in_hit_signatures = weights_sigpair %>% filter(is.hit=="hit") %>% select(Feature) %>% pull
        top5_in_hit_signatures = unique(top5_in_hit_signatures[duplicated(top5_in_hit_signatures)])
        top5_in_nohit_signatures = weights_sigpair %>% filter(is.hit=="no hit") %>% select(Feature) %>% pull %>% unique
        
        feature_hits = top5_in_hit_signatures[!top5_in_hit_signatures %in% top5_in_nohit_signatures]
        
        if(length(feature_hits) >=1){
          
          regfeats_cosmic_assoc_table = weights_sigpair %>% 
            filter(Feature %in% feature_hits) %>%         
            left_join(mutate(weights, Signature = gsub("\n", " ", Signature))) %>% 
            select(Signature, Feature, Weight, `Max. sim. COSMIC`) %>% 
            mutate("K" = optimal_k,
                   Sigpair_i = sigpair)
          
          # only record it IF the cosmic signature that is most common to the SBS component is the same for both signatures
          if(length(unique(gsub(": .*", "", regfeats_cosmic_assoc_table$`Max. sim. COSMIC`))) == 1){
            
            regfeats_cosmic_assoc[[optimal_k]] = regfeats_cosmic_assoc_table
          }
        }
      }
    }
  }
  
  
  ###################################################################################################
  
  ## write it for random forests
  write_tsv(exposures,
            paste0("exposures_weights/nEpochs",N_epochs,"_", optimal_k, "_exposures.tsv"))
  write_tsv(weights %>%
              rename("Chromatin feature" = "Regional\nfeature") %>%
              mutate(`Chromatin feature` = gsub("\n", "_", `Chromatin feature`)),
            paste0("exposures_weights/nEpochs",N_epochs,"_", optimal_k, "_weights.tsv"))
  
  
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
    theme(axis.text.x = element_text(angle = 45, hjust=1, size = 5),
          axis.text.y = element_text(size = 10),
          strip.text.x.top = element_text(size = 8, angle = 90, hjust=0),
          text = element_text(size = 10),
          strip.background = element_blank(),
          legend.key.size = unit(1, "cm"),
          legend.text = element_text(size = 10),
          legend.position = "top")
  
  
  #### weights plots
  
  ## regional features
  
  weights_plot_regfeat = ggplot(weights %>%
                                  filter(feature_group != "SBS") %>%
                                  select(-c(Feature, feature_group, contains("SBS"))),
                                aes(x = Weight,
                                    y = `Regional\nfeature`)) +
    scale_x_continuous(expand = c(0, 0),
                       breaks = seq(-10, 10, 0.1),
                       labels = function(x) round(x,1)) +
    scale_y_discrete(position = "right") +
    geom_col(aes(fill = `Regional\nfeature`)) +
    scale_fill_manual(values = jet_colors(length(unique(weights$`Regional\nfeature`)))) +
    guides(fill = guide_legend(override.aes = list(size=0.5),
                               nrow = 6)) +
    facet_wrap(~Signature, ncol = 1, scales = "free",
               strip.position="right") +
    xlab("Contribution (regional features)") +
    ylab("") +
    theme_classic() +
    theme(axis.ticks.y = element_blank(),
          axis.text.y = element_blank(),
          axis.line.y = element_blank(),
          axis.text.x = element_text(size = 10),
          text = element_text(size = 10),
          strip.background = element_blank(),
          strip.text.y.right = element_text(angle = 0),
          legend.title = element_blank(),
          legend.position = "top",
          legend.justification='right',
          legend.text = element_text(size = 7))
  
  ## SBS
  
  # to label the SBS plots with their COSMIC similarity
  max_sim_cosmic = weights %>%
    select(Signature,`Max. sim. COSMIC`) %>%
    distinct %>%
    arrange(Signature) %>%
    mutate(`Max. sim. COSMIC` = as.character(`Max. sim. COSMIC`)) %>%
    deframe()
  
  weights_plot_SBS = ggplot(weights %>%
                              filter(feature_group == "SBS") %>%
                              select(-c(Feature, feature_group, `Regional\nfeature`)),
                            aes(x = Weight,
                                y = SBS96)) +
    scale_x_continuous(expand = c(0, 0),
                       breaks = seq(-1, 1, 0.1),
                       labels = function(x) round(x, 1)) +
    geom_col(aes(fill = `SBS group`)) +
    scale_fill_manual(values = c("#00bfeb", "black", "#f3282f", "#cdc9ca", "#a1cc6b", "#f1c6c5")) +
    guides(fill = guide_legend(override.aes = list(size=1),
                               nrow = 6)) +
    facet_wrap(~Signature, ncol = 1, scales = "free",
               strip.position="right") +
    xlab("Contribution (SBS)") +
    ylab("") +
    theme_classic() +
    theme(axis.ticks.y = element_blank(),
          axis.text.y = element_blank(),
          axis.line.y = element_blank(),
          axis.text.x = element_text(size = 8),
          text = element_text(size = 10),
          strip.background = element_blank(),
          strip.text.y.right = element_blank(),
          legend.title = element_blank(),
          legend.position = "top",
          plot.margin = margin(0,0,0,-0.2, "cm"))
  
  # barplot of SBS cosmic top similarities
  barplot_cosmic = ggplot(weights %>%
                            filter(feature_group == "SBS") %>%
                            select(-c(Feature, feature_group, `Regional\nfeature`)) %>%
                            select(Signature, `Max. sim. COSMIC`) %>% 
                            distinct %>% 
                            separate(`Max. sim. COSMIC`, into = c("top-1", "top-2"), sep = " / ") %>% 
                            pivot_longer(cols = contains("top-"),
                                         names_to = "Max. sim. COSMIC",
                                         values_to = "COSMIC name and similarity") %>% 
                            separate(`COSMIC name and similarity`, into = c("COSMIC", "cosine similarity"), sep = ": ") %>% 
                            select(-`Max. sim. COSMIC`) %>% 
                            mutate(`cosine similarity` = as.numeric(`cosine similarity`)) %>% 
                            arrange(Signature, desc(`cosine similarity`)) %>% 
                            # remove SBS names if cos. sim is 0
                            mutate(COSMIC = ifelse(`cosine similarity` == 0, "", COSMIC)),
                          aes(x = COSMIC,
                              y = `cosine similarity`)) +
    coord_flip() +
    scale_y_continuous(n.breaks = 2) +
    geom_col() +
    facet_wrap(~Signature, ncol = 1, scales = "free_y") +
    theme_classic() +
    xlab("") +
    ggtitle("Top similar\nCOSMIC SBS") +
    ylab("cos similarity") +
    theme(text = element_text(size = 10),
          axis.text.x = element_text(size = 8, angle = 0),
          strip.text = element_blank(),
          strip.background = element_blank(),
          plot.title = element_text(hjust=0.5),
          plot.margin = margin(3,0,0.2,-0.1, "cm"))
  
  combined_plots = plot_grid(plot_grid(exposures_plot, ncol = 1),
                             NULL,
                             plot_grid(weights_plot_regfeat, ncol = 1),
                             NULL,
                             plot_grid(weights_plot_SBS, ncol = 1),
                             NULL,
                             plot_grid(barplot_cosmic, ncol = 1),
                             NULL,
                             ncol = 8,
                             rel_widths = c(1, 0.02, 0.2, -0.005, 0.1, -0.001, 0.1, 0.01))
  ggsave(paste0("plots/exposures_weights_plot__nFeatures_", N_features, "__nEpochs_",N_epochs,"__batchSize_", batch_size, "__l1Size_", l1_size, "__validationPerc_", validation_perc, "__normalization_", normalized, "__allow_negative_weights_", negative_weights_allowed, "__", optimal_k, ".jpeg"),
         plot = combined_plots,
         device = "jpeg",
         width = 25,
         height = 14,
         dpi = 600)
}

write_tsv(bind_rows(real_deltas_vs_bootstrap_hits) %>% 
            arrange(`Δ cosine similarity (standardized)`),
          paste0("plots/cos_sim_deltas/nEpochs",N_epochs,"_real_deltas_vs_bootstrap_hits.tsv"))

write_tsv(bind_rows(regfeats_cosmic_assoc) %>% 
            arrange(K, Sigpair_i),
          paste0("plots/cos_sim_deltas/nEpochs",N_epochs,"_regfeats_cosmic_assoc.tsv"))
