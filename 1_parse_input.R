library(tidyverse)
library(conflicted)
conflict_prefer("filter", "dplyr")
conflict_prefer("rename", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("map", "purrr")
conflict_prefer("extract", "magrittr")
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



### first write the original coefficients
read_tsv("../coefficients_resampling/original_coeff.tsv") %>% 
  pivot_wider(names_from = feature_name, values_from = estimate) %>% 
  write_tsv("datasets/original_coeff.tsv")



### now go with resampled ones

set.seed(1)
nIters = 1000 # 1 for each epoch in autoencoder training
dir.create("datasets")

for(training_fraction in c(0.7, 0.8, 0.9)){

  outputDir = paste0("datasets/validation_perc_", 100*(1-training_fraction))
  dir.create(outputDir)

  ####
  ## we want that the samples that are assigned to validation ((1-training_fraction)*100 %) keep the proportions of `alteration`
  alteration_freqs = samples_info %>% 
    pull(alteration) %>% 
    table %>% as.data.frame %>%
    `colnames<-`(c("alteration", "freq")) %>% 
    # remove altered_pathway_or_treatment_types for which taking a (1-training_fraction)*100 % thereof is <1 sample
    mutate(freq_in_validation = freq * (1-training_fraction)) %>% 
    mutate(freq_in_validation = round(freq_in_validation)) %>% 
    filter(freq_in_validation >= 0.99)
  
  # remove alterations for which taking a (1-training_fraction)*100 % thereof is <1 sample
  samples_info_tmp = samples_info %>% 
    filter(alteration %in% alteration_freqs$alteration)
  
  # store the latter here
  samples_info_tmp2 = samples_info %>% 
    filter(!alteration %in% alteration_freqs$alteration) %>% 
    ## do the same as before but with altered_pathway_or_treatment_type
    pull(altered_pathway_or_treatment_type) %>% 
    table %>% as.data.frame %>%
    `colnames<-`(c("altered_pathway_or_treatment_type", "freq")) %>% 
    # remove altered_pathway_or_treatment_types for which taking a (1-training_fraction)*100 % thereof is <1 sample
    mutate(freq_in_validation = freq * (1-training_fraction)) %>% 
    mutate(freq_in_validation = round(freq_in_validation)) %>% 
    filter(freq_in_validation >= 0.99)
  
  validation_samples_tmp1 = samples_info_tmp %>% 
    inner_join(alteration_freqs, by = "alteration") %>% 
    select(Sample, alteration, freq_in_validation) %>% 
    distinct() %>% 
    group_by(alteration) %>% 
    group_map(~slice_sample(., n = .x$freq_in_validation[1])) %>%
    bind_rows() %>% 
    ungroup() %>% 
    select(Sample) %>% 
    mutate(train_val = "validation")
  validation_samples_tmp2 = samples_info_tmp %>% 
    inner_join(samples_info_tmp2, by = "altered_pathway_or_treatment_type") %>% 
    select(Sample, altered_pathway_or_treatment_type, freq_in_validation) %>% 
    distinct() %>% 
    group_by(altered_pathway_or_treatment_type) %>% 
    group_map(~slice_sample(., n = .x$freq_in_validation[1])) %>%
    bind_rows() %>% 
    ungroup() %>% 
    select(Sample) %>% 
    mutate(train_val = "validation")
  validation_samples = bind_rows(validation_samples_tmp1,
                                 validation_samples_tmp2) %>% 
    distinct
  
  # remaining are training
  training_samples = samples_info %>% 
    select(Sample) %>% 
    distinct %>% 
    filter(!Sample %in% validation_samples$Sample) %>% 
    mutate(train_val = "training")
  
  # store info on which samples are for training and which for validation
  train_val_sets = bind_rows(training_samples,
                             validation_samples)
  
  
  
  ####
  # load nIters permutations of the coefficient matrix, generated for the NMF
  perm_coeff_iters = read_tsv(paste0("../coefficients_resampling/permuted_coefficients_", nIters, "iters.tsv")) %>% 
    left_join(train_val_sets) %>% 
    group_split(train_val)
  
  names(perm_coeff_iters) = c(unique(perm_coeff_iters[[1]]$train_val),
                              unique(perm_coeff_iters[[2]]$train_val))
  
  perm_coeff_iters_training = perm_coeff_iters$training %>% 
    select(-train_val) %>% 
    # reshape
    pivot_wider(names_from = feature_name, values_from = resampled_estimate) %>% 
    # split as a list, by iteration
    group_split(nIter) %>% 
    map(~select(.x, -nIter))
  
  perm_coeff_iters_validation = perm_coeff_iters$validation %>% 
    select(-train_val) %>% 
    # reshape
    pivot_wider(names_from = feature_name, values_from = resampled_estimate) %>% 
    # split as a list, by iteration
    group_split(nIter) %>% 
    map(~select(.x, -nIter))
  
  
  ## write each permuted TRAINING table as input for MUSE-XAE
  for(iter in seq(1, nIters)){
  
    write_tsv(perm_coeff_iters_training[[iter]],
              paste0(outputDir,"/perm_coeff_iter", iter, "_training.tsv"))
  }
  
  # write a single table for validation
  write_tsv(perm_coeff_iters_validation[[1]],
            paste0(outputDir,"/perm_coeff_validation.tsv"))
  gc()
}
