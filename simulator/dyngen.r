#' Step 1: Define backbone and other parameters
library(tidyverse)
library(dyngen)

set.seed(10)
model <- 
  initialise_model(
    num_tfs = 12,
    num_targets = 30,
    num_hks = 15,
    backbone = backbone_bifurcating(),
    verbose = TRUE,
    download_cache_dir = "~/.cache/dyngen",
    num_cores = 
  )

plot_backbone_statenet(model)

plot_backbone_modulenet(model)

names(list_backbones())

#' Step 2: Generate transcription factors (TFs)
model <- generate_tf_network(model)
## Generating TF network
plot_feature_network(model, show_targets = FALSE)

#' Step 3: Sample target genes and housekeeping genes (HKs)
model <- generate_feature_network(model)
## Sampling feature network from real network
plot_feature_network(model)
plot_feature_network(model, show_hks = TRUE)

#' Step 4: Generate kinetics
model <- generate_kinetics(model)
## Generating kinetics for 72 features
## Generating formulae
plot_feature_network(model)
plot_feature_network(model, show_hks = TRUE)

#' Step 5: Simulate gold standard
model <- generate_gold_standard(model)
plot_gold_simulations(model) + scale_colour_brewer(palette = "Dark2")
plot_gold_expression(model, what = "x") # mrna
plot_gold_expression(model, label_changing = FALSE) # premrna, mrna, and protein

#' Step 6: Simulate cells.
model <- generate_cells(model)
plot_simulations(model)
plot_gold_simulations(model) + scale_colour_brewer(palette = "Dark2")
plot_gold_mappings(model, do_facet = FALSE) + scale_colour_brewer(palette = "Dark2")
plot_simulation_expression(model, 1:4, what = "x")

#' Step 7: Experiment emulation
model <- generate_experiment(model)

#' Step 8: Convert to a dynwrap object
dataset <- wrap_dataset(model)
library(dynplot)
plot_dimred(dataset)