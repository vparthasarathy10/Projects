# Set working directory to the project folder (ADJUST AS NECESSARY)
setwd("~/Desktop/src/Code")

# We assume the user has the necessary packages installed
library(ggplot2)
library(plm)
library(did)
library(lubridate)
library(dplyr)
library(mp)

# Sourcing the first script
source("Generate_Spring_Forward_Plots.R")

# Save plots from the first script
 for (i in 1:length(att_forward_plots)) {
  p <- att_forward_plots[[i]]
  file_name <- paste0("~/Desktop/src/Output/", "forward_plot", i+1989, ".png")
  ggsave(file_name, p)
 }

# Reset working directory to code folder (ADJUST AS NECESSARY)
setwd("~/Desktop/src/Code")

# Sourcing the second script
source("Generate_Fall_Back_Plots.R")

# Save plots from the second script
for (i in 1:length(att_back_plots)) {
  p <- att_back_plots[[i]]
  file_name <- paste0("~/Desktop/src/Output/", "back_plot", i+1989, ".png")
  ggsave(file_name, p)
}

# Save aggregate immediate treatment effects plot
ggsave("~/Desktop/src/Output/Aggregate_Immediate_Effects.png", plot)
