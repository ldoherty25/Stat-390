# Best Models



# primary checks ----

# load required packages
library(tidyverse)
library(tidymodels)
library(forecast)
library(caret)
library(timetk)
library(doMC)
library(dplyr)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)

# loading files
load("data_frames/arima_combined_models_df_maria.rda")
load("data_frames/multivar_prophet_maria.rda")
load("data_frames/univar_prophet_combined_average_metrics_maria.rda")
load("data_frames/xgboost_maria.rda")


# creating a final set of metrics ----

# binding the tables together
combined_metrics <- bind_rows(
  arima_combined_models_df,
  multivar_prophet_maria,
  univar_prophet_combined_average_metrics,
  xgboost_maria
)

# selecting best per country
best_rmse_per_country <- combined_metrics %>%
  group_by(Country) %>%
  dplyr::slice(which.min(RMSE)) %>%
  ungroup()

# removing unwanted columns
final_metrics <- best_rmse_per_country %>% 
  select(-c(p, d, q, P, D, Q, Best_Model, period, MAPE)) %>% 
  arrange(RMSE)

# printing the resulting table
print(final_metrics)



# saving files ----
save(final_metrics, file = "data_frames/final_metrics.rda")