## Multivariate XGBoost



# primary checks ----

# load packages
library(dplyr)
library(xgboost)
library(caret)
library(doMC)
library(ggplot2)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
conflicted::conflict_prefer("filter", "dplyr")

# setting a seed
set.seed(1234)



# loading files ----
load("xgboost_components/xgb_model_ii_maria.rda")
load("data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")



# building metric set for each country ----
turkey_testing_df <- preprocessed_covid_multi_imputed %>%
  filter(country == "Peru") %>%
  janitor::clean_names() %>%
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  arrange(ds) %>%
  select(-country, -date) # Keep 'ds' for now to ensure we have a date column for arranging

# ckecking for character columns and converting them to factors then to numeric
turkey_test_data <- turkey_testing_df %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric) %>%
  select(-ds, -y)

# ensuring the data is all numeric now
if (any(sapply(turkey_test_data, is.character))) {
  stop("Data still contains non-numeric columns.")
}

# converting to matrix for prediction
turkey_test_data_matrix <- as.matrix(turkey_test_data)

# making predictions
turkey_predictions <- predict(xgb_model_ii, turkey_test_data_matrix)

# extracting the actual values for 'y' from testing dataframe
turkey_actuals <- turkey_testing_df$y

# calculating performance metrics
turkey_RMSE <- sqrt(mean((turkey_predictions - turkey_actuals)^2))
turkey_MAE <- mean(abs(turkey_predictions - turkey_actuals))
turkey_MSE <- mean((turkey_predictions - turkey_actuals)^2)
turkey_MAPE <- mean(abs((turkey_predictions - turkey_actuals) / turkey_actuals), na.rm = TRUE)
turkey_MASE <- mean(abs(turkey_predictions - turkey_actuals)) / mean(abs(diff(na.omit(turkey_actuals))), na.rm = TRUE)

# creating a data frame for the metrics
turkey_metrics_table <- data.frame(
  Country = "turkey",
  Best_Model = "Multivariate XGBoost",
  RMSE = turkey_RMSE, 
  MAE = turkey_MAE, 
  MSE = turkey_MSE, 
  MAPE = turkey_MAPE, 
  MASE = turkey_MASE
)

# printing metrics
print(turkey_metrics_table)



# merging country metrics----

xgboost_maria <- rbind(bolivia_metrics_table,
                       brazil_metrics_table,
                       russia_metrics_table,
                       us_metrics_table,
                       iran_metrics_table,
                       saudi_metrics_table,
                       colombia_metrics_table,
                       mexico_metrics_table,
                       turkey_metrics_table,
                       peru_metrics_table)

xgboost_maria <- xgboost_maria %>%
  arrange(RMSE)

save(xgboost_maria, file = "data_frames/xgboost_maria.rda")
