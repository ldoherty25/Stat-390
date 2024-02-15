## ARIMA

# primary checks ----

# load packages
library(tidyverse)
library(tidymodels)
library(reshape2)
library(lubridate)
library(forecast)
library(modelr)
library(purrr)
library(zoo)
library(TTR)
library(randomForest)
library(caret)
library(imputeTS)
library(doMC)
library(patchwork)
library(seastests)
library(gridExtra)
library(timetk)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)


## load data ----
load("data/preprocessed/univariate/not_split/brazil.rda")



# splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
total_days <- nrow(brazil)
train_days <- ceiling(0.9 * total_days)
test_days <- ceiling((total_days - train_days))

# creating folds
br_folds <- time_series_cv(
  brazil,
  date_var = date,
  initial = train_days,
  assess = test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
br_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()



# applying ARIMA model ----

# storing RMSE
rmse_results <- numeric(length(br_folds$splits))

# fitting and calculating RMSE
for (i in seq_along(br_folds$splits)) {
  fold <- br_folds$splits[[i]]
  train_data <- fold$data[fold$in_id, ]
  test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  arima_model <- arima(train_data$owid_new_deaths, order = c(1, 1, 1))
  
  # forecasting with ARIMA
  forecast_values <- forecast(arima_model, h = nrow(test_data))
  
  # calculating and storing RMSE
  rmse <- sqrt(mean((forecast_values$mean - test_data$owid_new_deaths)^2))
  rmse_results[i] <- rmse
}

# printing RMSE
print(rmse_results)

# # visualizing each fold
# for (i in seq_along(br_folds$splits)) {
#   fold <- br_folds$splits[[i]]
#   train_data <- fold$data[fold$in_id, ]
#   test_data <- fold$data[fold$out_id, ]
#   
#   # printing testing and training data
#   print(paste("Fold", i))
#   print(head(train_data))
#   print(head(test_data))
# }