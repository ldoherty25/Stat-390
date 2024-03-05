### AUTO-ARIMA UNIVARIATE ###

library(tidyverse)
library(stats)
library(forecast)
library(lubridate)
library(modelr)
library(purrr)
library(zoo)
library(randomForest)
library(caret)
library(imputeTS)
library(doMC)
library(patchwork)
library(seastests)
library(gridExtra)
library(timetk)
library(tidymodels)
library(tseries)
library(xts)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

#setting the seed
set.seed(1234)

#loading the univariate datasets
load("data/preprocessed/univariate/bolivia.rda")
load("data/preprocessed/univariate/brazil.rda")
load("data/preprocessed/univariate/colombia.rda")
load("data/preprocessed/univariate/iran.rda")
load("data/preprocessed/univariate/mexico.rda")
load("data/preprocessed/univariate/peru.rda")
load("data/preprocessed/univariate/russia.rda")
load("data/preprocessed/univariate/saudi.rda")
load("data/preprocessed/univariate/turkey.rda")
load("data/preprocessed/univariate/us.rda")

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

#setting the seed
set.seed(1234)

#############################################################################################################
## BOLIVIA ##

# Convert to a data frame
bolivia_auto_df <- as.data.frame(bolivia)

# Training and testing split
bolivia_auto_split_index <- floor(0.9 * nrow(bolivia_auto_df))
bolivia_auto_train_data <- bolivia_auto_df[1:bolivia_auto_split_index, ]
bolivia_auto_test_data <- bolivia_auto_df[(bolivia_auto_split_index + 1):nrow(bolivia_auto_df), ]

bolivia_auto_cv_folds <- createFolds(bolivia_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
bolivia_auto_curr_metrics <- data.frame(
  bolivia_auto_curr_fold = seq_along(bolivia_auto_cv_folds),
  RMSE = numeric(length(bolivia_auto_cv_folds)),
  MAE = numeric(length(bolivia_auto_cv_folds)),
  MSE = numeric(length(bolivia_auto_cv_folds)),
  MASE = numeric(length(bolivia_auto_cv_folds)),
  MAPE = numeric(length(bolivia_auto_cv_folds))
)

for (fold in seq_along(bolivia_auto_cv_folds)) {
  bolivia_auto_fold_train_data <- bolivia_auto_df[-bolivia_auto_cv_folds[[fold]], ]
  bolivia_auto_fold_validation_data <- bolivia_auto_df[bolivia_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  bolivia_auto_fold_model <- auto.arima(bolivia_auto_fold_train_data$owid_new_deaths)
  bolivia_auto_fold_forecast <- forecast(bolivia_auto_fold_model, h = nrow(bolivia_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  bolivia_auto_fold_accuracy <- forecast::accuracy(bolivia_auto_fold_forecast, bolivia_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  bolivia_auto_curr_metrics$RMSE[fold] <- sqrt(mean((bolivia_auto_fold_forecast$mean - bolivia_auto_fold_validation_data$owid_new_deaths)^2))
  bolivia_auto_curr_metrics$MAE[fold] <- mean(abs(bolivia_auto_fold_forecast$mean - bolivia_auto_fold_validation_data$owid_new_deaths))
  bolivia_auto_curr_metrics$MSE[fold] <- mean((bolivia_auto_fold_forecast$mean - bolivia_auto_fold_validation_data$owid_new_deaths)^2)
  bolivia_auto_curr_metrics$MASE[fold] <- bolivia_auto_fold_accuracy[1, "MASE"]
  bolivia_auto_curr_metrics$MAPE[fold] <- bolivia_auto_fold_accuracy[1, "MAPE"]
}

print(bolivia_auto_curr_metrics)

# Convert to a timeseries object
# bolivia_auto_ts <- as.ts(bolivia$date)
# 
# # Training and testing split
# bolivia_auto_split_index <- floor(0.9 * length(bolivia_auto_ts))
# bolivia_auto_train_data <- window(bolivia_auto_ts, end = bolivia_auto_split_index)
# bolivia_auto_test_data <- window(bolivia_auto_ts, start = bolivia_auto_split_index + 1)
# 
# # Modeling on training data
# bolivia_auto_arima_model <- auto.arima(bolivia_auto_train_data)
# 
# # # Modeling on the testing data
# bolivia_auto_arima_forecast <- forecast(bolivia_auto_arima_model, h = length(bolivia_auto_test_data))
# # 
# # # Evaluating the model with chosen metrics
# bolivia_auto_RMSE <- sqrt(mean((bolivia_auto_arima_forecast$mean - bolivia_auto_test_data)^2))
# bolivia_auto_MAE <- mean(abs(bolivia_auto_arima_forecast$mean - bolivia_auto_test_data))
# bolivia_auto_MSE <- mean((bolivia_auto_arima_forecast$mean - bolivia_auto_test_data)^2)
# bolivia_auto_accuracy <- forecast::accuracy(bolivia_auto_arima_forecast, window(bolivia_auto_test_data, start = 1))
# bolivia_auto_MASE <- bolivia_auto_accuracy[,"MAE"] / bolivia_auto_accuracy[,"RMSE"]
# bolivia_auto_MAPE <- bolivia_auto_accuracy[,"MAPE"]
# 
# ## Want to store the metrics in a dataframe:
# #initializing the datarame:
# bolivia_auto_metrics <- data.frame(
#   RMSE = numeric(0),
#   MAE = numeric(0),
#   MSE = numeric(0),
#   MASE = numeric(0),
#   MAPE = numeric(0)
# )
# #binding the metrics calculated above:
# bolivia_auto_metrics <- rbind(bolivia_auto_metrics, c(bolivia_auto_RMSE, bolivia_auto_MAE, 
#                                                       bolivia_auto_MSE, bolivia_auto_MASE, 
#                                                       bolivia_auto_MAPE))
# print(bolivia_auto_metrics)
# 
# bolivia_auto_cv_folds <- createFolds(window(bolivia_auto_train_data, start = 1), k = 5, list = TRUE, returnTrain = FALSE)
# 
# # bolivia_auto_cv_RMSE <- numeric(length = length(bolivia_auto_cv_folds))
# # bolivia_auto_cv_MAE <- numeric(length = length(bolivia_auto_cv_folds))
# # bolivia_auto_cv_MSE <- numeric(length = length(bolivia_auto_cv_folds))
# # bolivia_auto_cv_MASE <- numeric(length = length(bolivia_auto_cv_folds))
# # bolivia_auto_cv_MAPE <- numeric(length = length(bolivia_auto_cv_folds))
# 
# #empty dataframe for metrics of each fold
# bolivia_auto_curr_metrics <- data.frame(
#   bolivia_auto_curr_fold = seq_along(bolivia_auto_cv_folds),
#   RMSE = numeric(length(bolivia_auto_cv_folds)),
#   MAE = numeric(length(bolivia_auto_cv_folds)),
#   MSE = numeric(length(bolivia_auto_cv_folds)),
#   MASE = numeric(length(bolivia_auto_cv_folds)),
#   MAPE = numeric(length(bolivia_auto_cv_folds))
# )
# 
# for (fold in seq_along(bolivia_auto_cv_folds)) {
#   bolivia_auto_fold_train_data <- window(bolivia_auto_train_data, end = bolivia_auto_cv_folds[[fold]][1] - 1)
#   bolivia_auto_fold_validation_data <- window(bolivia_auto_train_data, 
#                                               start = bolivia_auto_cv_folds[[fold]][1], 
#                                               end = bolivia_auto_cv_folds[[fold]][length(bolivia_auto_cv_folds[[fold]])] - 1)
#   bolivia_auto_fold_validation_data_df <- as.data.frame(bolivia_auto_fold_validation_data)
#   
#   # Create a forecast for the current fold
#   bolivia_auto_fold_model <- auto.arima(bolivia_auto_fold_train_data)
#   bolivia_auto_fold_forecast <- forecast(bolivia_auto_fold_model, h = length(bolivia_auto_cv_folds[[fold]]))
#   
#   # Accuracy metrics for the current fold
#   bolivia_auto_fold_accuracy <- forecast::accuracy(bolivia_auto_fold_forecast, bolivia_auto_fold_validation_data_df$owid_new_deaths)
#   
#   # Store metrics in the data frame
#   bolivia_auto_curr_metrics$bolivia_auto_curr_fold[fold] <- fold
#   bolivia_auto_curr_metrics$RMSE[fold] <- sqrt(mean((bolivia_auto_fold_forecast$mean - bolivia_auto_fold_validation_data_df$owid_new_deaths)^2))
#   bolivia_auto_curr_metrics$MAE[fold] <- mean(abs(bolivia_auto_fold_forecast$mean - bolivia_auto_fold_validation_data_df$owid_new_deaths))
#   bolivia_auto_curr_metrics$MSE[fold] <- mean((bolivia_auto_fold_forecast$mean - bolivia_auto_fold_validation_data_df$owid_new_deaths)^2)
#   bolivia_auto_curr_metrics$MASE[fold] <- bolivia_auto_fold_accuracy[1, "MASE"]
#   bolivia_auto_curr_metrics$MAPE[fold] <- bolivia_auto_fold_accuracy[1, "MAPE"]
# }
#   
# print(bolivia_auto_curr_metrics)

# Evaluate cross-validation results
# bolivia_auto_fold_RMSE <- sqrt(mean((bolivia_auto_fold_res - bolivia_auto_test_data)^2))
# bolivia_auto_fold_MAE <- mean(abs(bolivia_auto_fold_res - bolivia_auto_test_data))
# bolivia_auto_fold_MSE <- mean((bolivia_auto_fold_res - bolivia_auto_test_data)^2)
# bolivia_auto_fold_accuracy <- forecast::accuracy(bolivia_auto_fold_forecast, window(bolivia_auto_test_data, start = 1))
# 
# # Print cross-validation results
# cat("CV RMSE:", cv_rmse, "\n")
# cat("CV MAE:", cv_mae, "\n")
# cat("CV MSE:", cv_mse, "\n")

###########################################################################################################
## BRAZIL ##

# Convert to a data frame
brazil_auto_df <- as.data.frame(brazil)

# Training and testing split
brazil_auto_split_index <- floor(0.9 * nrow(brazil_auto_df))
brazil_auto_train_data <- brazil_auto_df[1:brazil_auto_split_index, ]
brazil_auto_test_data <- brazil_auto_df[(brazil_auto_split_index + 1):nrow(brazil_auto_df), ]

brazil_auto_cv_folds <- createFolds(brazil_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
brazil_auto_curr_metrics <- data.frame(
  brazil_auto_curr_fold = seq_along(brazil_auto_cv_folds),
  RMSE = numeric(length(brazil_auto_cv_folds)),
  MAE = numeric(length(brazil_auto_cv_folds)),
  MSE = numeric(length(brazil_auto_cv_folds)),
  MASE = numeric(length(brazil_auto_cv_folds)),
  MAPE = numeric(length(brazil_auto_cv_folds))
)

for (fold in seq_along(brazil_auto_cv_folds)) {
  brazil_auto_fold_train_data <- brazil_auto_df[-brazil_auto_cv_folds[[fold]], ]
  brazil_auto_fold_validation_data <- brazil_auto_df[brazil_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  brazil_auto_fold_model <- auto.arima(brazil_auto_fold_train_data$owid_new_deaths)
  brazil_auto_fold_forecast <- forecast(brazil_auto_fold_model, h = nrow(brazil_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  brazil_auto_fold_accuracy <- forecast::accuracy(brazil_auto_fold_forecast, brazil_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  brazil_auto_curr_metrics$RMSE[fold] <- sqrt(mean((brazil_auto_fold_forecast$mean - brazil_auto_fold_validation_data$owid_new_deaths)^2))
  brazil_auto_curr_metrics$MAE[fold] <- mean(abs(brazil_auto_fold_forecast$mean - brazil_auto_fold_validation_data$owid_new_deaths))
  brazil_auto_curr_metrics$MSE[fold] <- mean((brazil_auto_fold_forecast$mean - brazil_auto_fold_validation_data$owid_new_deaths)^2)
  brazil_auto_curr_metrics$MASE[fold] <- brazil_auto_fold_accuracy[1, "MASE"]
  brazil_auto_curr_metrics$MAPE[fold] <- brazil_auto_fold_accuracy[1, "MAPE"]
}

print(brazil_auto_curr_metrics)

#############################################################################################################
## COLOMBIA ##

# Convert to a data frame
colombia_auto_df <- as.data.frame(colombia)

# Training and testing split
colombia_auto_split_index <- floor(0.9 * nrow(colombia_auto_df))
colombia_auto_train_data <- colombia_auto_df[1:colombia_auto_split_index, ]
colombia_auto_test_data <- colombia_auto_df[(colombia_auto_split_index + 1):nrow(colombia_auto_df), ]

colombia_auto_cv_folds <- createFolds(colombia_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
colombia_auto_curr_metrics <- data.frame(
  colombia_auto_curr_fold = seq_along(colombia_auto_cv_folds),
  RMSE = numeric(length(colombia_auto_cv_folds)),
  MAE = numeric(length(colombia_auto_cv_folds)),
  MSE = numeric(length(colombia_auto_cv_folds)),
  MASE = numeric(length(colombia_auto_cv_folds)),
  MAPE = numeric(length(colombia_auto_cv_folds))
)

for (fold in seq_along(colombia_auto_cv_folds)) {
  colombia_auto_fold_train_data <- colombia_auto_df[-colombia_auto_cv_folds[[fold]], ]
  colombia_auto_fold_validation_data <- colombia_auto_df[colombia_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  colombia_auto_fold_model <- auto.arima(colombia_auto_fold_train_data$owid_new_deaths)
  colombia_auto_fold_forecast <- forecast(colombia_auto_fold_model, h = nrow(colombia_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  colombia_auto_fold_accuracy <- forecast::accuracy(colombia_auto_fold_forecast, colombia_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  colombia_auto_curr_metrics$RMSE[fold] <- sqrt(mean((colombia_auto_fold_forecast$mean - colombia_auto_fold_validation_data$owid_new_deaths)^2))
  colombia_auto_curr_metrics$MAE[fold] <- mean(abs(colombia_auto_fold_forecast$mean - colombia_auto_fold_validation_data$owid_new_deaths))
  colombia_auto_curr_metrics$MSE[fold] <- mean((colombia_auto_fold_forecast$mean - colombia_auto_fold_validation_data$owid_new_deaths)^2)
  colombia_auto_curr_metrics$MASE[fold] <- colombia_auto_fold_accuracy[1, "MASE"]
  colombia_auto_curr_metrics$MAPE[fold] <- colombia_auto_fold_accuracy[1, "MAPE"]
}

print(colombia_auto_curr_metrics)

##########################################################################################################################
## IRAN ##

# Convert to a data frame
iran_auto_df <- as.data.frame(iran)

# Training and testing split
iran_auto_split_index <- floor(0.9 * nrow(iran_auto_df))
iran_auto_train_data <- iran_auto_df[1:iran_auto_split_index, ]
iran_auto_test_data <- iran_auto_df[(iran_auto_split_index + 1):nrow(iran_auto_df), ]

iran_auto_cv_folds <- createFolds(iran_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
iran_auto_curr_metrics <- data.frame(
  iran_auto_curr_fold = seq_along(iran_auto_cv_folds),
  RMSE = numeric(length(iran_auto_cv_folds)),
  MAE = numeric(length(iran_auto_cv_folds)),
  MSE = numeric(length(iran_auto_cv_folds)),
  MASE = numeric(length(iran_auto_cv_folds)),
  MAPE = numeric(length(iran_auto_cv_folds))
)

for (fold in seq_along(iran_auto_cv_folds)) {
  iran_auto_fold_train_data <- iran_auto_df[-iran_auto_cv_folds[[fold]], ]
  iran_auto_fold_validation_data <- iran_auto_df[iran_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  iran_auto_fold_model <- auto.arima(iran_auto_fold_train_data$owid_new_deaths)
  iran_auto_fold_forecast <- forecast(iran_auto_fold_model, h = nrow(iran_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  iran_auto_fold_accuracy <- forecast::accuracy(iran_auto_fold_forecast, iran_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  iran_auto_curr_metrics$RMSE[fold] <- sqrt(mean((iran_auto_fold_forecast$mean - iran_auto_fold_validation_data$owid_new_deaths)^2))
  iran_auto_curr_metrics$MAE[fold] <- mean(abs(iran_auto_fold_forecast$mean - iran_auto_fold_validation_data$owid_new_deaths))
  iran_auto_curr_metrics$MSE[fold] <- mean((iran_auto_fold_forecast$mean - iran_auto_fold_validation_data$owid_new_deaths)^2)
  iran_auto_curr_metrics$MASE[fold] <- iran_auto_fold_accuracy[1, "MASE"]
  iran_auto_curr_metrics$MAPE[fold] <- iran_auto_fold_accuracy[1, "MAPE"]
}

print(iran_auto_curr_metrics)

#####################################################################################################################
## MEXICO ##

# Convert to a data frame
mexico_auto_df <- as.data.frame(mexico)

# Training and testing split
mexico_auto_split_index <- floor(0.9 * nrow(mexico_auto_df))
mexico_auto_train_data <- mexico_auto_df[1:mexico_auto_split_index, ]
mexico_auto_test_data <- mexico_auto_df[(mexico_auto_split_index + 1):nrow(mexico_auto_df), ]

mexico_auto_cv_folds <- createFolds(mexico_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
mexico_auto_curr_metrics <- data.frame(
  mexico_auto_curr_fold = seq_along(mexico_auto_cv_folds),
  RMSE = numeric(length(mexico_auto_cv_folds)),
  MAE = numeric(length(mexico_auto_cv_folds)),
  MSE = numeric(length(mexico_auto_cv_folds)),
  MASE = numeric(length(mexico_auto_cv_folds)),
  MAPE = numeric(length(mexico_auto_cv_folds))
)

for (fold in seq_along(mexico_auto_cv_folds)) {
  mexico_auto_fold_train_data <- mexico_auto_df[-mexico_auto_cv_folds[[fold]], ]
  mexico_auto_fold_validation_data <- mexico_auto_df[mexico_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  mexico_auto_fold_model <- auto.arima(mexico_auto_fold_train_data$owid_new_deaths)
  mexico_auto_fold_forecast <- forecast(mexico_auto_fold_model, h = nrow(mexico_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  mexico_auto_fold_accuracy <- forecast::accuracy(mexico_auto_fold_forecast, mexico_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  mexico_auto_curr_metrics$RMSE[fold] <- sqrt(mean((mexico_auto_fold_forecast$mean - mexico_auto_fold_validation_data$owid_new_deaths)^2))
  mexico_auto_curr_metrics$MAE[fold] <- mean(abs(mexico_auto_fold_forecast$mean - mexico_auto_fold_validation_data$owid_new_deaths))
  mexico_auto_curr_metrics$MSE[fold] <- mean((mexico_auto_fold_forecast$mean - mexico_auto_fold_validation_data$owid_new_deaths)^2)
  mexico_auto_curr_metrics$MASE[fold] <- mexico_auto_fold_accuracy[1, "MASE"]
  mexico_auto_curr_metrics$MAPE[fold] <- mexico_auto_fold_accuracy[1, "MAPE"]
}

print(mexico_auto_curr_metrics)

#####################################################################################################################
## PERU ##

# Convert to a data frame
peru_auto_df <- as.data.frame(peru)

# Training and testing split
peru_auto_split_index <- floor(0.9 * nrow(peru_auto_df))
peru_auto_train_data <- peru_auto_df[1:peru_auto_split_index, ]
peru_auto_test_data <- peru_auto_df[(peru_auto_split_index + 1):nrow(peru_auto_df), ]

peru_auto_cv_folds <- createFolds(peru_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
peru_auto_curr_metrics <- data.frame(
  peru_auto_curr_fold = seq_along(peru_auto_cv_folds),
  RMSE = numeric(length(peru_auto_cv_folds)),
  MAE = numeric(length(peru_auto_cv_folds)),
  MSE = numeric(length(peru_auto_cv_folds)),
  MASE = numeric(length(peru_auto_cv_folds)),
  MAPE = numeric(length(peru_auto_cv_folds))
)

for (fold in seq_along(peru_auto_cv_folds)) {
  peru_auto_fold_train_data <- peru_auto_df[-peru_auto_cv_folds[[fold]], ]
  peru_auto_fold_validation_data <- peru_auto_df[peru_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  peru_auto_fold_model <- auto.arima(peru_auto_fold_train_data$owid_new_deaths)
  peru_auto_fold_forecast <- forecast(peru_auto_fold_model, h = nrow(peru_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  peru_auto_fold_accuracy <- forecast::accuracy(peru_auto_fold_forecast, peru_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  peru_auto_curr_metrics$RMSE[fold] <- sqrt(mean((peru_auto_fold_forecast$mean - peru_auto_fold_validation_data$owid_new_deaths)^2))
  peru_auto_curr_metrics$MAE[fold] <- mean(abs(peru_auto_fold_forecast$mean - peru_auto_fold_validation_data$owid_new_deaths))
  peru_auto_curr_metrics$MSE[fold] <- mean((peru_auto_fold_forecast$mean - peru_auto_fold_validation_data$owid_new_deaths)^2)
  peru_auto_curr_metrics$MASE[fold] <- peru_auto_fold_accuracy[1, "MASE"]
  peru_auto_curr_metrics$MAPE[fold] <- peru_auto_fold_accuracy[1, "MAPE"]
}

print(peru_auto_curr_metrics)

########################################################################################################################
## RUSSIA ##

# Convert to a data frame
russia_auto_df <- as.data.frame(russia)

# Training and testing split
russia_auto_split_index <- floor(0.9 * nrow(russia_auto_df))
russia_auto_train_data <- russia_auto_df[1:russia_auto_split_index, ]
russia_auto_test_data <- russia_auto_df[(russia_auto_split_index + 1):nrow(russia_auto_df), ]

russia_auto_cv_folds <- createFolds(russia_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
russia_auto_curr_metrics <- data.frame(
  russia_auto_curr_fold = seq_along(russia_auto_cv_folds),
  RMSE = numeric(length(russia_auto_cv_folds)),
  MAE = numeric(length(russia_auto_cv_folds)),
  MSE = numeric(length(russia_auto_cv_folds)),
  MASE = numeric(length(russia_auto_cv_folds)),
  MAPE = numeric(length(russia_auto_cv_folds))
)

for (fold in seq_along(russia_auto_cv_folds)) {
  russia_auto_fold_train_data <- russia_auto_df[-russia_auto_cv_folds[[fold]], ]
  russia_auto_fold_validation_data <- russia_auto_df[russia_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  russia_auto_fold_model <- auto.arima(russia_auto_fold_train_data$owid_new_deaths)
  russia_auto_fold_forecast <- forecast(russia_auto_fold_model, h = nrow(russia_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  russia_auto_fold_accuracy <- forecast::accuracy(russia_auto_fold_forecast, russia_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  russia_auto_curr_metrics$RMSE[fold] <- sqrt(mean((russia_auto_fold_forecast$mean - russia_auto_fold_validation_data$owid_new_deaths)^2))
  russia_auto_curr_metrics$MAE[fold] <- mean(abs(russia_auto_fold_forecast$mean - russia_auto_fold_validation_data$owid_new_deaths))
  russia_auto_curr_metrics$MSE[fold] <- mean((russia_auto_fold_forecast$mean - russia_auto_fold_validation_data$owid_new_deaths)^2)
  russia_auto_curr_metrics$MASE[fold] <- russia_auto_fold_accuracy[1, "MASE"]
  russia_auto_curr_metrics$MAPE[fold] <- russia_auto_fold_accuracy[1, "MAPE"]
}

print(russia_auto_curr_metrics)

#######################################################################################################################
## SAUDI ##

# Convert to a data frame
saudi_auto_df <- as.data.frame(saudi)

# Training and testing split
saudi_auto_split_index <- floor(0.9 * nrow(saudi_auto_df))
saudi_auto_train_data <- saudi_auto_df[1:saudi_auto_split_index, ]
saudi_auto_test_data <- saudi_auto_df[(saudi_auto_split_index + 1):nrow(saudi_auto_df), ]

saudi_auto_cv_folds <- createFolds(saudi_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
saudi_auto_curr_metrics <- data.frame(
  saudi_auto_curr_fold = seq_along(saudi_auto_cv_folds),
  RMSE = numeric(length(saudi_auto_cv_folds)),
  MAE = numeric(length(saudi_auto_cv_folds)),
  MSE = numeric(length(saudi_auto_cv_folds)),
  MASE = numeric(length(saudi_auto_cv_folds)),
  MAPE = numeric(length(saudi_auto_cv_folds))
)

for (fold in seq_along(saudi_auto_cv_folds)) {
  saudi_auto_fold_train_data <- saudi_auto_df[-saudi_auto_cv_folds[[fold]], ]
  saudi_auto_fold_validation_data <- saudi_auto_df[saudi_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  saudi_auto_fold_model <- auto.arima(saudi_auto_fold_train_data$owid_new_deaths)
  saudi_auto_fold_forecast <- forecast(saudi_auto_fold_model, h = nrow(saudi_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  saudi_auto_fold_accuracy <- forecast::accuracy(saudi_auto_fold_forecast, saudi_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  saudi_auto_curr_metrics$RMSE[fold] <- sqrt(mean((saudi_auto_fold_forecast$mean - saudi_auto_fold_validation_data$owid_new_deaths)^2))
  saudi_auto_curr_metrics$MAE[fold] <- mean(abs(saudi_auto_fold_forecast$mean - saudi_auto_fold_validation_data$owid_new_deaths))
  saudi_auto_curr_metrics$MSE[fold] <- mean((saudi_auto_fold_forecast$mean - saudi_auto_fold_validation_data$owid_new_deaths)^2)
  saudi_auto_curr_metrics$MASE[fold] <- saudi_auto_fold_accuracy[1, "MASE"]
  saudi_auto_curr_metrics$MAPE[fold] <- saudi_auto_fold_accuracy[1, "MAPE"]
}

print(saudi_auto_curr_metrics)

#######################################################################################################################
## TURKEY ##

# Convert to a data frame
turkey_auto_df <- as.data.frame(turkey)

# Training and testing split
turkey_auto_split_index <- floor(0.9 * nrow(turkey_auto_df))
turkey_auto_train_data <- turkey_auto_df[1:turkey_auto_split_index, ]
turkey_auto_test_data <- turkey_auto_df[(turkey_auto_split_index + 1):nrow(turkey_auto_df), ]

turkey_auto_cv_folds <- createFolds(turkey_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
turkey_auto_curr_metrics <- data.frame(
  turkey_auto_curr_fold = seq_along(turkey_auto_cv_folds),
  RMSE = numeric(length(turkey_auto_cv_folds)),
  MAE = numeric(length(turkey_auto_cv_folds)),
  MSE = numeric(length(turkey_auto_cv_folds)),
  MASE = numeric(length(turkey_auto_cv_folds)),
  MAPE = numeric(length(turkey_auto_cv_folds))
)

for (fold in seq_along(turkey_auto_cv_folds)) {
  turkey_auto_fold_train_data <- turkey_auto_df[-turkey_auto_cv_folds[[fold]], ]
  turkey_auto_fold_validation_data <- turkey_auto_df[turkey_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  turkey_auto_fold_model <- auto.arima(turkey_auto_fold_train_data$owid_new_deaths)
  turkey_auto_fold_forecast <- forecast(turkey_auto_fold_model, h = nrow(turkey_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  turkey_auto_fold_accuracy <- forecast::accuracy(turkey_auto_fold_forecast, turkey_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  turkey_auto_curr_metrics$RMSE[fold] <- sqrt(mean((turkey_auto_fold_forecast$mean - turkey_auto_fold_validation_data$owid_new_deaths)^2))
  turkey_auto_curr_metrics$MAE[fold] <- mean(abs(turkey_auto_fold_forecast$mean - turkey_auto_fold_validation_data$owid_new_deaths))
  turkey_auto_curr_metrics$MSE[fold] <- mean((turkey_auto_fold_forecast$mean - turkey_auto_fold_validation_data$owid_new_deaths)^2)
  turkey_auto_curr_metrics$MASE[fold] <- turkey_auto_fold_accuracy[1, "MASE"]
  turkey_auto_curr_metrics$MAPE[fold] <- turkey_auto_fold_accuracy[1, "MAPE"]
}

print(turkey_auto_curr_metrics)

######################################################################################################################
## US ##

# Convert to a data frame
us_auto_df <- as.data.frame(us)

# Training and testing split
us_auto_split_index <- floor(0.9 * nrow(us_auto_df))
us_auto_train_data <- us_auto_df[1:us_auto_split_index, ]
us_auto_test_data <- us_auto_df[(us_auto_split_index + 1):nrow(us_auto_df), ]

us_auto_cv_folds <- createFolds(us_auto_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Create an empty data frame with numeric columns
us_auto_curr_metrics <- data.frame(
  us_auto_curr_fold = seq_along(us_auto_cv_folds),
  RMSE = numeric(length(us_auto_cv_folds)),
  MAE = numeric(length(us_auto_cv_folds)),
  MSE = numeric(length(us_auto_cv_folds)),
  MASE = numeric(length(us_auto_cv_folds)),
  MAPE = numeric(length(us_auto_cv_folds))
)

for (fold in seq_along(us_auto_cv_folds)) {
  us_auto_fold_train_data <- us_auto_df[-us_auto_cv_folds[[fold]], ]
  us_auto_fold_validation_data <- us_auto_df[us_auto_cv_folds[[fold]], ]
  
  # Create a forecast for the current fold
  us_auto_fold_model <- auto.arima(us_auto_fold_train_data$owid_new_deaths)
  us_auto_fold_forecast <- forecast(us_auto_fold_model, h = nrow(us_auto_fold_validation_data))
  
  # Accuracy metrics for the current fold
  us_auto_fold_accuracy <- forecast::accuracy(us_auto_fold_forecast, us_auto_fold_validation_data$owid_new_deaths)
  
  # Store metrics in the data frame
  us_auto_curr_metrics$RMSE[fold] <- sqrt(mean((us_auto_fold_forecast$mean - us_auto_fold_validation_data$owid_new_deaths)^2))
  us_auto_curr_metrics$MAE[fold] <- mean(abs(us_auto_fold_forecast$mean - us_auto_fold_validation_data$owid_new_deaths))
  us_auto_curr_metrics$MSE[fold] <- mean((us_auto_fold_forecast$mean - us_auto_fold_validation_data$owid_new_deaths)^2)
  us_auto_curr_metrics$MASE[fold] <- us_auto_fold_accuracy[1, "MASE"]
  us_auto_curr_metrics$MAPE[fold] <- us_auto_fold_accuracy[1, "MAPE"]
}

print(us_auto_curr_metrics)
