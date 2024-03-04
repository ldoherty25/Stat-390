#### XGBOOST UNIVAR ###

library(tidyverse)
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
library(stats)
library(xgboost)
library(data.table)
library(tseries)

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


## BOLIVIA ##
#to avoid confusion
bolivia_boost <- as.data.table(bolivia)

## Feature engineering: Create lagged features

# bolivia_lags <- 3
# for (i in 1:bolivia_lags) {
#   col_name <- paste0("lag_", i)
#   bolivia[, (col_name) := shift(owid_new_deaths, i, type = "lag")]
# }
#had to use a different way because of Mac and package version difficulties

bolivia_lags <- 3
bolivia_lagged_cols <- paste0("lag_", 1:bolivia_lags)

for (i in 1:bolivia_lags) {
  bolivia_col_name <- bolivia_lagged_cols[i]
  set(bolivia_boost, j = bolivia_col_name, value = shift(bolivia_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
bolivia_boost <- na.omit(bolivia_boost)

## Create training and testing data
bolivia_train_prop <- 0.9
bolivia_split_index <- floor(nrow(bolivia_boost) * bolivia_train_prop)
bolivia_train_data <- bolivia_boost[1:bolivia_split_index, ]
bolivia_test_data <- bolivia_boost[(bolivia_split_index + 1):nrow(bolivia_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
bolivia_train_matrix <- xgb.DMatrix(data = as.matrix(bolivia_train_data[, -(1:2)]), 
                                    label = bolivia_train_data$owid_new_deaths)
bolivia_test_matrix <- xgb.DMatrix(data = as.matrix(bolivia_test_data[, -(1:2)]), 
                                   label = bolivia_test_data$owid_new_deaths)


## Define XGBoost parameters
bolivia_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
bolivia_xgb_model <- xgboost(params = bolivia_boost_params, data = bolivia_train_matrix, 
                             nrounds = bolivia_boost_params[[5]])

## Making predictions on the test set
bolivia_preds <- predict(bolivia_xgb_model, bolivia_test_matrix)

## Evaluating the model with the chosen metrics
bolivia_boost_RMSE <- sqrt(mean((bolivia_preds - bolivia_test_data$owid_new_deaths)^2))
bolivia_boost_MAE <- mean(abs(bolivia_preds - bolivia_test_data$owid_new_deaths))
bolivia_boost_MSE <- mean((bolivia_preds - bolivia_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
bolivia_accuracy_metrics <- forecast::accuracy(bolivia_preds, bolivia_test_data$owid_new_deaths)
bolivia_boost_MAPE <- bolivia_accuracy_metrics[,"MAPE"]
#bolivia_boost_MASE <- bolivia_accuracy_metrics[,"MASE"]
#for some reason MASE is not calculated using accuracy(). So let's calculate it using RMSE and MAE
bolivia_boost_MASE <- bolivia_accuracy_metrics[,"MAE"] / bolivia_accuracy_metrics[,"RMSE"]
#bolivia_boost_MASE <- MASE(bolivia_preds, bolivia_test_data$owid_new_deaths, na.rm = TRUE)
#bolivia_boost_MAPE <- MAPE(bolivia_preds, bolivia_test_data$owid_new_deaths)

## Want to store the metrics in a dataframe:
#initializing the datarame:
bolivia_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
bolivia_boost_metrics <- rbind(bolivia_boost_metrics, c(bolivia_boost_RMSE, bolivia_boost_MAE, 
                                                        bolivia_boost_MSE, bolivia_boost_MASE, 
                                                        bolivia_boost_MAPE))
bolivia_boost_metrics <- bolivia_boost_metrics %>%
  rename(RMSE = X6.54803732319967, MAE = X5.05849538530622, MSE = X42.876792786016,
         MASE = X0.77252085405561, MAPE = X26.6302684767642)
print(bolivia_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", bolivia_boost_RMSE, "\n")
cat("MAE:", bolivia_boost_MAE, "\n")
cat("MSE:", bolivia_boost_MSE, "\n")
cat("MASE:", bolivia_boost_MASE, "\n")
cat("MAPE:", bolivia_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
bolivia_boost_cv_folds <- createFolds(bolivia_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
bolivia_boost_cv_RMSE <- numeric(length = length(bolivia_boost_cv_folds))
bolivia_boost_cv_MAE <- numeric(length = length(bolivia_boost_cv_folds))
bolivia_boost_cv_MSE <- numeric(length = length(bolivia_boost_cv_folds))
bolivia_boost_cv_MASE <- numeric(length = length(bolivia_boost_cv_folds))
bolivia_boost_cv_MAPE <- numeric(length = length(bolivia_boost_cv_folds))

#empty vector for metrics of each fold
bolivia_boost_curr_metrics <- data.frame(
  bolivia_curr_fold = integer(length(bolivia_boost_cv_folds)),
  RMSE = bolivia_boost_cv_RMSE,
  MAE = bolivia_boost_cv_MAE,
  MSE = bolivia_boost_cv_MSE,
  MASE = bolivia_boost_cv_MASE,
  MAPE = bolivia_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(bolivia_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  bolivia_fold_train_data <- bolivia_train_data[-bolivia_boost_cv_folds[[fold]], ]
  bolivia_fold_validation_data <- bolivia_train_data[bolivia_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  bolivia_fold_train_matrix <- xgb.DMatrix(data = as.matrix(bolivia_fold_train_data[, -(1:2)]), 
                                           label = bolivia_fold_train_data$owid_new_deaths)
  bolivia_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(bolivia_fold_validation_data[, -(1:2)]), 
                                                label = bolivia_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  bolivia_fold_xgb_model <- xgboost(params = bolivia_boost_params, data = bolivia_fold_train_matrix, 
                                    nrounds = bolivia_boost_params[[5]])
  
  # Make predictions on the validation set
  bolivia_fold_preds <- predict(bolivia_fold_xgb_model, bolivia_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  bolivia_boost_curr_metrics$bolivia_curr_fold[fold] <- fold
  bolivia_boost_cv_accuracy_metrics <- forecast::accuracy(bolivia_fold_preds, bolivia_fold_validation_data$owid_new_deaths)
  bolivia_boost_curr_metrics$RMSE[fold] <- sqrt(mean((bolivia_fold_preds - bolivia_fold_validation_data$owid_new_deaths)^2))
  bolivia_boost_curr_metrics$MAE[fold] <- mean(abs(bolivia_fold_preds - bolivia_fold_validation_data$owid_new_deaths))
  bolivia_boost_curr_metrics$MSE[fold] <- mean((bolivia_fold_preds - bolivia_fold_validation_data$owid_new_deaths)^2)
  bolivia_boost_curr_metrics$MASE[fold] <- bolivia_boost_cv_accuracy_metrics[,"MAE"] / bolivia_boost_cv_accuracy_metrics[,"RMSE"]
  bolivia_boost_curr_metrics$MAPE[fold] <- bolivia_boost_cv_accuracy_metrics[,"MAPE"]
    
}

print(bolivia_boost_curr_metrics)


##########################################################################################################
## BRAZIL ##

#to avoid confusion
brazil_boost <- as.data.table(brazil)

## Feature engineering: Create lagged features
brazil_lags <- 7
brazil_lagged_cols <- paste0("lag_", 1:brazil_lags)

for (i in 1:brazil_lags) {
  brazil_col_name <- brazil_lagged_cols[i]
  set(brazil_boost, j = brazil_col_name, value = shift(brazil_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
brazil_boost <- na.omit(brazil_boost)

## Create training and testing data
brazil_train_prop <- 0.9
brazil_split_index <- floor(nrow(brazil_boost) * brazil_train_prop)
brazil_train_data <- brazil_boost[1:brazil_split_index, ]
brazil_test_data <- brazil_boost[(brazil_split_index + 1):nrow(brazil_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
brazil_train_matrix <- xgb.DMatrix(data = as.matrix(brazil_train_data[, -(1:2)]), 
                                    label = brazil_train_data$owid_new_deaths)
brazil_test_matrix <- xgb.DMatrix(data = as.matrix(brazil_test_data[, -(1:2)]), 
                                   label = brazil_test_data$owid_new_deaths)


## Define XGBoost parameters
brazil_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
brazil_xgb_model <- xgboost(params = brazil_boost_params, data = brazil_train_matrix, 
                             nrounds = brazil_boost_params[[5]])

## Making predictions on the test set
brazil_preds <- predict(brazil_xgb_model, brazil_test_matrix)

## Evaluating the model with the chosen metrics
brazil_boost_RMSE <- sqrt(mean((brazil_preds - brazil_test_data$owid_new_deaths)^2))
brazil_boost_MAE <- mean(abs(brazil_preds - brazil_test_data$owid_new_deaths))
brazil_boost_MSE <- mean((brazil_preds - brazil_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
brazil_accuracy_metrics <- forecast::accuracy(brazil_preds, brazil_test_data$owid_new_deaths)
brazil_boost_MAPE <- brazil_accuracy_metrics[,"MAPE"]
brazil_boost_MASE <- brazil_accuracy_metrics[,"MAE"] / brazil_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
brazil_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
brazil_boost_metrics <- rbind(brazil_boost_metrics, c(brazil_boost_RMSE, brazil_boost_MAE, 
                                                      brazil_boost_MSE, brazil_boost_MASE, 
                                                      brazil_boost_MAPE))
brazil_boost_metrics <- brazil_boost_metrics %>%
  rename(RMSE = X202.630175524217, MAE = X167.351117220792, MSE = X41058.9880329749,
         MASE = X0.825894350571648, MAPE = X43.491330074258)
print(brazil_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", brazil_boost_RMSE, "\n")
cat("MAE:", brazil_boost_MAE, "\n")
cat("MSE:", brazil_boost_MSE, "\n")
cat("MASE:", brazil_boost_MASE, "\n")
cat("MAPE:", brazil_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
brazil_boost_cv_folds <- createFolds(brazil_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
brazil_boost_cv_RMSE <- numeric(length = length(brazil_boost_cv_folds))
brazil_boost_cv_MAE <- numeric(length = length(brazil_boost_cv_folds))
brazil_boost_cv_MSE <- numeric(length = length(brazil_boost_cv_folds))
brazil_boost_cv_MASE <- numeric(length = length(brazil_boost_cv_folds))
brazil_boost_cv_MAPE <- numeric(length = length(brazil_boost_cv_folds))

#empty vector for metrics of each fold
brazil_boost_curr_metrics <- data.frame(
  brazil_curr_fold = integer(length(brazil_boost_cv_folds)),
  RMSE = brazil_boost_cv_RMSE,
  MAE = brazil_boost_cv_MAE,
  MSE = brazil_boost_cv_MSE,
  MASE = brazil_boost_cv_MASE,
  MAPE = brazil_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(brazil_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  brazil_fold_train_data <- brazil_train_data[-brazil_boost_cv_folds[[fold]], ]
  brazil_fold_validation_data <- brazil_train_data[brazil_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  brazil_fold_train_matrix <- xgb.DMatrix(data = as.matrix(brazil_fold_train_data[, -(1:2)]), 
                                           label = brazil_fold_train_data$owid_new_deaths)
  brazil_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(brazil_fold_validation_data[, -(1:2)]), 
                                                label = brazil_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  brazil_fold_xgb_model <- xgboost(params = brazil_boost_params, data = brazil_fold_train_matrix, 
                                    nrounds = brazil_boost_params[[5]])
  
  # Make predictions on the validation set
  brazil_fold_preds <- predict(brazil_fold_xgb_model, brazil_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  brazil_boost_curr_metrics$brazil_curr_fold[fold] <- fold
  brazil_boost_cv_accuracy_metrics <- forecast::accuracy(brazil_fold_preds, brazil_fold_validation_data$owid_new_deaths)
  brazil_boost_curr_metrics$RMSE[fold] <- sqrt(mean((brazil_fold_preds - brazil_fold_validation_data$owid_new_deaths)^2))
  brazil_boost_curr_metrics$MAE[fold] <- mean(abs(brazil_fold_preds - brazil_fold_validation_data$owid_new_deaths))
  brazil_boost_curr_metrics$MSE[fold] <- mean((brazil_fold_preds - brazil_fold_validation_data$owid_new_deaths)^2)
  brazil_boost_curr_metrics$MASE[fold] <- brazil_boost_cv_accuracy_metrics[,"MAE"] / brazil_boost_cv_accuracy_metrics[,"RMSE"]
  brazil_boost_curr_metrics$MAPE[fold] <- brazil_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(brazil_boost_curr_metrics)

####################################################################################################
## COLOMBIA ##

#to avoid confusion
colombia_boost <- as.data.table(colombia)

## Feature engineering: Create lagged features
colombia_lags <- 3
colombia_lagged_cols <- paste0("lag_", 1:colombia_lags)

for (i in 1:colombia_lags) {
  colombia_col_name <- colombia_lagged_cols[i]
  set(colombia_boost, j = colombia_col_name, value = shift(colombia_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
colombia_boost <- na.omit(colombia_boost)

## Create training and testing data
colombia_train_prop <- 0.9
colombia_split_index <- floor(nrow(colombia_boost) * colombia_train_prop)
colombia_train_data <- colombia_boost[1:colombia_split_index, ]
colombia_test_data <- colombia_boost[(colombia_split_index + 1):nrow(colombia_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
colombia_train_matrix <- xgb.DMatrix(data = as.matrix(colombia_train_data[, -(1:2)]), 
                                   label = colombia_train_data$owid_new_deaths)
colombia_test_matrix <- xgb.DMatrix(data = as.matrix(colombia_test_data[, -(1:2)]), 
                                  label = colombia_test_data$owid_new_deaths)


## Define XGBoost parameters
colombia_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
colombia_xgb_model <- xgboost(params = colombia_boost_params, data = colombia_train_matrix, 
                            nrounds = colombia_boost_params[[5]])

## Making predictions on the test set
colombia_preds <- predict(colombia_xgb_model, colombia_test_matrix)

## Evaluating the model with the chosen metrics
colombia_boost_RMSE <- sqrt(mean((colombia_preds - colombia_test_data$owid_new_deaths)^2))
colombia_boost_MAE <- mean(abs(colombia_preds - colombia_test_data$owid_new_deaths))
colombia_boost_MSE <- mean((colombia_preds - colombia_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
colombia_accuracy_metrics <- forecast::accuracy(colombia_preds, colombia_test_data$owid_new_deaths)
colombia_boost_MAPE <- colombia_accuracy_metrics[,"MAPE"]
colombia_boost_MASE <- colombia_accuracy_metrics[,"MAE"] / colombia_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
colombia_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
colombia_boost_metrics <- rbind(colombia_boost_metrics, c(colombia_boost_RMSE, colombia_boost_MAE, 
                                                          colombia_boost_MSE, colombia_boost_MASE, 
                                                          colombia_boost_MAPE))
colombia_boost_metrics <- colombia_boost_metrics %>%
  rename(RMSE = X21.9438904364818, MAE = X18.951526295055, MSE = X481.534327488319,
         MASE = X0.863635659771069, MAPE = X10.9296566714466)
print(colombia_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", colombia_boost_RMSE, "\n")
cat("MAE:", colombia_boost_MAE, "\n")
cat("MSE:", colombia_boost_MSE, "\n")
cat("MASE:", colombia_boost_MASE, "\n")
cat("MAPE:", colombia_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
colombia_boost_cv_folds <- createFolds(colombia_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
colombia_boost_cv_RMSE <- numeric(length = length(colombia_boost_cv_folds))
colombia_boost_cv_MAE <- numeric(length = length(colombia_boost_cv_folds))
colombia_boost_cv_MSE <- numeric(length = length(colombia_boost_cv_folds))
colombia_boost_cv_MASE <- numeric(length = length(colombia_boost_cv_folds))
colombia_boost_cv_MAPE <- numeric(length = length(colombia_boost_cv_folds))

#empty vector for metrics of each fold
colombia_boost_curr_metrics <- data.frame(
  colombia_curr_fold = integer(length(colombia_boost_cv_folds)),
  RMSE = colombia_boost_cv_RMSE,
  MAE = colombia_boost_cv_MAE,
  MSE = colombia_boost_cv_MSE,
  MASE = colombia_boost_cv_MASE,
  MAPE = colombia_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(colombia_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  colombia_fold_train_data <- colombia_train_data[-colombia_boost_cv_folds[[fold]], ]
  colombia_fold_validation_data <- colombia_train_data[colombia_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  colombia_fold_train_matrix <- xgb.DMatrix(data = as.matrix(colombia_fold_train_data[, -(1:2)]), 
                                          label = colombia_fold_train_data$owid_new_deaths)
  colombia_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(colombia_fold_validation_data[, -(1:2)]), 
                                               label = colombia_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  colombia_fold_xgb_model <- xgboost(params = colombia_boost_params, data = colombia_fold_train_matrix, 
                                   nrounds = colombia_boost_params[[5]])
  
  # Make predictions on the validation set
  colombia_fold_preds <- predict(colombia_fold_xgb_model, colombia_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  colombia_boost_curr_metrics$colombia_curr_fold[fold] <- fold
  colombia_boost_cv_accuracy_metrics <- forecast::accuracy(colombia_fold_preds, colombia_fold_validation_data$owid_new_deaths)
  colombia_boost_curr_metrics$RMSE[fold] <- sqrt(mean((colombia_fold_preds - colombia_fold_validation_data$owid_new_deaths)^2))
  colombia_boost_curr_metrics$MAE[fold] <- mean(abs(colombia_fold_preds - colombia_fold_validation_data$owid_new_deaths))
  colombia_boost_curr_metrics$MSE[fold] <- mean((colombia_fold_preds - colombia_fold_validation_data$owid_new_deaths)^2)
  colombia_boost_curr_metrics$MASE[fold] <- colombia_boost_cv_accuracy_metrics[,"MAE"] / colombia_boost_cv_accuracy_metrics[,"RMSE"]
  colombia_boost_curr_metrics$MAPE[fold] <- colombia_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(colombia_boost_curr_metrics)

########################################################################################################
## IRAN ##

#to avoid confusion
iran_boost <- as.data.table(iran)

## Feature engineering: Create lagged features
iran_lags <- 3
iran_lagged_cols <- paste0("lag_", 1:iran_lags)

for (i in 1:iran_lags) {
  iran_col_name <- iran_lagged_cols[i]
  set(iran_boost, j = iran_col_name, value = shift(iran_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
iran_boost <- na.omit(iran_boost)

## Create training and testing data
iran_train_prop <- 0.9
iran_split_index <- floor(nrow(iran_boost) * iran_train_prop)
iran_train_data <- iran_boost[1:iran_split_index, ]
iran_test_data <- iran_boost[(iran_split_index + 1):nrow(iran_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
iran_train_matrix <- xgb.DMatrix(data = as.matrix(iran_train_data[, -(1:2)]), 
                                     label = iran_train_data$owid_new_deaths)
iran_test_matrix <- xgb.DMatrix(data = as.matrix(iran_test_data[, -(1:2)]), 
                                    label = iran_test_data$owid_new_deaths)


## Define XGBoost parameters
iran_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
iran_xgb_model <- xgboost(params = iran_boost_params, data = iran_train_matrix, 
                              nrounds = iran_boost_params[[5]])

## Making predictions on the test set
iran_preds <- predict(iran_xgb_model, iran_test_matrix)

## Evaluating the model with the chosen metrics
iran_boost_RMSE <- sqrt(mean((iran_preds - iran_test_data$owid_new_deaths)^2))
iran_boost_MAE <- mean(abs(iran_preds - iran_test_data$owid_new_deaths))
iran_boost_MSE <- mean((iran_preds - iran_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
iran_accuracy_metrics <- forecast::accuracy(iran_preds, iran_test_data$owid_new_deaths)
iran_boost_MAPE <- iran_accuracy_metrics[,"MAPE"]
iran_boost_MASE <- iran_accuracy_metrics[,"MAE"] / iran_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
iran_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
iran_boost_metrics <- rbind(iran_boost_metrics, c(iran_boost_RMSE, iran_boost_MAE, 
                                                  iran_boost_MSE, iran_boost_MASE, 
                                                  iran_boost_MAPE))
iran_boost_metrics <- iran_boost_metrics %>%
  rename(RMSE = X75.1928428714595, MAE = X65.6308099365234, MSE = X5653.963619092,
         MASE = X0.872833203669634, MAPE = X22.9312503167492)
print(iran_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", iran_boost_RMSE, "\n")
cat("MAE:", iran_boost_MAE, "\n")
cat("MSE:", iran_boost_MSE, "\n")
cat("MASE:", iran_boost_MASE, "\n")
cat("MAPE:", iran_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
iran_boost_cv_folds <- createFolds(iran_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
iran_boost_cv_RMSE <- numeric(length = length(iran_boost_cv_folds))
iran_boost_cv_MAE <- numeric(length = length(iran_boost_cv_folds))
iran_boost_cv_MSE <- numeric(length = length(iran_boost_cv_folds))
iran_boost_cv_MASE <- numeric(length = length(iran_boost_cv_folds))
iran_boost_cv_MAPE <- numeric(length = length(iran_boost_cv_folds))

#empty vector for metrics of each fold
iran_boost_curr_metrics <- data.frame(
  iran_curr_fold = integer(length(iran_boost_cv_folds)),
  RMSE = iran_boost_cv_RMSE,
  MAE = iran_boost_cv_MAE,
  MSE = iran_boost_cv_MSE,
  MASE = iran_boost_cv_MASE,
  MAPE = iran_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(iran_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  iran_fold_train_data <- iran_train_data[-iran_boost_cv_folds[[fold]], ]
  iran_fold_validation_data <- iran_train_data[iran_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  iran_fold_train_matrix <- xgb.DMatrix(data = as.matrix(iran_fold_train_data[, -(1:2)]), 
                                            label = iran_fold_train_data$owid_new_deaths)
  iran_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(iran_fold_validation_data[, -(1:2)]), 
                                                 label = iran_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  iran_fold_xgb_model <- xgboost(params = iran_boost_params, data = iran_fold_train_matrix, 
                                     nrounds = iran_boost_params[[5]])
  
  # Make predictions on the validation set
  iran_fold_preds <- predict(iran_fold_xgb_model, iran_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  iran_boost_curr_metrics$iran_curr_fold[fold] <- fold
  iran_boost_cv_accuracy_metrics <- forecast::accuracy(iran_fold_preds, iran_fold_validation_data$owid_new_deaths)
  iran_boost_curr_metrics$RMSE[fold] <- sqrt(mean((iran_fold_preds - iran_fold_validation_data$owid_new_deaths)^2))
  iran_boost_curr_metrics$MAE[fold] <- mean(abs(iran_fold_preds - iran_fold_validation_data$owid_new_deaths))
  iran_boost_curr_metrics$MSE[fold] <- mean((iran_fold_preds - iran_fold_validation_data$owid_new_deaths)^2)
  iran_boost_curr_metrics$MASE[fold] <- iran_boost_cv_accuracy_metrics[,"MAE"] / iran_boost_cv_accuracy_metrics[,"RMSE"]
  iran_boost_curr_metrics$MAPE[fold] <- iran_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(iran_boost_curr_metrics)

#############################################################################################################
## MEXICO ##

#to avoid confusion
mexico_boost <- as.data.table(mexico)

## Feature engineering: Create lagged features
mexico_lags <- 7
mexico_lagged_cols <- paste0("lag_", 1:mexico_lags)

for (i in 1:mexico_lags) {
  mexico_col_name <- mexico_lagged_cols[i]
  set(mexico_boost, j = mexico_col_name, value = shift(mexico_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
mexico_boost <- na.omit(mexico_boost)

## Create training and testing data
mexico_train_prop <- 0.9
mexico_split_index <- floor(nrow(mexico_boost) * mexico_train_prop)
mexico_train_data <- mexico_boost[1:mexico_split_index, ]
mexico_test_data <- mexico_boost[(mexico_split_index + 1):nrow(mexico_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
mexico_train_matrix <- xgb.DMatrix(data = as.matrix(mexico_train_data[, -(1:2)]), 
                                 label = mexico_train_data$owid_new_deaths)
mexico_test_matrix <- xgb.DMatrix(data = as.matrix(mexico_test_data[, -(1:2)]), 
                                label = mexico_test_data$owid_new_deaths)


## Define XGBoost parameters
mexico_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
mexico_xgb_model <- xgboost(params = mexico_boost_params, data = mexico_train_matrix, 
                          nrounds = mexico_boost_params[[5]])

## Making predictions on the test set
mexico_preds <- predict(mexico_xgb_model, mexico_test_matrix)

## Evaluating the model with the chosen metrics
mexico_boost_RMSE <- sqrt(mean((mexico_preds - mexico_test_data$owid_new_deaths)^2))
mexico_boost_MAE <- mean(abs(mexico_preds - mexico_test_data$owid_new_deaths))
mexico_boost_MSE <- mean((mexico_preds - mexico_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
mexico_accuracy_metrics <- forecast::accuracy(mexico_preds, mexico_test_data$owid_new_deaths)
mexico_boost_MAPE <- mexico_accuracy_metrics[,"MAPE"]
mexico_boost_MASE <- mexico_accuracy_metrics[,"MAE"] / mexico_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
mexico_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
mexico_boost_metrics <- rbind(mexico_boost_metrics, c(mexico_boost_RMSE, mexico_boost_MAE, 
                                                      mexico_boost_MSE, mexico_boost_MASE, 
                                                      mexico_boost_MAPE))
mexico_boost_metrics <- mexico_boost_metrics %>%
  rename(RMSE = X161.500184751876, MAE = X125.155769694935, MSE = X26082.3096748899,
         MASE = X0.774957439752908, MAPE = Inf.)
print(mexico_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", mexico_boost_RMSE, "\n")
cat("MAE:", mexico_boost_MAE, "\n")
cat("MSE:", mexico_boost_MSE, "\n")
cat("MASE:", mexico_boost_MASE, "\n")
cat("MAPE:", mexico_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
mexico_boost_cv_folds <- createFolds(mexico_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
mexico_boost_cv_RMSE <- numeric(length = length(mexico_boost_cv_folds))
mexico_boost_cv_MAE <- numeric(length = length(mexico_boost_cv_folds))
mexico_boost_cv_MSE <- numeric(length = length(mexico_boost_cv_folds))
mexico_boost_cv_MASE <- numeric(length = length(mexico_boost_cv_folds))
mexico_boost_cv_MAPE <- numeric(length = length(mexico_boost_cv_folds))

#empty vector for metrics of each fold
mexico_boost_curr_metrics <- data.frame(
  mexico_curr_fold = integer(length(mexico_boost_cv_folds)),
  RMSE = mexico_boost_cv_RMSE,
  MAE = mexico_boost_cv_MAE,
  MSE = mexico_boost_cv_MSE,
  MASE = mexico_boost_cv_MASE,
  MAPE = mexico_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(mexico_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  mexico_fold_train_data <- mexico_train_data[-mexico_boost_cv_folds[[fold]], ]
  mexico_fold_validation_data <- mexico_train_data[mexico_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  mexico_fold_train_matrix <- xgb.DMatrix(data = as.matrix(mexico_fold_train_data[, -(1:2)]), 
                                        label = mexico_fold_train_data$owid_new_deaths)
  mexico_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(mexico_fold_validation_data[, -(1:2)]), 
                                             label = mexico_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  mexico_fold_xgb_model <- xgboost(params = mexico_boost_params, data = mexico_fold_train_matrix, 
                                 nrounds = mexico_boost_params[[5]])
  
  # Make predictions on the validation set
  mexico_fold_preds <- predict(mexico_fold_xgb_model, mexico_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  mexico_boost_curr_metrics$mexico_curr_fold[fold] <- fold
  mexico_boost_cv_accuracy_metrics <- forecast::accuracy(mexico_fold_preds, mexico_fold_validation_data$owid_new_deaths)
  mexico_boost_curr_metrics$RMSE[fold] <- sqrt(mean((mexico_fold_preds - mexico_fold_validation_data$owid_new_deaths)^2))
  mexico_boost_curr_metrics$MAE[fold] <- mean(abs(mexico_fold_preds - mexico_fold_validation_data$owid_new_deaths))
  mexico_boost_curr_metrics$MSE[fold] <- mean((mexico_fold_preds - mexico_fold_validation_data$owid_new_deaths)^2)
  mexico_boost_curr_metrics$MASE[fold] <- mexico_boost_cv_accuracy_metrics[,"MAE"] / mexico_boost_cv_accuracy_metrics[,"RMSE"]
  mexico_boost_curr_metrics$MAPE[fold] <- mexico_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(mexico_boost_curr_metrics)

############################################################################################################
## PERU ##

#to avoid confusion
peru_boost <- as.data.table(peru)

## Feature engineering: Create lagged features
peru_lags <- 3
peru_lagged_cols <- paste0("lag_", 1:peru_lags)

for (i in 1:peru_lags) {
  peru_col_name <- peru_lagged_cols[i]
  set(peru_boost, j = peru_col_name, value = shift(peru_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
peru_boost <- na.omit(peru_boost)

## Create training and testing data
peru_train_prop <- 0.9
peru_split_index <- floor(nrow(peru_boost) * peru_train_prop)
peru_train_data <- peru_boost[1:peru_split_index, ]
peru_test_data <- peru_boost[(peru_split_index + 1):nrow(peru_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
peru_train_matrix <- xgb.DMatrix(data = as.matrix(peru_train_data[, -(1:2)]), 
                                 label = peru_train_data$owid_new_deaths)
peru_test_matrix <- xgb.DMatrix(data = as.matrix(peru_test_data[, -(1:2)]), 
                                label = peru_test_data$owid_new_deaths)


## Define XGBoost parameters
peru_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
peru_xgb_model <- xgboost(params = peru_boost_params, data = peru_train_matrix, 
                          nrounds = peru_boost_params[[5]])

## Making predictions on the test set
peru_preds <- predict(peru_xgb_model, peru_test_matrix)

## Evaluating the model with the chosen metrics
peru_boost_RMSE <- sqrt(mean((peru_preds - peru_test_data$owid_new_deaths)^2))
peru_boost_MAE <- mean(abs(peru_preds - peru_test_data$owid_new_deaths))
peru_boost_MSE <- mean((peru_preds - peru_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
peru_accuracy_metrics <- forecast::accuracy(peru_preds, peru_test_data$owid_new_deaths)
peru_boost_MAPE <- peru_accuracy_metrics[,"MAPE"]
peru_boost_MASE <- peru_accuracy_metrics[,"MAE"] / peru_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
peru_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
peru_boost_metrics <- rbind(peru_boost_metrics, c(peru_boost_RMSE, peru_boost_MAE, 
                                                  peru_boost_MSE, peru_boost_MASE, 
                                                  peru_boost_MAPE))
peru_boost_metrics <- peru_boost_metrics %>%
  rename(RMSE = X14.0763170997536, MAE = X11.0081795779142, MSE = X198.142703092815,
         MASE = X0.782035492657867, MAPE = X18.7327980378166)
print(peru_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", peru_boost_RMSE, "\n")
cat("MAE:", peru_boost_MAE, "\n")
cat("MSE:", peru_boost_MSE, "\n")
cat("MASE:", peru_boost_MASE, "\n")
cat("MAPE:", peru_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
peru_boost_cv_folds <- createFolds(peru_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
peru_boost_cv_RMSE <- numeric(length = length(peru_boost_cv_folds))
peru_boost_cv_MAE <- numeric(length = length(peru_boost_cv_folds))
peru_boost_cv_MSE <- numeric(length = length(peru_boost_cv_folds))
peru_boost_cv_MASE <- numeric(length = length(peru_boost_cv_folds))
peru_boost_cv_MAPE <- numeric(length = length(peru_boost_cv_folds))

#empty vector for metrics of each fold
peru_boost_curr_metrics <- data.frame(
  peru_curr_fold = integer(length(peru_boost_cv_folds)),
  RMSE = peru_boost_cv_RMSE,
  MAE = peru_boost_cv_MAE,
  MSE = peru_boost_cv_MSE,
  MASE = peru_boost_cv_MASE,
  MAPE = peru_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(peru_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  peru_fold_train_data <- peru_train_data[-peru_boost_cv_folds[[fold]], ]
  peru_fold_validation_data <- peru_train_data[peru_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  peru_fold_train_matrix <- xgb.DMatrix(data = as.matrix(peru_fold_train_data[, -(1:2)]), 
                                        label = peru_fold_train_data$owid_new_deaths)
  peru_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(peru_fold_validation_data[, -(1:2)]), 
                                             label = peru_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  peru_fold_xgb_model <- xgboost(params = peru_boost_params, data = peru_fold_train_matrix, 
                                 nrounds = peru_boost_params[[5]])
  
  # Make predictions on the validation set
  peru_fold_preds <- predict(peru_fold_xgb_model, peru_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  peru_boost_curr_metrics$peru_curr_fold[fold] <- fold
  peru_boost_cv_accuracy_metrics <- forecast::accuracy(peru_fold_preds, peru_fold_validation_data$owid_new_deaths)
  peru_boost_curr_metrics$RMSE[fold] <- sqrt(mean((peru_fold_preds - peru_fold_validation_data$owid_new_deaths)^2))
  peru_boost_curr_metrics$MAE[fold] <- mean(abs(peru_fold_preds - peru_fold_validation_data$owid_new_deaths))
  peru_boost_curr_metrics$MSE[fold] <- mean((peru_fold_preds - peru_fold_validation_data$owid_new_deaths)^2)
  peru_boost_curr_metrics$MASE[fold] <- peru_boost_cv_accuracy_metrics[,"MAE"] / peru_boost_cv_accuracy_metrics[,"RMSE"]
  peru_boost_curr_metrics$MAPE[fold] <- peru_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(peru_boost_curr_metrics)

#########################################################################################################
## RUSSIA ##

#to avoid confusion
russia_boost <- as.data.table(russia)

## Feature engineering: Create lagged features
russia_lags <- 7
russia_lagged_cols <- paste0("lag_", 1:russia_lags)

for (i in 1:russia_lags) {
  russia_col_name <- russia_lagged_cols[i]
  set(russia_boost, j = russia_col_name, value = shift(russia_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
russia_boost <- na.omit(russia_boost)

## Create training and testing data
russia_train_prop <- 0.9
russia_split_index <- floor(nrow(russia_boost) * russia_train_prop)
russia_train_data <- russia_boost[1:russia_split_index, ]
russia_test_data <- russia_boost[(russia_split_index + 1):nrow(russia_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
russia_train_matrix <- xgb.DMatrix(data = as.matrix(russia_train_data[, -(1:2)]), 
                                   label = russia_train_data$owid_new_deaths)
russia_test_matrix <- xgb.DMatrix(data = as.matrix(russia_test_data[, -(1:2)]), 
                                  label = russia_test_data$owid_new_deaths)


## Define XGBoost parameters
russia_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
russia_xgb_model <- xgboost(params = russia_boost_params, data = russia_train_matrix, 
                            nrounds = russia_boost_params[[5]])

## Making predictions on the test set
russia_preds <- predict(russia_xgb_model, russia_test_matrix)

## Evaluating the model with the chosen metrics
russia_boost_RMSE <- sqrt(mean((russia_preds - russia_test_data$owid_new_deaths)^2))
russia_boost_MAE <- mean(abs(russia_preds - russia_test_data$owid_new_deaths))
russia_boost_MSE <- mean((russia_preds - russia_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
russia_accuracy_metrics <- forecast::accuracy(russia_preds, russia_test_data$owid_new_deaths)
russia_boost_MAPE <- russia_accuracy_metrics[,"MAPE"]
russia_boost_MASE <- russia_accuracy_metrics[,"MAE"] / russia_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
russia_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
russia_boost_metrics <- rbind(russia_boost_metrics, c(russia_boost_RMSE, russia_boost_MAE, 
                                                      russia_boost_MSE, russia_boost_MASE, 
                                                      russia_boost_MAPE))
russia_boost_metrics <- russia_boost_metrics %>%
  rename(RMSE = X88.5218529123345, MAE = X74.1649024269798, MSE = X7836.11844303298,
         MASE = X0.837814618503605, MAPE = X28.5100921032552)
print(russia_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", russia_boost_RMSE, "\n")
cat("MAE:", russia_boost_MAE, "\n")
cat("MSE:", russia_boost_MSE, "\n")
cat("MASE:", russia_boost_MASE, "\n")
cat("MAPE:", russia_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
russia_boost_cv_folds <- createFolds(russia_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
russia_boost_cv_RMSE <- numeric(length = length(russia_boost_cv_folds))
russia_boost_cv_MAE <- numeric(length = length(russia_boost_cv_folds))
russia_boost_cv_MSE <- numeric(length = length(russia_boost_cv_folds))
russia_boost_cv_MASE <- numeric(length = length(russia_boost_cv_folds))
russia_boost_cv_MAPE <- numeric(length = length(russia_boost_cv_folds))

#empty vector for metrics of each fold
russia_boost_curr_metrics <- data.frame(
  russia_curr_fold = integer(length(russia_boost_cv_folds)),
  RMSE = russia_boost_cv_RMSE,
  MAE = russia_boost_cv_MAE,
  MSE = russia_boost_cv_MSE,
  MASE = russia_boost_cv_MASE,
  MAPE = russia_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(russia_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  russia_fold_train_data <- russia_train_data[-russia_boost_cv_folds[[fold]], ]
  russia_fold_validation_data <- russia_train_data[russia_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  russia_fold_train_matrix <- xgb.DMatrix(data = as.matrix(russia_fold_train_data[, -(1:2)]), 
                                          label = russia_fold_train_data$owid_new_deaths)
  russia_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(russia_fold_validation_data[, -(1:2)]), 
                                               label = russia_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  russia_fold_xgb_model <- xgboost(params = russia_boost_params, data = russia_fold_train_matrix, 
                                   nrounds = russia_boost_params[[5]])
  
  # Make predictions on the validation set
  russia_fold_preds <- predict(russia_fold_xgb_model, russia_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  russia_boost_curr_metrics$russia_curr_fold[fold] <- fold
  russia_boost_cv_accuracy_metrics <- forecast::accuracy(russia_fold_preds, russia_fold_validation_data$owid_new_deaths)
  russia_boost_curr_metrics$RMSE[fold] <- sqrt(mean((russia_fold_preds - russia_fold_validation_data$owid_new_deaths)^2))
  russia_boost_curr_metrics$MAE[fold] <- mean(abs(russia_fold_preds - russia_fold_validation_data$owid_new_deaths))
  russia_boost_curr_metrics$MSE[fold] <- mean((russia_fold_preds - russia_fold_validation_data$owid_new_deaths)^2)
  russia_boost_curr_metrics$MASE[fold] <- russia_boost_cv_accuracy_metrics[,"MAE"] / russia_boost_cv_accuracy_metrics[,"RMSE"]
  russia_boost_curr_metrics$MAPE[fold] <- russia_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(russia_boost_curr_metrics)

##########################################################################################################
## SAUDI ##

#to avoid confusion
saudi_boost <- as.data.table(saudi)

## Feature engineering: Create lagged features
saudi_lags <- 3
saudi_lagged_cols <- paste0("lag_", 1:saudi_lags)

for (i in 1:saudi_lags) {
  saudi_col_name <- saudi_lagged_cols[i]
  set(saudi_boost, j = saudi_col_name, value = shift(saudi_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
saudi_boost <- na.omit(saudi_boost)

## Create training and testing data
saudi_train_prop <- 0.9
saudi_split_index <- floor(nrow(saudi_boost) * saudi_train_prop)
saudi_train_data <- saudi_boost[1:saudi_split_index, ]
saudi_test_data <- saudi_boost[(saudi_split_index + 1):nrow(saudi_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
saudi_train_matrix <- xgb.DMatrix(data = as.matrix(saudi_train_data[, -(1:2)]), 
                                 label = saudi_train_data$owid_new_deaths)
saudi_test_matrix <- xgb.DMatrix(data = as.matrix(saudi_test_data[, -(1:2)]), 
                                label = saudi_test_data$owid_new_deaths)


## Define XGBoost parameters
saudi_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
saudi_xgb_model <- xgboost(params = saudi_boost_params, data = saudi_train_matrix, 
                          nrounds = saudi_boost_params[[5]])

## Making predictions on the test set
saudi_preds <- predict(saudi_xgb_model, saudi_test_matrix)

## Evaluating the model with the chosen metrics
saudi_boost_RMSE <- sqrt(mean((saudi_preds - saudi_test_data$owid_new_deaths)^2))
saudi_boost_MAE <- mean(abs(saudi_preds - saudi_test_data$owid_new_deaths))
saudi_boost_MSE <- mean((saudi_preds - saudi_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
saudi_accuracy_metrics <- forecast::accuracy(saudi_preds, saudi_test_data$owid_new_deaths)
saudi_boost_MAPE <- saudi_accuracy_metrics[,"MAPE"]
saudi_boost_MASE <- saudi_accuracy_metrics[,"MAE"] / saudi_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
saudi_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
saudi_boost_metrics <- rbind(saudi_boost_metrics, c(saudi_boost_RMSE, saudi_boost_MAE, 
                                                    saudi_boost_MSE, saudi_boost_MASE, 
                                                    saudi_boost_MAPE))
saudi_boost_metrics <- saudi_boost_metrics %>%
  rename(RMSE = X5.03174230368827, MAE = X4.21138043837114, MSE = X25.3184306107261,
         MASE = X0.836962663068853, MAPE = X23.9729218633717)
print(saudi_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", saudi_boost_RMSE, "\n")
cat("MAE:", saudi_boost_MAE, "\n")
cat("MSE:", saudi_boost_MSE, "\n")
cat("MASE:", saudi_boost_MASE, "\n")
cat("MAPE:", saudi_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
saudi_boost_cv_folds <- createFolds(saudi_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
saudi_boost_cv_RMSE <- numeric(length = length(saudi_boost_cv_folds))
saudi_boost_cv_MAE <- numeric(length = length(saudi_boost_cv_folds))
saudi_boost_cv_MSE <- numeric(length = length(saudi_boost_cv_folds))
saudi_boost_cv_MASE <- numeric(length = length(saudi_boost_cv_folds))
saudi_boost_cv_MAPE <- numeric(length = length(saudi_boost_cv_folds))

#empty vector for metrics of each fold
saudi_boost_curr_metrics <- data.frame(
  saudi_curr_fold = integer(length(saudi_boost_cv_folds)),
  RMSE = saudi_boost_cv_RMSE,
  MAE = saudi_boost_cv_MAE,
  MSE = saudi_boost_cv_MSE,
  MASE = saudi_boost_cv_MASE,
  MAPE = saudi_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(saudi_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  saudi_fold_train_data <- saudi_train_data[-saudi_boost_cv_folds[[fold]], ]
  saudi_fold_validation_data <- saudi_train_data[saudi_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  saudi_fold_train_matrix <- xgb.DMatrix(data = as.matrix(saudi_fold_train_data[, -(1:2)]), 
                                        label = saudi_fold_train_data$owid_new_deaths)
  saudi_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(saudi_fold_validation_data[, -(1:2)]), 
                                             label = saudi_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  saudi_fold_xgb_model <- xgboost(params = saudi_boost_params, data = saudi_fold_train_matrix, 
                                 nrounds = saudi_boost_params[[5]])
  
  # Make predictions on the validation set
  saudi_fold_preds <- predict(saudi_fold_xgb_model, saudi_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  saudi_boost_curr_metrics$saudi_curr_fold[fold] <- fold
  saudi_boost_cv_accuracy_metrics <- forecast::accuracy(saudi_fold_preds, saudi_fold_validation_data$owid_new_deaths)
  saudi_boost_curr_metrics$RMSE[fold] <- sqrt(mean((saudi_fold_preds - saudi_fold_validation_data$owid_new_deaths)^2))
  saudi_boost_curr_metrics$MAE[fold] <- mean(abs(saudi_fold_preds - saudi_fold_validation_data$owid_new_deaths))
  saudi_boost_curr_metrics$MSE[fold] <- mean((saudi_fold_preds - saudi_fold_validation_data$owid_new_deaths)^2)
  saudi_boost_curr_metrics$MASE[fold] <- saudi_boost_cv_accuracy_metrics[,"MAE"] / saudi_boost_cv_accuracy_metrics[,"RMSE"]
  saudi_boost_curr_metrics$MAPE[fold] <- saudi_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(saudi_boost_curr_metrics)

##########################################################################################################
## TURKEY ##

#to avoid confusion
turkey_boost <- as.data.table(turkey)

## Feature engineering: Create lagged features
turkey_lags <- 3
turkey_lagged_cols <- paste0("lag_", 1:turkey_lags)

for (i in 1:turkey_lags) {
  turkey_col_name <- turkey_lagged_cols[i]
  set(turkey_boost, j = turkey_col_name, value = shift(turkey_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
turkey_boost <- na.omit(turkey_boost)

## Create training and testing data
turkey_train_prop <- 0.9
turkey_split_index <- floor(nrow(turkey_boost) * turkey_train_prop)
turkey_train_data <- turkey_boost[1:turkey_split_index, ]
turkey_test_data <- turkey_boost[(turkey_split_index + 1):nrow(turkey_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
turkey_train_matrix <- xgb.DMatrix(data = as.matrix(turkey_train_data[, -(1:2)]), 
                                  label = turkey_train_data$owid_new_deaths)
turkey_test_matrix <- xgb.DMatrix(data = as.matrix(turkey_test_data[, -(1:2)]), 
                                 label = turkey_test_data$owid_new_deaths)


## Define XGBoost parameters
turkey_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
turkey_xgb_model <- xgboost(params = turkey_boost_params, data = turkey_train_matrix, 
                           nrounds = turkey_boost_params[[5]])

## Making predictions on the test set
turkey_preds <- predict(turkey_xgb_model, turkey_test_matrix)

## Evaluating the model with the chosen metrics
turkey_boost_RMSE <- sqrt(mean((turkey_preds - turkey_test_data$owid_new_deaths)^2))
turkey_boost_MAE <- mean(abs(turkey_preds - turkey_test_data$owid_new_deaths))
turkey_boost_MSE <- mean((turkey_preds - turkey_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
turkey_accuracy_metrics <- forecast::accuracy(turkey_preds, turkey_test_data$owid_new_deaths)
turkey_boost_MAPE <- turkey_accuracy_metrics[,"MAPE"]
turkey_boost_MASE <- turkey_accuracy_metrics[,"MAE"] / turkey_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
turkey_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
turkey_boost_metrics <- rbind(turkey_boost_metrics, c(turkey_boost_RMSE, turkey_boost_MAE, 
                                                      turkey_boost_MSE, turkey_boost_MASE, 
                                                      turkey_boost_MAPE))
turkey_boost_metrics <- turkey_boost_metrics %>%
  rename(RMSE = X5.41536659474868, MAE = X4.22350244936736, MSE = X29.3261953555199,
         MASE = X0.779910718041308, MAPE = X6.45837580001237)
print(turkey_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", turkey_boost_RMSE, "\n")
cat("MAE:", turkey_boost_MAE, "\n")
cat("MSE:", turkey_boost_MSE, "\n")
cat("MASE:", turkey_boost_MASE, "\n")
cat("MAPE:", turkey_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
turkey_boost_cv_folds <- createFolds(turkey_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
turkey_boost_cv_RMSE <- numeric(length = length(turkey_boost_cv_folds))
turkey_boost_cv_MAE <- numeric(length = length(turkey_boost_cv_folds))
turkey_boost_cv_MSE <- numeric(length = length(turkey_boost_cv_folds))
turkey_boost_cv_MASE <- numeric(length = length(turkey_boost_cv_folds))
turkey_boost_cv_MAPE <- numeric(length = length(turkey_boost_cv_folds))

#empty vector for metrics of each fold
turkey_boost_curr_metrics <- data.frame(
  turkey_curr_fold = integer(length(turkey_boost_cv_folds)),
  RMSE = turkey_boost_cv_RMSE,
  MAE = turkey_boost_cv_MAE,
  MSE = turkey_boost_cv_MSE,
  MASE = turkey_boost_cv_MASE,
  MAPE = turkey_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(turkey_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  turkey_fold_train_data <- turkey_train_data[-turkey_boost_cv_folds[[fold]], ]
  turkey_fold_validation_data <- turkey_train_data[turkey_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  turkey_fold_train_matrix <- xgb.DMatrix(data = as.matrix(turkey_fold_train_data[, -(1:2)]), 
                                         label = turkey_fold_train_data$owid_new_deaths)
  turkey_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(turkey_fold_validation_data[, -(1:2)]), 
                                              label = turkey_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  turkey_fold_xgb_model <- xgboost(params = turkey_boost_params, data = turkey_fold_train_matrix, 
                                  nrounds = turkey_boost_params[[5]])
  
  # Make predictions on the validation set
  turkey_fold_preds <- predict(turkey_fold_xgb_model, turkey_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  turkey_boost_curr_metrics$turkey_curr_fold[fold] <- fold
  turkey_boost_cv_accuracy_metrics <- forecast::accuracy(turkey_fold_preds, turkey_fold_validation_data$owid_new_deaths)
  turkey_boost_curr_metrics$RMSE[fold] <- sqrt(mean((turkey_fold_preds - turkey_fold_validation_data$owid_new_deaths)^2))
  turkey_boost_curr_metrics$MAE[fold] <- mean(abs(turkey_fold_preds - turkey_fold_validation_data$owid_new_deaths))
  turkey_boost_curr_metrics$MSE[fold] <- mean((turkey_fold_preds - turkey_fold_validation_data$owid_new_deaths)^2)
  turkey_boost_curr_metrics$MASE[fold] <- turkey_boost_cv_accuracy_metrics[,"MAE"] / turkey_boost_cv_accuracy_metrics[,"RMSE"]
  turkey_boost_curr_metrics$MAPE[fold] <- turkey_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(turkey_boost_curr_metrics)

######################################################################################################
## US ##

#to avoid confusion
us_boost <- as.data.table(us)

## Feature engineering: Create lagged features
us_lags <- 7
us_lagged_cols <- paste0("lag_", 1:us_lags)

for (i in 1:us_lags) {
  us_col_name <- us_lagged_cols[i]
  set(us_boost, j = us_col_name, value = shift(us_boost$owid_new_deaths, i, type = "lag"))
}

## Remove rows with missing values introduced by lagging
us_boost <- na.omit(us_boost)

## Create training and testing data
us_train_prop <- 0.9
us_split_index <- floor(nrow(us_boost) * us_train_prop)
us_train_data <- us_boost[1:us_split_index, ]
us_test_data <- us_boost[(us_split_index + 1):nrow(us_boost), ]

## Convert data to xgb.DMatrix format for a chance of better performance
us_train_matrix <- xgb.DMatrix(data = as.matrix(us_train_data[, -(1:2)]), 
                                   label = us_train_data$owid_new_deaths)
us_test_matrix <- xgb.DMatrix(data = as.matrix(us_test_data[, -(1:2)]), 
                                  label = us_test_data$owid_new_deaths)


## Define XGBoost parameters
us_boost_params <- list(
  objective = "reg:squarederror", #our objective is regression
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 100
)

## Training the XGBoost model
us_xgb_model <- xgboost(params = us_boost_params, data = us_train_matrix, 
                            nrounds = us_boost_params[[5]])

## Making predictions on the test set
us_preds <- predict(us_xgb_model, us_test_matrix)

## Evaluating the model with the chosen metrics
us_boost_RMSE <- sqrt(mean((us_preds - us_test_data$owid_new_deaths)^2))
us_boost_MAE <- mean(abs(us_preds - us_test_data$owid_new_deaths))
us_boost_MSE <- mean((us_preds - us_test_data$owid_new_deaths)^2)
#to calculate MASE and MAPE, need to use the accuracy function from forecast package so:
us_accuracy_metrics <- forecast::accuracy(us_preds, us_test_data$owid_new_deaths)
us_boost_MAPE <- us_accuracy_metrics[,"MAPE"]
us_boost_MASE <- us_accuracy_metrics[,"MAE"] / us_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
us_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
us_boost_metrics <- rbind(us_boost_metrics, c(us_boost_RMSE, us_boost_MAE, 
                                                      us_boost_MSE, us_boost_MASE, 
                                                      us_boost_MAPE))
us_boost_metrics <- us_boost_metrics %>%
  rename(RMSE = X118.080676896666, MAE = X92.2981287638346, MSE = X13943.0462563748,
         MASE = X0.781653113697901, MAPE = X14.4958513177829)
print(us_boost_metrics)

## Printing evaluation metrics to the console to double check
cat("RMSE:", us_boost_RMSE, "\n")
cat("MAE:", us_boost_MAE, "\n")
cat("MSE:", us_boost_MSE, "\n")
cat("MASE:", us_boost_MASE, "\n")
cat("MAPE:", us_boost_MAPE, "\n")

## Going to use k-fold cross validation to assess xgboost model's performance on diff subsets of the
##training data

# Performing k-fold cross-validation
set.seed(1234)
#changing fold method for this model cause easier
us_boost_cv_folds <- createFolds(us_train_data$owid_new_deaths, k = 5, list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
us_boost_cv_RMSE <- numeric(length = length(us_boost_cv_folds))
us_boost_cv_MAE <- numeric(length = length(us_boost_cv_folds))
us_boost_cv_MSE <- numeric(length = length(us_boost_cv_folds))
us_boost_cv_MASE <- numeric(length = length(us_boost_cv_folds))
us_boost_cv_MAPE <- numeric(length = length(us_boost_cv_folds))

#empty dataframe for metrics of each fold
us_boost_curr_metrics <- data.frame(
  us_curr_fold = integer(length(us_boost_cv_folds)),
  RMSE = us_boost_cv_RMSE,
  MAE = us_boost_cv_MAE,
  MSE = us_boost_cv_MSE,
  MASE = us_boost_cv_MASE,
  MAPE = us_boost_cv_MAPE
)

# Fitting across folds
for (fold in seq_along(us_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  us_fold_train_data <- us_train_data[-us_boost_cv_folds[[fold]], ]
  us_fold_validation_data <- us_train_data[us_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  us_fold_train_matrix <- xgb.DMatrix(data = as.matrix(us_fold_train_data[, -(1:2)]), 
                                          label = us_fold_train_data$owid_new_deaths)
  us_fold_validation_matrix <- xgb.DMatrix(data = as.matrix(us_fold_validation_data[, -(1:2)]), 
                                               label = us_fold_validation_data$owid_new_deaths)
  
  # Train the XGBoost model on the current fold
  us_fold_xgb_model <- xgboost(params = us_boost_params, data = us_fold_train_matrix, 
                                   nrounds = us_boost_params[[5]])
  
  # Make predictions on the validation set
  us_fold_preds <- predict(us_fold_xgb_model, us_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  us_boost_curr_metrics$us_curr_fold[fold] <- fold
  us_boost_cv_accuracy_metrics <- forecast::accuracy(us_fold_preds, us_fold_validation_data$owid_new_deaths)
  us_boost_curr_metrics$RMSE[fold] <- sqrt(mean((us_fold_preds - us_fold_validation_data$owid_new_deaths)^2))
  us_boost_curr_metrics$MAE[fold] <- mean(abs(us_fold_preds - us_fold_validation_data$owid_new_deaths))
  us_boost_curr_metrics$MSE[fold] <- mean((us_fold_preds - us_fold_validation_data$owid_new_deaths)^2)
  us_boost_curr_metrics$MASE[fold] <- us_boost_cv_accuracy_metrics[,"MAE"] / us_boost_cv_accuracy_metrics[,"RMSE"]
  us_boost_curr_metrics$MAPE[fold] <- us_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(us_boost_curr_metrics)
