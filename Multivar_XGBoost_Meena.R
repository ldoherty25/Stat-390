### MULTIVARIATE XGBOOST MEENA ###

#calling the packages
library(tidyverse)
library(xgboost)
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

load("data/preprocessed/multivariate/preprocessed_covid_multi_imputed.rda")

preprocessed_covid_multi_imputed$date <- as.POSIXct(preprocessed_covid_multi_imputed$date)

## Need to change categorical vars into numeric for the purposes of the model
preprocessed_covid_multi_imputed$country_label <- as.numeric(as.factor(preprocessed_covid_multi_imputed$country))
preprocessed_covid_multi_imputed$weekday_label <- as.numeric(as.factor(preprocessed_covid_multi_imputed$weekday))
# One hot encoding to get rid of NA's induced by coercion
country_encoding <- model.matrix(~ country_label - 1, data = preprocessed_covid_multi_imputed)
weekday_encoding <- model.matrix(~ weekday_label - 1, data = preprocessed_covid_multi_imputed)
# Combining the onehot encoding
onehot_encoded_data <- cbind(preprocessed_covid_multi_imputed, country_encoding, weekday_encoding)

## Converting into xts object with encoded categorical vars, excluding the categorical and factorized vars
multivar_boost <- xts(onehot_encoded_data[, !(colnames(onehot_encoded_data) %in% c("country", "weekday", "country_label", "weekday_label"))],
                      order.by = onehot_encoded_data$date)
#for some reason the columns are classified as categorical, so forcing them to be numeric (except date column):
numeric_cols <- colnames(multivar_boost)[-1]  # Exclude the "date" column
multivar_boost <- apply(multivar_boost[, numeric_cols], 2, as.numeric)

#Splitting the training and testing data
multivar_split_index <- floor(0.8 * nrow(multivar_boost))
multivar_boost_train_data <- multivar_boost[1:multivar_split_index, ]
multivar_boost_test_data <- multivar_boost[(multivar_split_index + 1):nrow(multivar_boost), ]

multivar_boost_params <- list(
  objective = "reg:squarederror",
  booster = "gbtree",
  max_depth = 3,
  eta = 0.1,
  nrounds = 150
)

# print(class(multivar_boost_train_data))
# print(str(multivar_boost_train_data))


# Training the model on the training data

#have to remove the time index from multivar_boost_train_data
multivar_boost_train_data_matrix <- as.matrix(multivar_boost_train_data[, -1])

# Creating a DMatrix without using the xts structure
multivar_boost_train_matrix <- xgb.DMatrix(data = multivar_boost_train_data_matrix, 
                                           label = multivar_boost_train_data[, "owid_new_deaths"])

multivar_xgb_model <- xgboost(params = multivar_boost_params, data = multivar_boost_train_matrix, 
                              nrounds = multivar_boost_params[[5]])

# Making preds on the test set
multivar_boost_test_data_matrix <- as.matrix(multivar_boost_test_data[, -1])
multivar_boost_test_matrix <- xgb.DMatrix(data = multivar_boost_test_data_matrix, 
                                          label = multivar_boost_test_data[, "owid_new_deaths"])
multivar_boost_preds <- predict(multivar_xgb_model, multivar_boost_test_matrix)

## Evaluating the model with the chosen metrics
multivar_boost_RMSE <- sqrt(mean((multivar_boost_preds - multivar_boost_test_data[, "owid_new_deaths"])^2))
multivar_boost_MAE <- mean(abs(multivar_boost_preds - multivar_boost_test_data[, "owid_new_deaths"]))
multivar_boost_MSE <- mean((multivar_boost_preds - multivar_boost_test_data[, "owid_new_deaths"])^2)
multivar_boost_accuracy_metrics <- forecast::accuracy(multivar_boost_preds, multivar_boost_test_data[, "owid_new_deaths"])
multivar_boost_MAPE <- multivar_boost_accuracy_metrics[,"MAPE"]
multivar_boost_MASE <- multivar_boost_accuracy_metrics[,"MAE"] / multivar_boost_accuracy_metrics[,"RMSE"]

## Want to store the metrics in a dataframe:
#initializing the datarame:
multivar_boost_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)
#binding the metrics calculated above:
multivar_boost_metrics <- rbind(multivar_boost_metrics, c(multivar_boost_RMSE, multivar_boost_MAE, 
                                              multivar_boost_MSE, multivar_boost_MASE, 
                                              multivar_boost_MAPE))
print(multivar_boost_metrics)

# Let's use cross-validation folds

set.seed(1234)

multivar_boost_cv_folds <- createFolds(multivar_boost_train_data[, "owid_new_deaths"], k = 10, 
                                       list = TRUE, returnTrain = FALSE)

# Initializing metrics for each fold:
multivar_boost_cv_RMSE <- numeric(length = length(multivar_boost_cv_folds))
multivar_boost_cv_MAE <- numeric(length = length(multivar_boost_cv_folds))
multivar_boost_cv_MSE <- numeric(length = length(multivar_boost_cv_folds))
multivar_boost_cv_MASE <- numeric(length = length(multivar_boost_cv_folds))
multivar_boost_cv_MAPE <- numeric(length = length(multivar_boost_cv_folds))

#empty dataframe for metrics of each fold
multivar_boost_curr_metrics <- data.frame(
  multivar_boost_curr_fold = integer(length(multivar_boost_cv_folds)),
  RMSE = multivar_boost_cv_RMSE,
  MAE = multivar_boost_cv_MAE,
  MSE = multivar_boost_cv_MSE,
  MASE = multivar_boost_cv_MASE,
  MAPE = multivar_boost_cv_MAPE
)

for (fold in seq_along(multivar_boost_cv_folds)) {
  #Splitting data into training and validation sets based on the current fold
  multivar_boost_fold_train_data <- multivar_boost_train_data[-multivar_boost_cv_folds[[fold]], ]
  multivar_boost_fold_validation_data <- multivar_boost_train_data[multivar_boost_cv_folds[[fold]], ]
  
  # Convert data to xgb.DMatrix format
  multivar_boost_fold_train_matrix <- xgb.DMatrix(data = multivar_boost_fold_train_data[, -(1:2)], 
                                      label = multivar_boost_fold_train_data[, "owid_new_deaths"])
  multivar_boost_fold_validation_matrix <- xgb.DMatrix(data = multivar_boost_fold_validation_data[, -(1:2)], 
                                           label = multivar_boost_fold_validation_data[, "owid_new_deaths"])
  
  # Train the XGBoost model on the current fold
  multivar_boost_fold_xgb_model <- xgboost(params = multivar_boost_params, data = multivar_boost_fold_train_matrix, 
                               nrounds = multivar_boost_params[[5]])
  
  # Make predictions on the validation set
  multivar_boost_fold_preds <- predict(multivar_boost_fold_xgb_model, multivar_boost_fold_validation_matrix)
  
  # Evaluate the model on the current fold
  multivar_boost_curr_metrics$multivar_boost_curr_fold[fold] <- fold
  multivar_boost_cv_accuracy_metrics <- forecast::accuracy(multivar_boost_fold_preds, multivar_boost_fold_validation_data[, "owid_new_deaths"])
  multivar_boost_curr_metrics$RMSE[fold] <- sqrt(mean((multivar_boost_fold_preds - multivar_boost_fold_validation_data[, "owid_new_deaths"])^2))
  multivar_boost_curr_metrics$MAE[fold] <- mean(abs(multivar_boost_fold_preds - multivar_boost_fold_validation_data[, "owid_new_deaths"]))
  multivar_boost_curr_metrics$MSE[fold] <- mean((multivar_boost_fold_preds - multivar_boost_fold_validation_data[, "owid_new_deaths"])^2)
  multivar_boost_curr_metrics$MASE[fold] <- multivar_boost_cv_accuracy_metrics[,"MAE"] / multivar_boost_cv_accuracy_metrics[,"RMSE"]
  multivar_boost_curr_metrics$MAPE[fold] <- multivar_boost_cv_accuracy_metrics[,"MAPE"]
  
}

print(multivar_boost_curr_metrics)


