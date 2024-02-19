## UNIVARIATE PROPHET

#calling the packages
library(tidyverse)
library(prophet)
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

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

#setting the seed
set.seed(1234)

#loading the univariate datasets
load("data/preprocessed/univariate/not_split/bolivia.rda")
load("data/preprocessed/univariate/not_split/brazil.rda")
load("data/preprocessed/univariate/not_split/colombia.rda")
load("data/preprocessed/univariate/not_split/iran.rda")
load("data/preprocessed/univariate/not_split/mexico.rda")
load("data/preprocessed/univariate/not_split/peru.rda")
load("data/preprocessed/univariate/not_split/russia.rda")
load("data/preprocessed/univariate/not_split/saudi.rda")
load("data/preprocessed/univariate/not_split/turkey.rda")
load("data/preprocessed/univariate/not_split/us.rda")

## Modeling for Bolivia ---
#We have to rename the variables for Prophet to comprehend the vars
bolivia <- bolivia%>%
  rename(ds = date, y = owid_new_deaths)
#Splitting the data into training and testing sets
bolivia_train_size <- nrow(bolivia)
bolivia_train_set <- ceiling(0.9 * train_size)
bolivia_test_set <- ceiling((train_size - train_set))

#Creating time series folds in training data

bolivia_folds <- time_series_cv(
  bolivia,
  date_var = ds,
  initial = train_set,
  assess = test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
bolivia_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

#using prophet to fit the model to the training dataset

fitting_process <- function(folds) {
  for (i in seq_along(folds$splits)) {
    fold <- folds$splits[[i]]
    train_data <- fold$data[fold$in_id, ]
    test_data <- fold$data[fold$out_id, ]
  
    #initializing a dataset for all the metrics we will calculate
    all_metrics <- data.frame()
  
    # Fit the model on the training data for the current fold
    model <- prophet(train_data,
                     yearly.seasonality = FALSE,
                     weekly.seasonality = "auto",
                     daily.seasonality = FALSE,
                     fit = TRUE,
                     mcmc.samples = 0)
  
    # Make predictions on the test data for the current fold
    future <- make_future_dataframe(model, periods = nrow(test_data))
    forecast <- predict(model, future)
    
    # enforcing non-negativity on forecasted values
    #future$yhat <- pmax(future$yhat, 0)
    
    # Evaluate the model and store metrics
    metrics <- data.frame(
      #Errors = forecast$yhat - test_data$y,
      #Mean_train_diff = mean(abs(diff(train_data$y))),
      #MASE = mean(abs(Errors)) / Mean_train_diff,
      RMSE = sqrt(mean((forecast$yhat - test_data$y)^2)),
      MAE = mean(abs(forecast$yhat - test_data$y)),
      MSE = mean((forecast$yhat - test_data$y)^2),
      MAPE = mean(abs((test_data$y - forecast$yhat) / test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
  }
  return(all_metrics)
}

bolivia_metrics <- fitting_process(bolivia_folds)

#error I'm getting:
#1: In forecast$yhat - test_data$y :longer object length is not a multiple of shorter object length

#Display average metrics across all folds
cat("Average RMSE:", mean(bolivia_metrics$RMSE), "\n")
cat("Average MAE:", mean(bolivia_metrics$MAE), "\n")
cat("Average MSE:", mean(bolivia_metrics$MSE), "\n")
cat("Average MAPE:", mean(bolivia_metrics$MAPE), "\n")

# Fit the final model on the entire training dataset
bolivia_final_prophet <- prophet(bolivia_train_set,
                                 yearly.seasonality = FALSE,
                                 weekly.seasonality = "auto",
                                 daily.seasonality = FALSE,
                                 fit = TRUE,
                                 mcmc.samples = 0)
#getting error: Error in as.environment(where) : invalid 'pos' argument

# Make predictions on the test dataset
bolivia_future <- make_future_dataframe(bolivia_final_model, periods = nrow(bolivia_test_data))
bolivia_forecast <- predict(bolivia_final_model, bolivia_future)

# Evaluate the model on the test dataset
bolivia_test_metrics <- data.frame(
  RMSE = sqrt(mean((bolivia_forecast$yhat[-(1:train_size)] - bolivia_test_data$y)^2)),
  MAE = mean(abs(bolivia_forecast$yhat[-(1:bolivia_train_size)] - bolivia_test_data$y)),
  MSE = mean((bolivia_forecast$yhat[-(1:bolivia_train_size)] - bolivia_test_data$y)^2),
  MAPE = mean(abs((bolivia_test_data$y - bolivia_forecast$yhat[-(1:bolivia_train_size)]) / bolivia_test_data$y)) * 100
)

# Display metrics on the test dataset
cat("\nTest RMSE:", bolivia_test_metrics$RMSE, "\n")
cat("Test MAE:", bolivia_test_metrics$MAE, "\n")
cat("Test MSE:", bolivia_test_metrics$MSE, "\n")
cat("Test MAPE:", bolivia_test_metrics$MAPE, "\n")

