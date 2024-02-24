## Univariate Prophet

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
library(e1071)
library(prophet)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)



# bolivia ----


## load data ----
load("data/preprocessed/univariate/not_split/bolivia.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
bolivia_total_days <- nrow(bolivia)
bolivia_train_days <- ceiling(0.9 * bolivia_total_days)
bolivia_test_days <- ceiling((bolivia_total_days - bolivia_train_days))

# creating folds
bolivia_folds <- time_series_cv(
  bolivia,
  date_var = date,
  initial = bolivia_train_days,
  assess = bolivia_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
bolivia_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying prophet model ----

# creating metrics vector
bolivia_rmse_results <- numeric(length(bolivia_folds$splits))
bolivia_mae_results <- numeric(length(bolivia_folds$splits))
bolivia_mse_results <- numeric(length(bolivia_folds$splits))
bolivia_mape_results <- numeric(length(bolivia_folds$splits))
bolivia_mase_results <- numeric(length(bolivia_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(bolivia_folds$splits)) {
  fold <- bolivia_folds$splits[[i]]
  bolivia_train_data <- fold$data[fold$in_id, ]
  bolivia_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  bolivia_df_univar_prophet <- data.frame(ds = bolivia_train_data$date, y = bolivia_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  bolivia_univar_prophet_model<- prophet(bolivia_df_univar_prophet)
  
  # making future dataframe for forecasting
  bolivia_univar_prophet_future <- make_future_dataframe(bolivia_univar_prophet_model, periods = nrow(bolivia_test_data))
  
  # forecasting with Prophet
  bolivia_univar_prophet_forecast <- predict(bolivia_univar_prophet_model, bolivia_univar_prophet_future)
  
  # extracting forecasted values
  bolivia_forecast_values <- bolivia_univar_prophet_forecast %>% filter(ds %in% bolivia_test_data$date)
  
  # calculating evaluation metrics
  bolivia_errors <- bolivia_forecast_values$yhat - bolivia_test_data$owid_new_deaths
  bolivia_rmse_results[i] <- sqrt(mean(bolivia_errors^2))
  bolivia_mae_results[i] <- mean(abs(bolivia_errors))
  bolivia_mse_results[i] <- mean(bolivia_errors^2)
  bolivia_mape_results[i] <- mean(abs(bolivia_errors / bolivia_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  bolivia_mean_train_diff <- mean(abs(diff(bolivia_train_data$owid_new_deaths)))
  bolivia_mase_results[i] <- mean(abs(bolivia_errors)) / bolivia_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(bolivia_rmse_results)))
print(paste("MAE:", mean(bolivia_mae_results)))
print(paste("MSE:", mean(bolivia_mse_results)))
print(paste("MAPE:", mean(bolivia_mape_results)))
print(paste("MASE:", mean(bolivia_mase_results)))


## producing a plot ----

# combining training and test data for plotting
bolivia_all_dates <- c(bolivia_train_data$date, bolivia_test_data$date)
bolivia_all_values <- c(bolivia_train_data$owid_new_deaths, bolivia_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(bolivia_all_dates, bolivia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(bolivia_test_data$date, bolivia_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
bolivia_fitted_forecast <- predict(bolivia_univar_prophet_model, bolivia_df_univar_prophet)

# plotting the actual and forecasted values
plot(bolivia_all_dates, bolivia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(bolivia_test_data$date, bolivia_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(bolivia_df_univar_prophet$ds, bolivia_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# brazil ----


## load data ----
load("data/preprocessed/univariate/not_split/brazil.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
brazil_total_days <- nrow(brazil)
brazil_train_days <- ceiling(0.9 * brazil_total_days)
brazil_test_days <- ceiling((brazil_total_days - brazil_train_days))

# creating folds
brazil_folds <- time_series_cv(
  brazil,
  date_var = date,
  initial = brazil_train_days,
  assess = brazil_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
brazil_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying prophet model ----

# creating metrics vector
brazil_rmse_results <- numeric(length(brazil_folds$splits))
brazil_mae_results <- numeric(length(brazil_folds$splits))
brazil_mse_results <- numeric(length(brazil_folds$splits))
brazil_mape_results <- numeric(length(brazil_folds$splits))
brazil_mase_results <- numeric(length(brazil_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(brazil_folds$splits)) {
  fold <- brazil_folds$splits[[i]]
  brazil_train_data <- fold$data[fold$in_id, ]
  brazil_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  brazil_df_univar_prophet <- data.frame(ds = brazil_train_data$date, y = brazil_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  brazil_univar_prophet_model<- prophet(brazil_df_univar_prophet)
  
  # making future dataframe for forecasting
  brazil_univar_prophet_future <- make_future_dataframe(brazil_univar_prophet_model, periods = nrow(brazil_test_data))
  
  # forecasting with Prophet
  brazil_univar_prophet_forecast <- predict(brazil_univar_prophet_model, brazil_univar_prophet_future)
  
  # extracting forecasted values
  brazil_forecast_values <- brazil_univar_prophet_forecast %>% filter(ds %in% brazil_test_data$date)
  
  # calculating evaluation metrics
  brazil_errors <- brazil_forecast_values$yhat - brazil_test_data$owid_new_deaths
  brazil_rmse_results[i] <- sqrt(mean(brazil_errors^2))
  brazil_mae_results[i] <- mean(abs(brazil_errors))
  brazil_mse_results[i] <- mean(brazil_errors^2)
  brazil_mape_results[i] <- mean(abs(brazil_errors / brazil_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  brazil_mean_train_diff <- mean(abs(diff(brazil_train_data$owid_new_deaths)))
  brazil_mase_results[i] <- mean(abs(brazil_errors)) / brazil_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(brazil_rmse_results)))
print(paste("MAE:", mean(brazil_mae_results)))
print(paste("MSE:", mean(brazil_mse_results)))
print(paste("MAPE:", mean(brazil_mape_results)))
print(paste("MASE:", mean(brazil_mase_results)))


## producing a plot ----

# combining training and test data for plotting
brazil_all_dates <- c(brazil_train_data$date, brazil_test_data$date)
brazil_all_values <- c(brazil_train_data$owid_new_deaths, brazil_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(brazil_all_dates, brazil_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(brazil_test_data$date, brazil_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
brazil_fitted_forecast <- predict(brazil_univar_prophet_model, brazil_df_univar_prophet)

# plotting the actual and forecasted values
plot(brazil_all_dates, brazil_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(brazil_test_data$date, brazil_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(brazil_df_univar_prophet$ds, brazil_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)