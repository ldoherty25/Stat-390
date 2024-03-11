## ARIMA and Auto-ARIMA



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


## applying ARIMA model ----

# creating metrics vector
bolivia_rmse_results <- numeric(length(bolivia_folds$splits))
bolivia_mae_results <- numeric(length(bolivia_folds$splits))
bolivia_mse_results <- numeric(length(bolivia_folds$splits))
bolivia_mase_results <- numeric(length(bolivia_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(bolivia_folds$splits)) {
  fold <- bolivia_folds$splits[[i]]
  bolivia_train_data <- fold$data[fold$in_id, ]
  bolivia_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  bolivia_arima_model <- arima(bolivia_train_data$owid_new_deaths,
                              order = c(2, 1, 0))
  
  # forecasting with ARIMA
  bolivia_forecast_values <- forecast(bolivia_arima_model, h = nrow(bolivia_test_data))
  
  # enforcing non-negativity on forecasted values
  bolivia_forecast_values$mean <- pmax(bolivia_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  bolivia_errors <- bolivia_forecast_values$mean - bolivia_test_data$owid_new_deaths
  bolivia_rmse_results[i] <- sqrt(mean(bolivia_errors^2))
  bolivia_mae_results[i] <- mean(abs(bolivia_errors))
  bolivia_mse_results[i] <- mean(bolivia_errors^2)
  
  # calculating MASE
  bolivia_mean_train_diff <- mean(abs(diff(bolivia_train_data$owid_new_deaths)))
  bolivia_mase_results[i] <- mean(abs(bolivia_errors)) / bolivia_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(bolivia_rmse_results)))
print(paste("MAE:", mean(bolivia_mae_results)))
print(paste("MSE:", mean(bolivia_mse_results)))
print(paste("MASE:", mean(bolivia_mase_results)))

# retrieving the fitted values for the training set
bolivia_fitted_values <- fitted(bolivia_arima_model)

# enforcing non-negativity on fitted values
bolivia_fitted_values <- pmax(bolivia_fitted_values, 0)

# combining training and test data for plotting
bolivia_all_dates <- c(bolivia_train_data$date, bolivia_test_data$date)
bolivia_all_values <- c(bolivia_train_data$owid_new_deaths, bolivia_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(bolivia_all_dates, bolivia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(bolivia_test_data$date, bolivia_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(bolivia_train_data$date, bolivia_fitted_values, col = "red", lty = 1, lwd = 2)

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


## applying ARIMA model ----

# creating metrics vector
brazil_rmse_results <- numeric(length(brazil_folds$splits))
brazil_mae_results <- numeric(length(brazil_folds$splits))
brazil_mse_results <- numeric(length(brazil_folds$splits))
brazil_mase_results <- numeric(length(brazil_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(brazil_folds$splits)) {
  fold <- brazil_folds$splits[[i]]
  brazil_train_data <- fold$data[fold$in_id, ]
  brazil_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  brazil_arima_model <- arima(brazil_train_data$owid_new_deaths,
                              order = c(2, 1, 2))
  
  # forecasting with ARIMA
  brazil_forecast_values <- forecast(brazil_arima_model, h = nrow(brazil_test_data))
  
  # enforcing non-negativity on forecasted values
  brazil_forecast_values$mean <- pmax(brazil_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  brazil_errors <- brazil_forecast_values$mean - brazil_test_data$owid_new_deaths
  brazil_rmse_results[i] <- sqrt(mean(brazil_errors^2))
  brazil_mae_results[i] <- mean(abs(brazil_errors))
  brazil_mse_results[i] <- mean(brazil_errors^2)
  
  # calculating MASE
  brazil_mean_train_diff <- mean(abs(diff(brazil_train_data$owid_new_deaths)))
  brazil_mase_results[i] <- mean(abs(brazil_errors)) / brazil_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(brazil_rmse_results)))
print(paste("MAE:", mean(brazil_mae_results)))
print(paste("MSE:", mean(brazil_mse_results)))
print(paste("MASE:", mean(brazil_mase_results)))

# retrieving the fitted values for the training set
brazil_fitted_values <- fitted(brazil_arima_model)

# enforcing non-negativity on fitted values
brazil_fitted_values <- pmax(brazil_fitted_values, 0)

# combining training and test data for plotting
brazil_all_dates <- c(brazil_train_data$date, brazil_test_data$date)
brazil_all_values <- c(brazil_train_data$owid_new_deaths, brazil_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(brazil_all_dates, brazil_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(brazil_test_data$date, brazil_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(brazil_train_data$date, brazil_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)


## applying SARIMA model ----

# creating metrics vector
brazil_rmse_results_sarima <- numeric(length(brazil_folds$splits))
brazil_mae_results_sarima <- numeric(length(brazil_folds$splits))
brazil_mse_results_sarima <- numeric(length(brazil_folds$splits))
brazil_mase_results_sarima <- numeric(length(brazil_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(brazil_folds$splits)) {
  fold <- brazil_folds$splits[[i]]
  brazil_train_data <- fold$data[fold$in_id, ]
  brazil_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  brazil_sarima_model <- arima(brazil_train_data$owid_new_deaths,
                              order = c(0, 2, 2),
                              seasonal = list(order = c(1, 0, 0), period = 7))
  
  # forecasting with ARIMA
  brazil_forecast_values_sarima <- forecast(brazil_sarima_model, h = nrow(brazil_test_data))
  
  # enforcing non-negativity on forecasted values
  brazil_forecast_values_sarima$mean <- pmax(brazil_forecast_values_sarima$mean, 0)
  
  # calculating evaluation metrics
  brazil_errors_sarima <- brazil_forecast_values_sarima$mean - brazil_test_data$owid_new_deaths
  brazil_rmse_results_sarima[i] <- sqrt(mean(brazil_errors_sarima^2))
  brazil_mae_results_sarima[i] <- mean(abs(brazil_errors_sarima))
  brazil_mse_results_sarima[i] <- mean(brazil_errors_sarima^2)
  
  # calculating MASE
  brazil_mean_train_diff_sarima <- mean(abs(diff(brazil_train_data$owid_new_deaths)))
  brazil_mase_results_sarima[i] <- mean(abs(brazil_errors_sarima)) / brazil_mean_train_diff_sarima
}

# printing metrics
print(paste("RMSE:", mean(brazil_rmse_results_sarima)))
print(paste("MAE:", mean(brazil_mae_results_sarima)))
print(paste("MSE:", mean(brazil_mse_results_sarima)))
print(paste("MASE:", mean(brazil_mase_results_sarima)))

# retrieving the fitted values for the training set
brazil_fitted_values_sarima <- fitted(brazil_sarima_model)

# enforcing non-negativity on fitted values
brazil_fitted_values_sarima <- pmax(brazil_fitted_values_sarima, 0)

# combining training and test data for plotting
brazil_all_dates <- c(brazil_train_data$date, brazil_test_data$date)
brazil_all_values <- c(brazil_train_data$owid_new_deaths, brazil_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(brazil_all_dates, brazil_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(brazil_test_data$date, brazil_forecast_values_sarima$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(brazil_train_data$date, brazil_fitted_values_sarima, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# colombia ----


## load data ----
load("data/preprocessed/univariate/not_split/colombia.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
colombia_total_days <- nrow(colombia)
colombia_train_days <- ceiling(0.9 * colombia_total_days)
colombia_test_days <- ceiling((colombia_total_days - colombia_train_days))

# creating folds
colombia_folds <- time_series_cv(
  colombia,
  date_var = date,
  initial = colombia_train_days,
  assess = colombia_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
colombia_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
colombia_rmse_results <- numeric(length(colombia_folds$splits))
colombia_mae_results <- numeric(length(colombia_folds$splits))
colombia_mse_results <- numeric(length(colombia_folds$splits))
colombia_mase_results <- numeric(length(colombia_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(colombia_folds$splits)) {
  fold <- colombia_folds$splits[[i]]
  colombia_train_data <- fold$data[fold$in_id, ]
  colombia_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  colombia_arima_model <- arima(colombia_train_data$owid_new_deaths,
                                order = c(0, 2, 1))
  # forecasting with ARIMA
  colombia_forecast_values <- forecast(colombia_arima_model, h = nrow(colombia_test_data))
  
  # enforcing non-negativity on forecasted values
  colombia_forecast_values$mean <- pmax(colombia_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  colombia_errors <- colombia_forecast_values$mean - colombia_test_data$owid_new_deaths
  colombia_rmse_results[i] <- sqrt(mean(colombia_errors^2))
  colombia_mae_results[i] <- mean(abs(colombia_errors))
  colombia_mse_results[i] <- mean(colombia_errors^2)
  
  # calculating MASE
  colombia_mean_train_diff <- mean(abs(diff(colombia_train_data$owid_new_deaths)))
  colombia_mase_results[i] <- mean(abs(colombia_errors)) / colombia_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(colombia_rmse_results)))
print(paste("MAE:", mean(colombia_mae_results)))
print(paste("MSE:", mean(colombia_mse_results)))
print(paste("MASE:", mean(colombia_mase_results)))

# retrieving the fitted values for the training set
colombia_fitted_values <- fitted(colombia_arima_model)

# enforcing non-negativity on fitted values
colombia_fitted_values <- pmax(colombia_fitted_values, 0)

# combining training and test data for plotting
colombia_all_dates <- c(colombia_train_data$date, colombia_test_data$date)
colombia_all_values <- c(colombia_train_data$owid_new_deaths, colombia_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(colombia_all_dates, colombia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(colombia_test_data$date, colombia_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(colombia_train_data$date, colombia_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# iran ----

## load data ----
load("data/preprocessed/univariate/not_split/iran.rda")

## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
iran_total_days <- nrow(iran)
iran_train_days <- ceiling(0.9 * iran_total_days)
iran_test_days <- ceiling((iran_total_days - iran_train_days))

# creating folds
iran_folds <- time_series_cv(
  iran,
  date_var = date,
  initial = iran_train_days,
  assess = iran_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
iran_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
iran_rmse_results <- numeric(length(iran_folds$splits))
iran_mae_results <- numeric(length(iran_folds$splits))
iran_mse_results <- numeric(length(iran_folds$splits))
iran_mase_results <- numeric(length(iran_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(iran_folds$splits)) {
  fold <- iran_folds$splits[[i]]
  iran_train_data <- fold$data[fold$in_id, ]
  iran_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  iran_arima_model <- arima(iran_train_data$owid_new_deaths,
                            order = c(2, 2, 0))
  
  # forecasting with ARIMA
  iran_forecast_values <- forecast(iran_arima_model, h = nrow(iran_test_data))
  
  # enforcing non-negativity on forecasted values
  iran_forecast_values$mean <- pmax(iran_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  iran_errors <- iran_forecast_values$mean - iran_test_data$owid_new_deaths
  iran_rmse_results[i] <- sqrt(mean(iran_errors^2))
  iran_mae_results[i] <- mean(abs(iran_errors))
  iran_mse_results[i] <- mean(iran_errors^2)
  
  # calculating MASE
  iran_mean_train_diff <- mean(abs(diff(iran_train_data$owid_new_deaths)))
  iran_mase_results[i] <- mean(abs(iran_errors)) / iran_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(iran_rmse_results)))
print(paste("MAE:", mean(iran_mae_results)))
print(paste("MSE:", mean(iran_mse_results)))
print(paste("MASE:", mean(iran_mase_results)))

# retrieving the fitted values for the training set
iran_fitted_values <- fitted(iran_arima_model)

# enforcing non-negativity on fitted values
iran_fitted_values <- pmax(iran_fitted_values, 0)

# combining training and test data for plotting
iran_all_dates <- c(iran_train_data$date, iran_test_data$date)
iran_all_values <- c(iran_train_data$owid_new_deaths, iran_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(iran_all_dates, iran_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(iran_test_data$date, iran_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(iran_train_data$date, iran_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# mexico ----

## load data ----
load("data/preprocessed/univariate/not_split/mexico.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
mexico_total_days <- nrow(mexico)
mexico_train_days <- ceiling(0.9 * mexico_total_days)
mexico_test_days <- ceiling((mexico_total_days - mexico_train_days))

# creating folds
mexico_folds <- time_series_cv(
  mexico,
  date_var = date,
  initial = mexico_train_days,
  assess = mexico_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
mexico_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
mexico_rmse_results <- numeric(length(mexico_folds$splits))
mexico_mae_results <- numeric(length(mexico_folds$splits))
mexico_mse_results <- numeric(length(mexico_folds$splits))
mexico_mase_results <- numeric(length(mexico_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(mexico_folds$splits)) {
  fold <- mexico_folds$splits[[i]]
  mexico_train_data <- fold$data[fold$in_id, ]
  mexico_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  mexico_arima_model <- arima(mexico_train_data$owid_new_deaths,
                              order = c(0, 0, 0))
  
  # forecasting with ARIMA
  mexico_forecast_values <- forecast(mexico_arima_model, h = nrow(mexico_test_data))
  
  # enforcing non-negativity on forecasted values
  mexico_forecast_values$mean <- pmax(mexico_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  mexico_errors <- mexico_forecast_values$mean - mexico_test_data$owid_new_deaths
  mexico_rmse_results[i] <- sqrt(mean(mexico_errors^2))
  mexico_mae_results[i] <- mean(abs(mexico_errors))
  mexico_mse_results[i] <- mean(mexico_errors^2)
  
  # calculating MASE
  mexico_mean_train_diff <- mean(abs(diff(mexico_train_data$owid_new_deaths)))
  mexico_mase_results[i] <- mean(abs(mexico_errors)) / mexico_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(mexico_rmse_results)))
print(paste("MAE:", mean(mexico_mae_results)))
print(paste("MSE:", mean(mexico_mse_results)))
print(paste("MASE:", mean(mexico_mase_results)))

# retrieving the fitted values for the training set
mexico_fitted_values <- fitted(mexico_arima_model)

# enforcing non-negativity on fitted values
mexico_fitted_values <- pmax(mexico_fitted_values, 0)

# combining training and test data for plotting
mexico_all_dates <- c(mexico_train_data$date, mexico_test_data$date)
mexico_all_values <- c(mexico_train_data$owid_new_deaths, mexico_test_data$owid_new_deaths)

## producing a plot ----

# plotting actual values for both training and test data
plot(mexico_all_dates, mexico_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(mexico_test_data$date, mexico_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(mexico_train_data$date, mexico_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)


## applying SARIMA model ----

# creating metrics vector
mexico_rmse_results_sarima <- numeric(length(mexico_folds$splits))
mexico_mae_results_sarima <- numeric(length(mexico_folds$splits))
mexico_mse_results_sarima <- numeric(length(mexico_folds$splits))
mexico_mase_results_sarima <- numeric(length(mexico_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(mexico_folds$splits)) {
  fold <- mexico_folds$splits[[i]]
  mexico_train_data <- fold$data[fold$in_id, ]
  mexico_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  mexico_sarima_model <- arima(mexico_train_data$owid_new_deaths,
                              order = c(0, 0, 0),
                              seasonal = list(order = c(1, 1, 0), period = 7))
  
  # forecasting with ARIMA
  mexico_forecast_values_sarima <- forecast(mexico_sarima_model, h = nrow(mexico_test_data))
  
  # enforcing non-negativity on forecasted values
  mexico_forecast_values_sarima$mean <- pmax(mexico_forecast_values_sarima$mean, 0)
  
  # calculating evaluation metrics
  mexico_errors_sarima <- mexico_forecast_values_sarima$mean - mexico_test_data$owid_new_deaths
  mexico_rmse_results_sarima[i] <- sqrt(mean(mexico_errors_sarima^2))
  mexico_mae_results_sarima[i] <- mean(abs(mexico_errors_sarima))
  mexico_mse_results_sarima[i] <- mean(mexico_errors_sarima^2)
  
  # calculating MASE
  mexico_mean_train_diff_sarima <- mean(abs(diff(mexico_train_data$owid_new_deaths)))
  mexico_mase_results_sarima[i] <- mean(abs(mexico_errors_sarima)) / mexico_mean_train_diff_sarima
}

# printing metrics
print(paste("RMSE:", mean(mexico_rmse_results_sarima)))
print(paste("MAE:", mean(mexico_mae_results_sarima)))
print(paste("MSE:", mean(mexico_mse_results_sarima)))
print(paste("MASE:", mean(mexico_mase_results_sarima)))

# retrieving the fitted values for the training set
mexico_fitted_values_sarima <- fitted(mexico_sarima_model)

# enforcing non-negativity on fitted values
mexico_fitted_values_sarima <- pmax(mexico_fitted_values_sarima, 0)

# combining training and test data for plotting
mexico_all_dates <- c(mexico_train_data$date, mexico_test_data$date)
mexico_all_values <- c(mexico_train_data$owid_new_deaths, mexico_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(mexico_all_dates, mexico_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(mexico_test_data$date, mexico_forecast_values_sarima$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(mexico_train_data$date, mexico_fitted_values_sarima, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# peru ----


## load data ----
load("data/preprocessed/univariate/not_split/peru.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
peru_total_days <- nrow(peru)
peru_train_days <- ceiling(0.9 * peru_total_days)
peru_test_days <- ceiling((peru_total_days - peru_train_days))

# creating folds
peru_folds <- time_series_cv(
  peru,
  date_var = date,
  initial = peru_train_days,
  assess = peru_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
peru_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
peru_rmse_results <- numeric(length(peru_folds$splits))
peru_mae_results <- numeric(length(peru_folds$splits))
peru_mse_results <- numeric(length(peru_folds$splits))
peru_mase_results <- numeric(length(peru_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(peru_folds$splits)) {
  fold <- peru_folds$splits[[i]]
  peru_train_data <- fold$data[fold$in_id, ]
  peru_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  peru_arima_model <- arima(peru_train_data$owid_new_deaths,
                            order = c(2, 1, 0))
  
  # forecasting with ARIMA
  peru_forecast_values <- forecast(peru_arima_model, h = nrow(peru_test_data))
  
  # enforcing non-negativity on forecasted values
  peru_forecast_values$mean <- pmax(peru_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  peru_errors <- peru_forecast_values$mean - peru_test_data$owid_new_deaths
  peru_rmse_results[i] <- sqrt(mean(peru_errors^2))
  peru_mae_results[i] <- mean(abs(peru_errors))
  peru_mse_results[i] <- mean(peru_errors^2)
  
  # calculating MASE
  peru_mean_train_diff <- mean(abs(diff(peru_train_data$owid_new_deaths)))
  peru_mase_results[i] <- mean(abs(peru_errors)) / peru_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(peru_rmse_results)))
print(paste("MAE:", mean(peru_mae_results)))
print(paste("MSE:", mean(peru_mse_results)))
print(paste("MASE:", mean(peru_mase_results)))

# retrieving the fitted values for the training set
peru_fitted_values <- fitted(peru_arima_model)

# enforcing non-negativity on fitted values
peru_fitted_values <- pmax(peru_fitted_values, 0)

# combining training and test data for plotting
peru_all_dates <- c(peru_train_data$date, peru_test_data$date)
peru_all_values <- c(peru_train_data$owid_new_deaths, peru_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(peru_all_dates, peru_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(peru_test_data$date, peru_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(peru_train_data$date, peru_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# russia ----


## load data ----
load("data/preprocessed/univariate/not_split/russia.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
russia_total_days <- nrow(russia)
russia_train_days <- ceiling(0.9 * russia_total_days)
russia_test_days <- ceiling((russia_total_days - russia_train_days))

# creating folds
russia_folds <- time_series_cv(
  russia,
  date_var = date,
  initial = russia_train_days,
  assess = russia_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
russia_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
russia_rmse_results <- numeric(length(russia_folds$splits))
russia_mae_results <- numeric(length(russia_folds$splits))
russia_mse_results <- numeric(length(russia_folds$splits))
russia_mase_results <- numeric(length(russia_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(russia_folds$splits)) {
  fold <- russia_folds$splits[[i]]
  russia_train_data <- fold$data[fold$in_id, ]
  russia_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  russia_arima_model <- arima(russia_train_data$owid_new_deaths,
                              order = c(1, 2, 2))
  
  # forecasting with ARIMA
  russia_forecast_values <- forecast(russia_arima_model, h = nrow(russia_test_data))
  
  # enforcing non-negativity on forecasted values
  russia_forecast_values$mean <- pmax(russia_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  russia_errors <- russia_forecast_values$mean - russia_test_data$owid_new_deaths
  russia_rmse_results[i] <- sqrt(mean(russia_errors^2))
  russia_mae_results[i] <- mean(abs(russia_errors))
  russia_mse_results[i] <- mean(russia_errors^2)
  
  # calculating MASE
  russia_mean_train_diff <- mean(abs(diff(russia_train_data$owid_new_deaths)))
  russia_mase_results[i] <- mean(abs(russia_errors)) / russia_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(russia_rmse_results)))
print(paste("MAE:", mean(russia_mae_results)))
print(paste("MSE:", mean(russia_mse_results)))
print(paste("MASE:", mean(russia_mase_results)))

# retrieving the fitted values for the training set
russia_fitted_values <- fitted(russia_arima_model)

# enforcing non-negativity on fitted values
russia_fitted_values <- pmax(russia_fitted_values, 0)

# combining training and test data for plotting
russia_all_dates <- c(russia_train_data$date, russia_test_data$date)
russia_all_values <- c(russia_train_data$owid_new_deaths, russia_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(russia_all_dates, russia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(russia_test_data$date, russia_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(russia_train_data$date, russia_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)


## applying SARIMA model ----

# creating metrics vector
russia_rmse_results_sarima <- numeric(length(russia_folds$splits))
russia_mae_results_sarima <- numeric(length(russia_folds$splits))
russia_mse_results_sarima <- numeric(length(russia_folds$splits))
russia_mase_results_sarima <- numeric(length(russia_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(russia_folds$splits)) {
  fold <- russia_folds$splits[[i]]
  russia_train_data <- fold$data[fold$in_id, ]
  russia_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  russia_sarima_model <- arima(russia_train_data$owid_new_deaths,
                               order = c(2, 2, 0),
                               seasonal = list(order = c(1, 1, 1), period = 7))
  
  # forecasting with ARIMA
  russia_forecast_values_sarima <- forecast(russia_sarima_model, h = nrow(russia_test_data))
  
  # enforcing non-negativity on forecasted values
  russia_forecast_values_sarima$mean <- pmax(russia_forecast_values_sarima$mean, 0)
  
  # calculating evaluation metrics
  russia_errors_sarima <- russia_forecast_values_sarima$mean - russia_test_data$owid_new_deaths
  russia_rmse_results_sarima[i] <- sqrt(mean(russia_errors_sarima^2))
  russia_mae_results_sarima[i] <- mean(abs(russia_errors_sarima))
  russia_mse_results_sarima[i] <- mean(russia_errors_sarima^2)
  
  # calculating MASE
  russia_mean_train_diff_sarima <- mean(abs(diff(russia_train_data$owid_new_deaths)))
  russia_mase_results_sarima[i] <- mean(abs(russia_errors_sarima)) / russia_mean_train_diff_sarima
}

# printing metrics
print(paste("RMSE:", mean(russia_rmse_results_sarima)))
print(paste("MAE:", mean(russia_mae_results_sarima)))
print(paste("MSE:", mean(russia_mse_results_sarima)))
print(paste("MASE:", mean(russia_mase_results_sarima)))

# retrieving the fitted values for the training set
russia_fitted_values_sarima <- fitted(russia_sarima_model)

# enforcing non-negativity on fitted values
russia_fitted_values_sarima <- pmax(russia_fitted_values_sarima, 0)

# combining training and test data for plotting
russia_all_dates <- c(russia_train_data$date, russia_test_data$date)
russia_all_values <- c(russia_train_data$owid_new_deaths, russia_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(russia_all_dates, russia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(russia_test_data$date, russia_forecast_values_sarima$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(russia_train_data$date, russia_fitted_values_sarima, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# saudi ----


## load data ----
load("data/preprocessed/univariate/not_split/saudi.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
saudi_total_days <- nrow(saudi)
saudi_train_days <- ceiling(0.9 * saudi_total_days)
saudi_test_days <- ceiling((saudi_total_days - saudi_train_days))

# creating folds
saudi_folds <- time_series_cv(
  saudi,
  date_var = date,
  initial = saudi_train_days,
  assess = saudi_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
saudi_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
saudi_rmse_results <- numeric(length(saudi_folds$splits))
saudi_mae_results <- numeric(length(saudi_folds$splits))
saudi_mse_results <- numeric(length(saudi_folds$splits))
saudi_mase_results <- numeric(length(saudi_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(saudi_folds$splits)) {
  fold <- saudi_folds$splits[[i]]
  saudi_train_data <- fold$data[fold$in_id, ]
  saudi_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  saudi_arima_model <- arima(saudi_train_data$owid_new_deaths,
                             order = c(1, 1, 0),
                             seasonal = list(order = c(1, 2, 2), period = 3))
  
  # forecasting with ARIMA
  saudi_forecast_values <- forecast(saudi_arima_model, h = nrow(saudi_test_data))
  
  # enforcing non-negativity on forecasted values
  saudi_forecast_values$mean <- pmax(saudi_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  saudi_errors <- saudi_forecast_values$mean - saudi_test_data$owid_new_deaths
  saudi_rmse_results[i] <- sqrt(mean(saudi_errors^2))
  saudi_mae_results[i] <- mean(abs(saudi_errors))
  saudi_mse_results[i] <- mean(saudi_errors^2)
  
  # calculating MASE
  saudi_mean_train_diff <- mean(abs(diff(saudi_train_data$owid_new_deaths)))
  saudi_mase_results[i] <- mean(abs(saudi_errors)) / saudi_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(saudi_rmse_results)))
print(paste("MAE:", mean(saudi_mae_results)))
print(paste("MSE:", mean(saudi_mse_results)))
print(paste("MASE:", mean(saudi_mase_results)))

# retrieving the fitted values for the training set
saudi_fitted_values <- fitted(saudi_arima_model)

# enforcing non-negativity on fitted values
saudi_fitted_values <- pmax(saudi_fitted_values, 0)

# combining training and test data for plotting
saudi_all_dates <- c(saudi_train_data$date, saudi_test_data$date)
saudi_all_values <- c(saudi_train_data$owid_new_deaths, saudi_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(saudi_all_dates, saudi_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(saudi_test_data$date, saudi_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(saudi_train_data$date, saudi_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# turkey ----


## load data ----
load("data/preprocessed/univariate/not_split/turkey.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
turkey_total_days <- nrow(turkey)
turkey_train_days <- ceiling(0.9 * turkey_total_days)
turkey_test_days <- ceiling((turkey_total_days - turkey_train_days))

# creating folds
turkey_folds <- time_series_cv(
  turkey,
  date_var = date,
  initial = turkey_train_days,
  assess = turkey_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
turkey_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
turkey_rmse_results <- numeric(length(turkey_folds$splits))
turkey_mae_results <- numeric(length(turkey_folds$splits))
turkey_mse_results <- numeric(length(turkey_folds$splits))
turkey_mase_results <- numeric(length(turkey_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(turkey_folds$splits)) {
  fold <- turkey_folds$splits[[i]]
  turkey_train_data <- fold$data[fold$in_id, ]
  turkey_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  turkey_arima_model <- arima(turkey_train_data$owid_new_deaths,
                              order = c(0, 2, 0))
  
  # forecasting with ARIMA
  turkey_forecast_values <- forecast(turkey_arima_model, h = nrow(turkey_test_data))
  
  # enforcing non-negativity on forecasted values
  turkey_forecast_values$mean <- pmax(turkey_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  turkey_errors <- turkey_forecast_values$mean - turkey_test_data$owid_new_deaths
  turkey_rmse_results[i] <- sqrt(mean(turkey_errors^2))
  turkey_mae_results[i] <- mean(abs(turkey_errors))
  turkey_mse_results[i] <- mean(turkey_errors^2)
  
  # calculating MASE
  turkey_mean_train_diff <- mean(abs(diff(turkey_train_data$owid_new_deaths)))
  turkey_mase_results[i] <- mean(abs(turkey_errors)) / turkey_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(turkey_rmse_results)))
print(paste("MAE:", mean(turkey_mae_results)))
print(paste("MSE:", mean(turkey_mse_results)))
print(paste("MASE:", mean(turkey_mase_results)))

# retrieving the fitted values for the training set
turkey_fitted_values <- fitted(turkey_arima_model)

# enforcing non-negativity on fitted values
turkey_fitted_values <- pmax(turkey_fitted_values, 0)

# combining training and test data for plotting
turkey_all_dates <- c(turkey_train_data$date, turkey_test_data$date)
turkey_all_values <- c(turkey_train_data$owid_new_deaths, turkey_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(turkey_all_dates, turkey_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(turkey_test_data$date, turkey_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(turkey_train_data$date, turkey_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# us ----


## load data ----
load("data/preprocessed/univariate/not_split/us.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
us_total_days <- nrow(us)
us_train_days <- ceiling(0.9 * us_total_days)
us_test_days <- ceiling((us_total_days - us_train_days))

# creating folds
us_folds <- time_series_cv(
  us,
  date_var = date,
  initial = us_train_days,
  assess = us_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
us_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
us_rmse_results <- numeric(length(us_folds$splits))
us_mae_results <- numeric(length(us_folds$splits))
us_mse_results <- numeric(length(us_folds$splits))
us_mase_results <- numeric(length(us_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(us_folds$splits)) {
  fold <- us_folds$splits[[i]]
  us_train_data <- fold$data[fold$in_id, ]
  us_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  us_arima_model <- arima(us_train_data$owid_new_deaths,
                          order = c(2, 0, 2))
  
  # forecasting with ARIMA
  us_forecast_values <- forecast(us_arima_model, h = nrow(us_test_data))
  
  # enforcing non-negativity on forecasted values
  us_forecast_values$mean <- pmax(us_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  us_errors <- us_forecast_values$mean - us_test_data$owid_new_deaths
  us_rmse_results[i] <- sqrt(mean(us_errors^2))
  us_mae_results[i] <- mean(abs(us_errors))
  us_mse_results[i] <- mean(us_errors^2)
  
  # calculating MASE
  us_mean_train_diff <- mean(abs(diff(us_train_data$owid_new_deaths)))
  us_mase_results[i] <- mean(abs(us_errors)) / us_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(us_rmse_results)))
print(paste("MAE:", mean(us_mae_results)))
print(paste("MSE:", mean(us_mse_results)))
print(paste("MASE:", mean(us_mase_results)))

# retrieving the fitted values for the training set
us_fitted_values <- fitted(us_arima_model)

# enforcing non-negativity on fitted values
us_fitted_values <- pmax(us_fitted_values, 0)

# combining training and test data for plotting
us_all_dates <- c(us_train_data$date, us_test_data$date)
us_all_values <- c(us_train_data$owid_new_deaths, us_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(us_all_dates, us_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(us_test_data$date, us_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(us_train_data$date, us_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



## applying SARIMA model ----

# creating metrics vector
us_rmse_results_sarima <- numeric(length(us_folds$splits))
us_mae_results_sarima <- numeric(length(us_folds$splits))
us_mse_results_sarima <- numeric(length(us_folds$splits))
us_mase_results_sarima <- numeric(length(us_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(us_folds$splits)) {
  fold <- us_folds$splits[[i]]
  us_train_data <- fold$data[fold$in_id, ]
  us_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  us_sarima_model <- arima(us_train_data$owid_new_deaths,
                               order = c(0, 0, 0),
                               seasonal = list(order = c(1, 1, 1), period = 7))
  
  # forecasting with ARIMA
  us_forecast_values_sarima <- forecast(us_sarima_model, h = nrow(us_test_data))
  
  # enforcing non-negativity on forecasted values
  us_forecast_values_sarima$mean <- pmax(us_forecast_values_sarima$mean, 0)
  
  # calculating evaluation metrics
  us_errors_sarima <- us_forecast_values_sarima$mean - us_test_data$owid_new_deaths
  us_rmse_results_sarima[i] <- sqrt(mean(us_errors_sarima^2))
  us_mae_results_sarima[i] <- mean(abs(us_errors_sarima))
  us_mse_results_sarima[i] <- mean(us_errors_sarima^2)
  
  # calculating MASE
  us_mean_train_diff_sarima <- mean(abs(diff(us_train_data$owid_new_deaths)))
  us_mase_results_sarima[i] <- mean(abs(us_errors_sarima)) / us_mean_train_diff_sarima
}

# printing metrics
print(paste("RMSE:", mean(us_rmse_results_sarima)))
print(paste("MAE:", mean(us_mae_results_sarima)))
print(paste("MSE:", mean(us_mse_results_sarima)))
print(paste("MASE:", mean(us_mase_results_sarima)))

# retrieving the fitted values for the training set
us_fitted_values_sarima <- fitted(us_sarima_model)

# enforcing non-negativity on fitted values
us_fitted_values_sarima <- pmax(us_fitted_values_sarima, 0)

# combining training and test data for plotting
us_all_dates <- c(us_train_data$date, us_test_data$date)
us_all_values <- c(us_train_data$owid_new_deaths, us_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(us_all_dates, us_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(us_test_data$date, us_forecast_values_sarima$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(us_train_data$date, us_fitted_values_sarima, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# finding best ARIMA parameters per country (changed variable names for reuse - switching countries) ----

# initializing parameters
min_rmse <- Inf
best_order <- NULL
best_seasonal <- NULL

# iterating over all combinations without a seasonal component
for (order_i in c(0, 1, 2)) {
  for (order_ii in c(0, 1, 2)) {
    for (order_iii in c(0, 1, 2)) {

      # initializing vectors to store RMSE
      rmse_results <- numeric(length(bolivia_folds$splits))

      # fitting to model and calculating metrics
      for (i in seq_along(bolivia_folds$splits)) {
        fold <- bolivia_folds$splits[[i]]
        train_data <- fold$data[fold$in_id, ]
        test_data <- fold$data[fold$out_id, ]

        # fitting to ARIMA without a seasonal component
        arima_model <- tryCatch({
          arima(train_data$owid_new_deaths,
                order = c(order_i, order_ii, order_iii))
        }, error = function(e) {
          NULL
        })

        if (!is.null(arima_model)) {
          # forecasting with ARIMA
          forecast_values <- forecast(arima_model, h = nrow(test_data))

          # enforcing non-negativity on forecasted values
          forecast_values$mean <- pmax(forecast_values$mean, 0)

          # computing RMSE
          errors <- forecast_values$mean - test_data$owid_new_deaths
          rmse_results[i] <- sqrt(mean(errors^2))
        } else {
          # setting RMSE to Inf if model fitting fails
          rmse_results[i] <- Inf
        }
      }

      # calculating average folds RMSE
      avg_rmse <- mean(rmse_results, na.rm = TRUE)

      # checking if current combo results in lower RMSE
      if (avg_rmse < min_rmse) {
        min_rmse <- avg_rmse
        best_order <- c(order_i, order_ii, order_iii)
      }
    }
  }
}

# printing lowest RMSE and best parameters
cat("Minimum RMSE:", min_rmse, "\n")
cat("Best order:", best_order, "\n")



# # finding best SARIMA parameters per country ----
# 
# # initializing parameters
# min_rmse <- Inf
# best_order <- NULL
# best_seasonal <- NULL
# 
# # iterating over all combinations
# for (order_i in c(0, 1, 2)) {
#   for (order_ii in c(0, 1, 2)) {
#     for (order_iii in c(0, 1, 2)) {
#       for (seasonal_order_i in c(0, 1)) {
#         for (seasonal_order_ii in c(0, 1)) {
#           for (seasonal_order_iii in c(0, 1)) {
#             for (seasonal_period in c(7)) {
#               seasonal <- list(order = c(seasonal_order_i, seasonal_order_ii, seasonal_order_iii),
#                                period = seasonal_period)
# 
#               # initializing vectors to store RMSE
#               rmse_results <- numeric(length(us_folds$splits))
# 
#               # fitting to model and calculating metrics
#               for (i in seq_along(us_folds$splits)) {
#                 fold <- us_folds$splits[[i]]
#                 train_data <- fold$data[fold$in_id, ]
#                 test_data <- fold$data[fold$out_id, ]
# 
#                 # fitting to ARIMA
#                 sarima_model <- tryCatch({
#                   arima(train_data$owid_new_deaths,
#                         order = c(order_i, order_ii, order_iii), # Use dynamic order
#                         seasonal = seasonal)
#                 }, error = function(e) {
#                   NULL
#                 })
# 
#                 if (!is.null(sarima_model)) {
# 
#                   # forecasting with ARIMA
#                   forecast_values <- forecast(sarima_model, h = nrow(test_data))
# 
#                   # enforcing non-negativity on forecasted values
#                   forecast_values$mean <- pmax(forecast_values$mean, 0)
# 
#                   # computing RMSE
#                   errors <- forecast_values$mean - test_data$owid_new_deaths
#                   rmse_results[i] <- sqrt(mean(errors^2))
#                 } else {
# 
#                   # setting RMSE to Inf if model fails
#                   rmse_results[i] <- Inf
#                 }
#               }
# 
#               # calculating average folds RMSE
#               avg_rmse <- mean(rmse_results, na.rm = TRUE)
# 
#               # checking if current combo results in lower RMSE
#               if (avg_rmse < min_rmse) {
#                 min_rmse <- avg_rmse
#                 best_order <- c(order_i, order_ii, order_iii)
#                 best_seasonal <- seasonal
#               }
#             }
#           }
#         }
#       }
#     }
#   }
# }
# 
# # printing lowest RMSE and best parameters
# cat("Minimum RMSE:", min_rmse, "\n")
# cat("Best order:", best_order, "\n")
# cat("Best seasonal:", best_seasonal$order, "\n")
# cat("Best seasonal period:", best_seasonal$period, "\n")





# YOU STOPPED HERE ----





# producing training and testing separate plots ----


## bolivia ----

# producing fitted training set plot
plot(bolivia_train_data$date, bolivia_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Bolivia", "Training Set"))
lines(bolivia_train_data$date, bolivia_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(bolivia_test_data$date, bolivia_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Bolivia", "Test Set"))
lines(bolivia_test_data$date, bolivia_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## brazil ----

# producing fitted training set plot
plot(brazil_train_data$date, brazil_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Brazil", "Training Set"))
lines(brazil_train_data$date, brazil_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(brazil_test_data$date, brazil_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Brazil", "Test Set"))
lines(brazil_test_data$date, brazil_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## iran ----

# producing fitted training set plot
plot(iran_train_data$date, iran_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Iran", "Training Set"))
lines(iran_train_data$date, iran_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(iran_test_data$date, iran_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Iran", "Test Set"))
lines(iran_test_data$date, iran_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## mexico ----

# producing fitted training set plot
plot(mexico_train_data$date, mexico_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Mexico", "Training Set"))
lines(mexico_train_data$date, mexico_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(mexico_test_data$date, mexico_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Mexico", "Test Set"))
lines(mexico_test_data$date, mexico_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## peru ----

# producing fitted training set plot
plot(peru_train_data$date, peru_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Peru", "Training Set"))
lines(peru_train_data$date, peru_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(peru_test_data$date, peru_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Peru", "Test Set"))
lines(peru_test_data$date, peru_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## russia ----

# producing fitted training set plot
plot(russia_train_data$date, russia_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Russia", "Training Set"))
lines(russia_train_data$date, russia_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(russia_test_data$date, russia_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Russia", "Test Set"))
lines(russia_test_data$date, russia_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## saudi ----

# producing fitted training set plot
plot(saudi_train_data$date, saudi_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Saudi", "Training Set"))
lines(saudi_train_data$date, saudi_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(saudi_test_data$date, saudi_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Saudi", "Test Set"))
lines(saudi_test_data$date, saudi_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## turkey ----

# producing fitted training set plot
plot(turkey_train_data$date, turkey_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Turkey", "Training Set"))
lines(turkey_train_data$date, turkey_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(turkey_test_data$date, turkey_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Turkey", "Test Set"))
lines(turkey_test_data$date, turkey_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## us ----

# producing fitted training set plot
plot(us_train_data$date, us_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("U.S.", "Training Set"))
lines(us_train_data$date, us_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(us_test_data$date, us_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("U.S.", "Test Set"))
lines(us_test_data$date, us_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)



# visualizing all metrics ----

# defining a list to hold each country's metrics
metrics_list <- list()

# calculating metrics for each country
for (country in c("bolivia", "brazil", "colombia", "iran",
                  "mexico", "peru", "russia", "saudi", "turkey", "us")) {

    # simulating the calculation of metrics
  rmse <- mean(get(paste0(country, "_rmse_results")))
  mae <- mean(get(paste0(country, "_mae_results")))
  mse <- mean(get(paste0(country, "_mse_results")))
  mase <- mean(get(paste0(country, "_mase_results")))
  
  # creating a data frame for the country's metrics
  if (tolower(country) == "us") {
    country_metrics_df <- data.frame(
      Country = "US",
      RMSE = rmse,
      MAE = mae,
      MSE = mse,
      MASE = mase
    )
  } else {
    country_metrics_df <- data.frame(
      Country = tools::toTitleCase(country),
      RMSE = rmse,
      MAE = mae,
      MSE = mse,
      MASE = mase
    )
  }
  
  # adding data frame to list
  metrics_list[[country]] <- country_metrics_df
}

# combining all the data frames into one
all_metrics_df <- do.call(rbind, metrics_list)

# rounding the metrics to 3 decimal points + handling NAs
num_cols <- c("RMSE", "MAE", "MSE", "MASE")
all_metrics_df[num_cols] <- lapply(all_metrics_df[num_cols], function(x) {
  ifelse(is.na(x), NA, round(x, 3))
})

# sorting by RMSE
all_metrics_df <- all_metrics_df[order(all_metrics_df$RMSE), ]

# printing data frame
print(all_metrics_df %>% DT::datatable())



# creating a metrics list for each country (training included) ----

# defining a list
train_metrics_list <- list()

# specifying countries
countries <- c("bolivia", "brazil", "colombia", "iran", "mexico", "peru", "russia", "saudi", "turkey", "us")

# looping to calculate training metrics
for (country in countries) {
  
  # getting the actual and fitted values
  actual_values <- get(paste0(country, "_train_data"))$owid_new_deaths
  fitted_values <- get(paste0(country, "_fitted_values"))
  
  # calculating errors
  errors <- actual_values - fitted_values
  
  # calculating metrics
  rmse <- sqrt(mean(errors^2))
  mae <- mean(abs(errors))
  mse <- mean(errors^2)
  
  # storing metrics in list
  train_metrics_list[[paste0(country, "_train_rmse")]] <- rmse
  train_metrics_list[[paste0(country, "_train_mae")]] <- mae
  train_metrics_list[[paste0(country, "_train_mse")]] <- mse
}

# creating a list to store comparison results
performance_comparison <- list()

# looping through each country
for (country in countries) {
  
  # retrieving average training metrics
  avg_train_rmse <- round(train_metrics_list[[paste0(country, "_train_rmse")]], 3)
  avg_train_mae <- round(train_metrics_list[[paste0(country, "_train_mae")]], 3)
  avg_train_mse <- round(train_metrics_list[[paste0(country, "_train_mse")]], 3)
  
  # retrieving average testing metrics
  avg_test_rmse <- round(mean(get(paste0(country, "_rmse_results"))), 3)
  avg_test_mae <- round(mean(get(paste0(country, "_mae_results"))), 3)
  avg_test_mse <- round(mean(get(paste0(country, "_mse_results"))), 3)
  
  # calculating differences and rounding
  rmse_diff <- round(avg_test_rmse - avg_train_rmse, 3)
  mae_diff <- round(avg_test_mae - avg_train_mae, 3)
  mse_diff <- round(avg_test_mse - avg_train_mse, 3)
  
  # diagnosing based on differences
  diagnosis <- ifelse(rmse_diff > 0.2 & mae_diff > 0.2 & mse_diff > 0.2, "Possible Overfitting",
                      ifelse(avg_train_rmse > 1 & avg_test_rmse > 1 & avg_train_mae > 1 & avg_test_mae > 1,
                             "Possible Underfitting", "Appropriate Fit"))
  
  # formatting country names
  formatted_country <- ifelse(country == "us", "US", tools::toTitleCase(country))
  
  # combining into data frame
  comparison_df <- data.frame(
    Country = formatted_country,
    Train_RMSE = avg_train_rmse,
    Test_RMSE = avg_test_rmse,
    RMSE_Difference = rmse_diff,
    Train_MAE = avg_train_mae,
    Test_MAE = avg_test_mae,
    MAE_Difference = mae_diff,
    Train_MSE = avg_train_mse,
    Test_MSE = avg_test_mse,
    MSE_Difference = mse_diff,
    Diagnosis = diagnosis
  )
  
  # adding to list
  performance_comparison[[country]] <- comparison_df
}

# combining all country comparisons
all_comparisons_df <- do.call(rbind, performance_comparison)

# rounding
numeric_columns <- sapply(all_comparisons_df, is.numeric)
all_comparisons_df[, numeric_columns] <- round(all_comparisons_df[, numeric_columns], 3)

# printing comparison
print(all_comparisons_df %>% DT::datatable(options = list(pageLength = 10)))



# determining whether log transformation is required ----

# placeholder for skewedness
skewness_values <- list()
transformation_advice <- list()
countries <- c("bolivia", "brazil", "colombia", "iran", "mexico", "peru", "russia", "saudi", "turkey", "us")
for (country in countries) {
  
  # replacing these placeholders with variable names
  forecast_values <- get(paste0(country, "_forecast_values"))$mean
  actual_values <- get(paste0(country, "_test_data"))$owid_new_deaths
  
  # calculating residuals
  residuals <- forecast_values - actual_values
  
  # calculating skewness
  skewness_val <- skewness(residuals)
  skewness_values[[country]] <- skewness_val
  
  # advice for transformation based on skewness
  advice <- ifelse(abs(skewness_val) > 1, "Consider Log Transformation", "No Transformation Needed")
  transformation_advice[[country]] <- advice
}

# printing values and advice
print(skewness_values)
print(transformation_advice)



# auto-ARIMA ----

# automatic ARIMA modeling
bolivia_auto_model <- auto.arima(bolivia_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
brazil_auto_model <- auto.arima(brazil_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
colombia_auto_model <- auto.arima(colombia_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
iran_auto_model <- auto.arima(iran_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
mexico_auto_model <- auto.arima(mexico_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
peru_auto_model <- auto.arima(peru_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
russia_auto_model <- auto.arima(russia_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
saudi_auto_model <- auto.arima(saudi_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
turkey_auto_model <- auto.arima(turkey_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
us_auto_model <- auto.arima(us_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

# viewing selected model's details
summary(bolivia_auto_model)
summary(brazil_auto_model)
summary(colombia_auto_model)
summary(iran_auto_model)
summary(mexico_auto_model)
summary(peru_auto_model)
summary(russia_auto_model)
summary(saudi_auto_model)
summary(turkey_auto_model)
summary(us_auto_model)



# displaying best models ----

# defining helper function to calculate metrics
calculate_metrics <- function(forecasted, actual, train) {
  errors <- forecasted - actual
  train_diff <- mean(abs(diff(train)))
  
  list(
    RMSE = sqrt(mean(errors^2)),
    MAE = mean(abs(errors)),
    MSE = mean(errors^2),
    MASE = mean(abs(errors)) / train_diff
  )
}

# extracting country names
countries_data <- list(bolivia = bolivia, brazil = brazil, colombia = colombia, iran = iran, mexico = mexico, peru = peru, russia = russia, saudi = saudi, turkey = turkey, us = us)
country_names <- names(countries_data)

# initializing empty list to store metrics
all_countries_metrics <- list()

# looping through each country
for (country_name in country_names) {
  
  # load country-specific data
  country_data <- get(paste0(country_name, "_train_data"))
  country_test_data <- get(paste0(country_name, "_test_data"))
  
  # ARIMA model
  arima_model <- get(paste0(country_name, "_arima_model"))
  
  # Auto-ARIMA model
  auto_model <- auto.arima(country_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
  
  # forecasting using both models
  arima_forecast <- forecast(arima_model, h = nrow(country_test_data))
  auto_forecast <- forecast(auto_model, h = nrow(country_test_data))
  
  # calculating metrics for ARIMA
  arima_metrics <- calculate_metrics(arima_forecast$mean, country_test_data$owid_new_deaths, country_data$owid_new_deaths)
  
  # calculating metrics for Auto-ARIMA
  auto_metrics <- calculate_metrics(auto_forecast$mean, country_test_data$owid_new_deaths, country_data$owid_new_deaths)
  
  # combining metrics into a data frame
  metrics_df <- data.frame(
    Country = country_name,
    Best_Model_RMSE = c("ARIMA", "Auto-ARIMA"),
    RMSE = c(arima_metrics$RMSE, auto_metrics$RMSE),
    MAE = c(arima_metrics$MAE, auto_metrics$MAE),
    MSE = c(arima_metrics$MSE, auto_metrics$MSE),
    MASE = c(arima_metrics$MASE, auto_metrics$MASE)
  )
  
  # capitalizing country name in metrics_df
  metrics_df$Country <- ifelse(tolower(metrics_df$Country) != "us", tools::toTitleCase(metrics_df$Country), "US")
  
  # storing in list
  all_countries_metrics[[country_name]] <- metrics_df
}

# combining all metrics into one data frame
arima_final_metrics_df <- do.call(rbind, all_countries_metrics)
arima_final_metrics_df <- arima_final_metrics_df %>%
  group_by(Country) %>%
  slice_min(order_by = RMSE, with_ties = FALSE) %>%
  ungroup()

arima_final_metrics_df <- arima_final_metrics_df %>%
  mutate(across(c(RMSE, MAE, MSE, MASE), round, 3))

arima_final_metrics_df %>% 
  DT::datatable()

# printing final metrics table
print(arima_final_metrics_df)

# removing row names
row.names(arima_final_metrics_df) <- NULL

# arima parameters
arima_params <- data.frame(
  Country = c("Bolivia", "Brazil", "Colombia", "Iran", "Mexico", "Peru", "Russia", "Saudi", "Turkey", "US"),
  p = c(0, 0, 0, 1, 1, 0, 1, 1, 0, 0),
  d = c(1, 1, 1, 1, 0, 1, 1, 1, 1, 0),
  q = c(1, 1, 0, 0, 0, 1, 0, 0, 0, 0),
  P = c(0, 0, 1, 1, 1, 0, 0, 1, 0, 1),
  D = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
  Q = c(0, 0, 1, 0, 0, 1, 0, 0, 0, 1),
  s = c(7, 7, 10, 10, 13, 3, 2, 3, 1, 7)
) %>% DT::datatable()

# auto-arima parameters
auto_arima_params <- data.frame(
  Country = c("Bolivia", "Brazil", "Colombia", "Iran", "Mexico", "Peru", "Russia", "Saudi", "Turkey", "US"),
  p = c(2, 3, 1, 2, 0, 1, 0, 3, 1, 2),
  d = c(1, 0, 1, 1, 1, 1, 1, 1, 1, 1),
  q = c(2, 3, 1, 0, 2, 3, 1, 0, 1, 2),
  P = c(1, 0, 1, 0, 1, 0, 2, 1, 0, 1),
  D = c(1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
  Q = c(1, 2, 1, 1, 1, 0, 2, 1, 2, 1),
  s = c(7, 7, 12, 12, 12, 4, 12, 12, 12, 7)
) %>% DT::datatable()

# initializing an empty list to store the metrics for each country
auto_arima_metrics_list <- list()

# looping through each country
for (country_name in country_names) {
  
  # loading country-specific training data
  country_train_data <- get(paste0(country_name, "_train_data"))
  country_test_data <- get(paste0(country_name, "_test_data"))
  
  # fitting the Auto-ARIMA model to the training data
  auto_model <- auto.arima(country_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
  
  # forecasting using the Auto-ARIMA model
  auto_forecast <- forecast(auto_model, h = nrow(country_test_data))
  
  # calculating metrics for Auto-ARIMA
  auto_metrics <- calculate_metrics(auto_forecast$mean, country_test_data$owid_new_deaths, country_train_data$owid_new_deaths)
  
  # creating a data frame for the Auto-ARIMA metrics
  auto_metrics_df <- data.frame(
    Country = ifelse(country_name == "us", "US", tools::toTitleCase(country_name)),
    RMSE = round(auto_metrics$RMSE, 3),
    MAE = round(auto_metrics$MAE, 3),
    MSE = round(auto_metrics$MSE, 3),
<<<<<<< HEAD
    MAPE = round(auto_metrics$MAPE, 3),
=======
>>>>>>> main
    MASE = round(auto_metrics$MASE, 3)
  )
  
  # storing the data frame in the list
  auto_arima_metrics_list[[country_name]] <- auto_metrics_df
}

# combining all the country-specific metrics into a single data frame
auto_arima_final_metrics_df <- do.call(rbind, auto_arima_metrics_list)

# removing row names for a cleaner look
row.names(auto_arima_final_metrics_df) <- NULL



# saving files ----
save(arima_final_metrics_df, file = "data_frames/maria_arima_final_metrics_df.rda")
<<<<<<< HEAD
save(auto_arima_final_metrics_df, file = "auto_arima_final_metrics_df.rda")
=======
save(auto_arima_final_metrics_df, file = "auto_arima_final_metrics_df.rda") 
>>>>>>> main
