## ARIMA and Auto-ARIMA
## Cross-validation and hyperparameter tuning are performed in document 1b.


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

# creating folds (ignore and refer to folds and tuning document for all countries)
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
                              order = c(0, 2, 1))
  
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
                              order = c(2, 0, 2),
                              seasonal = list(order = c(1, 1, 1), period = 7))
  
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
                                order = c(2, 1, 2))
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
                            order = c(2, 0, 0))
  
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
                              order = c(0, 2, 2))
  
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
                              order = c(2, 0, 2),
                              seasonal = list(order = c(1, 1, 1), period = 7))
  
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
                            order = c(0, 1, 0))
  
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
                              order = c(0, 1, 0))
  
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
                               order = c(0, 1, 0),
                               seasonal = list(order = c(0, 1, 2), period = 7))
  
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
                             order = c(0, 1, 1))
  
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
                              order = c(2, 1, 1))
  
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
                          order = c(1, 0, 0))
  
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
                               order = c(1, 0, 0),
                               seasonal = list(order = c(1, 0, 1), period = 7))
  
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



# auto-ARIMA ----

# creating a list of country data frames
countries_data <- list(
  bolivia = bolivia,
  brazil = brazil,
  colombia = colombia,
  iran = iran,
  mexico = mexico,
  peru = peru,
  russia = russia,
  saudi = saudi,
  turkey = turkey,
  us = us
)

# initializing a list to store the results
results <- list()

# looping through each country's data
for(country_name in names(countries_data)) {
  
  # getting the country data
  country_data <- countries_data[[country_name]]
  
  # ensuring the date column is of Date type
  country_data$date <- as.Date(country_data$date)
  
  # splitting data into training (80%) and testing (20%)
  split_index <- ceiling(nrow(country_data) * 0.8)
  train_data <- country_data[1:split_index, ]
  test_data <- country_data[(split_index + 1):nrow(country_data), ]
  
  # applying Auto-ARIMA model
  auto_arima_model <- auto.arima(train_data$owid_new_deaths)
  
  # forecasting on the test set
  forecast_values <- forecast(auto_arima_model, h = nrow(test_data))
  
  # storing the results
  results[[country_name]] <- list(
    model = auto_arima_model,
    forecast = forecast_values,
    test_data = test_data
  )
  
  rmse_value <- sqrt(mean((forecast_values$mean - test_data$owid_new_deaths)^2))
  cat(country_name, "RMSE:", rmse_value, "\n")
}



# creating table metrics ----


## arima ----

# defining countries
countries <- c("bolivia", "brazil", "colombia", "iran", "mexico", "peru", "russia", "saudi", "turkey", "us")
sarima_countries <- c("brazil", "mexico", "russia", "us")

# initialize empty data frames for ARIMA and SARIMA metrics
arima_metrics <- data.frame(Country=character(), RMSE=numeric(), MSE=numeric(), MAE=numeric(), MASE=numeric(), p=integer(), d=integer(), q=integer(), stringsAsFactors=FALSE)
sarima_metrics <- data.frame(Country=character(), RMSE=numeric(), MSE=numeric(), MAE=numeric(), MASE=numeric(), p=integer(), d=integer(), q=integer(), P=integer(), D=integer(), Q=integer(), stringsAsFactors=FALSE)

for (country in countries) {
  rmse <- get(paste0(country, "_rmse_results"))
  mse <- get(paste0(country, "_mse_results"))
  mae <- get(paste0(country, "_mae_results"))
  mase <- get(paste0(country, "_mase_results"))
  arima_country_metrics <- data.frame(Country = country, RMSE = rmse, MSE = mse, MAE = mae, MASE = mase)
  arima_metrics <- rbind(arima_metrics, arima_country_metrics)
}

# adding column
arima_metrics <- arima_metrics %>%
  mutate(Model_Type = "ARIMA") %>% 
  select(Country, Model_Type, RMSE, MSE, MASE, MAE)

# printing the final metrics data frame
print(arima_metrics)


## sarima ----

for (country in sarima_countries) {
  rmse <- get(paste0(country, "_rmse_results_sarima"))
  mse <- get(paste0(country, "_mse_results_sarima"))
  mae <- get(paste0(country, "_mae_results_sarima"))
  mase <- get(paste0(country, "_mase_results_sarima"))
  sarima_country_metrics <- data.frame(Country = country, RMSE = rmse, MSE = mse, MAE = mae, MASE = mase)
  sarima_metrics <- rbind(sarima_metrics, sarima_country_metrics)
}

# adding column
sarima_metrics <- sarima_metrics %>%
  mutate(Model_Type = "SARIMA") %>% 
  select(Country, Model_Type, RMSE, MSE, MASE, MAE)

# printing the final metrics data frame
print(sarima_metrics)


## auto-arima ----

# initializing an empty data frame to store the metrics for all countries
auto_arima_metrics <- data.frame(Country=character(), RMSE=numeric(), MSE=numeric(), MAE=numeric(), MASE=numeric(), stringsAsFactors=FALSE)

# looping through the results to calculate metrics and accumulate them
for(country_name in names(results)) {
  
  # retrieving forecasted values and test data for the current country
  forecast_values <- results[[country_name]]$forecast
  test_data <- results[[country_name]]$test_data
  
  # calculating metrics
  errors <- forecast_values$mean - test_data$owid_new_deaths
  rmse_value <- sqrt(mean(errors^2))
  mse_value <- mean(errors^2)
  mae_value <- mean(abs(errors))
  mase_value <- mean(abs(errors)) / mean(abs(diff(test_data$owid_new_deaths)))
  
  # creating a data frame row for the current country's metrics
  country_metrics <- data.frame(Country = country_name, RMSE = rmse_value, MSE = mse_value, MAE = mae_value, MASE = mase_value)
  
  # accumulating the metrics in the main data frame
  auto_arima_metrics <- rbind(auto_arima_metrics, country_metrics)
}

# adding column
auto_arima_metrics <- auto_arima_metrics %>%
  mutate(Model_Type = "Auto-ARIMA") %>% 
  select(Country, Model_Type, RMSE, MSE, MASE, MAE)

# printing the final metrics data frame
print(auto_arima_metrics)



# displaying best models ----

# binding previously made tables
arima_all_family_models <- rbind(arima_metrics, sarima_metrics, auto_arima_metrics) %>% 
  arrange(RMSE)

# removing row names
row.names(arima_all_family_models) <- NULL


## best per country ----

# grouping by country and selecting the best model by RMSE
best_arima_family_models <- arima_all_family_models %>%
  group_by(Country) %>%
  filter(RMSE == min(RMSE)) %>%
  ungroup()

# removing row names for a cleaner look
row.names(best_arima_family_models) <- NULL


## hyperparameters ----

### ARIMA ----

arima_best_params <- list(
  list(Country = "saudi", Model_Type = "ARIMA", p = "0", d = "1", q = "1"),
  list(Country = "bolivia", Model_Type = "ARIMA", p = "0", d = "2", q = "1"),
  list(Country = "colombia", Model_Type = "ARIMA", p = "2", d = "1", q = "2"),
  list(Country = "peru", Model_Type = "ARIMA", p = "0", d = "1", q = "0")
)

# converting the list of lists into a data frame
arima_best_params_df <- do.call(rbind, lapply(arima_best_params, as.data.frame, stringsAsFactors = FALSE))

# row names as numbers
row.names(arima_best_params_df) <- NULL

### SARIMA ----
sarima_best_params <- list(
  list(Country = "russia", Model_Type = "SARIMA", p = "0", d = "1", q = "0", P = "0", D = "1", Q = "2", period = "7"),
  list(Country = "brazil", Model_Type = "SARIMA", p = "2", d = "0", q = "2", P = "1", D = "1", Q = "1", period = "7"),
  list(Country = "us", Model_Type = "SARIMA", p = "1", d = "0", q = "0", P = "1", D = "0", Q = "1", period = "7")
)
sarima_best_params_df <- do.call(rbind, lapply(sarima_best_params, as.data.frame, stringsAsFactors = FALSE))
row.names(sarima_best_params_df) <- NULL

### Auto-ARIMA ----
auto_arima_best_params <- list(
  list(Country = "turkey", Model_Type = "Auto-ARIMA", p = "1", d = "1", q = "1"),
  list(Country = "iran", Model_Type = "Auto-ARIMA", p = "1", d = "1", q = "3"),
  list(Country = "mexico", Model_Type = "Auto-ARIMA", p = "1", d = "1", q = "3")
)
auto_arima_best_params_df <- do.call(rbind, lapply(auto_arima_best_params, as.data.frame, stringsAsFactors = FALSE))
row.names(auto_arima_best_params_df) <- NULL

### combining the models and parameters data frames ----
arima_combined_models_df <- best_arima_family_models %>%
  full_join(arima_best_params_df, by = "Country") %>%
  full_join(sarima_best_params_df, by = "Country") %>%
  full_join(auto_arima_best_params_df, by = "Country")

# identifying Model_Type columns if they've been suffixed
model_type_cols <- names(arima_combined_models_df)[grepl("Model_Type", names(arima_combined_models_df))]

# choosing the first Model_Type
if (length(model_type_cols) > 1) {
  arima_combined_models_df <- arima_combined_models_df %>%
    rename(Model_Type = !!model_type_cols[1]) %>%
    select(-one_of(model_type_cols[-1]))
} else if (length(model_type_cols) == 1) {
  arima_combined_models_df <- arima_combined_models_df %>%
    rename(Model_Type = !!model_type_cols[1])
}

# cleaning up any other inconsistencies
arima_combined_models_df <- arima_combined_models_df %>%
  distinct() %>%
  arrange(Country)

# displaying the cleaned-up data frame
print(arima_combined_models_df)

# merging the columns with coalesce
arima_combined_models_df <- arima_combined_models_df %>%
  mutate(
    p = coalesce(p.x, p.y, p),
    d = coalesce(d.x, d.y, d),
    q = coalesce(q.x, q.y, q)
  ) %>%
  select(Country, Model_Type, RMSE, MSE, MAE, MASE, p, d, q, P, D, Q, period) %>%
  arrange(Country) 

# reordering
arima_all_family_models <- arima_all_family_models %>% 
  select(Country, Model_Type, RMSE, MSE, MAE, MASE)



# saving files ----
save(arima_all_family_models, file = "data_frames/arima_all_family_models_maria.rda")
save(arima_combined_models_df, file = "data_frames/arima_combined_models_df_maria.rda")
