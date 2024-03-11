### COPY FOR CODING PURPOSES
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

bolivia_copy <- bolivia
save(bolivia_copy, file = "data/preprocessed/univariate/not_split/bolivia_copy.rda")


## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
bolivia_copy_total_days <- nrow(bolivia_copy)
bolivia_copy_train_days <- ceiling(0.9 * bolivia_copy_total_days)
bolivia_copy_test_days <- ceiling((bolivia_copy_total_days - bolivia_copy_train_days))

# creating folds
bolivia_copy_folds <- time_series_cv(
  bolivia_copy,
  date_var = date,
  initial = bolivia_copy_train_days,
  assess = bolivia_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
bolivia_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
bolivia_copy_rmse_results <- numeric(length(bolivia_copy_folds$splits))
bolivia_copy_mae_results <- numeric(length(bolivia_copy_folds$splits))
bolivia_copy_mse_results <- numeric(length(bolivia_copy_folds$splits))
bolivia_copy_mase_results <- numeric(length(bolivia_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(bolivia_copy_folds$splits)) {
  fold <- bolivia_copy_folds$splits[[i]]
  bolivia_copy_train_data <- fold$data[fold$in_id, ]
  bolivia_copy_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  bolivia_copy_arima_model <- arima(bolivia_copy_train_data$owid_new_deaths,
                               order = c(0, 1, 1))
  
  # forecasting with ARIMA
  bolivia_copy_forecast_values <- forecast(bolivia_copy_arima_model, h = nrow(bolivia_copy_test_data))
  
  # enforcing non-negativity on forecasted values
  bolivia_copy_forecast_values$mean <- pmax(bolivia_copy_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  bolivia_copy_errors <- bolivia_copy_forecast_values$mean - bolivia_copy_test_data$owid_new_deaths
  bolivia_copy_rmse_results[i] <- sqrt(mean(bolivia_copy_errors^2))
  bolivia_copy_mae_results[i] <- mean(abs(bolivia_copy_errors))
  bolivia_copy_mse_results[i] <- mean(bolivia_copy_errors^2)
  
  # calculating MASE
  bolivia_copy_mean_train_diff <- mean(abs(diff(bolivia_copy_train_data$owid_new_deaths)))
  bolivia_copy_mase_results[i] <- mean(abs(bolivia_copy_errors)) / bolivia_copy_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(bolivia_copy_rmse_results)))
print(paste("MAE:", mean(bolivia_copy_mae_results)))
print(paste("MSE:", mean(bolivia_copy_mse_results)))
print(paste("MASE:", mean(bolivia_copy_mase_results)))

# retrieving the fitted values for the training set
bolivia_copy_fitted_values <- fitted(bolivia_copy_arima_model)

# enforcing non-negativity on fitted values
bolivia_copy_fitted_values <- pmax(bolivia_copy_fitted_values, 0)

# combining training and test data for plotting
bolivia_copy_all_dates <- c(bolivia_copy_train_data$date, bolivia_copy_test_data$date)
bolivia_copy_all_values <- c(bolivia_copy_train_data$owid_new_deaths, bolivia_copy_test_data$owid_new_deaths)


## producing a plot ----
options(repr.plot.width = 10, repr.plot.height = 6)  # Adjust values as needed

# plotting actual values for both training and test data
plot(bolivia_copy_all_dates, bolivia_copy_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(bolivia_copy_test_data$date, bolivia_copy_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(bolivia_copy_train_data$date, bolivia_copy_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# brazil ----


## load data ----
load("data/preprocessed/univariate/not_split/brazil.rda")

brazil_copy <- brazil
save(brazil_copy, file = "data/preprocessed/univariate/not_split/brazil_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
brazil_copy_total_days <- nrow(brazil_copy)
brazil_copy_train_days <- ceiling(0.9 * brazil_copy_total_days)
brazil_copy_test_days <- ceiling((brazil_copy_total_days - brazil_copy_train_days))

# creating folds
brazil_copy_folds <- time_series_cv(
  brazil_copy,
  date_var = date,
  initial = brazil_copy_train_days,
  assess = brazil_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
brazil_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

### ARIMA
brazil_copy_arima_rmse_results <- numeric(length(brazil_copy_folds$splits))
brazil_copy_arima_mae_results <- numeric(length(brazil_copy_folds$splits))
brazil_copy_arima_mse_results <- numeric(length(brazil_copy_folds$splits))
brazil_copy_arima_mase_results <- numeric(length(brazil_copy_folds$splits))

# ARIMA : fitting to model and calculating metrics
for (i in seq_along(brazil_copy_folds$splits)) {
  arima_copy_fold <- brazil_copy_folds$splits[[i]]
  brazil_copy_arima_train_data <- arima_copy_fold$data[arima_copy_fold$in_id, ]
  brazil_copy_arima_test_data <- arima_copy_fold$data[arima_copy_fold$out_id, ]
  
  # fitting to ARIMA model
  brazil_copy_arima_model <- arima(brazil_copy_arima_train_data$owid_new_deaths,
                              order = c(0, 1, 1)) 
  
  # forecasting with ARIMA
  brazil_copy_arima_forecast_values <- forecast(brazil_copy_arima_model, h = nrow(brazil_copy_arima_test_data))
  
  # enforcing non-negativity on forecasted values
  brazil_copy_arima_forecast_values$mean <- pmax(brazil_copy_arima_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  brazil_copy_arima_errors <- brazil_copy_arima_forecast_values$mean - brazil_copy_arima_test_data$owid_new_deaths
  brazil_copy_arima_rmse_results[i] <- sqrt(mean(brazil_copy_arima_errors^2))
  brazil_copy_arima_mae_results[i] <- mean(abs(brazil_copy_arima_errors))
  brazil_copy_arima_mse_results[i] <- mean(brazil_copy_arima_errors^2)
  
  # calculating MASE
  brazil_copy_arima_mean_train_diff <- mean(abs(diff(brazil_copy_arima_train_data$owid_new_deaths)))
  brazil_copy_arima_mase_results[i] <- mean(abs(brazil_copy_arima_errors)) / brazil_copy_arima_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(brazil_copy_arima_rmse_results)))
print(paste("MAE:", mean(brazil_copy_arima_mae_results)))
print(paste("MSE:", mean(brazil_copy_arima_mse_results)))
print(paste("MASE:", mean(brazil_copy_arima_mase_results)))

# retrieving the fitted values for the training set
brazil_copy_arima_fitted_values <- fitted(brazil_copy_arima_model)

# enforcing non-negativity on fitted values
brazil_copy_arima_fitted_values <- pmax(brazil_copy_arima_fitted_values, 0)

# combining training and test data for plotting
brazil_copy_arima_all_dates <- c(brazil_copy_arima_train_data$date, brazil_copy_arima_test_data$date)
brazil_copy_arima_all_values <- c(brazil_copy_arima_train_data$owid_new_deaths, brazil_copy_arima_test_data$owid_new_deaths)

## producing a plot ----

options(repr.plot.width = 10, repr.plot.height = 6)
# plotting actual values for both training and test data
plot(brazil_copy_arima_all_dates, brazil_copy_arima_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(brazil_copy_arima_test_data$date, brazil_copy_arima_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(brazil_copy_arima_train_data$date, brazil_copy_arima_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)

##----------------

### SARIMA
# creating metrics vector
brazil_copy_sarima_rmse_results <- numeric(length(brazil_copy_folds$splits))
brazil_copy_sarima_mae_results <- numeric(length(brazil_copy_folds$splits))
brazil_copy_sarima_mse_results <- numeric(length(brazil_copy_folds$splits))
brazil_copy_sarima_mase_results <- numeric(length(brazil_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(brazil_copy_folds$splits)) {
  sarima_copy_fold <- brazil_copy_folds$splits[[i]]
  brazil_copy_sarima_train_data <- sarima_copy_fold$data[sarima_copy_fold$in_id, ]
  brazil_copy_sarima_test_data <- sarima_copy_fold$data[sarima_copy_fold$out_id, ]
  
  # fitting to ARIMA model
  brazil_copy_sarima_model <- arima(brazil_copy_sarima_train_data$owid_new_deaths,
                              order = c(0, 1, 1),
                              seasonal = list(order = c(0, 1, 0), period = 7)) #SARIMA bc saw a weekly pattern
  
  # forecasting with ARIMA
  brazil_copy_sarima_forecast_values <- forecast(brazil_copy_sarima_model, h = nrow(brazil_copy_sarima_test_data))
  
  # enforcing non-negativity on forecasted values
  brazil_copy_sarima_forecast_values$mean <- pmax(brazil_copy_sarima_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  brazil_copy_sarima_errors <- brazil_copy_sarima_forecast_values$mean - brazil_copy_sarima_test_data$owid_new_deaths
  brazil_copy_sarima_rmse_results[i] <- sqrt(mean(brazil_copy_sarima_errors^2))
  brazil_copy_sarima_mae_results[i] <- mean(abs(brazil_copy_sarima_errors))
  brazil_copy_sarima_mse_results[i] <- mean(brazil_copy_sarima_errors^2)
  
  # calculating MASE
  brazil_copy_sarima_mean_train_diff <- mean(abs(diff(brazil_copy_sarima_train_data$owid_new_deaths)))
  brazil_copy_sarima_mase_results[i] <- mean(abs(brazil_copy_sarima_errors)) / brazil_copy_sarima_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(brazil_copy_sarima_rmse_results)))
print(paste("MAE:", mean(brazil_copy_sarima_mae_results)))
print(paste("MSE:", mean(brazil_copy_sarima_mse_results)))
print(paste("MASE:", mean(brazil_copy_sarima_mase_results)))

# retrieving the fitted values for the training set
brazil_copy_sarima_fitted_values <- fitted(brazil_copy_sarima_model)

# enforcing non-negativity on fitted values
brazil_copy_sarima_fitted_values <- pmax(brazil_copy_sarima_fitted_values, 0)

# combining training and test data for plotting
brazil_copy_sarima_all_dates <- c(brazil_copy_sarima_train_data$date, brazil_copy_sarima_test_data$date)
brazil_copy_sarima_all_values <- c(brazil_copy_sarima_train_data$owid_new_deaths, brazil_copy_sarima_test_data$owid_new_deaths)


## producing a plot ----
options(repr.plot.width = 10, repr.plot.height = 6)
# plotting actual values for both training and test data
plot(brazil_copy_sarima_all_dates, brazil_copy_sarima_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(brazil_copy_sarima_test_data$date, brazil_copy_sarima_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(brazil_copy_sarima_train_data$date, brazil_copy_sarima_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# colombia ----


## load data ----
load("data/preprocessed/univariate/not_split/colombia.rda")

colombia_copy <- colombia
save(colombia_copy, file = "data/preprocessed/univariate/not_split/colombia_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
colombia_copy_total_days <- nrow(colombia_copy)
colombia_copy_train_days <- ceiling(0.9 * colombia_copy_total_days)
colombia_copy_test_days <- ceiling((colombia_copy_total_days - colombia_copy_train_days))

# creating folds
colombia_copy_folds <- time_series_cv(
  colombia_copy,
  date_var = date,
  initial = colombia_copy_train_days,
  assess = colombia_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
colombia_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
colombia_copy_rmse_results <- numeric(length(colombia_copy_folds$splits))
colombia_copy_mae_results <- numeric(length(colombia_copy_folds$splits))
colombia_copy_mse_results <- numeric(length(colombia_copy_folds$splits))
colombia_copy_mase_results <- numeric(length(colombia_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(colombia_copy_folds$splits)) {
  fold <- colombia_copy_folds$splits[[i]]
  colombia_copy_train_data <- fold$data[fold$in_id, ]
  colombia_copy_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  colombia_copy_arima_model <- arima(colombia_copy_train_data$owid_new_deaths,
                                order = c(0, 1, 0))
  
  # forecasting with ARIMA
  colombia_copy_forecast_values <- forecast(colombia_copy_arima_model, h = nrow(colombia_copy_test_data))
  
  # enforcing non-negativity on forecasted values
  colombia_copy_forecast_values$mean <- pmax(colombia_copy_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  colombia_copy_errors <- colombia_copy_forecast_values$mean - colombia_copy_test_data$owid_new_deaths
  colombia_copy_rmse_results[i] <- sqrt(mean(colombia_copy_errors^2))
  colombia_copy_mae_results[i] <- mean(abs(colombia_copy_errors))
  colombia_copy_mse_results[i] <- mean(colombia_copy_errors^2)
  
  # calculating MASE
  colombia_copy_mean_train_diff <- mean(abs(diff(colombia_copy_train_data$owid_new_deaths)))
  colombia_copy_mase_results[i] <- mean(abs(colombia_copy_errors)) / colombia_copy_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(colombia_copy_rmse_results)))
print(paste("MAE:", mean(colombia_copy_mae_results)))
print(paste("MSE:", mean(colombia_copy_mse_results)))
print(paste("MASE:", mean(colombia_copy_mase_results)))

# retrieving the fitted values for the training set
colombia_copy_fitted_values <- fitted(colombia_copy_arima_model)

# enforcing non-negativity on fitted values
colombia_copy_fitted_values <- pmax(colombia_copy_fitted_values, 0)

# combining training and test data for plotting
colombia_copy_all_dates <- c(colombia_copy_train_data$date, colombia_copy_test_data$date)
colombia_copy_all_values <- c(colombia_copy_train_data$owid_new_deaths, colombia_copy_test_data$owid_new_deaths)


## producing a plot ----
options(repr.plot.width = 10, repr.plot.height = 6)
# plotting actual values for both training and test data
plot(colombia_copy_all_dates, colombia_copy_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(colombia_copy_test_data$date, colombia_copy_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(colombia_copy_train_data$date, colombia_copy_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# iran ----


## load data ----
load("data/preprocessed/univariate/not_split/iran.rda")

iran_copy <- iran
save(colombia_copy, file = "data/preprocessed/univariate/not_split/iran_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
iran_copy_total_days <- nrow(iran_copy)
iran_copy_train_days <- ceiling(0.9 * iran_copy_total_days)
iran_copy_test_days <- ceiling((iran_copy_total_days - iran_copy_train_days))

# creating folds
iran_copy_folds <- time_series_cv(
  iran_copy,
  date_var = date,
  initial = iran_copy_train_days,
  assess = iran_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
iran_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
iran_copy_rmse_results <- numeric(length(iran_copy_folds$splits))
iran_copy_mae_results <- numeric(length(iran_copy_folds$splits))
iran_copy_mse_results <- numeric(length(iran_copy_folds$splits))
iran_copy_mase_results <- numeric(length(iran_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(iran_copy_folds$splits)) {
  fold <- iran_copy_folds$splits[[i]]
  iran_copy_train_data <- fold$data[fold$in_id, ]
  iran_copy_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  iran_copy_arima_model <- arima(iran_copy_train_data$owid_new_deaths,
                            order = c(1, 1, 0))
  
  # forecasting with ARIMA
  iran_copy_forecast_values <- forecast(iran_copy_arima_model, h = nrow(iran_copy_test_data))
  
  # enforcing non-negativity on forecasted values
  iran_copy_forecast_values$mean <- pmax(iran_copy_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  iran_copy_errors <- iran_copy_forecast_values$mean - iran_copy_test_data$owid_new_deaths
  iran_copy_rmse_results[i] <- sqrt(mean(iran_copy_errors^2))
  iran_copy_mae_results[i] <- mean(abs(iran_copy_errors))
  iran_copy_mse_results[i] <- mean(iran_copy_errors^2)
  
  # calculating MASE
  iran_copy_mean_train_diff <- mean(abs(diff(iran_copy_train_data$owid_new_deaths)))
  iran_copy_mase_results[i] <- mean(abs(iran_copy_errors)) / iran_copy_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(iran_copy_rmse_results)))
print(paste("MAE:", mean(iran_copy_mae_results)))
print(paste("MSE:", mean(iran_copy_mse_results)))
print(paste("MASE:", mean(iran_copy_mase_results)))

# retrieving the fitted values for the training set
iran_copy_fitted_values <- fitted(iran_copy_arima_model)

# enforcing non-negativity on fitted values
iran_copy_fitted_values <- pmax(iran_copy_fitted_values, 0)

# combining training and test data for plotting
iran_copy_all_dates <- c(iran_copy_train_data$date, iran_copy_test_data$date)
iran_copy_all_values <- c(iran_copy_train_data$owid_new_deaths, iran_copy_test_data$owid_new_deaths)


## producing a plot ----
options(repr.plot.width = 10, repr.plot.height = 6)
# plotting actual values for both training and test data
plot(iran_copy_all_dates, iran_copy_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(iran_copy_test_data$date, iran_copy_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(iran_copy_train_data$date, iran_copy_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# mexico ----


## load data ----
load("data/preprocessed/univariate/not_split/mexico.rda")

mexico_copy <- mexico
save(mexico_copy, file = "data/preprocessed/univariate/not_split/mexico_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
mexico_copy_total_days <- nrow(mexico_copy)
mexico_copy_train_days <- ceiling(0.9 * mexico_copy_total_days)
mexico_copy_test_days <- ceiling((mexico_copy_total_days - mexico_copy_train_days))

# creating folds
mexico_copy_folds <- time_series_cv(
  mexico_copy,
  date_var = date,
  initial = mexico_copy_train_days,
  assess = mexico_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
mexico_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
mexico_copy_arima_rmse_results <- numeric(length(mexico_copy_folds$splits))
mexico_copy_arima_mae_results <- numeric(length(mexico_copy_folds$splits))
mexico_copy_arima_mse_results <- numeric(length(mexico_copy_folds$splits))
mexico_copy_arima_mase_results <- numeric(length(mexico_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(mexico_copy_folds$splits)) {
  arima_copy_fold <- mexico_copy_folds$splits[[i]]
  mexico_copy_arima_train_data <- arima_copy_fold$data[arima_copy_fold$in_id, ]
  mexico_copy_arima_test_data <- arima_copy_fold$data[arima_copy_fold$out_id, ]
  
  # fitting to ARIMA model
  mexico_copy_arima_model <- arima(mexico_copy_arima_train_data$owid_new_deaths,
                              order = c(1, 0, 0)) #found weekly so changed to 7
  
  # forecasting with ARIMA
  mexico_copy_arima_forecast_values <- forecast(mexico_copy_arima_model, h = nrow(mexico_copy_arima_test_data))
  
  # enforcing non-negativity on forecasted values
  mexico_copy_arima_forecast_values$mean <- pmax(mexico_copy_arima_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  mexico_copy_arima_errors <- mexico_copy_arima_forecast_values$mean - mexico_copy_arima_test_data$owid_new_deaths
  mexico_copy_arima_rmse_results[i] <- sqrt(mean(mexico_copy_arima_errors^2))
  mexico_copy_arima_mae_results[i] <- mean(abs(mexico_copy_arima_errors))
  mexico_copy_arima_mse_results[i] <- mean(mexico_copy_arima_errors^2)
  
  # calculating MASE
  mexico_copy_arima_mean_train_diff <- mean(abs(diff(mexico_copy_arima_train_data$owid_new_deaths)))
  mexico_copy_arima_mase_results[i] <- mean(abs(mexico_copy_arima_errors)) / mexico_copy_arima_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(mexico_copy_arima_rmse_results)))
print(paste("MAE:", mean(mexico_copy_arima_mae_results)))
print(paste("MSE:", mean(mexico_copy_arima_mse_results)))
print(paste("MASE:", mean(mexico_copy_arima_mase_results)))

# retrieving the fitted values for the training set
mexico_copy_arima_fitted_values <- fitted(mexico_copy_arima_model)

# enforcing non-negativity on fitted values
mexico_copy_arima_fitted_values <- pmax(mexico_copy_arima_fitted_values, 0)

# combining training and test data for plotting
mexico_copy_arima_all_dates <- c(mexico_copy_arima_train_data$date, mexico_copy_arima_test_data$date)
mexico_copy_arima_all_values <- c(mexico_copy_arima_train_data$owid_new_deaths, mexico_copy_arima_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(mexico_copy_arima_all_dates, mexico_copy_arima_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(mexico_copy_arima_test_data$date, mexico_copy_arima_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(mexico_copy_arima_train_data$date, mexico_copy_arima_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)

### SARIMA --------------------

# creating metrics vector
mexico_copy_sarima_rmse_results <- numeric(length(mexico_copy_folds$splits))
mexico_copy_sarima_mae_results <- numeric(length(mexico_copy_folds$splits))
mexico_copy_sarima_mse_results <- numeric(length(mexico_copy_folds$splits))
mexico_copy_sarima_mase_results <- numeric(length(mexico_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(mexico_copy_folds$splits)) {
  sarima_copy_fold <- mexico_copy_folds$splits[[i]]
  mexico_copy_sarima_train_data <- sarima_copy_fold$data[sarima_copy_fold$in_id, ]
  mexico_copy_sarima_test_data <- sarima_copy_fold$data[sarima_copy_fold$out_id, ]
  
  # fitting to ARIMA model
  mexico_copy_sarima_model <- arima(mexico_copy_sarima_train_data$owid_new_deaths,
                              order = c(1, 0, 0),
                              seasonal = list(order = c(1, 1, 0), period = 7)) #found weekly so changed to 7
  
  # forecasting with ARIMA
  mexico_copy_sarima_forecast_values <- forecast(mexico_copy_sarima_model, h = nrow(mexico_copy_sarima_test_data))
  
  # enforcing non-negativity on forecasted values
  mexico_copy_sarima_forecast_values$mean <- pmax(mexico_copy_sarima_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  mexico_copy_sarima_errors <- mexico_copy_sarima_forecast_values$mean - mexico_copy_sarima_test_data$owid_new_deaths
  mexico_copy_sarima_rmse_results[i] <- sqrt(mean(mexico_copy_sarima_errors^2))
  mexico_copy_sarima_mae_results[i] <- mean(abs(mexico_copy_sarima_errors))
  mexico_copy_sarima_mse_results[i] <- mean(mexico_copy_sarima_errors^2)
  
  # calculating MASE
  mexico_copy_sarima_mean_train_diff <- mean(abs(diff(mexico_copy_sarima_train_data$owid_new_deaths)))
  mexico_copy_sarima_mase_results[i] <- mean(abs(mexico_copy_sarima_errors)) / mexico_copy_sarima_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(mexico_copy_sarima_rmse_results)))
print(paste("MAE:", mean(mexico_copy_sarima_mae_results)))
print(paste("MSE:", mean(mexico_copy_sarima_mse_results)))
print(paste("MASE:", mean(mexico_copy_sarima_mase_results)))

# retrieving the fitted values for the training set
mexico_copy_sarima_fitted_values <- fitted(mexico_copy_sarima_model)

# enforcing non-negativity on fitted values
mexico_copy_sarima_fitted_values <- pmax(mexico_copy_sarima_fitted_values, 0)

# combining training and test data for plotting
mexico_copy_sarima_all_dates <- c(mexico_copy_sarima_train_data$date, mexico_copy_sarima_test_data$date)
mexico_copy_sarima_all_values <- c(mexico_copy_sarima_train_data$owid_new_deaths, mexico_copy_sarima_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(mexico_copy_sarima_all_dates, mexico_copy_sarima_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(mexico_copy_sarima_test_data$date, mexico_copy_sarima_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(mexico_copy_sarima_train_data$date, mexico_copy_sarima_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# peru ----


## load data ----
load("data/preprocessed/univariate/not_split/peru.rda")

peru_copy <- peru
save(peru_copy, file = "data/preprocessed/univariate/not_split/peru_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
peru_copy_total_days <- nrow(peru_copy)
peru_copy_train_days <- ceiling(0.9 * peru_copy_total_days)
peru_copy_test_days <- ceiling((peru_copy_total_days - peru_copy_train_days))

# creating folds
peru_copy_folds <- time_series_cv(
  peru_copy,
  date_var = date,
  initial = peru_copy_train_days,
  assess = peru_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
peru_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
peru_copy_rmse_results <- numeric(length(peru_copy_folds$splits))
peru_copy_mae_results <- numeric(length(peru_copy_folds$splits))
peru_copy_mse_results <- numeric(length(peru_copy_folds$splits))
peru_copy_mase_results <- numeric(length(peru_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(peru_copy_folds$splits)) {
  fold <- peru_copy_folds$splits[[i]]
  peru_copy_train_data <- fold$data[fold$in_id, ]
  peru_copy_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  peru_copy_arima_model <- arima(peru_copy_train_data$owid_new_deaths,
                            order = c(0, 1, 1))
  
  # forecasting with ARIMA
  peru_copy_forecast_values <- forecast(peru_copy_arima_model, h = nrow(peru_copy_test_data))
  
  # enforcing non-negativity on forecasted values
  peru_copy_forecast_values$mean <- pmax(peru_copy_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  peru_copy_errors <- peru_copy_forecast_values$mean - peru_copy_test_data$owid_new_deaths
  peru_copy_rmse_results[i] <- sqrt(mean(peru_copy_errors^2))
  peru_copy_mae_results[i] <- mean(abs(peru_copy_errors))
  peru_copy_mse_results[i] <- mean(peru_copy_errors^2)
  
  # calculating MASE
  peru_copy_mean_train_diff <- mean(abs(diff(peru_copy_train_data$owid_new_deaths)))
  peru_copy_mase_results[i] <- mean(abs(peru_copy_errors)) / peru_copy_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(peru_copy_rmse_results)))
print(paste("MAE:", mean(peru_copy_mae_results)))
print(paste("MSE:", mean(peru_copy_mse_results)))
print(paste("MASE:", mean(peru_copy_mase_results)))

# retrieving the fitted values for the training set
peru_copy_fitted_values <- fitted(peru_copy_arima_model)

# enforcing non-negativity on fitted values
peru_copy_fitted_values <- pmax(peru_copy_fitted_values, 0)

# combining training and test data for plotting
peru_copy_all_dates <- c(peru_copy_train_data$date, peru_copy_test_data$date)
peru_copy_all_values <- c(peru_copy_train_data$owid_new_deaths, peru_copy_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(peru_copy_all_dates, peru_copy_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(peru_copy_test_data$date, peru_copy_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(peru_copy_train_data$date, peru_copy_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# russia ----


## load data ----
load("data/preprocessed/univariate/not_split/russia.rda")

russia_copy <- russia
save(russia_copy, file = "data/preprocessed/univariate/not_split/russia_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
russia_copy_total_days <- nrow(russia_copy)
russia_copy_train_days <- ceiling(0.9 * russia_copy_total_days)
russia_copy_test_days <- ceiling((russia_copy_total_days - russia_copy_train_days))

# creating folds
russia_copy_folds <- time_series_cv(
  russia_copy,
  date_var = date,
  initial = russia_copy_train_days,
  assess = russia_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
russia_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
russia_copy_arima_rmse_results <- numeric(length(russia_copy_folds$splits))
russia_copy_arima_mae_results <- numeric(length(russia_copy_folds$splits))
russia_copy_arima_mse_results <- numeric(length(russia_copy_folds$splits))
russia_copy_arima_mase_results <- numeric(length(russia_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(russia_copy_folds$splits)) {
  arima_copy_fold <- russia_copy_folds$splits[[i]]
  russia_copy_arima_train_data <- arima_copy_fold$data[arima_copy_fold$in_id, ]
  russia_copy_arima_test_data <- arima_copy_fold$data[arima_copy_fold$out_id, ]
  
  # fitting to ARIMA model
  russia_copy_arima_model <- arima(russia_copy_arima_train_data$owid_new_deaths,
                              order = c(1, 1, 0)) 
  
  # forecasting with ARIMA
  russia_copy_arima_forecast_values <- forecast(russia_copy_arima_model, h = nrow(russia_copy_arima_test_data))
  
  # enforcing non-negativity on forecasted values
  russia_copy_arima_forecast_values$mean <- pmax(russia_copy_arima_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  russia_copy_arima_errors <- russia_copy_arima_forecast_values$mean - russia_copy_arima_test_data$owid_new_deaths
  russia_copy_arima_rmse_results[i] <- sqrt(mean(russia_copy_arima_errors^2))
  russia_copy_arima_mae_results[i] <- mean(abs(russia_copy_arima_errors))
  russia_copy_arima_mse_results[i] <- mean(russia_copy_arima_errors^2)
  
  # calculating MASE
  russia_copy_arima_mean_train_diff <- mean(abs(diff(russia_copy_arima_train_data$owid_new_deaths)))
  russia_copy_arima_mase_results[i] <- mean(abs(russia_copy_arima_errors)) / russia_copy_arima_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(russia_copy_arima_rmse_results)))
print(paste("MAE:", mean(russia_copy_arima_mae_results)))
print(paste("MSE:", mean(russia_copy_arima_mse_results)))
print(paste("MASE:", mean(russia_copy_arima_mase_results)))

# retrieving the fitted values for the training set
russia_copy_arima_fitted_values <- fitted(russia_copy_arima_model)

# enforcing non-negativity on fitted values
russia_copy_arima_fitted_values <- pmax(russia_copy_arima_fitted_values, 0)

# combining training and test data for plotting
russia_copy_arima_all_dates <- c(russia_copy_arima_train_data$date, russia_copy_arima_test_data$date)
russia_copy_arima_all_values <- c(russia_copy_arima_train_data$owid_new_deaths, russia_copy_arima_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(russia_copy_arima_all_dates, russia_copy_arima_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(russia_copy_arima_test_data$date, russia_copy_arima_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(russia_copy_arima_train_data$date, russia_copy_arima_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)

### SARIMA ---------------------------

# creating metrics vector
russia_copy_sarima_rmse_results <- numeric(length(russia_copy_folds$splits))
russia_copy_sarima_mae_results <- numeric(length(russia_copy_folds$splits))
russia_copy_sarima_mse_results <- numeric(length(russia_copy_folds$splits))
russia_copy_sarima_mase_results <- numeric(length(russia_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(russia_copy_folds$splits)) {
  sarima_copy_fold <- russia_copy_folds$splits[[i]]
  russia_copy_sarima_train_data <- sarima_copy_fold$data[sarima_copy_fold$in_id, ]
  russia_copy_sarima_test_data <- sarima_copy_fold$data[sarima_copy_fold$out_id, ]
  
  # fitting to ARIMA model
  russia_copy_sarima_model <- arima(russia_copy_sarima_train_data$owid_new_deaths,
                              order = c(1, 1, 0),
                              seasonal = list(order = c(0, 1, 0), period = 7)) # changed to 7 (per feedback)
  
  # forecasting with ARIMA
  russia_copy_sarima_forecast_values <- forecast(russia_copy_sarima_model, h = nrow(russia_copy_sarima_test_data))
  
  # enforcing non-negativity on forecasted values
  russia_copy_sarima_forecast_values$mean <- pmax(russia_copy_sarima_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  russia_copy_sarima_errors <- russia_copy_sarima_forecast_values$mean - russia_copy_sarima_test_data$owid_new_deaths
  russia_copy_sarima_rmse_results[i] <- sqrt(mean(russia_copy_sarima_errors^2))
  russia_copy_sarima_mae_results[i] <- mean(abs(russia_copy_sarima_errors))
  russia_copy_sarima_mse_results[i] <- mean(russia_copy_sarima_errors^2)
  
  # calculating MASE
  russia_copy_sarima_mean_train_diff <- mean(abs(diff(russia_copy_sarima_train_data$owid_new_deaths)))
  russia_copy_sarima_mase_results[i] <- mean(abs(russia_copy_sarima_errors)) / russia_copy_sarima_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(russia_copy_sarima_rmse_results)))
print(paste("MAE:", mean(russia_copy_sarima_mae_results)))
print(paste("MSE:", mean(russia_copy_sarima_mse_results)))
print(paste("MASE:", mean(russia_copy_sarima_mase_results)))

# retrieving the fitted values for the training set
russia_copy_sarima_fitted_values <- fitted(russia_copy_sarima_model)

# enforcing non-negativity on fitted values
russia_copy_sarima_fitted_values <- pmax(russia_copy_sarima_fitted_values, 0)

# combining training and test data for plotting
russia_copy_sarima_all_dates <- c(russia_copy_sarima_train_data$date, russia_copy_sarima_test_data$date)
russia_copy_sarima_all_values <- c(russia_copy_sarima_train_data$owid_new_deaths, russia_copy_sarima_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(russia_copy_sarima_all_dates, russia_copy_sarima_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(russia_copy_sarima_test_data$date, russia_copy_sarima_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(russia_copy_sarima_train_data$date, russia_copy_sarima_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)

# saudi ----


## load data ----
load("data/preprocessed/univariate/not_split/saudi.rda")

saudi_copy <- saudi
save(saudi_copy, file = "data/preprocessed/univariate/not_split/saudi_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
saudi_copy_total_days <- nrow(saudi_copy)
saudi_copy_train_days <- ceiling(0.9 * saudi_copy_total_days)
saudi_copy_test_days <- ceiling((saudi_copy_total_days - saudi_copy_train_days))

# creating folds
saudi_copy_folds <- time_series_cv(
  saudi_copy,
  date_var = date,
  initial = saudi_copy_train_days,
  assess = saudi_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
saudi_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
saudi_copy_rmse_results <- numeric(length(saudi_copy_folds$splits))
saudi_copy_mae_results <- numeric(length(saudi_copy_folds$splits))
saudi_copy_mse_results <- numeric(length(saudi_copy_folds$splits))
saudi_copy_mase_results <- numeric(length(saudi_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(saudi_copy_folds$splits)) {
  fold <- saudi_copy_folds$splits[[i]]
  saudi_copy_train_data <- fold$data[fold$in_id, ]
  saudi_copy_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  saudi_copy_arima_model <- arima(saudi_copy_train_data$owid_new_deaths,
                             order = c(1, 1, 0))
  
  # forecasting with ARIMA
  saudi_copy_forecast_values <- forecast(saudi_copy_arima_model, h = nrow(saudi_copy_test_data))
  
  # enforcing non-negativity on forecasted values
  saudi_copy_forecast_values$mean <- pmax(saudi_copy_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  saudi_copy_errors <- saudi_copy_forecast_values$mean - saudi_copy_test_data$owid_new_deaths
  saudi_copy_rmse_results[i] <- sqrt(mean(saudi_copy_errors^2))
  saudi_copy_mae_results[i] <- mean(abs(saudi_copy_errors))
  saudi_copy_mse_results[i] <- mean(saudi_copy_errors^2)
  
  # calculating MASE
  saudi_copy_mean_train_diff <- mean(abs(diff(saudi_copy_train_data$owid_new_deaths)))
  saudi_copy_mase_results[i] <- mean(abs(saudi_copy_errors)) / saudi_copy_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(saudi_copy_rmse_results)))
print(paste("MAE:", mean(saudi_copy_mae_results)))
print(paste("MSE:", mean(saudi_copy_mse_results)))
print(paste("MASE:", mean(saudi_copy_mase_results)))

# retrieving the fitted values for the training set
saudi_copy_fitted_values <- fitted(saudi_copy_arima_model)

# enforcing non-negativity on fitted values
saudi_copy_fitted_values <- pmax(saudi_copy_fitted_values, 0)

# combining training and test data for plotting
saudi_copy_all_dates <- c(saudi_copy_train_data$date, saudi_copy_test_data$date)
saudi_copy_all_values <- c(saudi_copy_train_data$owid_new_deaths, saudi_copy_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(saudi_copy_all_dates, saudi_copy_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(saudi_copy_test_data$date, saudi_copy_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(saudi_copy_train_data$date, saudi_copy_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# turkey ----


## load data ----
load("data/preprocessed/univariate/not_split/turkey.rda")

turkey_copy <- turkey
save(turkey_copy, file = "data/preprocessed/univariate/not_split/turkey_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
turkey_copy_total_days <- nrow(turkey_copy)
turkey_copy_train_days <- ceiling(0.9 * turkey_copy_total_days)
turkey_copy_test_days <- ceiling((turkey_copy_total_days - turkey_copy_train_days))

# creating folds
turkey_copy_folds <- time_series_cv(
  turkey_copy,
  date_var = date,
  initial = turkey_copy_train_days,
  assess = turkey_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
turkey_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
turkey_copy_rmse_results <- numeric(length(turkey_copy_folds$splits))
turkey_copy_mae_results <- numeric(length(turkey_copy_folds$splits))
turkey_copy_mse_results <- numeric(length(turkey_copy_folds$splits))
turkey_copy_mase_results <- numeric(length(turkey_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(turkey_copy_folds$splits)) {
  fold <- turkey_copy_folds$splits[[i]]
  turkey_copy_train_data <- fold$data[fold$in_id, ]
  turkey_copy_test_data <- fold$data[fold$out_id, ]
  
  # fitting to ARIMA model
  turkey_copy_arima_model <- arima(turkey_copy_train_data$owid_new_deaths,
                              order = c(0, 1, 0))
  
  # forecasting with ARIMA
  turkey_copy_forecast_values <- forecast(turkey_copy_arima_model, h = nrow(turkey_copy_test_data))
  
  # enforcing non-negativity on forecasted values
  turkey_copy_forecast_values$mean <- pmax(turkey_copy_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  turkey_copy_errors <- turkey_copy_forecast_values$mean - turkey_copy_test_data$owid_new_deaths
  turkey_copy_rmse_results[i] <- sqrt(mean(turkey_copy_errors^2))
  turkey_copy_mae_results[i] <- mean(abs(turkey_copy_errors))
  turkey_copy_mse_results[i] <- mean(turkey_copy_errors^2)
  
  # calculating MASE
  turkey_copy_mean_train_diff <- mean(abs(diff(turkey_copy_train_data$owid_new_deaths)))
  turkey_copy_mase_results[i] <- mean(abs(turkey_copy_errors)) / turkey_copy_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(turkey_copy_rmse_results)))
print(paste("MAE:", mean(turkey_copy_mae_results)))
print(paste("MSE:", mean(turkey_copy_mse_results)))
print(paste("MASE:", mean(turkey_copy_mase_results)))

# retrieving the fitted values for the training set
turkey_copy_fitted_values <- fitted(turkey_copy_arima_model)

# enforcing non-negativity on fitted values
turkey_copy_fitted_values <- pmax(turkey_copy_fitted_values, 0)

# combining training and test data for plotting
turkey_copy_all_dates <- c(turkey_copy_train_data$date, turkey_copy_test_data$date)
turkey_copy_all_values <- c(turkey_copy_train_data$owid_new_deaths, turkey_copy_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(turkey_copy_all_dates, turkey_copy_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(turkey_copy_test_data$date, turkey_copy_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(turkey_copy_train_data$date, turkey_copy_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# us ----


## load data ----
load("data/preprocessed/univariate/not_split/us.rda")

us_copy <- us
save(us_copy, file = "data/preprocessed/univariate/not_split/us_copy.rda")
## splitting into training and testing sets----

# calculating the number of dates for time_series_cv parameters
us_copy_total_days <- nrow(us_copy)
us_copy_train_days <- ceiling(0.9 * us_copy_total_days)
us_copy_test_days <- ceiling((us_copy_total_days - us_copy_train_days))

# creating folds
us_copy_folds <- time_series_cv(
  us_copy,
  date_var = date,
  initial = us_copy_train_days,
  assess = us_copy_test_days,
  fold = 1,
  slice_limit = 1)

# filtering by slice
us_copy_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()


## applying ARIMA model ----

# creating metrics vector
us_copy_arima_rmse_results <- numeric(length(us_copy_folds$splits))
us_copy_arima_mae_results <- numeric(length(us_copy_folds$splits))
us_copy_arima_mse_results <- numeric(length(us_copy_folds$splits))
us_copy_arima_mase_results <- numeric(length(us_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(us_copy_folds$splits)) {
  arima_copy_fold <- us_copy_folds$splits[[i]]
  us_copy_arima_train_data <- arima_copy_fold$data[arima_copy_fold$in_id, ]
  us_copy_arima_test_data <- arima_copy_fold$data[arima_copy_fold$out_id, ]
  
  # fitting to ARIMA model
  us_copy_arima_model <- arima(us_copy_arima_train_data$owid_new_deaths,
                          order = c(0, 0, 0))
  
  # forecasting with ARIMA
  us_copy_arima_forecast_values <- forecast(us_copy_arima_model, h = nrow(us_copy_arima_test_data))
  
  # enforcing non-negativity on forecasted values
  us_copy_arima_forecast_values$mean <- pmax(us_copy_arima_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  us_copy_arima_errors <- us_copy_arima_forecast_values$mean - us_copy_arima_test_data$owid_new_deaths
  us_copy_arima_rmse_results[i] <- sqrt(mean(us_copy_arima_errors^2))
  us_copy_arima_mae_results[i] <- mean(abs(us_copy_arima_errors))
  us_copy_arima_mse_results[i] <- mean(us_copy_arima_errors^2)
  
  # calculating MASE
  us_copy_arima_mean_train_diff <- mean(abs(diff(us_copy_arima_train_data$owid_new_deaths)))
  us_copy_arima_mase_results[i] <- mean(abs(us_copy_arima_errors)) / us_copy_arima_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(us_copy_arima_rmse_results)))
print(paste("MAE:", mean(us_copy_arima_mae_results)))
print(paste("MSE:", mean(us_copy_arima_mse_results)))
print(paste("MASE:", mean(us_copy_arima_mase_results)))

# retrieving the fitted values for the training set
us_copy_arima_fitted_values <- fitted(us_copy_arima_model)

# enforcing non-negativity on fitted values
us_copy_arima_fitted_values <- pmax(us_copy_arima_fitted_values, 0)

# combining training and test data for plotting
us_copy_arima_all_dates <- c(us_copy_arima_train_data$date, us_copy_arima_test_data$date)
us_copy_arima_all_values <- c(us_copy_arima_train_data$owid_new_deaths, us_copy_arima_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(us_copy_arima_all_dates, us_copy_arima_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(us_copy_arima_test_data$date, us_copy_arima_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(us_copy_arima_train_data$date, us_copy_arima_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)

### SARIMA ---------------------------

# creating metrics vector
us_copy_sarima_rmse_results <- numeric(length(us_copy_folds$splits))
us_copy_sarima_mae_results <- numeric(length(us_copy_folds$splits))
us_copy_sarima_mse_results <- numeric(length(us_copy_folds$splits))
us_copy_sarima_mase_results <- numeric(length(us_copy_folds$splits))

# fitting to model and calculating metrics
for (i in seq_along(us_copy_folds$splits)) {
  sarima_copy_fold <- us_copy_folds$splits[[i]]
  us_copy_sarima_train_data <- sarima_copy_fold$data[sarima_copy_fold$in_id, ]
  us_copy_sarima_test_data <- sarima_copy_fold$data[sarima_copy_fold$out_id, ]
  
  # fitting to ARIMA model
  us_copy_sarima_model <- arima(us_copy_sarima_train_data$owid_new_deaths,
                          order = c(0, 0, 0),
                          seasonal = list(order = c(1, 1, 1), period = 7))
  
  # forecasting with ARIMA
  us_copy_sarima_forecast_values <- forecast(us_copy_sarima_model, h = nrow(us_copy_sarima_test_data))
  
  # enforcing non-negativity on forecasted values
  us_copy_sarima_forecast_values$mean <- pmax(us_copy_sarima_forecast_values$mean, 0)
  
  # calculating evaluation metrics
  us_copy_sarima_errors <- us_copy_sarima_forecast_values$mean - us_copy_sarima_test_data$owid_new_deaths
  us_copy_sarima_rmse_results[i] <- sqrt(mean(us_copy_sarima_errors^2))
  us_copy_sarima_mae_results[i] <- mean(abs(us_copy_sarima_errors))
  us_copy_sarima_mse_results[i] <- mean(us_copy_sarima_errors^2)
  
  # calculating MASE
  us_copy_sarima_mean_train_diff <- mean(abs(diff(us_copy_sarima_train_data$owid_new_deaths)))
  us_copy_sarima_mase_results[i] <- mean(abs(us_copy_sarima_errors)) / us_copy_sarima_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(us_copy_sarima_rmse_results)))
print(paste("MAE:", mean(us_copy_sarima_mae_results)))
print(paste("MSE:", mean(us_copy_sarima_mse_results)))
print(paste("MASE:", mean(us_copy_sarima_mase_results)))

# retrieving the fitted values for the training set
us_copy_sarima_fitted_values <- fitted(us_copy_sarima_model)

# enforcing non-negativity on fitted values
us_copy_sarima_fitted_values <- pmax(us_copy_sarima_fitted_values, 0)

# combining training and test data for plotting
us_copy_sarima_all_dates <- c(us_copy_sarima_train_data$date, us_copy_sarima_test_data$date)
us_copy_sarima_all_values <- c(us_copy_sarima_train_data$owid_new_deaths, us_copy_sarima_test_data$owid_new_deaths)


## producing a plot ----

# plotting actual values for both training and test data
plot(us_copy_sarima_all_dates, us_copy_sarima_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(us_copy_sarima_test_data$date, us_copy_sarima_forecast_values$mean, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(us_copy_sarima_train_data$date, us_copy_sarima_fitted_values, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)

# # finding best parameters per country ----
# 
# ## load data ----
# load("data/preprocessed/univariate/not_split/us.rda")
# 
# 
# ## splitting into training and testing sets ----
# 
# # calculating the number of dates for time_series_cv parameters
# total_days <- nrow(us)
# train_days <- ceiling(0.9 * total_days)
# test_days <- ceiling((total_days - train_days))
# 
# # creating folds
# us_folds <- time_series_cv(
#   us,
#   date_var = date,
#   initial = train_days,
#   assess = test_days,
#   fold = 1,
#   slice_limit = 1)
# 
# # filtering by slice
# us_folds %>% tk_time_series_cv_plan() %>%
#   filter(.id == "Slice2") %>%
#   nrow()
# 
# # defining grid of parameter values to search over
# orders <- c(0, 1, 2)
# seasonals <- list(order = c(0, 1), period = 7)
# 
# # initializing parameters
# min_rmse <- Inf
# best_order <- NULL
# best_seasonal <- NULL
# 
# # iterating over all combinations
# for (order_i in c(0, 1)) {
#   for (order_ii in c(0, 1)) {
#     for (order_iii in c(0, 1)) {
#       for (seasonal_order_i in c(0, 1)) {
#         for (seasonal_order_ii in c(0, 1)) {
#           for (seasonal_order_iii in c(0, 1)) {
#             for (seasonal_period in 1:14) {
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
#                 arima_model <- tryCatch({
#                   arima(train_data$owid_new_deaths,
#                         order = c(order_i, order_ii, order_iii), # Use dynamic order
#                         seasonal = seasonal)
#                 }, error = function(e) {
#                   NULL
#                 })
# 
#                 if (!is.null(arima_model)) {
# 
#                   # forecasting with ARIMA
#                   forecast_values <- forecast(arima_model, h = nrow(test_data))
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



# producing training and testing separate plots ----


## bolivia ----

# producing fitted training set plot
plot(bolivia_copy_train_data$date, bolivia_copy_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Bolivia", "Training Set"))
lines(bolivia_copy_train_data$date, bolivia_copy_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(bolivia_copy_test_data$date, bolivia_copy_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Bolivia", "Test Set"))
lines(bolivia_copy_test_data$date, bolivia_copy_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## brazil ----

# producing fitted training set plot
plot(brazil_copy_arima_train_data$date, brazil_copy_arima_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Brazil", "Training Set"))
lines(brazil_copy_arima_train_data$date, brazil_copy_arima_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(brazil_copy_arima_test_data$date, brazil_copy_arima_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Brazil", "Test Set"))
lines(brazil_copy_arima_test_data$date, brazil_copy_arima_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

### SARIMA -----

plot(brazil_copy_sarima_train_data$date, brazil_copy_sarima_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Brazil", "Training Set"))
lines(brazil_copy_sarima_train_data$date, brazil_copy_sarima_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(brazil_copy_sarima_test_data$date, brazil_copy_sarima_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Brazil", "Test Set"))
lines(brazil_copy_sarima_test_data$date, brazil_copy_sarima_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## iran ----

# producing fitted training set plot
plot(iran_copy_train_data$date, iran_copy_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Iran", "Training Set"))
lines(iran_copy_train_data$date, iran_copy_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(iran_copy_test_data$date, iran_copy_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Iran", "Test Set"))
lines(iran_copy_test_data$date, iran_copy_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## mexico ----

# producing fitted training set plot
plot(mexico_copy_arima_train_data$date, mexico_copy_arima_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Mexico", "Training Set"))
lines(mexico_copy_arima_train_data$date, mexico_copy_arima_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(mexico_copy_arima_test_data$date, mexico_copy_arima_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Mexico", "Test Set"))
lines(mexico_copy_arima_test_data$date, mexico_copy_arima_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

### SARIMA ----------

plot(mexico_copy_sarima_train_data$date, mexico_copy_sarima_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Mexico", "Training Set"))
lines(mexico_copy_sarima_train_data$date, mexico_copy_sarima_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(mexico_copy_sarima_test_data$date, mexico_copy_sarima_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Mexico", "Test Set"))
lines(mexico_copy_sarima_test_data$date, mexico_copy_sarima_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## peru ----

# producing fitted training set plot
plot(peru_copy_train_data$date, peru_copy_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Peru", "Training Set"))
lines(peru_copy_train_data$date, peru_copy_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(peru_copy_test_data$date, peru_copy_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Peru", "Test Set"))
lines(peru_copy_test_data$date, peru_copy_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## russia ----

# producing fitted training set plot
plot(russia_copy_arima_train_data$date, russia_copy_arima_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Russia", "Training Set"))
lines(russia_copy_arima_train_data$date, russia_copy_arima_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(russia_copy_arima_test_data$date, russia_copy_arima_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Russia", "Test Set"))
lines(russia_copy_arima_test_data$date, russia_copy_arima_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

### SARIMA -------------

plot(russia_copy_sarima_train_data$date, russia_copy_sarima_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Russia", "Training Set"))
lines(russia_copy_sarima_train_data$date, russia_copy_sarima_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(russia_copy_sarima_test_data$date, russia_copy_sarima_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Russia", "Test Set"))
lines(russia_copy_sarima_test_data$date, russia_copy_sarima_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## saudi ----

# producing fitted training set plot
plot(saudi_copy_train_data$date, saudi_copy_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Saudi", "Training Set"))
lines(saudi_copy_train_data$date, saudi_copy_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(saudi_copy_test_data$date, saudi_copy_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Saudi", "Test Set"))
lines(saudi_copy_test_data$date, saudi_copy_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## turkey ----

# producing fitted training set plot
plot(turkey_copy_train_data$date, turkey_copy_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Turkey", "Training Set"))
lines(turkey_copy_train_data$date, turkey_copy_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(turkey_copy_test_data$date, turkey_copy_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("Turkey", "Test Set"))
lines(turkey_copy_test_data$date, turkey_copy_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)


## us ----

# producing fitted training set plot
plot(us_copy_arima_train_data$date, us_copy_arima_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("U.S.", "Training Set"))
lines(us_copy_arima_train_data$date, us_copy_arima_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(us_copy_arima_test_data$date, us_copy_arima_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("U.S.", "Test Set"))
lines(us_copy_arima_test_data$date, us_copy_arima_forecast_values$mean, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Forecast"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

### SARIMA ------------

plot(us_copy_sarima_train_data$date, us_copy_sarima_train_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("U.S.", "Training Set"))
lines(us_copy_sarima_train_data$date, us_copy_sarima_fitted_values, col = "red", lty = 2)
legend("topright", legend = c("Actual", "Fitted"), col = c("blue", "red"), lty = 1:2, cex = 0.8)

# producing forecasting plot
plot(us_copy_sarima_test_data$date, us_copy_sarima_test_data$owid_new_deaths, type = "l", col = "blue", xlab = "Date",
     ylab = "New Deaths", main = paste("U.S.", "Test Set"))
lines(us_copy_sarima_test_data$date, us_copy_sarima_forecast_values$mean, col = "red", lty = 2)
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
  # Get the actual and fitted values
  actual_values <- get(paste0(country, "_copy_train_data"))$owid_new_deaths
  fitted_values <- get(paste0(country, "_copy_fitted_values"))
  
  # calculating errors
  errors <- actual_values - fitted_values
  
  # calculating metrics
  rmse <- sqrt(mean(errors^2))
  mae <- mean(abs(errors))
  mse <- mean(errors^2)
  
  # storing metrics in list
  train_metrics_list[[paste0(country, "_copy_train_rmse")]] <- rmse
  train_metrics_list[[paste0(country, "_copy_train_mae")]] <- mae
  train_metrics_list[[paste0(country, "_copy_train_mse")]] <- mse
}

# creating a list to store comparison results
performance_comparison <- list()

# looping through each country
for (country in countries) {
  
  # retrieving average training metrics
  avg_train_rmse <- round(train_metrics_list[[paste0(country, "_train_rmse")]], 3)
  avg_train_mae <- round(train_metrics_list[[paste0(country, "_train_mae")]], 3)
  avg_train_mse <- round(train_metrics_list[[paste0(country, "_train_mse")]], 3)
  
  # retrievingaverage testing metrics
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
bolivia_copy_auto_model <- auto.arima(bolivia_copy_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
brazil_copy_auto_model <- auto.arima(brazil_copy_arima_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
colombia_copy_auto_model <- auto.arima(colombia_copy_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
iran_copy_auto_model <- auto.arima(iran_copy_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
mexico_copy_auto_model <- auto.arima(mexico_copy_arima_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
peru_copy_auto_model <- auto.arima(peru_copy_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
russia_copy_auto_model <- auto.arima(russia_copy_arima_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
saudi_copy_auto_model <- auto.arima(saudi_copy_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
turkey_copy_auto_model <- auto.arima(turkey_copy_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
us_copy_auto_model <- auto.arima(us_copy_arima_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

# viewing selected model's details
summary(bolivia_copy_auto_model)
summary(brazil_copy_auto_model)
summary(colombia_copy_auto_model)
summary(iran_copy_auto_model)
summary(mexico_copy_auto_model)
summary(peru_copy_auto_model)
summary(russia_copy_auto_model)
summary(saudi_copy_auto_model)
summary(turkey_copy_auto_model)
summary(us_copy_auto_model)



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
countries_copy_data <- list(bolivia_copy = bolivia, brazil_copy = brazil, 
                            colombia_copy = colombia, iran_copy = iran, 
                            mexico_copy = mexico, peru_copy = peru, 
                            russia_copy = russia, saudi_copy = saudi, 
                            turkey_copy = turkey, us_copy = us)
country_copy_names <- names(countries_copy_data)

# initializing empty list to store metrics
all_copy_countries_metrics <- list()

# looping through each country
for (country_name in country_copy_names) {
  
  # load country-specific data
  country_copy_data <- get(paste0(country_name, "_train_data"))
  country_copy_test_data <- get(paste0(country_name, "_test_data"))
  
  # ARIMA model
  arima_copy_model <- get(paste0(country_name, "_arima_model"))
  
  # Auto-ARIMA model
  auto_copy_model <- auto.arima(country_copy_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
  
  # forecasting using both models
  arima_copy_forecast <- forecast(arima_copy_model, h = nrow(country_copy_test_data))
  auto_copy_forecast <- forecast(auto_copy_model, h = nrow(country_copy_test_data))
  
  # calculating metrics for ARIMA
  arima_copy_metrics <- calculate_metrics(arima_copy_forecast$mean, country_copy_test_data$owid_new_deaths, country_copy_data$owid_new_deaths)
  
  # calculating metrics for Auto-ARIMA
  auto_copy_metrics <- calculate_metrics(auto_copy_forecast$mean, country_copy_test_data$owid_new_deaths, country_copy_data$owid_new_deaths)
  
  # combining metrics into a data frame
  metrics_copy_df <- data.frame(
    Country = country_name,
    Best_copy_Model_RMSE = c("ARIMA", "Auto-ARIMA"),
    RMSE = c(arima_copy_metrics$RMSE, auto_copy_metrics$RMSE),
    MAE = c(arima_copy_metrics$MAE, auto_copy_metrics$MAE),
    MSE = c(arima_copy_metrics$MSE, auto_copy_metrics$MSE),
    MASE = c(arima_copy_metrics$MASE, auto_copy_metrics$MASE)
  )
  
  # capitalizing country name in metrics_df
  metrics_copy_df$Country <- ifelse(tolower(metrics_copy_df$Country) != "us", tools::toTitleCase(metrics_copy_df$Country), "US")
  
  # storing in list
  all_copy_countries_metrics[[country_name]] <- metrics_copy_df
}

# combining all metrics into one data frame
arima_copy_final_metrics_df <- do.call(rbind, all_copy_countries_metrics)

arima_copy_final_metrics_df <- arima_copy_final_metrics_df %>%
  group_by(Country) %>%
  slice_min(order_by = RMSE, with_ties = FALSE) %>%
  ungroup()

arima_copy_final_metrics_df <- arima_copy_final_metrics_df %>%
  mutate(across(c(RMSE, MAE, MSE, MASE), round, 3))

arima_copy_final_metrics_df %>% 
  DT::datatable()

# printing final metrics table
print(arima_copy_final_metrics_df)

# removing row names
row.names(arima_copy_final_metrics_df) <- NULL

# arima parameters
arima_copy_params <- data.frame(
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
auto_arima_copy_params <- data.frame(
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
auto_arima_copy_metrics_list <- list()

# looping through each country
for (country_name in country_copy_names) {
  
  # loading country-specific training data
  country_copy_train_data <- get(paste0(country_name, "_train_data"))
  country_copy_test_data <- get(paste0(country_name, "_test_data"))
  
  # fitting the Auto-ARIMA model to the training data
  auto_copy_model <- auto.arima(country_copy_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)
  
  # forecasting using the Auto-ARIMA model
  auto_copy_forecast <- forecast(auto_copy_model, h = nrow(country_copy_test_data))
  
  # calculating metrics for Auto-ARIMA
  auto_copy_metrics <- calculate_metrics(auto_copy_forecast$mean, country_copy_test_data$owid_new_deaths, country_copy_train_data$owid_new_deaths)
  
  # creating a data frame for the Auto-ARIMA metrics
  auto_copy_metrics_df <- data.frame(
    Country = ifelse(country_name == "us", "US", tools::toTitleCase(country_name)),
    RMSE = round(auto_copy_metrics$RMSE, 3),
    MAE = round(auto_copy_metrics$MAE, 3),
    MSE = round(auto_copy_metrics$MSE, 3),
    MASE = round(auto_copy_metrics$MASE, 3)
  )
  
  # storing the data frame in the list
  auto_arima_copy_metrics_list[[country_name]] <- auto_copy_metrics_df
}

# combining all the country-specific metrics into a single data frame
auto_arima_copy_final_metrics_df <- do.call(rbind, auto_arima_copy_metrics_list)

# removing row names for a cleaner look
row.names(auto_arima_copy_final_metrics_df) <- NULL



# saving files ----
save(arima_copy_final_metrics_df, file = "data_frames/arima_copy_final_metrics_df.rda")
save(auto_arima_copy__final_metrics_df, file = "data_frames/auto_arima_copy_final_metrics_df.rda")