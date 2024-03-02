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
plot(bolivia_all_dates, bolivia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Bolivia Prophet Model: Fitted and Forecasted Values")

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
plot(brazil_all_dates, brazil_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Brazil Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(brazil_test_data$date, brazil_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(brazil_df_univar_prophet$ds, brazil_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

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


## applying prophet model ----

# creating metrics vector
colombia_rmse_results <- numeric(length(colombia_folds$splits))
colombia_mae_results <- numeric(length(colombia_folds$splits))
colombia_mse_results <- numeric(length(colombia_folds$splits))
colombia_mape_results <- numeric(length(colombia_folds$splits))
colombia_mase_results <- numeric(length(colombia_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(colombia_folds$splits)) {
  fold <- colombia_folds$splits[[i]]
  colombia_train_data <- fold$data[fold$in_id, ]
  colombia_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  colombia_df_univar_prophet <- data.frame(ds = colombia_train_data$date, y = colombia_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  colombia_univar_prophet_model<- prophet(colombia_df_univar_prophet)
  
  # making future dataframe for forecasting
  colombia_univar_prophet_future <- make_future_dataframe(colombia_univar_prophet_model, periods = nrow(colombia_test_data))
  
  # forecasting with Prophet
  colombia_univar_prophet_forecast <- predict(colombia_univar_prophet_model, colombia_univar_prophet_future)
  
  # extracting forecasted values
  colombia_forecast_values <- colombia_univar_prophet_forecast %>% filter(ds %in% colombia_test_data$date)
  
  # calculating evaluation metrics
  colombia_errors <- colombia_forecast_values$yhat - colombia_test_data$owid_new_deaths
  colombia_rmse_results[i] <- sqrt(mean(colombia_errors^2))
  colombia_mae_results[i] <- mean(abs(colombia_errors))
  colombia_mse_results[i] <- mean(colombia_errors^2)
  colombia_mape_results[i] <- mean(abs(colombia_errors / colombia_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  colombia_mean_train_diff <- mean(abs(diff(colombia_train_data$owid_new_deaths)))
  colombia_mase_results[i] <- mean(abs(colombia_errors)) / colombia_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(colombia_rmse_results)))
print(paste("MAE:", mean(colombia_mae_results)))
print(paste("MSE:", mean(colombia_mse_results)))
print(paste("MAPE:", mean(colombia_mape_results)))
print(paste("MASE:", mean(colombia_mase_results)))


## producing a plot ----

# combining training and test data for plotting
colombia_all_dates <- c(colombia_train_data$date, colombia_test_data$date)
colombia_all_values <- c(colombia_train_data$owid_new_deaths, colombia_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(colombia_all_dates, colombia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(colombia_test_data$date, colombia_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
colombia_fitted_forecast <- predict(colombia_univar_prophet_model, colombia_df_univar_prophet)

# plotting the actual and forecasted values
plot(colombia_all_dates, colombia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Colombia Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(colombia_test_data$date, colombia_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(colombia_df_univar_prophet$ds, colombia_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

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


## applying prophet model ----

# creating metrics vector
iran_rmse_results <- numeric(length(iran_folds$splits))
iran_mae_results <- numeric(length(iran_folds$splits))
iran_mse_results <- numeric(length(iran_folds$splits))
iran_mape_results <- numeric(length(iran_folds$splits))
iran_mase_results <- numeric(length(iran_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(iran_folds$splits)) {
  fold <- iran_folds$splits[[i]]
  iran_train_data <- fold$data[fold$in_id, ]
  iran_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  iran_df_univar_prophet <- data.frame(ds = iran_train_data$date, y = iran_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  iran_univar_prophet_model<- prophet(iran_df_univar_prophet)
  
  # making future dataframe for forecasting
  iran_univar_prophet_future <- make_future_dataframe(iran_univar_prophet_model, periods = nrow(iran_test_data))
  
  # forecasting with Prophet
  iran_univar_prophet_forecast <- predict(iran_univar_prophet_model, iran_univar_prophet_future)
  
  # extracting forecasted values
  iran_forecast_values <- iran_univar_prophet_forecast %>% filter(ds %in% iran_test_data$date)
  
  # calculating evaluation metrics
  iran_errors <- iran_forecast_values$yhat - iran_test_data$owid_new_deaths
  iran_rmse_results[i] <- sqrt(mean(iran_errors^2))
  iran_mae_results[i] <- mean(abs(iran_errors))
  iran_mse_results[i] <- mean(iran_errors^2)
  iran_mape_results[i] <- mean(abs(iran_errors / iran_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  iran_mean_train_diff <- mean(abs(diff(iran_train_data$owid_new_deaths)))
  iran_mase_results[i] <- mean(abs(iran_errors)) / iran_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(iran_rmse_results)))
print(paste("MAE:", mean(iran_mae_results)))
print(paste("MSE:", mean(iran_mse_results)))
print(paste("MAPE:", mean(iran_mape_results)))
print(paste("MASE:", mean(iran_mase_results)))


## producing a plot ----

# combining training and test data for plotting
iran_all_dates <- c(iran_train_data$date, iran_test_data$date)
iran_all_values <- c(iran_train_data$owid_new_deaths, iran_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(iran_all_dates, iran_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(iran_test_data$date, iran_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
iran_fitted_forecast <- predict(iran_univar_prophet_model, iran_df_univar_prophet)

# plotting the actual and forecasted values
plot(iran_all_dates, iran_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Iran Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(iran_test_data$date, iran_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(iran_df_univar_prophet$ds, iran_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

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


## applying prophet model ----

# creating metrics vector
mexico_rmse_results <- numeric(length(mexico_folds$splits))
mexico_mae_results <- numeric(length(mexico_folds$splits))
mexico_mse_results <- numeric(length(mexico_folds$splits))
mexico_mape_results <- numeric(length(mexico_folds$splits))
mexico_mase_results <- numeric(length(mexico_folds$splits))

epsilon <- 1e-8

# fitting model and calculating metrics
for (i in seq_along(mexico_folds$splits)) {
  fold <- mexico_folds$splits[[i]]
  mexico_train_data <- fold$data[fold$in_id, ]
  mexico_test_data <- fold$data[fold$out_id, ]
  
  # Preparing data for Prophet
  mexico_df_univar_prophet <- data.frame(ds = mexico_train_data$date, y = mexico_train_data$owid_new_deaths)
  
  # Fitting to Prophet model
  mexico_univar_prophet_model <- prophet(mexico_df_univar_prophet)
  
  # Making future dataframe for forecasting
  mexico_univar_prophet_future <- make_future_dataframe(mexico_univar_prophet_model, periods = nrow(mexico_test_data))
  
  # Forecasting with Prophet
  mexico_univar_prophet_forecast <- predict(mexico_univar_prophet_model, mexico_univar_prophet_future)
  
  # Aligning forecasted values with actual values by date
  aligned_forecast <- merge(mexico_test_data, mexico_univar_prophet_forecast, by.x = "date", by.y = "ds", all.x = TRUE)
  
  # Calculating evaluation metrics
  mexico_errors <- aligned_forecast$yhat - aligned_forecast$owid_new_deaths
  mexico_rmse_results[i] <- sqrt(mean(mexico_errors^2, na.rm = TRUE))
  mexico_mae_results[i] <- mean(abs(mexico_errors), na.rm = TRUE)
  mexico_mse_results[i] <- mean(mexico_errors^2, na.rm = TRUE)
  mexico_mape_results[i] <- mean(abs(mexico_errors / (aligned_forecast$owid_new_deaths + epsilon)), na.rm = TRUE) * 100
  mexico_mean_train_diff <- mean(abs(diff(mexico_train_data$owid_new_deaths)))
  mexico_mase_results[i] <- mean(abs(mexico_errors), na.rm = TRUE) / mexico_mean_train_diff

  aligned_forecast <- merge(mexico_test_data, mexico_univar_prophet_forecast, by.x = "date", by.y = "ds", all.x = TRUE)  
}

# printing metrics
print(paste("RMSE:", mean(mexico_rmse_results)))
print(paste("MAE:", mean(mexico_mae_results)))
print(paste("MSE:", mean(mexico_mse_results)))
print(paste("MAPE:", mean(mexico_mape_results)))
print(paste("MASE:", mean(mexico_mase_results)))


## producing a plot ----

# combining training and test data for plotting
mexico_all_dates <- c(mexico_train_data$date, mexico_test_data$date)
mexico_all_values <- c(mexico_train_data$owid_new_deaths, mexico_test_data$owid_new_deaths)

# potting actual values for both training and test data
plot(mexico_all_dates, mexico_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# get predictions for the training period to extract fitted values
mexico_fitted_forecast <- predict(mexico_univar_prophet_model, mexico_df_univar_prophet)

# plotting the actual and forecasted values
plot(mexico_all_dates, mexico_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Mexico Prophet Model: Fitted and Forecasted Values")

# plotting fitted training data
lines(mexico_df_univar_prophet$ds, mexico_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)

# ensuring aligned_forecast is defined outside the loop and contains the forecasted values for plotting
lines(aligned_forecast$date, aligned_forecast$yhat, col = "blue", lty = 2, lwd = 2)



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


## applying prophet model ----

# creating metrics vector
peru_rmse_results <- numeric(length(peru_folds$splits))
peru_mae_results <- numeric(length(peru_folds$splits))
peru_mse_results <- numeric(length(peru_folds$splits))
peru_mape_results <- numeric(length(peru_folds$splits))
peru_mase_results <- numeric(length(peru_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(peru_folds$splits)) {
  fold <- peru_folds$splits[[i]]
  peru_train_data <- fold$data[fold$in_id, ]
  peru_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  peru_df_univar_prophet <- data.frame(ds = peru_train_data$date, y = peru_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  peru_univar_prophet_model<- prophet(peru_df_univar_prophet)
  
  # making future dataframe for forecasting
  peru_univar_prophet_future <- make_future_dataframe(peru_univar_prophet_model, periods = nrow(peru_test_data))
  
  # forecasting with Prophet
  peru_univar_prophet_forecast <- predict(peru_univar_prophet_model, peru_univar_prophet_future)
  
  # extracting forecasted values
  peru_forecast_values <- peru_univar_prophet_forecast %>% filter(ds %in% peru_test_data$date)
  
  # calculating evaluation metrics
  peru_errors <- peru_forecast_values$yhat - peru_test_data$owid_new_deaths
  peru_rmse_results[i] <- sqrt(mean(peru_errors^2))
  peru_mae_results[i] <- mean(abs(peru_errors))
  peru_mse_results[i] <- mean(peru_errors^2)
  peru_mape_results[i] <- mean(abs(peru_errors / peru_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  peru_mean_train_diff <- mean(abs(diff(peru_train_data$owid_new_deaths)))
  peru_mase_results[i] <- mean(abs(peru_errors)) / peru_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(peru_rmse_results)))
print(paste("MAE:", mean(peru_mae_results)))
print(paste("MSE:", mean(peru_mse_results)))
print(paste("MAPE:", mean(peru_mape_results)))
print(paste("MASE:", mean(peru_mase_results)))


## producing a plot ----

# combining training and test data for plotting
peru_all_dates <- c(peru_train_data$date, peru_test_data$date)
peru_all_values <- c(peru_train_data$owid_new_deaths, peru_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(peru_all_dates, peru_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(peru_test_data$date, peru_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
peru_fitted_forecast <- predict(peru_univar_prophet_model, peru_df_univar_prophet)

# plotting the actual and forecasted values
plot(peru_all_dates, peru_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Peru Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(peru_test_data$date, peru_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(peru_df_univar_prophet$ds, peru_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

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


## applying prophet model ----

# creating metrics vector
russia_rmse_results <- numeric(length(russia_folds$splits))
russia_mae_results <- numeric(length(russia_folds$splits))
russia_mse_results <- numeric(length(russia_folds$splits))
russia_mape_results <- numeric(length(russia_folds$splits))
russia_mase_results <- numeric(length(russia_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(russia_folds$splits)) {
  fold <- russia_folds$splits[[i]]
  russia_train_data <- fold$data[fold$in_id, ]
  russia_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  russia_df_univar_prophet <- data.frame(ds = russia_train_data$date, y = russia_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  russia_univar_prophet_model<- prophet(russia_df_univar_prophet)
  
  # making future dataframe for forecasting
  russia_univar_prophet_future <- make_future_dataframe(russia_univar_prophet_model, periods = nrow(russia_test_data))
  
  # forecasting with Prophet
  russia_univar_prophet_forecast <- predict(russia_univar_prophet_model, russia_univar_prophet_future)
  
  # extracting forecasted values
  russia_forecast_values <- russia_univar_prophet_forecast %>% filter(ds %in% russia_test_data$date)
  
  # calculating evaluation metrics
  russia_errors <- russia_forecast_values$yhat - russia_test_data$owid_new_deaths
  russia_rmse_results[i] <- sqrt(mean(russia_errors^2))
  russia_mae_results[i] <- mean(abs(russia_errors))
  russia_mse_results[i] <- mean(russia_errors^2)
  russia_mape_results[i] <- mean(abs(russia_errors / russia_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  russia_mean_train_diff <- mean(abs(diff(russia_train_data$owid_new_deaths)))
  russia_mase_results[i] <- mean(abs(russia_errors)) / russia_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(russia_rmse_results)))
print(paste("MAE:", mean(russia_mae_results)))
print(paste("MSE:", mean(russia_mse_results)))
print(paste("MAPE:", mean(russia_mape_results)))
print(paste("MASE:", mean(russia_mase_results)))


## producing a plot ----

# combining training and test data for plotting
russia_all_dates <- c(russia_train_data$date, russia_test_data$date)
russia_all_values <- c(russia_train_data$owid_new_deaths, russia_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(russia_all_dates, russia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(russia_test_data$date, russia_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
russia_fitted_forecast <- predict(russia_univar_prophet_model, russia_df_univar_prophet)

# plotting the actual and forecasted values
plot(russia_all_dates, russia_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Russia Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(russia_test_data$date, russia_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(russia_df_univar_prophet$ds, russia_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

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


## applying prophet model ----

# creating metrics vector
saudi_rmse_results <- numeric(length(saudi_folds$splits))
saudi_mae_results <- numeric(length(saudi_folds$splits))
saudi_mse_results <- numeric(length(saudi_folds$splits))
saudi_mape_results <- numeric(length(saudi_folds$splits))
saudi_mase_results <- numeric(length(saudi_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(saudi_folds$splits)) {
  fold <- saudi_folds$splits[[i]]
  saudi_train_data <- fold$data[fold$in_id, ]
  saudi_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  saudi_df_univar_prophet <- data.frame(ds = saudi_train_data$date, y = saudi_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  saudi_univar_prophet_model<- prophet(saudi_df_univar_prophet)
  
  # making future dataframe for forecasting
  saudi_univar_prophet_future <- make_future_dataframe(saudi_univar_prophet_model, periods = nrow(saudi_test_data))
  
  # forecasting with Prophet
  saudi_univar_prophet_forecast <- predict(saudi_univar_prophet_model, saudi_univar_prophet_future)
  
  # extracting forecasted values
  saudi_forecast_values <- saudi_univar_prophet_forecast %>% filter(ds %in% saudi_test_data$date)
  
  # calculating evaluation metrics
  saudi_errors <- saudi_forecast_values$yhat - saudi_test_data$owid_new_deaths
  saudi_rmse_results[i] <- sqrt(mean(saudi_errors^2))
  saudi_mae_results[i] <- mean(abs(saudi_errors))
  saudi_mse_results[i] <- mean(saudi_errors^2)
  saudi_mape_results[i] <- mean(abs(saudi_errors / saudi_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  saudi_mean_train_diff <- mean(abs(diff(saudi_train_data$owid_new_deaths)))
  saudi_mase_results[i] <- mean(abs(saudi_errors)) / saudi_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(saudi_rmse_results)))
print(paste("MAE:", mean(saudi_mae_results)))
print(paste("MSE:", mean(saudi_mse_results)))
print(paste("MAPE:", mean(saudi_mape_results)))
print(paste("MASE:", mean(saudi_mase_results)))


## producing a plot ----

# combining training and test data for plotting
saudi_all_dates <- c(saudi_train_data$date, saudi_test_data$date)
saudi_all_values <- c(saudi_train_data$owid_new_deaths, saudi_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(saudi_all_dates, saudi_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(saudi_test_data$date, saudi_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
saudi_fitted_forecast <- predict(saudi_univar_prophet_model, saudi_df_univar_prophet)

# plotting the actual and forecasted values
plot(saudi_all_dates, saudi_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Saudi Arabia Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(saudi_test_data$date, saudi_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(saudi_df_univar_prophet$ds, saudi_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

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


## applying prophet model ----

# creating metrics vector
turkey_rmse_results <- numeric(length(turkey_folds$splits))
turkey_mae_results <- numeric(length(turkey_folds$splits))
turkey_mse_results <- numeric(length(turkey_folds$splits))
turkey_mape_results <- numeric(length(turkey_folds$splits))
turkey_mase_results <- numeric(length(turkey_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(turkey_folds$splits)) {
  fold <- turkey_folds$splits[[i]]
  turkey_train_data <- fold$data[fold$in_id, ]
  turkey_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  turkey_df_univar_prophet <- data.frame(ds = turkey_train_data$date, y = turkey_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  turkey_univar_prophet_model<- prophet(turkey_df_univar_prophet)
  
  # making future dataframe for forecasting
  turkey_univar_prophet_future <- make_future_dataframe(turkey_univar_prophet_model, periods = nrow(turkey_test_data))
  
  # forecasting with Prophet
  turkey_univar_prophet_forecast <- predict(turkey_univar_prophet_model, turkey_univar_prophet_future)
  
  # extracting forecasted values
  turkey_forecast_values <- turkey_univar_prophet_forecast %>% filter(ds %in% turkey_test_data$date)
  
  # calculating evaluation metrics
  turkey_errors <- turkey_forecast_values$yhat - turkey_test_data$owid_new_deaths
  turkey_rmse_results[i] <- sqrt(mean(turkey_errors^2))
  turkey_mae_results[i] <- mean(abs(turkey_errors))
  turkey_mse_results[i] <- mean(turkey_errors^2)
  turkey_mape_results[i] <- mean(abs(turkey_errors / turkey_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  turkey_mean_train_diff <- mean(abs(diff(turkey_train_data$owid_new_deaths)))
  turkey_mase_results[i] <- mean(abs(turkey_errors)) / turkey_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(turkey_rmse_results)))
print(paste("MAE:", mean(turkey_mae_results)))
print(paste("MSE:", mean(turkey_mse_results)))
print(paste("MAPE:", mean(turkey_mape_results)))
print(paste("MASE:", mean(turkey_mase_results)))


## producing a plot ----

# combining training and test data for plotting
turkey_all_dates <- c(turkey_train_data$date, turkey_test_data$date)
turkey_all_values <- c(turkey_train_data$owid_new_deaths, turkey_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(turkey_all_dates, turkey_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(turkey_test_data$date, turkey_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
turkey_fitted_forecast <- predict(turkey_univar_prophet_model, turkey_df_univar_prophet)

# plotting the actual and forecasted values
plot(turkey_all_dates, turkey_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "Turkey Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(turkey_test_data$date, turkey_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(turkey_df_univar_prophet$ds, turkey_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

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


## applying prophet model ----

# creating metrics vector
us_rmse_results <- numeric(length(us_folds$splits))
us_mae_results <- numeric(length(us_folds$splits))
us_mse_results <- numeric(length(us_folds$splits))
us_mape_results <- numeric(length(us_folds$splits))
us_mase_results <- numeric(length(us_folds$splits))

# fitting model and calculating metrics
for (i in seq_along(us_folds$splits)) {
  fold <- us_folds$splits[[i]]
  us_train_data <- fold$data[fold$in_id, ]
  us_test_data <- fold$data[fold$out_id, ]
  
  # preparing data for Prophet
  us_df_univar_prophet <- data.frame(ds = us_train_data$date, y = us_train_data$owid_new_deaths)
  
  # fitting to Prophet model
  us_univar_prophet_model<- prophet(us_df_univar_prophet)
  
  # making future dataframe for forecasting
  us_univar_prophet_future <- make_future_dataframe(us_univar_prophet_model, periods = nrow(us_test_data))
  
  # forecasting with Prophet
  us_univar_prophet_forecast <- predict(us_univar_prophet_model, us_univar_prophet_future)
  
  # extracting forecasted values
  us_forecast_values <- us_univar_prophet_forecast %>% filter(ds %in% us_test_data$date)
  
  # calculating evaluation metrics
  us_errors <- us_forecast_values$yhat - us_test_data$owid_new_deaths
  us_rmse_results[i] <- sqrt(mean(us_errors^2))
  us_mae_results[i] <- mean(abs(us_errors))
  us_mse_results[i] <- mean(us_errors^2)
  us_mape_results[i] <- mean(abs(us_errors / us_test_data$owid_new_deaths)) * 100
  
  # calculating MASE
  us_mean_train_diff <- mean(abs(diff(us_train_data$owid_new_deaths)))
  us_mase_results[i] <- mean(abs(us_errors)) / us_mean_train_diff
}

# printing metrics
print(paste("RMSE:", mean(us_rmse_results)))
print(paste("MAE:", mean(us_mae_results)))
print(paste("MSE:", mean(us_mse_results)))
print(paste("MAPE:", mean(us_mape_results)))
print(paste("MASE:", mean(us_mase_results)))


## producing a plot ----

# combining training and test data for plotting
us_all_dates <- c(us_train_data$date, us_test_data$date)
us_all_values <- c(us_train_data$owid_new_deaths, us_test_data$owid_new_deaths)

# plotting actual values for both training and test data
plot(us_all_dates, us_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths")

# plotting the forecasted values for test data
lines(us_test_data$date, us_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# get predictions for the training period to extract fitted values
us_fitted_forecast <- predict(us_univar_prophet_model, us_df_univar_prophet)

# plotting the actual and forecasted values
plot(us_all_dates, us_all_values, type = "l", col = "black", lwd = 2, xlab = "Date", ylab = "New Deaths", main = "U.S. Prophet Model: Fitted and Forecasted Values")

# plotting forecasted values for the test data
lines(us_test_data$date, us_forecast_values$yhat, col = "blue", lty = 2, lwd = 2)

# plotting fitted training data
lines(us_df_univar_prophet$ds, us_fitted_forecast$yhat, col = "red", lty = 1, lwd = 2)

# adding legend
legend("topright", legend = c("Actual", "Forecast", "Training Fit"), col = c("black", "blue", "red"), lty = c(1, 2, 1), lwd = 2)



# displaying best models ----

# defining list to hold each country's metrics
prophet_metrics_list <- list()

# calculating metrics for each country
countries <- c("bolivia", "brazil", "colombia", "iran", "mexico", "peru", "russia", "saudi", "turkey", "us")

for (country in countries) {
  
  # copying calculation of metrics
  prophet_rmse <- mean(get(paste0(country, "_rmse_results")))
  prophet_mae <- mean(get(paste0(country, "_mae_results")))
  prophet_mse <- mean(get(paste0(country, "_mse_results")))
  prophet_mape <- mean(get(paste0(country, "_mape_results")))
  prophet_mase <- mean(get(paste0(country, "_mase_results")))
  
  # creating data frame for country's metrics
  country_metrics_df <- data.frame(
    Country = ifelse(tolower(country) == "us", "US", tools::toTitleCase(country)),
    RMSE = prophet_rmse,
    MAE = prophet_mae,
    MSE = prophet_mse,
    MAPE = prophet_mape,
    MASE = prophet_mase
  )
  
  # adding data frame to list
  prophet_metrics_list[[country]] <- country_metrics_df
}

# combining all data frames into one
prophet_all_metrics_df <- do.call(rbind, prophet_metrics_list)

# rounding metrics to 3 decimal points + handle NAs
num_cols <- c("RMSE", "MAE", "MSE", "MAPE", "MASE")
prophet_all_metrics_df[num_cols] <- lapply(prophet_all_metrics_df[num_cols], function(x) ifelse(is.na(x), NA, round(x, 3)))

# creating a new variable
prophet_all_metrics_df <- prophet_all_metrics_df %>%
  mutate(Best_Model_RMSE = "Prophet") %>%
  select(Country, Best_Model_RMSE, everything())

# sorting by RMSE
prophet_all_metrics_df <- prophet_all_metrics_df %>%
  arrange(RMSE)

# removing row names
row.names(prophet_all_metrics_df) <- NULL

# proper formatting
prophet_all_metrics_df %>%
  DT::datatable(rownames = FALSE)
  


# saving files ----
save(prophet_all_metrics_df, file = "data_frames/maria_prophet_final_metrics_df.rda")