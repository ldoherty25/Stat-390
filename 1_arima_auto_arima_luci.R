## ARIMA and Auto-ARIMA

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


##########################################################################################################################################################################################################################################################


## Bolivia model

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
  slice_limit = 1
)

# filtering by slice

bolivia_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

bolivia_rmse_results <- numeric(length(bolivia_folds$splits))
bolivia_mae_results <- numeric(length(bolivia_folds$splits))
bolivia_mse_results <- numeric(length(bolivia_folds$splits))
bolivia_mape_results <- numeric(length(bolivia_folds$splits))

# arima model fit

for (i in seq_along(bolivia_folds$splits)) {
  fold <- bolivia_folds$splits[[i]]
  bolivia_train_data = fold$data[fold$in_id, ]
  bolivia_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
    
  bolivia_arima <- arima(bolivia_train_data$owid_new_deaths,
                           order = c(2, 0, 0),
                           seasonal = list(order = c(0, 1, 0), period = 7))
    
  bolivia_forecast <- forecast(bolivia_arima, h = nrow(bolivia_test_data))
  bolivia_forecast$mean <- pmax(bolivia_forecast$mean, 0)
    
  bolivia_errors <- bolivia_forecast$mean - bolivia_test_data$owid_new_deaths
  bolivia_rmse_results[i] <- sqrt(mean(bolivia_errors^2))
  bolivia_mae_results[i] <- mean(abs(bolivia_errors))
  bolivia_mse_results[i] <- mean(bolivia_errors^2)
  bolivia_mape_results[i] <- mean(abs(bolivia_errors / bolivia_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(bolivia_rmse_results)))
print(paste("MAE:", mean(bolivia_mae_results)))
print(paste("MSE:", mean(bolivia_mse_results)))
print(paste("MAPE:", mean(bolivia_mape_results)))

bolivia_fitted <- fitted(bolivia_arima)

bolivia_fitted <- pmax(bolivia_fitted, 0)

bolivia_all_dates <- c(bolivia_train_data$date, bolivia_test_data$date)
bolivia_all_values <- c(bolivia_train_data$owid_new_deaths, bolivia_test_data$owid_new_deaths)

plot(bolivia_all_dates, bolivia_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Bolivia Arima Model") + 
  lines(bolivia_test_data$date, bolivia_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(bolivia_train_data$date, bolivia_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## brazil model

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
  slice_limit = 1
)

# filtering by slice

brazil_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

brazil_rmse_results <- numeric(length(brazil_folds$splits))
brazil_mae_results <- numeric(length(brazil_folds$splits))
brazil_mse_results <- numeric(length(brazil_folds$splits))
brazil_mape_results <- numeric(length(brazil_folds$splits))

# arima model fit

for (i in seq_along(brazil_folds$splits)) {
  fold <- brazil_folds$splits[[i]]
  brazil_train_data = fold$data[fold$in_id, ]
  brazil_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  brazil_arima <- arima(brazil_train_data$owid_new_deaths,
                         order = c(0, 1, 1),
                         seasonal = list(order = c(1, 1, 0), period = 7))
  
  brazil_forecast <- forecast(brazil_arima, h = nrow(brazil_test_data))
  brazil_forecast$mean <- pmax(brazil_forecast$mean, 0)
  
  brazil_errors <- brazil_forecast$mean - brazil_test_data$owid_new_deaths
  brazil_rmse_results[i] <- sqrt(mean(brazil_errors^2))
  brazil_mae_results[i] <- mean(abs(brazil_errors))
  brazil_mse_results[i] <- mean(brazil_errors^2)
  brazil_mape_results[i] <- mean(abs(brazil_errors / brazil_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(brazil_rmse_results)))
print(paste("MAE:", mean(brazil_mae_results)))
print(paste("MSE:", mean(brazil_mse_results)))
print(paste("MAPE:", mean(brazil_mape_results)))

brazil_fitted <- fitted(brazil_arima)

brazil_fitted <- pmax(brazil_fitted, 0)

brazil_all_dates <- c(brazil_train_data$date, brazil_test_data$date)
brazil_all_values <- c(brazil_train_data$owid_new_deaths, brazil_test_data$owid_new_deaths)

plot(brazil_all_dates, brazil_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Brazil Arima Model") + 
  lines(brazil_test_data$date, brazil_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(brazil_train_data$date, brazil_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## colombia model

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
  slice_limit = 1
)

# filtering by slice

colombia_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

colombia_rmse_results <- numeric(length(colombia_folds$splits))
colombia_mae_results <- numeric(length(colombia_folds$splits))
colombia_mse_results <- numeric(length(colombia_folds$splits))
colombia_mape_results <- numeric(length(colombia_folds$splits))

# arima model fit

for (i in seq_along(colombia_folds$splits)) {
  fold <- colombia_folds$splits[[i]]
  colombia_train_data = fold$data[fold$in_id, ]
  colombia_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  colombia_arima <- arima(colombia_train_data$owid_new_deaths,
                         order = c(0, 0, 0),
                         seasonal = list(order = c(0, 1, 0), period = 7))
  
  colombia_forecast <- forecast(colombia_arima, h = nrow(colombia_test_data))
  colombia_forecast$mean <- pmax(colombia_forecast$mean, 0)
  
  colombia_errors <- colombia_forecast$mean - colombia_test_data$owid_new_deaths
  colombia_rmse_results[i] <- sqrt(mean(colombia_errors^2))
  colombia_mae_results[i] <- mean(abs(colombia_errors))
  colombia_mse_results[i] <- mean(colombia_errors^2)
  colombia_mape_results[i] <- mean(abs(colombia_errors / colombia_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(colombia_rmse_results)))
print(paste("MAE:", mean(colombia_mae_results)))
print(paste("MSE:", mean(colombia_mse_results)))
print(paste("MAPE:", mean(colombia_mape_results)))

colombia_fitted <- fitted(colombia_arima)

colombia_fitted <- pmax(colombia_fitted, 0)

colombia_all_dates <- c(colombia_train_data$date, colombia_test_data$date)
colombia_all_values <- c(colombia_train_data$owid_new_deaths, colombia_test_data$owid_new_deaths)

plot(colombia_all_dates, colombia_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Colombia Arima Model") + 
  lines(colombia_test_data$date, colombia_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(colombia_train_data$date, colombia_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## iran model

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
  slice_limit = 1
)

# filtering by slice

iran_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

iran_rmse_results <- numeric(length(iran_folds$splits))
iran_mae_results <- numeric(length(iran_folds$splits))
iran_mse_results <- numeric(length(iran_folds$splits))
iran_mape_results <- numeric(length(iran_folds$splits))

# arima model fit

for (i in seq_along(iran_folds$splits)) {
  fold <- iran_folds$splits[[i]]
  iran_train_data = fold$data[fold$in_id, ]
  iran_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  iran_arima <- arima(iran_train_data$owid_new_deaths,
                         order = c(0, 1, 0),
                         seasonal = list(order = c(1, 1, 0), period = 10))
  
  iran_forecast <- forecast(iran_arima, h = nrow(iran_test_data))
  iran_forecast$mean <- pmax(iran_forecast$mean, 0)
  
  iran_errors <- iran_forecast$mean - iran_test_data$owid_new_deaths
  iran_rmse_results[i] <- sqrt(mean(iran_errors^2))
  iran_mae_results[i] <- mean(abs(iran_errors))
  iran_mse_results[i] <- mean(iran_errors^2)
  iran_mape_results[i] <- mean(abs(iran_errors / iran_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(iran_rmse_results)))
print(paste("MAE:", mean(iran_mae_results)))
print(paste("MSE:", mean(iran_mse_results)))
print(paste("MAPE:", mean(iran_mape_results)))


iran_fitted <- fitted(iran_arima)

iran_fitted <- pmax(iran_fitted, 0)

iran_all_dates <- c(iran_train_data$date, iran_test_data$date)
iran_all_values <- c(iran_train_data$owid_new_deaths, iran_test_data$owid_new_deaths)

plot(iran_all_dates, iran_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Iran Arima Model") + 
  lines(iran_test_data$date, iran_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(iran_train_data$date, iran_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## mexico model

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
  slice_limit = 1
)

# filtering by slice

mexico_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

mexico_rmse_results <- numeric(length(mexico_folds$splits))
mexico_mae_results <- numeric(length(mexico_folds$splits))
mexico_mse_results <- numeric(length(mexico_folds$splits))
mexico_mape_results <- numeric(length(mexico_folds$splits))

# arima model fit

for (i in seq_along(mexico_folds$splits)) {
  fold <- mexico_folds$splits[[i]]
  mexico_train_data = fold$data[fold$in_id, ]
  mexico_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  mexico_arima <- arima(mexico_train_data$owid_new_deaths,
                         order = c(1, 0, 0),
                         seasonal = list(order = c(0, 1, 0), period = 13))
  
  mexico_forecast <- forecast(mexico_arima, h = nrow(mexico_test_data))
  mexico_forecast$mean <- pmax(mexico_forecast$mean, 0)
  
  mexico_errors <- mexico_forecast$mean - mexico_test_data$owid_new_deaths
  mexico_rmse_results[i] <- sqrt(mean(mexico_errors^2))
  mexico_mae_results[i] <- mean(abs(mexico_errors))
  mexico_mse_results[i] <- mean(mexico_errors^2)
  mexico_mape_results[i] <- mean(abs(mexico_errors / mexico_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(mexico_rmse_results)))
print(paste("MAE:", mean(mexico_mae_results)))
print(paste("MSE:", mean(mexico_mse_results)))
print(paste("MAPE:", mean(mexico_mape_results)))


mexico_fitted <- fitted(mexico_arima)

mexico_fitted <- pmax(mexico_fitted, 0)

mexico_all_dates <- c(mexico_train_data$date, mexico_test_data$date)
mexico_all_values <- c(mexico_train_data$owid_new_deaths, mexico_test_data$owid_new_deaths)

plot(mexico_all_dates, mexico_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Mexico Arima Model") + 
  lines(mexico_test_data$date, mexico_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(mexico_train_data$date, mexico_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## Peru model

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
  slice_limit = 1
)

# filtering by slice

peru_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

peru_rmse_results <- numeric(length(peru_folds$splits))
peru_mae_results <- numeric(length(peru_folds$splits))
peru_mse_results <- numeric(length(peru_folds$splits))
peru_mape_results <- numeric(length(peru_folds$splits))

# arima model fit

for (i in seq_along(peru_folds$splits)) {
  fold <- peru_folds$splits[[i]]
  peru_train_data = fold$data[fold$in_id, ]
  peru_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  peru_arima <- arima(peru_train_data$owid_new_deaths,
                         order = c(0, 0, 0),
                         seasonal = list(order = c(1, 1, 0), period = 7))
  
  peru_forecast <- forecast(peru_arima, h = nrow(peru_test_data))
  peru_forecast$mean <- pmax(peru_forecast$mean, 0)
  
  peru_errors <- peru_forecast$mean - peru_test_data$owid_new_deaths
  peru_rmse_results[i] <- sqrt(mean(peru_errors^2))
  peru_mae_results[i] <- mean(abs(peru_errors))
  peru_mse_results[i] <- mean(peru_errors^2)
  peru_mape_results[i] <- mean(abs(peru_errors / peru_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(peru_rmse_results)))
print(paste("MAE:", mean(peru_mae_results)))
print(paste("MSE:", mean(peru_mse_results)))
print(paste("MAPE:", mean(peru_mape_results)))

peru_fitted <- fitted(peru_arima)

peru_fitted <- pmax(peru_fitted, 0)

peru_all_dates <- c(peru_train_data$date, peru_test_data$date)
peru_all_values <- c(peru_train_data$owid_new_deaths, peru_test_data$owid_new_deaths)

plot(peru_all_dates, peru_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Peru Arima Model") + 
  lines(peru_test_data$date, peru_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(peru_train_data$date, peru_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## russia model

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
  slice_limit = 1
)

# filtering by slice

russia_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

russia_rmse_results <- numeric(length(russia_folds$splits))
russia_mae_results <- numeric(length(russia_folds$splits))
russia_mse_results <- numeric(length(russia_folds$splits))
russia_mape_results <- numeric(length(russia_folds$splits))

# arima model fit

for (i in seq_along(russia_folds$splits)) {
  fold <- russia_folds$splits[[i]]
  russia_train_data = fold$data[fold$in_id, ]
  russia_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  russia_arima <- arima(russia_train_data$owid_new_deaths,
                         order = c(1, 1, 0),
                         seasonal = list(order = c(0, 1, 0), period = 2))
  
  russia_forecast <- forecast(russia_arima, h = nrow(russia_test_data))
  russia_forecast$mean <- pmax(russia_forecast$mean, 0)
  
  russia_errors <- russia_forecast$mean - russia_test_data$owid_new_deaths
  russia_rmse_results[i] <- sqrt(mean(russia_errors^2))
  russia_mae_results[i] <- mean(abs(russia_errors))
  russia_mse_results[i] <- mean(russia_errors^2)
  russia_mape_results[i] <- mean(abs(russia_errors / russia_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(russia_rmse_results)))
print(paste("MAE:", mean(russia_mae_results)))
print(paste("MSE:", mean(russia_mse_results)))
print(paste("MAPE:", mean(russia_mape_results)))

russia_fitted <- fitted(russia_arima)

russia_fitted <- pmax(russia_fitted, 0)

russia_all_dates <- c(russia_train_data$date, russia_test_data$date)
russia_all_values <- c(russia_train_data$owid_new_deaths, russia_test_data$owid_new_deaths)

plot(russia_all_dates, russia_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Russia Arima Model") + 
  lines(russia_test_data$date, russia_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(russia_train_data$date, russia_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## saudi model

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
  slice_limit = 1
)

# filtering by slice

saudi_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

saudi_rmse_results <- numeric(length(saudi_folds$splits))
saudi_mae_results <- numeric(length(saudi_folds$splits))
saudi_mse_results <- numeric(length(saudi_folds$splits))
saudi_mape_results <- numeric(length(saudi_folds$splits))
saudi_mase_results <- numeric(length(saudi_folds$splits))

# arima model fit

for (i in seq_along(saudi_folds$splits)) {
  fold <- saudi_folds$splits[[i]]
  saudi_train_data = fold$data[fold$in_id, ]
  saudi_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  saudi_arima <- arima(saudi_train_data$owid_new_deaths,
                         order = c(0, 0, 0),
                         seasonal = list(order = c(1, 0, 0), period = 7))
  
  saudi_forecast <- forecast(saudi_arima, h = nrow(saudi_test_data))
  saudi_forecast$mean <- pmax(saudi_forecast$mean, 0)
  
  saudi_errors <- saudi_forecast$mean - saudi_test_data$owid_new_deaths
  saudi_rmse_results[i] <- sqrt(mean(saudi_errors^2))
  saudi_mae_results[i] <- mean(abs(saudi_errors))
  saudi_mse_results[i] <- mean(saudi_errors^2)
  saudi_mape_results[i] <- mean(abs(saudi_errors / saudi_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(saudi_rmse_results)))
print(paste("MAE:", mean(saudi_mae_results)))
print(paste("MSE:", mean(saudi_mse_results)))
print(paste("MAPE:", mean(saudi_mape_results)))

saudi_fitted <- fitted(saudi_arima)

saudi_fitted <- pmax(saudi_fitted, 0)

saudi_all_dates <- c(saudi_train_data$date, saudi_test_data$date)
saudi_all_values <- c(saudi_train_data$owid_new_deaths, saudi_test_data$owid_new_deaths)

plot(saudi_all_dates, saudi_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Suadi Arabia Arima Model") + 
  lines(saudi_test_data$date, saudi_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(saudi_train_data$date, saudi_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## turkey model

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
  slice_limit = 1
)

# filtering by slice

turkey_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

turkey_rmse_results <- numeric(length(turkey_folds$splits))
turkey_mae_results <- numeric(length(turkey_folds$splits))
turkey_mse_results <- numeric(length(turkey_folds$splits))
turkey_mape_results <- numeric(length(turkey_folds$splits))
turkey_mase_results <- numeric(length(turkey_folds$splits))

# arima model fit

for (i in seq_along(turkey_folds$splits)) {
  fold <- turkey_folds$splits[[i]]
  turkey_train_data = fold$data[fold$in_id, ]
  turkey_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  turkey_arima <- arima(turkey_train_data$owid_new_deaths,
                         order = c(0, 0, 0),
                         seasonal = list(order = c(0, 1, 0), period = 7))
  
  turkey_forecast <- forecast(turkey_arima, h = nrow(turkey_test_data))
  turkey_forecast$mean <- pmax(turkey_forecast$mean, 0)
  
  turkey_errors <- turkey_forecast$mean - turkey_test_data$owid_new_deaths
  turkey_rmse_results[i] <- sqrt(mean(turkey_errors^2))
  turkey_mae_results[i] <- mean(abs(turkey_errors))
  turkey_mse_results[i] <- mean(turkey_errors^2)
  turkey_mape_results[i] <- mean(abs(turkey_errors / turkey_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(turkey_rmse_results)))
print(paste("MAE:", mean(turkey_mae_results)))
print(paste("MSE:", mean(turkey_mse_results)))
print(paste("MAPE:", mean(turkey_mape_results)))

turkey_fitted <- fitted(turkey_arima)

turkey_fitted <- pmax(turkey_fitted, 0)

turkey_all_dates <- c(turkey_train_data$date, turkey_test_data$date)
turkey_all_values <- c(turkey_train_data$owid_new_deaths, turkey_test_data$owid_new_deaths)

plot(turkey_all_dates, turkey_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Turkey Arima Model") + 
  lines(turkey_test_data$date, turkey_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(turkey_train_data$date, turkey_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


########################################################################################################################################################################################################################################################


## us model

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
  slice_limit = 1
)

# filtering by slice

us_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

us_rmse_results <- numeric(length(us_folds$splits))
us_mae_results <- numeric(length(us_folds$splits))
us_mse_results <- numeric(length(us_folds$splits))
us_mape_results <- numeric(length(us_folds$splits))
us_mase_results <- numeric(length(us_folds$splits))

# arima model fit

for (i in seq_along(us_folds$splits)) {
  fold <- us_folds$splits[[i]]
  us_train_data = fold$data[fold$in_id, ]
  us_test_data = fold$data[fold$out_id, ]
  
  all_metrics <- data.frame()
  
  us_arima <- arima(us_train_data$owid_new_deaths,
                         order = c(2, 0, 0),
                         seasonal = list(order = c(0, 1, 0), period = 7))
  
  us_forecast <- forecast(us_arima, h = nrow(us_test_data))
  us_forecast$mean <- pmax(us_forecast$mean, 0)
  
  us_errors <- us_forecast$mean - us_test_data$owid_new_deaths
  us_rmse_results[i] <- sqrt(mean(us_errors^2))
  us_mae_results[i] <- mean(abs(us_errors))
  us_mse_results[i] <- mean(us_errors^2)
  us_mape_results[i] <- mean(abs(us_errors / us_test_data$owid_new_deaths)) * 100
  
}

print(paste("RMSE:", mean(us_rmse_results)))
print(paste("MAE:", mean(us_mae_results)))
print(paste("MSE:", mean(us_mse_results)))
print(paste("MAPE:", mean(us_mape_results)))

#us_metrics <- fitting(us_folds)

us_fitted <- fitted(us_arima)

us_fitted <- pmax(us_fitted, 0)

us_all_dates <- c(us_train_data$date, us_test_data$date)
us_all_values <- c(us_train_data$owid_new_deaths, us_test_data$owid_new_deaths)


plot(us_all_dates, us_all_values, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "US Arima Model") + 
  lines(us_test_data$date, us_forecast$mean, col = "blue", lty = 2, lwd = 2) + 
  lines(us_train_data$date, us_fitted, col = "red", lty = 1, lwd = 1) +
  legend("topright", 
         legend = c("Actual", "Forecast", "Training Fit"), 
         col = c("black", "blue", "red"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)

################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################


## Auto Arima


# Bolivia

bolivia_auto <- auto.arima(bolivia_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(bolivia_auto)

bolivia_auto_forecast <- forecast(bolivia_auto, h = nrow(bolivia_test_data))

bolivia_auto_rmse <- sqrt(mean((bolivia_auto_forecast$mean - bolivia_test_data$owid_new_deaths)^2))
bolivia_auto_mae <- mean(abs(bolivia_auto_forecast$mean - bolivia_test_data$owid_new_deaths))
bolivia_auto_mse <- mean((bolivia_auto_forecast$mean - bolivia_test_data$owid_new_deaths)^2)
bolivia_auto_mape <- mean(abs((bolivia_test_data$owid_new_deaths - bolivia_auto_forecast$mean) /
                                bolivia_test_data$owid_new_deaths)) * 100

cat("RMSE:", bolivia_auto_rmse, "\n")
cat("MAE:", bolivia_auto_mae, "\n")
cat("MSE:", bolivia_auto_mse, "\n")
cat("MAPE:", bolivia_auto_mape, "\n")

########################################################################################################################################################################################################################################################

# Brazil

brazil_auto <- auto.arima(brazil_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(brazil_auto)

brazil_auto_forecast <- forecast(brazil_auto, h = nrow(brazil_test_data))

brazil_auto_rmse <- sqrt(mean((brazil_auto_forecast$mean - brazil_test_data$owid_new_deaths)^2))
brazil_auto_mae <- mean(abs(brazil_auto_forecast$mean - brazil_test_data$owid_new_deaths))
brazil_auto_mse <- mean((brazil_auto_forecast$mean - brazil_test_data$owid_new_deaths)^2)
brazil_auto_mape <- mean(abs((brazil_test_data$owid_new_deaths - brazil_auto_forecast$mean) /
                                brazil_test_data$owid_new_deaths)) * 100

cat("RMSE:", brazil_auto_rmse, "\n")
cat("MAE:", brazil_auto_mae, "\n")
cat("MSE:", brazil_auto_mse, "\n")
cat("MAPE:", brazil_auto_mape, "\n")

########################################################################################################################################################################################################################################################

# Colombia

colombia_auto <- auto.arima(colombia_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(colombia_auto)

colombia_auto_forecast <- forecast(colombia_auto, h = nrow(colombia_test_data))

colombia_auto_rmse <- sqrt(mean((colombia_auto_forecast$mean - colombia_test_data$owid_new_deaths)^2))
colombia_auto_mae <- mean(abs(colombia_auto_forecast$mean - colombia_test_data$owid_new_deaths))
colombia_auto_mse <- mean((colombia_auto_forecast$mean - colombia_test_data$owid_new_deaths)^2)
colombia_auto_mape <- mean(abs((colombia_test_data$owid_new_deaths - colombia_auto_forecast$mean) /
                               colombia_test_data$owid_new_deaths)) * 100

cat("RMSE:", colombia_auto_rmse, "\n")
cat("MAE:", colombia_auto_mae, "\n")
cat("MSE:", colombia_auto_mse, "\n")
cat("MAPE:", colombia_auto_mape, "\n")

########################################################################################################################################################################################################################################################

# Iran

iran_auto <- auto.arima(iran_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(iran_auto)

iran_auto_forecast <- forecast(iran_auto, h = nrow(iran_test_data))

iran_auto_rmse <- sqrt(mean((iran_auto_forecast$mean - iran_test_data$owid_new_deaths)^2))
iran_auto_mae <- mean(abs(iran_auto_forecast$mean - iran_test_data$owid_new_deaths))
iran_auto_mse <- mean((iran_auto_forecast$mean - iran_test_data$owid_new_deaths)^2)
iran_auto_mape <- mean(abs((iran_test_data$owid_new_deaths - iran_auto_forecast$mean) /
                               iran_test_data$owid_new_deaths)) * 100

cat("RMSE:", iran_auto_rmse, "\n")
cat("MAE:", iran_auto_mae, "\n")
cat("MSE:", iran_auto_mse, "\n")
cat("MAPE:", iran_auto_mape, "\n")

########################################################################################################################################################################################################################################################

# Mexico

mexico_auto <- auto.arima(mexico_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(mexico_auto)

mexico_auto_forecast <- forecast(mexico_auto, h = nrow(mexico_test_data))

mexico_auto_rmse <- sqrt(mean((mexico_auto_forecast$mean - mexico_test_data$owid_new_deaths)^2))
mexico_auto_mae <- mean(abs(mexico_auto_forecast$mean - mexico_test_data$owid_new_deaths))
mexico_auto_mse <- mean((mexico_auto_forecast$mean - mexico_test_data$owid_new_deaths)^2)
mexico_auto_mape <- mean(abs((mexico_test_data$owid_new_deaths - mexico_auto_forecast$mean) /
                               mexico_test_data$owid_new_deaths)) * 100

cat("RMSE:", mexico_auto_rmse, "\n")
cat("MAE:", mexico_auto_mae, "\n")
cat("MSE:", mexico_auto_mse, "\n")
cat("MAPE:", mexico_auto_mape, "\n")

########################################################################################################################################################################################################################################################

# Peru

peru_auto <- auto.arima(peru_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(peru_auto)

peru_auto_forecast <- forecast(peru_auto, h = nrow(peru_test_data))

peru_auto_rmse <- sqrt(mean((peru_auto_forecast$mean - peru_test_data$owid_new_deaths)^2))
peru_auto_mae <- mean(abs(peru_auto_forecast$mean - peru_test_data$owid_new_deaths))
peru_auto_mse <- mean((peru_auto_forecast$mean - peru_test_data$owid_new_deaths)^2)
peru_auto_mape <- mean(abs((peru_test_data$owid_new_deaths - peru_auto_forecast$mean) /
                               peru_test_data$owid_new_deaths)) * 100

cat("RMSE:", peru_auto_rmse, "\n")
cat("MAE:", peru_auto_mae, "\n")
cat("MSE:", peru_auto_mse, "\n")
cat("MAPE:", peru_auto_mape, "\n")

#######################################################################################################################################################################################################################################################

# Russia

russia_auto <- auto.arima(russia_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(russia_auto)

russia_auto_forecast <- forecast(russia_auto, h = nrow(russia_test_data))

russia_auto_rmse <- sqrt(mean((russia_auto_forecast$mean - russia_test_data$owid_new_deaths)^2))
russia_auto_mae <- mean(abs(russia_auto_forecast$mean - russia_test_data$owid_new_deaths))
russia_auto_mse <- mean((russia_auto_forecast$mean - russia_test_data$owid_new_deaths)^2)
russia_auto_mape <- mean(abs((russia_test_data$owid_new_deaths - russia_auto_forecast$mean) /
                               russia_test_data$owid_new_deaths)) * 100

cat("RMSE:", russia_auto_rmse, "\n")
cat("MAE:", russia_auto_mae, "\n")
cat("MSE:", russia_auto_mse, "\n")
cat("MAPE:", russia_auto_mape, "\n")

########################################################################################################################################################################################################################################################

# Saudi

saudi_auto <- auto.arima(saudi_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(saudi_auto)

saudi_auto_forecast <- forecast(saudi_auto, h = nrow(saudi_test_data))

saudi_auto_rmse <- sqrt(mean((saudi_auto_forecast$mean - saudi_test_data$owid_new_deaths)^2))
saudi_auto_mae <- mean(abs(saudi_auto_forecast$mean - saudi_test_data$owid_new_deaths))
saudi_auto_mse <- mean((saudi_auto_forecast$mean - saudi_test_data$owid_new_deaths)^2)
saudi_auto_mape <- mean(abs((saudi_test_data$owid_new_deaths - saudi_auto_forecast$mean) /
                               saudi_test_data$owid_new_deaths)) * 100

cat("RMSE:", saudi_auto_rmse, "\n")
cat("MAE:", saudi_auto_mae, "\n")
cat("MSE:", saudi_auto_mse, "\n")
cat("MAPE:", saudi_auto_mape, "\n")

########################################################################################################################################################################################################################################################

# Turkey

turkey_auto <- auto.arima(turkey_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(turkey_auto)

turkey_auto_forecast <- forecast(turkey_auto, h = nrow(turkey_test_data))

turkey_auto_rmse <- sqrt(mean((turkey_auto_forecast$mean - turkey_test_data$owid_new_deaths)^2))
turkey_auto_mae <- mean(abs(turkey_auto_forecast$mean - turkey_test_data$owid_new_deaths))
turkey_auto_mse <- mean((turkey_auto_forecast$mean - turkey_test_data$owid_new_deaths)^2)
turkey_auto_mape <- mean(abs((turkey_test_data$owid_new_deaths - turkey_auto_forecast$mean) /
                               turkey_test_data$owid_new_deaths)) * 100

cat("RMSE:", turkey_auto_rmse, "\n")
cat("MAE:", turkey_auto_mae, "\n")
cat("MSE:", turkey_auto_mse, "\n")
cat("MAPE:", turkey_auto_mape, "\n")

########################################################################################################################################################################################################################################################

# US

us_auto <- auto.arima(us_train_data$owid_new_deaths, stepwise = FALSE, approximation = FALSE)

summary(us_auto)

us_auto_forecast <- forecast(us_auto, h = nrow(us_test_data))

us_auto_rmse <- sqrt(mean((us_auto_forecast$mean - us_test_data$owid_new_deaths)^2))
us_auto_mae <- mean(abs(us_auto_forecast$mean - us_test_data$owid_new_deaths))
us_auto_mse <- mean((us_auto_forecast$mean - us_test_data$owid_new_deaths)^2)
us_auto_mape <- mean(abs((us_test_data$owid_new_deaths - us_auto_forecast$mean) /
                               us_test_data$owid_new_deaths)) * 100

cat("RMSE:", us_auto_rmse, "\n")
cat("MAE:", us_auto_mae, "\n")
cat("MSE:", us_auto_mse, "\n")
cat("MAPE:", us_auto_mape, "\n")









