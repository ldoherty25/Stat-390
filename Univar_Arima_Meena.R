## UNIVARIATE ARIMA MEENA

## uploading packages
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


bolivia_arima <- bolivia %>%
  rename(ds = date, y = owid_new_deaths)
brazil_arima <- brazil %>%
  rename(ds = date, y = owid_new_deaths)
colombia_arima <- colombia %>%
  rename(ds = date, y = owid_new_deaths)
iran_arima <- iran %>%
  rename(ds = date, y = owid_new_deaths)
mexico_arima <- mexico %>%
  rename(ds = date, y = owid_new_deaths)
peru_arima <- peru %>%
  rename(ds = date, y = owid_new_deaths)
russia_arima <- russia %>%
  rename(ds = date, y = owid_new_deaths)
saudi_arima <- saudi %>%
  rename(ds = date, y = owid_new_deaths)
turkey_arima <- turkey %>%
  rename(ds = date, y = owid_new_deaths)
us_arima <- us %>%
  rename(ds = date, y = owid_new_deaths)


##############################################################################################################
## BOLIVIA ##

b_train_size <- nrow(bolivia_arima)
b_train_set <- ceiling(0.9 * b_train_size)
b_test_set <- ceiling((b_train_size - b_train_set))

# Create time series cross-validation folds
bolivia_arima_folds <- time_series_cv(
  bolivia_arima,
  date_var = ds,
  initial = b_train_set,
  assess = b_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
bolivia_arima_models <- list()

# creating an empty data frame to store evaluation metrics
bolivia_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
bolivia_best_order <- c(Inf, Inf, Inf)
bolivia_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 0:1) {
    for (q in 0:3) {
      
      bolivia_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      bolivia_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(bolivia_arima_folds$splits)) {
        b_fold <- bolivia_arima_folds$splits[[i]]
        b_train_fold <- b_fold$data[b_fold$in_id, ]
        b_test_fold <- b_fold$data[b_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        bolivia_model <- Arima(b_train_fold$y, order = bolivia_curr_order)
        
        # Make predictions on the test data
        bolivia_forecast <- forecast(bolivia_model, h = nrow(b_test_fold))
        
        # Evaluate the model and store metrics
        bolivia_curr_metrics <- c(
          bolivia_curr_metrics,
          RMSE = sqrt(mean((bolivia_forecast$mean - b_test_fold$y)^2)),
          MAE = mean(abs(bolivia_forecast$mean - b_test_fold$y)),
          MSE = mean((bolivia_forecast$mean - b_test_fold$y)^2),
          MASE = mean(abs(diff(b_train_fold$y)) / mean(abs(bolivia_forecast$mean - b_test_fold$y))),
          MAPE = mean(abs((b_test_fold$y - bolivia_forecast$mean) / b_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      bolivia_curr_avg_metric <- mean(bolivia_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (bolivia_curr_avg_metric < bolivia_best_metric) {
        bolivia_best_metric <- bolivia_curr_avg_metric
        bolivia_best_order <- bolivia_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      bolivia_arima_metrics <- rbind(bolivia_arima_metrics, c(order = paste(bolivia_curr_order, collapse = ","), 
                                                              bolivia_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", bolivia_best_order, "\n")

# Display or return the evaluation metrics
print(bolivia_arima_metrics)

### BEST one after a few runs and range changes was 0, 2, 2


#bolivia is of class dataframe, so we have to turn it into a timeseries object
# bolivia_ts <- bolivia %>%
#   ts(bolivia$owid_new_deaths, start = bolivia[1, 1], end = bolivia[nrow(bolivia), 1], frequency = 1) 
# #frequency = 7 because there are 7 days in a week, so 7 observations per week, which is the unit of time
# 
# #have to establish the training and testing datasets, then the folds
# #code is a little different from prophet because we're working with a time_series object
# # instead of a dataframe object
# bolivia_split_cutoff <- ceiling(0.9 * length(bolivia_ts))
# bolivia_train_set <- head(bolivia_ts, bolivia_split_cutoff)
# #making it univariate so we can use auto_arima
# bolivia_train_set_univar <- bolivia_train_set[, 2]
#   
# bolivia_test_set <- tail(bolivia_ts, length(bolivia_ts) - bolivia_split_cutoff + 1)
# bolivia_test_set_univar <- bolivia_test_set[, 2]
# 
# bolivia_arima_metrics <- c("RMSE", "MASE", "MAE", "MSE", "MAPE")
# 
# #creating a tuning grid for this one to find the best combo of p, d, q parameters using auto_arima
# bolivia_param_grid <- expand.grid(p = 0:2, d = 0:1, q = 0:2)
# #choosing small range of values for p, d, q to begin with. Can change depending on model results. 
# 
# #going to do time series cross validation differently since object of class time_series using caret
# #could use trainControl() but found it too difficult
# bolivia_cv_res <- matrix(data = NA, nrow = nrow(bolivia_param_grid), ncol = length(bolivia_arima_metrics),
#                          dimnames = list(NULL, bolivia_arima_metrics))
# 
# #going to use a loop to apply the model
# for (i in seq_len(nrow(bolivia_param_grid))) {
#   p <- bolivia_param_grid[i, "p"]
#   d <- bolivia_param_grid[i, "d"]
#   q <- bolivia_param_grid[i, "q"]
#   
#   bolivia_arima_model <- Arima(bolivia_train_set_univar, order = c(p, d, q),
#                                seasonal = list(order = c(0, 0, 0), period = 1)) 
#   #zero values because assuming no seasonality
#   bolivia_forecast_values <- forecast(bolivia_arima_model, h = length(bolivia_test_set_univar))
#   
#   for (j in seq_along(bolivia_arima_metrics)) {
#     bolivia_cv_res[i, j] <- eval(parse(text = bolivia_arima_metrics[j]))(bolivia_forecast_values, bolivia_test_set_univar)
#   }
# }
# 
# #training the model using the created folds: 
# arima_model <- train(
#   bolivia_train_set,
#   "ARIMA",
#   trControl = bolivia_arima_folds,
#   tuneGrid = bolivia_param_grid
# )

#####################################################################################################
## BRAZIL ##

brazil_train_size <- nrow(brazil_arima)
brazil_train_set <- ceiling(0.9 * brazil_train_size)
brazil_test_set <- ceiling((brazil_train_size - brazil_train_set))

# Create time series cross-validation folds
brazil_arima_folds <- time_series_cv(
  brazil_arima,
  date_var = ds,
  initial = brazil_train_set,
  assess = brazil_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
brazil_arima_models <- list()

# creating an empty data frame to store evaluation metrics
brazil_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
brazil_best_order <- c(Inf, Inf, Inf)
brazil_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 1:2) { #had to change this range because I was getting an error: non-stationary AR part from CSS
    for (q in 0:3) {
      
      brazil_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      brazil_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(brazil_arima_folds$splits)) {
        brazil_fold <- brazil_arima_folds$splits[[i]]
        brazil_train_fold <- brazil_fold$data[brazil_fold$in_id, ]
        brazil_test_fold <- brazil_fold$data[brazil_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        brazil_model <- Arima(brazil_train_fold$y, order = brazil_curr_order)
        
        # Make predictions on the test data
        brazil_forecast <- forecast(brazil_model, h = nrow(brazil_test_fold))
        
        # Evaluate the model and store metrics
        brazil_curr_metrics <- c(
          brazil_curr_metrics,
          RMSE = sqrt(mean((brazil_forecast$mean - brazil_test_fold$y)^2)),
          MAE = mean(abs(brazil_forecast$mean - brazil_test_fold$y)),
          MSE = mean((brazil_forecast$mean - brazil_test_fold$y)^2),
          MASE = mean(abs(diff(brazil_train_fold$y)) / mean(abs(brazil_forecast$mean - brazil_test_fold$y))),
          MAPE = mean(abs((brazil_test_fold$y - brazil_forecast$mean) / brazil_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      brazil_curr_avg_metric <- mean(brazil_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (brazil_curr_avg_metric < brazil_best_metric) {
        brazil_best_metric <- brazil_curr_avg_metric
        brazil_best_order <- brazil_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      brazil_arima_metrics <- rbind(brazil_arima_metrics, c(order = paste(brazil_curr_order, collapse = ","), 
                                                            brazil_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", brazil_best_order, "\n")

# Display or return the evaluation metrics
print(brazil_arima_metrics)

################################################################################################
## COLOMBIA ##

colombia_train_size <- nrow(colombia_arima)
colombia_train_set <- ceiling(0.9 * colombia_train_size)
colombia_test_set <- ceiling((colombia_train_size - colombia_train_set))

# Create time series cross-validation folds
colombia_arima_folds <- time_series_cv(
  colombia_arima,
  date_var = ds,
  initial = colombia_train_set,
  assess = colombia_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
colombia_arima_models <- list()

# creating an empty data frame to store evaluation metrics
colombia_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
colombia_best_order <- c(Inf, Inf, Inf)
colombia_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 1:2) { #had to change this range because I was getting an error: non-stationary AR part from CSS
    for (q in 0:3) {
      
      colombia_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      colombia_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(colombia_arima_folds$splits)) {
        colombia_fold <- colombia_arima_folds$splits[[i]]
        colombia_train_fold <- colombia_fold$data[colombia_fold$in_id, ]
        colombia_test_fold <- colombia_fold$data[colombia_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        colombia_model <- Arima(colombia_train_fold$y, order = colombia_curr_order)
        
        # Make predictions on the test data
        colombia_forecast <- forecast(colombia_model, h = nrow(colombia_test_fold))
        
        # Evaluate the model and store metrics
        colombia_curr_metrics <- c(
          colombia_curr_metrics,
          RMSE = sqrt(mean((colombia_forecast$mean - colombia_test_fold$y)^2)),
          MAE = mean(abs(colombia_forecast$mean - colombia_test_fold$y)),
          MSE = mean((colombia_forecast$mean - colombia_test_fold$y)^2),
          MASE = mean(abs(diff(colombia_train_fold$y)) / mean(abs(colombia_forecast$mean - colombia_test_fold$y))),
          MAPE = mean(abs((colombia_test_fold$y - colombia_forecast$mean) / colombia_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      colombia_curr_avg_metric <- mean(colombia_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (colombia_curr_avg_metric < colombia_best_metric) {
        colombia_best_metric <- colombia_curr_avg_metric
        colombia_best_order <- colombia_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      colombia_arima_metrics <- rbind(colombia_arima_metrics, c(order = paste(colombia_curr_order, collapse = ","), 
                                                                colombia_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", colombia_best_order, "\n")

# Display or return the evaluation metrics
print(colombia_arima_metrics)

################################################################################################
## IRAN ##

iran_train_size <- nrow(iran_arima)
iran_train_set <- ceiling(0.9 * iran_train_size)
iran_test_set <- ceiling((iran_train_size - iran_train_set))

# Create time series cross-validation folds
iran_arima_folds <- time_series_cv(
  iran_arima,
  date_var = ds,
  initial = iran_train_set,
  assess = iran_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
iran_arima_models <- list()

# creating an empty data frame to store evaluation metrics
iran_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
iran_best_order <- c(Inf, Inf, Inf)
iran_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 1:2) { #had to change this range because I was getting an error: non-stationary AR part from CSS
    for (q in 0:3) {
      
      iran_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      iran_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(iran_arima_folds$splits)) {
        iran_fold <- iran_arima_folds$splits[[i]]
        iran_train_fold <- iran_fold$data[iran_fold$in_id, ]
        iran_test_fold <- iran_fold$data[iran_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        iran_model <- Arima(iran_train_fold$y, order = iran_curr_order)
        
        # Make predictions on the test data
        iran_forecast <- forecast(iran_model, h = nrow(iran_test_fold))
        
        # Evaluate the model and store metrics
        iran_curr_metrics <- c(
          iran_curr_metrics,
          RMSE = sqrt(mean((iran_forecast$mean - iran_test_fold$y)^2)),
          MAE = mean(abs(iran_forecast$mean - iran_test_fold$y)),
          MSE = mean((iran_forecast$mean - iran_test_fold$y)^2),
          MASE = mean(abs(diff(iran_train_fold$y)) / mean(abs(iran_forecast$mean - iran_test_fold$y))),
          MAPE = mean(abs((iran_test_fold$y - iran_forecast$mean) / iran_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      iran_curr_avg_metric <- mean(iran_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (iran_curr_avg_metric < iran_best_metric) {
        iran_best_metric <- iran_curr_avg_metric
        iran_best_order <- iran_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      iran_arima_metrics <- rbind(iran_arima_metrics, c(order = paste(iran_curr_order, collapse = ","), 
                                                        iran_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", iran_best_order, "\n")

# Display or return the evaluation metrics
print(iran_arima_metrics)

#############################################################################################
## MEXICO ##

mexico_train_size <- nrow(mexico_arima)
mexico_train_set <- ceiling(0.9 * mexico_train_size)
mexico_test_set <- ceiling((mexico_train_size - mexico_train_set))

# Create time series cross-validation folds
mexico_arima_folds <- time_series_cv(
  mexico_arima,
  date_var = ds,
  initial = mexico_train_set,
  assess = mexico_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
mexico_arima_models <- list()

# creating an empty data frame to store evaluation metrics
mexico_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
mexico_best_order <- c(Inf, Inf, Inf)
mexico_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 0:1) { #had to change this range because I was getting an error: non-stationary AR part from CSS
    for (q in 0:3) {
      
      mexico_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      mexico_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(mexico_arima_folds$splits)) {
        mexico_fold <- mexico_arima_folds$splits[[i]]
        mexico_train_fold <- mexico_fold$data[mexico_fold$in_id, ]
        mexico_test_fold <- mexico_fold$data[mexico_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        mexico_model <- Arima(mexico_train_fold$y, order = mexico_curr_order)
        
        # Make predictions on the test data
        mexico_forecast <- forecast(mexico_model, h = nrow(mexico_test_fold))
        
        # Evaluate the model and store metrics
        mexico_curr_metrics <- c(
          mexico_curr_metrics,
          RMSE = sqrt(mean((mexico_forecast$mean - mexico_test_fold$y)^2)),
          MAE = mean(abs(mexico_forecast$mean - mexico_test_fold$y)),
          MSE = mean((mexico_forecast$mean - mexico_test_fold$y)^2),
          MASE = mean(abs(diff(mexico_train_fold$y)) / mean(abs(mexico_forecast$mean - mexico_test_fold$y))),
          MAPE = mean(abs((mexico_test_fold$y - mexico_forecast$mean) / mexico_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      mexico_curr_avg_metric <- mean(mexico_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (mexico_curr_avg_metric < mexico_best_metric) {
        mexico_best_metric <- mexico_curr_avg_metric
        mexico_best_order <- mexico_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      mexico_arima_metrics <- rbind(mexico_arima_metrics, c(order = paste(mexico_curr_order, collapse = ","), 
                                                            mexico_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", mexico_best_order, "\n")

# Display or return the evaluation metrics
print(mexico_arima_metrics)

#SOMETHING WRONG???

##################################################################################################
## PERU ##

peru_train_size <- nrow(peru_arima)
peru_train_set <- ceiling(0.9 * peru_train_size)
peru_test_set <- ceiling((peru_train_size - peru_train_set))

# Create time series cross-validation folds
peru_arima_folds <- time_series_cv(
  peru_arima,
  date_var = ds,
  initial = peru_train_set,
  assess = peru_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
peru_arima_models <- list()

# creating an empty data frame to store evaluation metrics
peru_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
peru_best_order <- c(Inf, Inf, Inf)
peru_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 1:2) { 
    for (q in 0:3) {
      
      peru_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      peru_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(peru_arima_folds$splits)) {
        peru_fold <- peru_arima_folds$splits[[i]]
        peru_train_fold <- peru_fold$data[peru_fold$in_id, ]
        peru_test_fold <- peru_fold$data[peru_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        peru_model <- Arima(peru_train_fold$y, order = peru_curr_order)
        
        # Make predictions on the test data
        peru_forecast <- forecast(peru_model, h = nrow(peru_test_fold))
        
        # Evaluate the model and store metrics
        peru_curr_metrics <- c(
          peru_curr_metrics,
          RMSE = sqrt(mean((peru_forecast$mean - peru_test_fold$y)^2)),
          MAE = mean(abs(peru_forecast$mean - peru_test_fold$y)),
          MSE = mean((peru_forecast$mean - peru_test_fold$y)^2),
          MASE = mean(abs(diff(peru_train_fold$y)) / mean(abs(peru_forecast$mean - peru_test_fold$y))),
          MAPE = mean(abs((peru_test_fold$y - peru_forecast$mean) / peru_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      peru_curr_avg_metric <- mean(peru_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (peru_curr_avg_metric < peru_best_metric) {
        peru_best_metric <- peru_curr_avg_metric
        peru_best_order <- peru_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      peru_arima_metrics <- rbind(peru_arima_metrics, c(order = paste(peru_curr_order, collapse = ","), 
                                                        peru_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", peru_best_order, "\n")

# Display or return the evaluation metrics
print(peru_arima_metrics)

################################################################################################
## RUSSIA ##

russia_train_size <- nrow(russia_arima)
russia_train_set <- ceiling(0.9 * russia_train_size)
russia_test_set <- ceiling((russia_train_size - russia_train_set))

# Create time series cross-validation folds
russia_arima_folds <- time_series_cv(
  russia_arima,
  date_var = ds,
  initial = russia_train_set,
  assess = russia_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
russia_arima_models <- list()

# creating an empty data frame to store evaluation metrics
russia_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
russia_best_order <- c(Inf, Inf, Inf)
russia_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 0:2) { #had to change this range because I was getting an error: non-stationary AR part from CSS
    for (q in 0:3) {
      
      russia_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      russia_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(russia_arima_folds$splits)) {
        russia_fold <- russia_arima_folds$splits[[i]]
        russia_train_fold <- russia_fold$data[russia_fold$in_id, ]
        russia_test_fold <- russia_fold$data[russia_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        russia_model <- Arima(russia_train_fold$y, order = russia_curr_order)
        
        # Make predictions on the test data
        russia_forecast <- forecast(russia_model, h = nrow(russia_test_fold))
        
        # Evaluate the model and store metrics
        russia_curr_metrics <- c(
          russia_curr_metrics,
          RMSE = sqrt(mean((russia_forecast$mean - russia_test_fold$y)^2)),
          MAE = mean(abs(russia_forecast$mean - russia_test_fold$y)),
          MSE = mean((russia_forecast$mean - russia_test_fold$y)^2),
          MASE = mean(abs(diff(russia_train_fold$y)) / mean(abs(russia_forecast$mean - russia_test_fold$y))),
          MAPE = mean(abs((russia_test_fold$y - russia_forecast$mean) / russia_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      russia_curr_avg_metric <- mean(russia_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (russia_curr_avg_metric < russia_best_metric) {
        russia_best_metric <- russia_curr_avg_metric
        russia_best_order <- russia_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      russia_arima_metrics <- rbind(russia_arima_metrics, c(order = paste(russia_curr_order, collapse = ","), 
                                                            russia_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", russia_best_order, "\n")

# Display or return the evaluation metrics
print(russia_arima_metrics)

# Error in stats::arima(x = x, order = order, seasonal = seasonal, include.mean = include.mean,  : 
#                         non-stationary AR part from CSS
#                       > adf.test(russia_train_fold$y)
#                       
#                       Augmented Dickey-Fuller Test
#                       
#                       data:  russia_train_fold$y
#                       Dickey-Fuller = -2.1096, Lag order = 5, p-value = 0.5302
#                       alternative hypothesis: stationary

###################################################################################################
## SAUDI ##

saudi_train_size <- nrow(saudi_arima)
saudi_train_set <- ceiling(0.9 * saudi_train_size)
saudi_test_set <- ceiling((saudi_train_size - saudi_train_set))

# Create time series cross-validation folds
saudi_arima_folds <- time_series_cv(
  saudi_arima,
  date_var = ds,
  initial = saudi_train_set,
  assess = saudi_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
saudi_arima_models <- list()

# creating an empty data frame to store evaluation metrics
saudi_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
saudi_best_order <- c(Inf, Inf, Inf)
saudi_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 1:2) { 
    for (q in 0:3) {
      
      saudi_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      saudi_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(saudi_arima_folds$splits)) {
        saudi_fold <- saudi_arima_folds$splits[[i]]
        saudi_train_fold <- saudi_fold$data[saudi_fold$in_id, ]
        saudi_test_fold <- saudi_fold$data[saudi_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        saudi_model <- Arima(saudi_train_fold$y, order = saudi_curr_order)
        
        # Make predictions on the test data
        saudi_forecast <- forecast(saudi_model, h = nrow(saudi_test_fold))
        
        # Evaluate the model and store metrics
        saudi_curr_metrics <- c(
          saudi_curr_metrics,
          RMSE = sqrt(mean((saudi_forecast$mean - saudi_test_fold$y)^2)),
          MAE = mean(abs(saudi_forecast$mean - saudi_test_fold$y)),
          MSE = mean((saudi_forecast$mean - saudi_test_fold$y)^2),
          MASE = mean(abs(diff(saudi_train_fold$y)) / mean(abs(saudi_forecast$mean - saudi_test_fold$y))),
          MAPE = mean(abs((saudi_test_fold$y - saudi_forecast$mean) / saudi_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      saudi_curr_avg_metric <- mean(saudi_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (saudi_curr_avg_metric < saudi_best_metric) {
        saudi_best_metric <- saudi_curr_avg_metric
        saudi_best_order <- saudi_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      saudi_arima_metrics <- rbind(saudi_arima_metrics, c(order = paste(saudi_curr_order, collapse = ","), 
                                                          saudi_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", saudi_best_order, "\n")

# Display or return the evaluation metrics
print(saudi_arima_metrics)

##############################################################################################
## TURKEY ##

turkey_train_size <- nrow(turkey_arima)
turkey_train_set <- ceiling(0.9 * turkey_train_size)
turkey_test_set <- ceiling((turkey_train_size - turkey_train_set))

# Create time series cross-validation folds
turkey_arima_folds <- time_series_cv(
  turkey_arima,
  date_var = ds,
  initial = turkey_train_set,
  assess = turkey_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
turkey_arima_models <- list()

# creating an empty data frame to store evaluation metrics
turkey_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
turkey_best_order <- c(Inf, Inf, Inf)
turkey_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 1:2) { 
    for (q in 0:3) {
      
      turkey_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      turkey_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(turkey_arima_folds$splits)) {
        turkey_fold <- turkey_arima_folds$splits[[i]]
        turkey_train_fold <- turkey_fold$data[turkey_fold$in_id, ]
        turkey_test_fold <- turkey_fold$data[turkey_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        turkey_model <- Arima(turkey_train_fold$y, order = turkey_curr_order)
        
        # Make predictions on the test data
        turkey_forecast <- forecast(turkey_model, h = nrow(turkey_test_fold))
        
        # Evaluate the model and store metrics
        turkey_curr_metrics <- c(
          turkey_curr_metrics,
          RMSE = sqrt(mean((turkey_forecast$mean - turkey_test_fold$y)^2)),
          MAE = mean(abs(turkey_forecast$mean - turkey_test_fold$y)),
          MSE = mean((turkey_forecast$mean - turkey_test_fold$y)^2),
          MASE = mean(abs(diff(turkey_train_fold$y)) / mean(abs(turkey_forecast$mean - turkey_test_fold$y))),
          MAPE = mean(abs((turkey_test_fold$y - turkey_forecast$mean) / turkey_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      turkey_curr_avg_metric <- mean(turkey_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (turkey_curr_avg_metric < turkey_best_metric) {
        turkey_best_metric <- turkey_curr_avg_metric
        turkey_best_order <- turkey_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      turkey_arima_metrics <- rbind(turkey_arima_metrics, c(order = paste(turkey_curr_order, collapse = ","), 
                                                            turkey_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", turkey_best_order, "\n")

# Display or return the evaluation metrics
print(turkey_arima_metrics)

#################################################################################################
## US ##

us_train_size <- nrow(us_arima)
us_train_set <- ceiling(0.9 * us_train_size)
us_test_set <- ceiling((us_train_size - us_train_set))

# Create time series cross-validation folds
us_arima_folds <- time_series_cv(
  us_arima,
  date_var = ds,
  initial = us_train_set,
  assess = us_test_set,
  fold = 1,
  slice_limit = 1
)

# creating an empty list to store ARIMA models
us_arima_models <- list()

# creating an empty data frame to store evaluation metrics
us_arima_metrics <- data.frame()

# Tuning p, d, q (hyperparameters)
us_best_order <- c(Inf, Inf, Inf)
us_best_metric <- Inf

# Loop through potential orders and fit ARIMA models
# keeping intial ranges small, can adjust depending on results
for (p in 0:3) {
  for (d in 1:2) { 
    for (q in 0:3) {
      
      us_curr_order <- c(p, d, q)
      
      # Initialize an empty vector to store metrics for the current order
      us_curr_metrics <- c()
      
      # Fit models and store them in the list
      for (i in seq_along(us_arima_folds$splits)) {
        us_fold <- us_arima_folds$splits[[i]]
        us_train_fold <- us_fold$data[us_fold$in_id, ]
        us_test_fold <- us_fold$data[us_fold$out_id, ]
        
        # Fit the ARIMA model on the training data for the current fold
        us_model <- Arima(us_train_fold$y, order = us_curr_order)
        
        # Make predictions on the test data
        us_forecast <- forecast(us_model, h = nrow(us_test_fold))
        
        # Evaluate the model and store metrics
        us_curr_metrics <- c(
          us_curr_metrics,
          RMSE = sqrt(mean((us_forecast$mean - us_test_fold$y)^2)),
          MAE = mean(abs(us_forecast$mean - us_test_fold$y)),
          MSE = mean((us_forecast$mean - us_test_fold$y)^2),
          MASE = mean(abs(diff(us_train_fold$y)) / mean(abs(us_forecast$mean - us_test_fold$y))),
          MAPE = mean(abs((us_test_fold$y - us_forecast$mean) / us_test_fold$y)) * 100
        )
      }
      
      # Calculate average metric for the current order
      us_curr_avg_metric <- mean(us_curr_metrics)
      
      # Check if the current order has a lower average metric
      if (us_curr_avg_metric < us_best_metric) {
        us_best_metric <- us_curr_avg_metric
        us_best_order <- us_curr_order
      }
      
      # Store the metrics for the current order in the data frame
      us_arima_metrics <- rbind(us_arima_metrics, c(order = paste(us_curr_order, collapse = ","), 
                                                            us_curr_avg_metric))
    }
  }
}

# Print the best order
cat("Best ARIMA Order:", us_best_order, "\n")

# Display or return the evaluation metrics
print(us_arima_metrics)
