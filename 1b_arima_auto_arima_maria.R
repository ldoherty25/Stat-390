# Folding & Tuning ARIMA/SARIMA



# primary checks ----

# load required packages
library(tidyverse)
library(tidymodels)
library(forecast)
library(caret)
library(timetk)
library(doMC)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)

# load data
load("data/preprocessed/univariate/not_split/us.rda")
us$date <- as.Date(us$date)



# ARIMA ----

# define the total number of observations
num_obs <- nrow(us)

# setup for cross-validation
num_folds <- 5
fold_length <- floor(num_obs / num_folds)

# generating start and end indices for each fold
fold_indices <- lapply(1:num_folds, function(x) {
  start <- (x - 1) * fold_length + 1
  end <- start + fold_length - 1
  c(start, end)
})

# initializing an empty data frame to store results
results <- data.frame(Fold = integer(), RMSE = numeric(), MAE = numeric())

for (i in seq_along(fold_indices)) {
  indices <- fold_indices[[i]]
  start_index <- indices[1]
  end_index <- indices[2]
  
  # adjusting the approach for the first fold
  training_set <- if (i == 1) {
    us[1:end_index, ]
  } else {
    us[1:(start_index - 1), ]
  }
  test_set <- us[start_index:end_index, ]
  
  # check if training set has enough data to fit the model
  if (nrow(training_set) > 3) {
    tryCatch({
      fit <- arima(training_set$owid_new_deaths, order = c(0, 0, 0))
      
      forecast_length <- nrow(test_set)
      forecast <- forecast(fit, h = forecast_length)
      
      actuals <- test_set$owid_new_deaths
      predictions <- forecast$mean
      rmse <- sqrt(mean((actuals - predictions)^2, na.rm = TRUE))
      mae <- mean(abs(actuals - predictions), na.rm = TRUE)
      
      results <- rbind(results, data.frame(Fold = i, RMSE = rmse, MAE = mae))
    }, error = function(e) {
      # optionally handle the error, e.g., by printing a message
      cat("error in fold", i, ": ", e$message, "\n")
    })
  } else {
    cat("skipping fold", i, "due to insufficient training data.\n")
  }
}

print(results)

# initializing parameters for best model tracking
min_rmse <- Inf
best_order <- NULL

# iterating over all combinations without a seasonal component
for (order_i in c(0, 1, 2)) {
  for (order_ii in c(0, 1, 2)) {
    for (order_iii in c(0, 1, 2)) {
      
      # initializing vector to store RMSE for each fold
      rmse_results <- numeric(length(fold_indices))
      
      # iterating over each fold
      for (i in seq_along(fold_indices)) {
        indices <- fold_indices[[i]]
        start_index <- indices[1]
        end_index <- indices[2]
        
        training_set <- if (i == 1) {
          us[1:end_index, ]  # adjust based on fold logic
        } else {
          us[1:(start_index - 1), ]
        }
        test_set <- us[start_index:end_index, ]
        
        # fitting ARIMA without a seasonal component, using tryCatch to handle errors
        arima_model <- tryCatch({
          arima(training_set$owid_new_deaths, order = c(order_i, order_ii, order_iii))
        }, error = function(e) {
          NULL  # in case of error, return NULL to indicate failure
        })
        
        if (!is.null(arima_model)) {
          # forecasting with ARIMA
          forecast_length <- nrow(test_set)
          forecast_values <- forecast(arima_model, h = forecast_length)
          
          # enforcing non-negativity on forecasted values
          forecast_values$mean <- pmax(forecast_values$mean, 0)
          
          # computing RMSE
          errors <- forecast_values$mean - test_set$owid_new_deaths
          rmse_results[i] <- sqrt(mean(errors^2))
        } else {
          # setting RMSE to Inf if model fitting fails
          rmse_results[i] <- Inf
        }
      }
      
      # calculating average folds RMSE
      avg_rmse <- mean(rmse_results, na.rm = TRUE)
      
      # updating best parameters if current combination is better
      if (avg_rmse < min_rmse) {
        min_rmse <- avg_rmse
        best_order <- c(order_i, order_ii, order_iii)
      }
    }
  }
}

# printing lowest RMSE and best parameters
cat("minimum RMSE:", min_rmse, "\n")
cat("best order:", best_order, "\n")



# SARIMA ----

# initializing parameters for best SARIMA model tracking
min_sarima_rmse <- Inf
best_sarima_order <- NULL
best_seasonal_order <- NULL

# iterating over all combinations of non-seasonal and seasonal components
for (p in 0:2) {
  for (d in 0:2) {
    for (q in 0:2) {
      for (P in 0:2) {
        for (D in 0:2) {
          for (Q in 0:2) {
            # initializing vector to store RMSE for each fold
            sarima_rmse_results <- numeric(length(fold_indices))
            
            # iterating over each fold
            for (i in seq_along(fold_indices)) {
              indices <- fold_indices[[i]]
              start_index <- indices[1]
              end_index <- indices[2]
              
              training_set <- if (i == 1) {
                us[1:end_index, ]
              } else {
                us[1:(start_index - 1), ]
              }
              test_set <- us[start_index:end_index, ]
              
              # fitting SARIMA with a seasonal component
              sarima_model <- tryCatch({
                arima(training_set$owid_new_deaths, 
                      order = c(p, d, q),
                      seasonal = list(order = c(P, D, Q), period = 7))
              }, error = function(e) {
                NULL  # in case of error, return NULL to indicate failure
              })
              
              if (!is.null(sarima_model)) {
                # forecasting with SARIMA
                forecast_length <- nrow(test_set)
                forecast_values <- forecast(sarima_model, h = forecast_length)
                
                # enforcing non-negativity on forecasted values
                forecast_values$mean <- pmax(forecast_values$mean, 0)
                
                # computing RMSE
                errors <- forecast_values$mean - test_set$owid_new_deaths
                sarima_rmse_results[i] <- sqrt(mean(errors^2))
              } else {
                # setting RMSE to Inf if model fitting fails
                sarima_rmse_results[i] <- Inf
              }
            }
            
            # calculating average folds RMSE
            avg_sarima_rmse <- mean(sarima_rmse_results, na.rm = TRUE)
            
            # updating best parameters if current combination is better
            if (avg_sarima_rmse < min_sarima_rmse) {
              min_sarima_rmse <- avg_sarima_rmse
              best_sarima_order <- c(p, d, q)
              best_seasonal_order <- c(P, D, Q)
            }
          }
        }
      }
    }
  }
}

# printing lowest SARIMA RMSE and best parameters
cat("minimum SARIMA RMSE:", min_sarima_rmse, "\n")
cat("best non-seasonal order:", best_sarima_order, "\n")
cat("best seasonal order:", best_seasonal_order, "\n")
