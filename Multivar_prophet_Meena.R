#### MEENA PROPHET MULTIVARIATE #### ------

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
library(tseries)
library(stats)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

#setting the seed
set.seed(1234)

load("data/preprocessed/multivariate/preprocessed_covid_multi_imputed.rda")

####################### changing variable names

# NOTE TO SELF: covariates = additional time-dependent variables that may influence the target variables 
#but are not the primary variables being forecasted.These covariates are provided as input features 
#to the model to improve its predictions.

## Need to change categorical variables to numeric
multivar_covid <- preprocessed_covid_multi_imputed %>%
  rename(ds = date, y = owid_new_deaths) %>%
  mutate(country_label = as.numeric(as.factor(multivar_covid$country)),
         weekday_label = as.numeric(as.factor(multivar_covid$weekday)))

# Get rid of the old categorical vars
multivar_covid <- multivar_covid %>%
  select(-c(country, weekday))

# Splitting into training and testing 
multivar_train_size <- nrow(multivar_covid)
multivar_train_set <- ceiling(0.8 * multivar_train_size)
multivar_test_set <- ceiling((multivar_train_size - multivar_train_set))

# Creating a tuning grid 
multi_param_grid <- expand.grid(
  yearly.seasonality = c(TRUE, FALSE),
  weekly.seasonality = c(TRUE, FALSE),
  daily.seasonality = c(TRUE, FALSE),
  seasonality.prior.scale = c(0.1, 0.5),
 # holidays.prior.scale = c(10, 20),
  changepoint.prior.scale = c(0.05, 0.1),
  interval.width = c(0.8, 0.95)
)

# Initializing a dataframe to store the metrics to be calculatedd
multivar_metrics <- data.frame(
  RMSE = numeric(0),
  MAE = numeric(0),
  MSE = numeric(0),
  MASE = numeric(0),
  MAPE = numeric(0)
)

# List of additional regressors to include. Adding custom socioeconomic features
multi_regressors <- c("policy_response_impact_cf", "vulnerability_index_cf")

# Iterate over the tuning grid
for (i in 1:nrow(multi_param_grid)) {
  multi_params <- multi_param_grid[i, ]
  
  # Train the Prophet model with specified parameters
  multi_prophet_model <- prophet(
    yearly.seasonality = multi_params$yearly.seasonality,
    weekly.seasonality = multi_params$weekly.seasonality,
    daily.seasonality = multi_params$daily.seasonality,
    seasonality.prior.scale = multi_params$seasonality.prior.scale,
    changepoint.prior.scale = multi_params$changepoint.prior.scale,
    interval.width = multi_params$interval.width
  )
  
  # Adding regressors
  for (regressor in multi_regressors) {
    multi_prophet_model <- add_regressor(multi_prophet_model, name = regressor)
  }
  
  # Fitting the model to training set
  multi_prophet_model <- fit.prophet(multi_prophet_model, multivar_train_set)
  
  # Make predictions on the test set
  multi_prophet_future <- make_future_dataframe(multi_prophet_model, periods = nrow(multivar_test_set))
  
  # Adding regressors for our predictions on the testing set
  for (regressor in multi_regressors) {
    multi_prophet_future[[regressor]] <- multi_future_regressor_vals
  }
  
  # Forecasting
  multi_prophet_forecast <- predict(multi_prophet_model, multi_prophet_future)
  
  # Evaluating with our chosen metrics
  multi_prophet_accuracy <- accuracy(multi_prophet_forecast, multivar_test_set$y)
  multivar_metrics <- rbind(multivar_metrics, c(multi_prophet_accuracy[1, ], multi_params))
}

# Print or analyze the results_df data frame
print(multi_var_metrics)

# # Create the training and testing datasets (80/20 split)
# split_index <- floor(0.8 * nrow(multivariate_data))
# train_data <- multivariate_data[1:split_index, ]
# test_data <- multivariate_data[(split_index + 1):nrow(multivariate_data), ]
