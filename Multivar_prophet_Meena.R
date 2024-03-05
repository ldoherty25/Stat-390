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

multivar_covid <- preprocessed_covid_multi_imputed %>%
  rename(ds = date, y = owid_new_deaths, covariate1 = policy_response_impact_cf)

# Splitting into training and testing 
multivar_train_size <- nrow(multivar_covid)
multivar_train_set <- ceiling(0.9 * multivar_train_size)
multivar_test_set <- ceiling((multivar_train_size - multivar_train_set))

# Creating time series cross-validation folds
# multivar_folds <- time_series_cv(
#   multivar_covid,
#   date_var = "ds",
#   initial = multivar_train_size,
#   assess = nrow(multivar_covid) - multivar_train_size,
#   fold = 10,  # Number of folds, adjust as needed
#   slice_limit = 50
# )

# trying a different method of manual splitting because getting this error: 
# Overlapping Timestamps Detected. Processing overlapping time series together using sliding windows.
# Error: There should be at least 48212 nrows in `data`



# Initializing an empty list to store the model
multivar_prophet_mods <- list()

# Initializing a dataframe to store the metrics to be calculated
multivar_metrics <- data.frame()


