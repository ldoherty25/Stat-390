## UNIVARIATE PROPHET MEENA

## UNIVARIATE PROPHET

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

## Modeling for Bolivia ---
#We have to rename the variables for Prophet to comprehend the vars
bolivia <- bolivia%>%
  rename(ds = date, y = owid_new_deaths)#%>%
  #column_to_rownames(var = "ds") #repurpose the date column to be used as the index of the dataframe

##don't think I need this here
#l = BoxCox.lambda(bolivia$y, method = "loglik")
#bolivia$y = BoxCox(bolivia$y, l)
#bolivia.m <- melt(bolivia, measure.vars=c("y", "y"))

b_train_size <- nrow(bolivia)
b_train_set <- ceiling(0.9 * b_train_size)
b_test_set <- ceiling((b_train_size - b_train_set))
##did it a different way in the end
#bolivia_fold_size <- floor(nrow(b_train_set) / num_folds)

bolivia_folds <- time_series_cv(
  bolivia,
  date_var = ds,
  initial = b_train_set,
  assess = b_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
bolivia_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

bolivia_all_metrics <- data.frame()

for (i in seq_along(bolivia_folds$splits)) {
  b_fold <- bolivia_folds$splits[[i]]
  b_train_fold <- b_fold$data[b_fold$in_id, ]
  b_test_fold <- b_fold$data[b_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  b_model <- prophet(b_train_fold, daily.seasonality = TRUE)
    
  # Make predictions on the test data
  bolivia_future <- make_future_dataframe(b_model, periods = nrow(b_test_fold))
  bolivia_forecast <- predict(b_model, bolivia_future)
  
  # Extract yhat and y values
  forecast_yhat <- bolivia_forecast$yhat[1:nrow(b_test_fold)]
  test_y <- b_test_fold$y
  
  print(length(bolivia_forecast$yhat))
  print(length(b_test_fold$y))
  
  #print(length(forecast_yhat))
  #print(length(test_y))
  
  print(dim(bolivia_forecast))
  print(dim(b_test_fold))
  
  b_metrics <- data.frame(
    RMSE = sqrt(mean((bolivia_forecast$yhat[1:nrow(b_test_fold)] - b_test_fold$y)^2)),
    MAE = mean(abs(bolivia_forecast$yhat[1:nrow(b_test_fold)] - b_test_fold$y)),
    MSE = mean((bolivia_forecast$yhat[1:nrow(b_test_fold)] - b_test_fold$y)^2),
    MASE = mean(abs(diff(b_train_fold$y)) / mean(abs(bolivia_forecast$yhat[1:nrow(b_test_fold)] - b_test_fold$y))),
    MAPE = mean(abs((b_test_fold$y - bolivia_forecast$yhat[1:nrow(b_test_fold)]) / b_test_fold$y)) * 100)
  # Accumulate metrics
  bolivia_all_metrics <- bind_rows(bolivia_all_metrics, b_metrics)
}
bolivia_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(bolivia_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(bolivia_all_metrics$MAE), "\n")
cat("Average MSE:", mean(bolivia_all_metrics$MSE), "\n")
cat("Average MASE:", mean(bolivia_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(bolivia_all_metrics$MAPE), "\n")

#plot.prophet()
    
#############################################################################################

## BRAZIL ##
brazil <- brazil%>%
  rename(ds = date, y = owid_new_deaths)#%>%

brazil_train_size <- nrow(brazil)
brazil_train_set <- ceiling(0.9 * brazil_train_size)
brazil_test_set <- ceiling((brazil_train_size - brazil_train_set))

brazil_folds <- time_series_cv(
  brazil,
  date_var = ds,
  initial = brazil_train_set,
  assess = brazil_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
brazil_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

brazil_all_metrics <- data.frame()

for (i in seq_along(brazil_folds$splits)) {
  brazil_fold <- brazil_folds$splits[[i]]
  brazil_train_fold <- brazil_fold$data[brazil_fold$in_id, ]
  brazil_test_fold <- brazil_fold$data[brazil_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  brazil_model <- prophet(brazil_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  brazil_future <- make_future_dataframe(brazil_model, periods = nrow(brazil_test_fold))
  brazil_forecast <- predict(brazil_model, brazil_future)
  
  # Extract yhat and y values
  brazil_forecast_yhat <- brazil_forecast$yhat[1:nrow(brazil_test_fold)]
  brazil_test_y <- brazil_test_fold$y
  
  brazil_metrics <- data.frame(
    RMSE = sqrt(mean((brazil_forecast$yhat[1:nrow(brazil_test_fold)] - brazil_test_fold$y)^2)),
    MAE = mean(abs(brazil_forecast$yhat[1:nrow(brazil_test_fold)] - brazil_test_fold$y)),
    MSE = mean((brazil_forecast$yhat[1:nrow(brazil_test_fold)] - brazil_test_fold$y)^2),
    MASE = mean(abs(diff(brazil_train_fold$y)) / mean(abs(brazil_forecast$yhat[1:nrow(brazil_test_fold)] - brazil_test_fold$y))),
    MAPE = mean(abs((brazil_test_fold$y - brazil_forecast$yhat[1:nrow(brazil_test_fold)]) / brazil_test_fold$y)) * 100)
  # Accumulate metrics
  brazil_all_metrics <- bind_rows(bolivia_all_metrics, brazil_metrics)
}
brazil_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(brazil_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(brazil_all_metrics$MAE), "\n")
cat("Average MSE:", mean(brazil_all_metrics$MSE), "\n")
cat("Average MASE:", mean(brazil_all_metrics$MSE), "\n")
cat("Average MAPE:", mean(brazil_all_metrics$MAPE), "\n")


###############################################################################################################
## COLOMBIA ##

colombia <- colombia%>%
  rename(ds = date, y = owid_new_deaths)#%>%

colombia_train_size <- nrow(colombia)
colombia_train_set <- ceiling(0.9 * colombia_train_size)
colombia_test_set <- ceiling((colombia_train_size - colombia_train_set))

colombia_folds <- time_series_cv(
  colombia,
  date_var = ds,
  initial = colombia_train_set,
  assess = colombia_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
colombia_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

colombia_all_metrics <- data.frame()

for (i in seq_along(colombia_folds$splits)) {
  colombia_fold <- colombia_folds$splits[[i]]
  colombia_train_fold <- colombia_fold$data[colombia_fold$in_id, ]
  colombia_test_fold <- colombia_fold$data[colombia_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  colombia_model <- prophet(colombia_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  colombia_future <- make_future_dataframe(colombia_model, periods = nrow(colombia_test_fold))
  colombia_forecast <- predict(colombia_model, colombia_future)
  
  # Extract yhat and y values
  colombia_forecast_yhat <- colombia_forecast$yhat[1:nrow(colombia_test_fold)]
  colombia_test_y <- colombia_test_fold$y
  
  colombia_metrics <- data.frame(
    RMSE = sqrt(mean((colombia_forecast$yhat[1:nrow(colombia_test_fold)] - colombia_test_fold$y)^2)),
    MAE = mean(abs(colombia_forecast$yhat[1:nrow(colombia_test_fold)] - colombia_test_fold$y)),
    MSE = mean((colombia_forecast$yhat[1:nrow(colombia_test_fold)] - colombia_test_fold$y)^2),
    MASE = mean(abs(diff(colombia_train_fold$y)) / mean(abs(colombia_forecast$yhat[1:nrow(colombia_test_fold)] - colombia_test_fold$y))),
    MAPE = mean(abs((colombia_test_fold$y - colombia_forecast$yhat[1:nrow(colombia_test_fold)]) / colombia_test_fold$y)) * 100)
  # Accumulate metrics
  colombia_all_metrics <- bind_rows(colombia_all_metrics, colombia_metrics)
}
colombia_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(colombia_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(colombia_all_metrics$MAE), "\n")
cat("Average MSE:", mean(colombia_all_metrics$MSE), "\n")
cat("Average MASE:", mean(colombia_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(colombia_all_metrics$MAPE), "\n")


##########################################################################################################
## IRAN ##

iran <- iran%>%
  rename(ds = date, y = owid_new_deaths)#%>%

iran_train_size <- nrow(colombia)
iran_train_set <- ceiling(0.9 * iran_train_size)
iran_test_set <- ceiling((iran_train_size - iran_train_set))

iran_folds <- time_series_cv(
  iran,
  date_var = ds,
  initial = iran_train_set,
  assess = iran_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
iran_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

iran_all_metrics <- data.frame()

for (i in seq_along(iran_folds$splits)) {
  iran_fold <- iran_folds$splits[[i]]
  iran_train_fold <- iran_fold$data[iran_fold$in_id, ]
  iran_test_fold <- iran_fold$data[iran_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  iran_model <- prophet(iran_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  iran_future <- make_future_dataframe(iran_model, periods = nrow(iran_test_fold))
  iran_forecast <- predict(iran_model, iran_future)
  
  # Extract yhat and y values
  iran_forecast_yhat <- iran_forecast$yhat[1:nrow(iran_test_fold)]
  iran_test_y <- iran_test_fold$y
  
  iran_metrics <- data.frame(
    RMSE = sqrt(mean((iran_forecast$yhat[1:nrow(iran_test_fold)] - iran_test_fold$y)^2)),
    MAE = mean(abs(iran_forecast$yhat[1:nrow(iran_test_fold)] - iran_test_fold$y)),
    MSE = mean((iran_forecast$yhat[1:nrow(iran_test_fold)] - iran_test_fold$y)^2),
    MASE = mean(abs(diff(iran_train_fold$y)) / mean(abs(iran_forecast$yhat[1:nrow(iran_test_fold)] - iran_test_fold$y))),
    MAPE = mean(abs((iran_test_fold$y - iran_forecast$yhat[1:nrow(iran_test_fold)]) / iran_test_fold$y)) * 100)
  # Accumulate metrics
  iran_all_metrics <- bind_rows(iran_all_metrics, iran_metrics)
}
iran_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(iran_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(iran_all_metrics$MAE), "\n")
cat("Average MSE:", mean(iran_all_metrics$MSE), "\n")
cat("Average MASE:", mean(iran_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(iran_all_metrics$MAPE), "\n")


#####################################################################################################
## MEXICO ##

mexico <- mexico%>%
  rename(ds = date, y = owid_new_deaths)#%>%

mexico_train_size <- nrow(mexico)
mexico_train_set <- ceiling(0.9 * mexico_train_size)
mexico_test_set <- ceiling((mexico_train_size - mexico_train_set))

mexico_folds <- time_series_cv(
  mexico,
  date_var = ds,
  initial = mexico_train_set,
  assess = mexico_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
mexico_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

mexico_all_metrics <- data.frame()

for (i in seq_along(mexico_folds$splits)) {
  mexico_fold <- mexico_folds$splits[[i]]
  mexico_train_fold <- mexico_fold$data[mexico_fold$in_id, ]
  mexico_test_fold <- mexico_fold$data[mexico_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  mexico_model <- prophet(mexico_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  mexico_future <- make_future_dataframe(mexico_model, periods = nrow(mexico_test_fold))
  mexico_forecast <- predict(mexico_model, mexico_future)
  
  # Extract yhat and y values
  mexico_forecast_yhat <- mexico_forecast$yhat[1:nrow(mexico_test_fold)]
  mexico_test_y <- mexico_test_fold$y
  
  mexico_metrics <- data.frame(
    RMSE = sqrt(mean((mexico_forecast$yhat[1:nrow(mexico_test_fold)] - mexico_test_fold$y)^2)),
    MAE = mean(abs(mexico_forecast$yhat[1:nrow(mexico_test_fold)] - mexico_test_fold$y)),
    MSE = mean((mexico_forecast$yhat[1:nrow(mexico_test_fold)] - mexico_test_fold$y)^2),
    MASE = mean(abs(diff(mexico_train_fold$y)) / mean(abs(mexico_forecast$yhat[1:nrow(mexico_test_fold)] - mexico_test_fold$y))),
    MAPE = mean(abs((mexico_test_fold$y - mexico_forecast$yhat[1:nrow(mexico_test_fold)]) / mexico_test_fold$y)) * 100)
  # Accumulate metrics
  mexico_all_metrics <- bind_rows(mexico_all_metrics, mexico_metrics)
}
mexico_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(mexico_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(mexico_all_metrics$MAE), "\n")
cat("Average MSE:", mean(mexico_all_metrics$MSE), "\n")
cat("Average MASE:", mean(mexico_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(mexico_all_metrics$MAPE), "\n")


##############################################################################################################
## PERU ##

peru <- peru%>%
  rename(ds = date, y = owid_new_deaths)#%>%

peru_train_size <- nrow(peru)
peru_train_set <- ceiling(0.9 * peru_train_size)
peru_test_set <- ceiling((peru_train_size - peru_train_set))

peru_folds <- time_series_cv(
  peru,
  date_var = ds,
  initial = peru_train_set,
  assess = peru_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
peru_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

peru_all_metrics <- data.frame()

for (i in seq_along(peru_folds$splits)) {
  peru_fold <- peru_folds$splits[[i]]
  peru_train_fold <- peru_fold$data[peru_fold$in_id, ]
  peru_test_fold <- peru_fold$data[peru_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  peru_model <- prophet(peru_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  peru_future <- make_future_dataframe(peru_model, periods = nrow(peru_test_fold))
  peru_forecast <- predict(peru_model, peru_future)
  
  # Extract yhat and y values
  peru_forecast_yhat <- peru_forecast$yhat[1:nrow(peru_test_fold)]
  peru_test_y <- peru_test_fold$y
  
  peru_metrics <- data.frame(
    RMSE = sqrt(mean((peru_forecast$yhat[1:nrow(peru_test_fold)] - peru_test_fold$y)^2)),
    MAE = mean(abs(peru_forecast$yhat[1:nrow(peru_test_fold)] - peru_test_fold$y)),
    MSE = mean((peru_forecast$yhat[1:nrow(peru_test_fold)] - peru_test_fold$y)^2),
    MASE = mean(abs(diff(peru_train_fold$y)) / mean(abs(peru_forecast$yhat[1:nrow(peru_test_fold)] - peru_test_fold$y))),
    MAPE = mean(abs((peru_test_fold$y - peru_forecast$yhat[1:nrow(peru_test_fold)]) / peru_test_fold$y)) * 100)
  # Accumulate metrics
  peru_all_metrics <- bind_rows(peru_all_metrics, peru_metrics)
}
peru_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(peru_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(peru_all_metrics$MAE), "\n")
cat("Average MSE:", mean(peru_all_metrics$MSE), "\n")
cat("Average MASE:", mean(peru_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(peru_all_metrics$MAPE), "\n")


####################################################################################################
## RUSSIA ##

russia <- russia%>%
  rename(ds = date, y = owid_new_deaths)#%>%

russia_train_size <- nrow(russia)
russia_train_set <- ceiling(0.9 * russia_train_size)
russia_test_set <- ceiling((russia_train_size - russia_train_set))

russia_folds <- time_series_cv(
  russia,
  date_var = ds,
  initial = russia_train_set,
  assess = russia_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
russia_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

russia_all_metrics <- data.frame()

for (i in seq_along(russia_folds$splits)) {
  russia_fold <- russia_folds$splits[[i]]
  russia_train_fold <- russia_fold$data[russia_fold$in_id, ]
  russia_test_fold <- russia_fold$data[russia_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  russia_model <- prophet(russia_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  russia_future <- make_future_dataframe(russia_model, periods = nrow(russia_test_fold))
  russia_forecast <- predict(russia_model, russia_future)
  
  # Extract yhat and y values
  russia_forecast_yhat <- russia_forecast$yhat[1:nrow(russia_test_fold)]
  russia_test_y <- russia_test_fold$y
  
  russia_metrics <- data.frame(
    RMSE = sqrt(mean((russia_forecast$yhat[1:nrow(russia_test_fold)] - russia_test_fold$y)^2)),
    MAE = mean(abs(russia_forecast$yhat[1:nrow(russia_test_fold)] - russia_test_fold$y)),
    MSE = mean((russia_forecast$yhat[1:nrow(russia_test_fold)] - russia_test_fold$y)^2),
    MASE = mean(abs(diff(russia_train_fold$y)) / mean(abs(russia_forecast$yhat[1:nrow(russia_test_fold)] - russia_test_fold$y))),
    MAPE = mean(abs((russia_test_fold$y - russia_forecast$yhat[1:nrow(russia_test_fold)]) / russia_test_fold$y)) * 100)
  # Accumulate metrics
  russia_all_metrics <- bind_rows(russia_all_metrics, russia_metrics)
}
russia_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(russia_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(russia_all_metrics$MAE), "\n")
cat("Average MSE:", mean(russia_all_metrics$MSE), "\n")
cat("Average MASE:", mean(russia_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(russia_all_metrics$MAPE), "\n")


####################################################################################################
## SAUDI ##

saudi <- saudi%>%
  rename(ds = date, y = owid_new_deaths)#%>%

saudi_train_size <- nrow(saudi)
saudi_train_set <- ceiling(0.9 * saudi_train_size)
saudi_test_set <- ceiling((saudi_train_size - saudi_train_set))

saudi_folds <- time_series_cv(
  saudi,
  date_var = ds,
  initial = saudi_train_set,
  assess = saudi_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
saudi_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

saudi_all_metrics <- data.frame()

for (i in seq_along(saudi_folds$splits)) {
  saudi_fold <- saudi_folds$splits[[i]]
  saudi_train_fold <- saudi_fold$data[saudi_fold$in_id, ]
  saudi_test_fold <- saudi_fold$data[saudi_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  saudi_model <- prophet(saudi_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  saudi_future <- make_future_dataframe(saudi_model, periods = nrow(saudi_test_fold))
  saudi_forecast <- predict(saudi_model, saudi_future)
  
  # Extract yhat and y values
  saudi_forecast_yhat <- saudi_forecast$yhat[1:nrow(saudi_test_fold)]
  saudi_test_y <- saudi_test_fold$y
  
  saudi_metrics <- data.frame(
    RMSE = sqrt(mean((saudi_forecast$yhat[1:nrow(saudi_test_fold)] - saudi_test_fold$y)^2)),
    MAE = mean(abs(saudi_forecast$yhat[1:nrow(saudi_test_fold)] - saudi_test_fold$y)),
    MSE = mean((saudi_forecast$yhat[1:nrow(saudi_test_fold)] - saudi_test_fold$y)^2),
    MASE = mean(abs(diff(saudi_train_fold$y)) / mean(abs(saudi_forecast$yhat[1:nrow(saudi_test_fold)] - saudi_test_fold$y))),
    MAPE = mean(abs((saudi_test_fold$y - saudi_forecast$yhat[1:nrow(saudi_test_fold)]) / saudi_test_fold$y)) * 100)
  # Accumulate metrics
  saudi_all_metrics <- bind_rows(saudi_all_metrics, saudi_metrics)
}
saudi_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(saudi_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(saudi_all_metrics$MAE), "\n")
cat("Average MSE:", mean(saudi_all_metrics$MSE), "\n")
cat("Average MASE:", mean(saudi_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(saudi_all_metrics$MAPE), "\n")


#################################################################################################################
## TURKEY ##

turkey <- turkey%>%
  rename(ds = date, y = owid_new_deaths)#%>%

turkey_train_size <- nrow(turkey)
turkey_train_set <- ceiling(0.9 * turkey_train_size)
turkey_test_set <- ceiling((turkey_train_size - turkey_train_set))

turkey_folds <- time_series_cv(
  turkey,
  date_var = ds,
  initial = turkey_train_set,
  assess = turkey_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
turkey_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

turkey_all_metrics <- data.frame()

for (i in seq_along(turkey_folds$splits)) {
  turkey_fold <- turkey_folds$splits[[i]]
  turkey_train_fold <- turkey_fold$data[turkey_fold$in_id, ]
  turkey_test_fold <- turkey_fold$data[turkey_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  turkey_model <- prophet(turkey_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  turkey_future <- make_future_dataframe(turkey_model, periods = nrow(turkey_test_fold))
  turkey_forecast <- predict(turkey_model, turkey_future)
  
  # Extract yhat and y values
  turkey_forecast_yhat <- turkey_forecast$yhat[1:nrow(turkey_test_fold)]
  turkey_test_y <- turkey_test_fold$y
  
  turkey_metrics <- data.frame(
    RMSE = sqrt(mean((turkey_forecast$yhat[1:nrow(turkey_test_fold)] - turkey_test_fold$y)^2)),
    MAE = mean(abs(turkey_forecast$yhat[1:nrow(turkey_test_fold)] - turkey_test_fold$y)),
    MSE = mean((turkey_forecast$yhat[1:nrow(turkey_test_fold)] - turkey_test_fold$y)^2),
    MASE = mean(abs(diff(turkey_train_fold$y)) / mean(abs(turkey_forecast$yhat[1:nrow(turkey_test_fold)] - turkey_test_fold$y))),
    MAPE = mean(abs((turkey_test_fold$y - turkey_forecast$yhat[1:nrow(turkey_test_fold)]) / turkey_test_fold$y)) * 100)
  # Accumulate metrics
  turkey_all_metrics <- bind_rows(turkey_all_metrics, turkey_metrics)
}
turkey_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(turkey_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(turkey_all_metrics$MAE), "\n")
cat("Average MSE:", mean(turkey_all_metrics$MSE), "\n")
cat("Average MASE:", mean(turkey_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(turkey_all_metrics$MAPE), "\n")


#############################################################################################################
## US ##

us <- us%>%
  rename(ds = date, y = owid_new_deaths)#%>%

us_train_size <- nrow(us)
us_train_set <- ceiling(0.9 * us_train_size)
us_test_set <- ceiling((us_train_size - us_train_set))

us_folds <- time_series_cv(
  us,
  date_var = ds,
  initial = us_train_set,
  assess = us_test_set,
  fold = 1,
  slice_limit = 1)

#Filtering by slice
us_folds %>% tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

us_all_metrics <- data.frame()

for (i in seq_along(us_folds$splits)) {
  us_fold <- us_folds$splits[[i]]
  us_train_fold <- us_fold$data[us_fold$in_id, ]
  us_test_fold <- us_fold$data[us_fold$out_id, ]
  
  #Fitting the model on the training data for the current fold
  us_model <- prophet(us_train_fold, daily.seasonality = TRUE)
  
  # Make predictions on the test data
  us_future <- make_future_dataframe(us_model, periods = nrow(us_test_fold))
  us_forecast <- predict(us_model, us_future)
  
  # Extract yhat and y values
  us_forecast_yhat <- us_forecast$yhat[1:nrow(us_test_fold)]
  us_test_y <- us_test_fold$y
  
  us_metrics <- data.frame(
    RMSE = sqrt(mean((us_forecast$yhat[1:nrow(us_test_fold)] - us_test_fold$y)^2)),
    MAE = mean(abs(us_forecast$yhat[1:nrow(us_test_fold)] - us_test_fold$y)),
    MSE = mean((us_forecast$yhat[1:nrow(us_test_fold)] - us_test_fold$y)^2),
    MASE = mean(abs(diff(us_train_fold$y)) / mean(abs(us_forecast$yhat[1:nrow(us_test_fold)] - us_test_fold$y))),
    MAPE = mean(abs((us_test_fold$y - us_forecast$yhat[1:nrow(us_test_fold)]) / us_test_fold$y)) * 100)
  # Accumulate metrics
  us_all_metrics <- bind_rows(us_all_metrics, us_metrics)
}
us_all_metrics

#Display average metrics across all folds
cat("Average RMSE:", mean(us_all_metrics$RMSE), "\n")
cat("Average MAE:", mean(us_all_metrics$MAE), "\n")
cat("Average MSE:", mean(us_all_metrics$MSE), "\n")
cat("Average MASE:", mean(us_all_metrics$MASE), "\n")
cat("Average MAPE:", mean(us_all_metrics$MAPE), "\n")
