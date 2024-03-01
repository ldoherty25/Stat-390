## Univariate Prophet

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

# set a seed
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

####################################################################################################################################################################################################################################################################  

## Bolivia model

bolivia <- bolivia %>%
  rename(ds = date, y = owid_new_deaths)

bolivia_total_days <- nrow(bolivia)
bolivia_train_days <- ceiling(0.9 * bolivia_total_days)
bolivia_test_days <- ceiling((bolivia_total_days - bolivia_train_days))

# creating folds

bolivia_folds <- time_series_cv(
  bolivia,
  date_var = ds,
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

# prophet model fit

fitting <- function(bolivia_folds) {
  for (i in seq_along(bolivia_folds$splits)) {
    fold <- bolivia_folds$splits[[i]]
    bolivia_train_data = fold$data[fold$in_id, ]
    bolivia_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    bolivia_prophet <- prophet(bolivia_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    bolivia_future <- make_future_dataframe(bolivia_prophet, periods = nrow(bolivia_test_data))
    bolivia_forecast <- predict(bolivia_prophet, bolivia_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((bolivia_forecast$yhat - bolivia_test_data$y)^2)),
      MAE = mean(abs(bolivia_forecast$yhat - bolivia_test_data$y)),
      MSE = mean((bolivia_forecast$yhat - bolivia_test_data$y)^2),
      MAPE = mean(abs((bolivia_test_data$y - bolivia_forecast$yhat) / bolivia_test_data$y)) * 100)
      all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

bolivia_metrics <- fitting(bolivia_folds)

cat("RMSE:", mean(bolivia_metrics$RMSE))
cat("MAE:", mean(bolivia_metrics$MAE))
cat("MSE:", mean(bolivia_metrics$MSE))
cat("MAPE:", mean(bolivia_metrics$MAPE))


bolivia_all_dates <- c(bolivia_train_data$ds, bolivia_test_data$ds)
bolivia_all_values <- c(bolivia_train_data$y, bolivia_test_data$y)

bolivia_forecast_df <- data.frame(
  date = bolivia_all_dates,
  actual_deaths = bolivia_all_values,
  forecasted_deaths = bolivia_forecast$yhat
)

bolivia_forecast_df_abs <- bolivia_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(bolivia_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")


##################################################################################################################################################################################################################################################################


## Brazil Model

brazil <- brazil %>%
  rename(ds = date, y = owid_new_deaths)

brazil_total_days <- nrow(brazil)
brazil_train_days <- ceiling(0.9 * brazil_total_days)
brazil_test_days <- ceiling((brazil_total_days - brazil_train_days))

# creating folds

brazil_folds <- time_series_cv(
  brazil,
  date_var = ds,
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

# prophet model

#fitting <- function(brazil_folds) {
for (i in seq_along(brazil_folds$splits)) {
  fold <- brazil_folds$splits[[i]]
  brazil_train_data = fold$data[fold$in_id, ]
  brazil_test_data = fold$data[fold$out_id, ]
    
    #all_metrics <- data.frame()
    
  brazil_prophet <- prophet(brazil_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
  brazil_future <- make_future_dataframe(brazil_prophet, periods = nrow(brazil_test_data))
  brazil_forecast <- predict(brazil_prophet, brazil_future)
    
#  metrics <- data.frame(
#    RMSE = sqrt(mean((brazil_test_data$y - brazil_forecast$yhat)^2)),
#    MAE = mean(abs(brazil_test_data$y - brazil_forecast$yhat)),
#    MSE = mean((brazil_test_data$y - brazil_forecast$yhat)^2),
#    MAPE = mean(abs((brazil_test_data$y - brazil_forecast$yhat) / brazil_test_data$y)) * 100)
#  all_metrics <- bind_rows(all_metrics, metrics)
    
}
#  return (all_metrics)
#}

brazil_metrics <- fitting(brazil_folds)

## In brazil_test_data$y - brazil_forecast$yhat :
## longer object length is not a multiple of shorter object length
## 2: In brazil_test_data$y - brazil_forecast$yhat :
##   longer object length is not a multiple of shorter object length
## 3: In brazil_test_data$y - brazil_forecast$yhat :
##   longer object length is not a multiple of shorter object length
## 4: In brazil_test_data$y - brazil_forecast$yhat :
##   longer object length is not a multiple of shorter object length
## 5: In (brazil_test_data$y - brazil_forecast$yhat)/brazil_test_data$y :
##   longer object length is not a multiple of shorter object length

cat("RMSE:", mean(brazil_metrics$RMSE))
cat("MAE:", mean(brazil_metrics$MAE))
cat("MSE:", mean(brazil_metrics$MSE))
cat("MAPE:", mean(brazil_metrics$MAPE))

brazil_all_dates <- c(brazil_train_data$ds, brazil_test_data$ds)
brazil_all_values <- c(brazil_train_data$y, brazil_test_data$y)

brazil_forecast_df <- data.frame(
  date = brazil_all_dates,
  actual_deaths = brazil_all_values,
  forecasted_deaths = brazil_forecast$yhat
)

brazil_forecast_df_abs <- brazil_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(brazil_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")

##################################################################################################################################################################################################################################################################


## Colombia

colombia <- colombia %>%
  rename(ds = date, y = owid_new_deaths)

colombia_total_days <- nrow(colombia)
colombia_train_days <- ceiling(0.9 * colombia_total_days)
colombia_test_days <- ceiling((colombia_total_days - colombia_train_days))

# creating folds

colombia_folds <- time_series_cv(
  colombia,
  date_var = ds,
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

# prophet model fit

fitting <- function(colombia_folds) {
  for (i in seq_along(colombia_folds$splits)) {
    fold <- colombia_folds$splits[[i]]
    colombia_train_data = fold$data[fold$in_id, ]
    colombia_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    colombia_prophet <- prophet(colombia_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    colombia_future <- make_future_dataframe(colombia_prophet, periods = nrow(colombia_test_data))
    colombia_forecast <- predict(colombia_prophet, colombia_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((colombia_forecast$yhat - colombia_test_data$y)^2)),
      MAE = mean(abs(colombia_forecast$yhat - colombia_test_data$y)),
      MSE = mean((colombia_forecast$yhat - colombia_test_data$y)^2),
      MAPE = mean(abs((colombia_test_data$y - colombia_forecast$yhat) / colombia_test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

colombia_metrics <- fitting(colombia_folds)

cat("RMSE:", mean(colombia_metrics$RMSE))
cat("MAE:", mean(colombia_metrics$MAE))
cat("MSE:", mean(colombia_metrics$MSE))
cat("MAPE:", mean(colombia_metrics$MAPE))

colombia_all_dates <- c(colombia_train_data$ds, colombia_test_data$ds)
colombia_all_values <- c(colombia_train_data$y, colombia_test_data$y)

colombia_forecast_df <- data.frame(
  date = colombia_all_dates,
  actual_deaths = colombia_all_values,
  forecasted_deaths = colombia_forecast$yhat
)

colombia_forecast_df_abs <- colombia_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(colombia_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")


##################################################################################################################################################################################################################################################################


## Iran

iran <- iran %>%
  rename(ds = date, y = owid_new_deaths)

iran_total_days <- nrow(iran)
iran_train_days <- ceiling(0.9 * iran_total_days)
iran_test_days <- ceiling((iran_total_days - iran_train_days))

# creating folds

iran_folds <- time_series_cv(
  iran,
  date_var = ds,
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

# prophet model fit

fitting <- function(iran_folds) {
  for (i in seq_along(iran_folds$splits)) {
    fold <- iran_folds$splits[[i]]
    iran_train_data = fold$data[fold$in_id, ]
    iran_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    iran_prophet <- prophet(iran_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    iran_future <- make_future_dataframe(iran_prophet, periods = nrow(iran_test_data))
    iran_forecast <- predict(iran_prophet, iran_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((iran_forecast$yhat - iran_test_data$y)^2)),
      MAE = mean(abs(iran_forecast$yhat - iran_test_data$y)),
      MSE = mean((iran_forecast$yhat - iran_test_data$y)^2),
      MAPE = mean(abs((iran_test_data$y - iran_forecast$yhat) / iran_test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

iran_metrics <- fitting(iran_folds)

cat("RMSE:", mean(iran_metrics$RMSE))
cat("MAE:", mean(iran_metrics$MAE))
cat("MSE:", mean(iran_metrics$MSE))
cat("MAPE:", mean(iran_metrics$MAPE))

iran_all_dates <- c(iran_train_data$ds, iran_test_data$ds)
iran_all_values <- c(iran_train_data$y, iran_test_data$y)

iran_forecast_df <- data.frame(
  date = iran_all_dates,
  actual_deaths = iran_all_values,
  forecasted_deaths = iran_forecast$yhat
)

iran_forecast_df_abs <- iran_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(iran_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")


##################################################################################################################################################################################################################################################################


## Mexico

mexico <- mexico %>%
  rename(ds = date, y = owid_new_deaths)

mexico_total_days <- nrow(mexico)
mexico_train_days <- ceiling(0.9 * mexico_total_days)
mexico_test_days <- ceiling((mexico_total_days - mexico_train_days))

# creating folds

mexico_folds <- time_series_cv(
  mexico,
  date_var = ds,
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

# prophet model fit

fitting <- function(mexico_folds) {
  for (i in seq_along(mexico_folds$splits)) {
    fold <- mexico_folds$splits[[i]]
    mexico_train_data = fold$data[fold$in_id, ]
    mexico_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    mexico_prophet <- prophet(mexico_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    mexico_future <- make_future_dataframe(mexico_prophet, periods = nrow(mexico_test_data))
    mexico_forecast <- predict(mexico_prophet, mexico_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((mexico_forecast$yhat - mexico_test_data$y)^2)),
      MAE = mean(abs(mexico_forecast$yhat - mexico_test_data$y)),
      MSE = mean((mexico_forecast$yhat - mexico_test_data$y)^2),
      MAPE = mean(abs((mexico_test_data$y - mexico_forecast$yhat) / mexico_test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

mexico_metrics <- fitting(mexico_folds)

cat("RMSE:", mean(mexico_metrics$RMSE))
cat("MAE:", mean(mexico_metrics$MAE))
cat("MSE:", mean(mexico_metrics$MSE))
cat("MAPE:", mean(mexico_metrics$MAPE))

mexico_all_dates <- c(mexico_train_data$ds, mexico_test_data$ds)
mexico_all_values <- c(mexico_train_data$y, mexico_test_data$y)

mexico_forecast_df <- data.frame(
  date = mexico_all_dates,
  actual_deaths = mexico_all_values,
  forecasted_deaths = mexico_forecast$yhat
)

mexico_forecast_df_abs <- mexico_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(mexico_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")


##################################################################################################################################################################################################################################################################


## Peru

peru <- peru %>%
  rename(ds = date, y = owid_new_deaths)

peru_total_days <- nrow(peru)
peru_train_days <- ceiling(0.9 * peru_total_days)
peru_test_days <- ceiling((peru_total_days - peru_train_days))

# creating folds

peru_folds <- time_series_cv(
  peru,
  date_var = ds,
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

# prophet model fit

fitting <- function(peru_folds) {
  for (i in seq_along(peru_folds$splits)) {
    fold <- peru_folds$splits[[i]]
    peru_train_data = fold$data[fold$in_id, ]
    peru_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    peru_prophet <- prophet(peru_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    peru_future <- make_future_dataframe(peru_prophet, periods = nrow(peru_test_data))
    peru_forecast <- predict(peru_prophet, peru_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((peru_forecast$yhat - peru_test_data$y)^2)),
      MAE = mean(abs(peru_forecast$yhat - peru_test_data$y)),
      MSE = mean((peru_forecast$yhat - peru_test_data$y)^2),
      MAPE = mean(abs((peru_test_data$y - peru_forecast$yhat) / peru_test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

peru_metrics <- fitting(peru_folds)

cat("RMSE:", mean(peru_metrics$RMSE))
cat("MAE:", mean(peru_metrics$MAE))
cat("MSE:", mean(peru_metrics$MSE))
cat("MAPE:", mean(peru_metrics$MAPE))

peru_all_dates <- c(peru_train_data$ds, peru_test_data$ds)
peru_all_values <- c(peru_train_data$y, peru_test_data$y)

peru_forecast_df <- data.frame(
  date = peru_all_dates,
  actual_deaths = peru_all_values,
  forecasted_deaths = peru_forecast$yhat
)

peru_forecast_df_abs <- peru_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(peru_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")


##################################################################################################################################################################################################################################################################

## Russia

russia <- russia %>%
  rename(ds = date, y = owid_new_deaths)

russia_total_days <- nrow(russia)
russia_train_days <- ceiling(0.9 * russia_total_days)
russia_test_days <- ceiling((russia_total_days - russia_train_days))

# creating folds

russia_folds <- time_series_cv(
  russia,
  date_var = ds,
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

# prophet model fit

fitting <- function(russia_folds) {
  for (i in seq_along(russia_folds$splits)) {
    fold <- russia_folds$splits[[i]]
    russia_train_data = fold$data[fold$in_id, ]
    russia_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    russia_prophet <- prophet(russia_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    russia_future <- make_future_dataframe(russia_prophet, periods = nrow(russia_test_data))
    russia_forecast <- predict(russia_prophet, russia_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((russia_forecast$yhat - russia_test_data$y)^2)),
      MAE = mean(abs(russia_forecast$yhat - russia_test_data$y)),
      MSE = mean((russia_forecast$yhat - russia_test_data$y)^2),
      MAPE = mean(abs((russia_test_data$y - russia_forecast$yhat) / russia_test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

russia_metrics <- fitting(russia_folds)

cat("RMSE:", mean(russia_metrics$RMSE))
cat("MAE:", mean(russia_metrics$MAE))
cat("MSE:", mean(russia_metrics$MSE))
cat("MAPE:", mean(russia_metrics$MAPE))

russia_all_dates <- c(russia_train_data$ds, russia_test_data$ds)
russia_all_values <- c(russia_train_data$y, russia_test_data$y)

russia_forecast_df <- data.frame(
  date = russia_all_dates,
  actual_deaths = russia_all_values,
  forecasted_deaths = russia_forecast$yhat
)

russia_forecast_df_abs <- russia_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(russia_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")

##################################################################################################################################################################################################################################################################


## Saudi Arabia

saudi <- saudi %>%
  rename(ds = date, y = owid_new_deaths)

saudi_total_days <- nrow(saudi)
saudi_train_days <- ceiling(0.9 * saudi_total_days)
saudi_test_days <- ceiling((saudi_total_days - saudi_train_days))

# creating folds

saudi_folds <- time_series_cv(
  saudi,
  date_var = ds,
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

# prophet model fit

fitting <- function(saudi_folds) {
  for (i in seq_along(saudi_folds$splits)) {
    fold <- saudi_folds$splits[[i]]
    saudi_train_data = fold$data[fold$in_id, ]
    saudi_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    saudi_prophet <- prophet(saudi_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    saudi_future <- make_future_dataframe(saudi_prophet, periods = nrow(saudi_test_data))
    saudi_forecast <- predict(saudi_prophet, saudi_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((saudi_forecast$yhat - saudi_test_data$y)^2)),
      MAE = mean(abs(saudi_forecast$yhat - saudi_test_data$y)),
      MSE = mean((saudi_forecast$yhat - saudi_test_data$y)^2),
      MAPE = mean(abs((saudi_test_data$y - saudi_forecast$yhat) / saudi_test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

saudi_metrics <- fitting(saudi_folds)

cat("RMSE:", mean(saudi_metrics$RMSE))
cat("MAE:", mean(saudi_metrics$MAE))
cat("MSE:", mean(saudi_metrics$MSE))
cat("MAPE:", mean(saudi_metrics$MAPE))

saudi_all_dates <- c(saudi_train_data$ds, saudi_test_data$ds)
saudi_all_values <- c(saudi_train_data$y, saudi_test_data$y)

saudi_forecast_df <- data.frame(
  date = saudi_all_dates,
  actual_deaths = saudi_all_values,
  forecasted_deaths = saudi_forecast$yhat
)

saudi_forecast_df_abs <- saudi_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(saudi_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")


##################################################################################################################################################################################################################################################################

## Turkey

turkey <- turkey %>%
  rename(ds = date, y = owid_new_deaths)

turkey_total_days <- nrow(turkey)
turkey_train_days <- ceiling(0.9 * turkey_total_days)
turkey_test_days <- ceiling((turkey_total_days - turkey_train_days))

# creating folds

turkey_folds <- time_series_cv(
  turkey,
  date_var = ds,
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

# prophet model fit

fitting <- function(turkey_folds) {
  for (i in seq_along(turkey_folds$splits)) {
    fold <- turkey_folds$splits[[i]]
    turkey_train_data = fold$data[fold$in_id, ]
    turkey_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    turkey_prophet <- prophet(turkey_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    turkey_future <- make_future_dataframe(turkey_prophet, periods = nrow(turkey_test_data))
    turkey_forecast <- predict(turkey_prophet, turkey_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((turkey_forecast$yhat - turkey_test_data$y)^2)),
      MAE = mean(abs(turkey_forecast$yhat - turkey_test_data$y)),
      MSE = mean((turkey_forecast$yhat - turkey_test_data$y)^2),
      MAPE = mean(abs((turkey_test_data$y - turkey_forecast$yhat) / turkey_test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

turkey_metrics <- fitting(turkey_folds)

cat("RMSE:", mean(turkey_metrics$RMSE))
cat("MAE:", mean(turkey_metrics$MAE))
cat("MSE:", mean(turkey_metrics$MSE))
cat("MAPE:", mean(turkey_metrics$MAPE))

turkey_all_dates <- c(turkey_train_data$ds, turkey_test_data$ds)
turkey_all_values <- c(turkey_train_data$y, turkey_test_data$y)

turkey_forecast_df <- data.frame(
  date = turkey_all_dates,
  actual_deaths = turkey_all_values,
  forecasted_deaths = turkey_forecast$yhat
)

turkey_forecast_df_abs <- turkey_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(turkey_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")


##################################################################################################################################################################################################################################################################

## US

us <- us %>%
  rename(ds = date, y = owid_new_deaths)

us_total_days <- nrow(us)
us_train_days <- ceiling(0.9 * us_total_days)
us_test_days <- ceiling((us_total_days - us_train_days))

# creating folds

us_folds <- time_series_cv(
  us,
  date_var = ds,
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

# prophet model fit

fitting <- function(us_folds) {
  for (i in seq_along(us_folds$splits)) {
    fold <- us_folds$splits[[i]]
    us_train_data = fold$data[fold$in_id, ]
    us_test_data = fold$data[fold$out_id, ]
    
    all_metrics <- data.frame()
    
    us_prophet <- prophet(us_train_data, 
                               yearly.seasonality = FALSE,
                               weekly.seasonality = "auto",
                               daily.seasonality = FALSE,
                               growth = "linear")
    
    us_future <- make_future_dataframe(us_prophet, periods = nrow(us_test_data))
    us_forecast <- predict(us_prophet, us_future)
    
    metrics <- data.frame(
      RMSE = sqrt(mean((us_forecast$yhat - us_test_data$y)^2)),
      MAE = mean(abs(us_forecast$yhat - us_test_data$y)),
      MSE = mean((us_forecast$yhat - us_test_data$y)^2),
      MAPE = mean(abs((us_test_data$y - us_forecast$yhat) / us_test_data$y)) * 100)
    all_metrics <- bind_rows(all_metrics, metrics)
    
  }
  return (all_metrics)
}

us_metrics <- fitting(us_folds)

cat("RMSE:", mean(us_metrics$RMSE))
cat("MAE:", mean(us_metrics$MAE))
cat("MSE:", mean(us_metrics$MSE))
cat("MAPE:", mean(us_metrics$MAPE))


us_all_dates <- c(us_train_data$ds, us_test_data$ds)
us_all_values <- c(us_train_data$y, us_test_data$y)

us_forecast_df <- data.frame(
  date = us_all_dates,
  actual_deaths = us_all_values,
  forecasted_deaths = us_forecast$yhat
)

us_forecast_df_abs <- us_forecast_df %>% 
  mutate(forecasted_deaths = abs(forecasted_deaths))

ggplot(us_forecast_df_abs, aes(date)) +
  geom_line(aes(y = actual_deaths), color = "red") +
  geom_line(aes(y = forecasted_deaths), color = "blue")


##################################################################################################################################################################################################################################################################
