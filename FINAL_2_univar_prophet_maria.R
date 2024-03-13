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

# loading data
load("data/preprocessed/univariate/not_split/bolivia.rda")

# preparing data 
bolivia_df <- bolivia %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_bolivia <- floor(nrow(bolivia_df) * 0.8)
period_bolivia <- floor((nrow(bolivia_df) - initial_bolivia) / 5)
horizon_bolivia <- nrow(bolivia_df) - initial_bolivia - period_bolivia * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_bolivia <- list()
plot_data_bolivia <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_bolivia + (i-1) * period_bolivia
  
  training_set <- bolivia_df[1:end_index, ]
  testing_set <- bolivia_df[(end_index + 1):(end_index + horizon_bolivia), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_bolivia[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_bolivia[[i]] <- data.frame(
    date = testing_set$ds,
    actual_bolivia = actuals,
    forecasted_bolivia = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_bolivia <- bind_rows(cv_results_bolivia)
print(cv_results_df_bolivia)

average_metrics_bolivia <- cv_results_df_bolivia %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_bolivia)

## combining all folds into one dataframe for plotting  ----
plot_df_bolivia <- bind_rows(plot_data_bolivia, .id = "fold")

## plotting the actual vs forecasted values  ----
ggplot(plot_df_bolivia, aes(x = date)) +
  geom_line(aes(y = actual_bolivia, color = "Actual")) +
  geom_line(aes(y = forecasted_bolivia, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "Bolivia: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# brazil ----

# loading data
load("data/preprocessed/univariate/not_split/brazil.rda")

# preparing data 
brazil_df <- brazil %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_brazil <- floor(nrow(brazil_df) * 0.8)
period_brazil <- floor((nrow(brazil_df) - initial_brazil) / 5)
horizon_brazil <- nrow(brazil_df) - initial_brazil - period_brazil * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_brazil <- list()
plot_data_brazil <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_brazil + (i-1) * period_brazil
  
  training_set <- brazil_df[1:end_index, ]
  testing_set <- brazil_df[(end_index + 1):(end_index + horizon_brazil), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_brazil[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_brazil[[i]] <- data.frame(
    date = testing_set$ds,
    actual_brazil = actuals,
    forecasted_brazil = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_brazil <- bind_rows(cv_results_brazil)
print(cv_results_df_brazil)

average_metrics_brazil <- cv_results_df_brazil %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_brazil)

## combining all folds into one dataframe for plotting  ----
plot_df_brazil <- bind_rows(plot_data_brazil, .id = "fold")

## plotting the actual vs forecasted values  ----
ggplot(plot_df_brazil, aes(x = date)) +
  geom_line(aes(y = actual_brazil, color = "Actual")) +
  geom_line(aes(y = forecasted_brazil, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "brazil: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# colombia ----

# loading data
load("data/preprocessed/univariate/not_split/colombia.rda")

# preparing data 
colombia_df <- colombia %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_colombia <- floor(nrow(colombia_df) * 0.8)
period_colombia <- floor((nrow(colombia_df) - initial_colombia) / 5)
horizon_colombia <- nrow(colombia_df) - initial_colombia - period_colombia * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_colombia <- list()
plot_data_colombia <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_colombia + (i-1) * period_colombia
  
  training_set <- colombia_df[1:end_index, ]
  testing_set <- colombia_df[(end_index + 1):(end_index + horizon_colombia), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_colombia[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_colombia[[i]] <- data.frame(
    date = testing_set$ds,
    actual_colombia = actuals,
    forecasted_colombia = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_colombia <- bind_rows(cv_results_colombia)
print(cv_results_df_colombia)

average_metrics_colombia <- cv_results_df_colombia %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_colombia)

## combining all folds into one dataframe for plotting  ----
plot_df_colombia <- bind_rows(plot_data_colombia, .id = "fold")

## plotting the actual vs forecasted values  ----
ggplot(plot_df_colombia, aes(x = date)) +
  geom_line(aes(y = actual_colombia, color = "Actual")) +
  geom_line(aes(y = forecasted_colombia, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "colombia: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# iran ----

# loading data
load("data/preprocessed/univariate/not_split/iran.rda")

# preparing data 
iran_df <- iran %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_iran <- floor(nrow(iran_df) * 0.8)
period_iran <- floor((nrow(iran_df) - initial_iran) / 5)
horizon_iran <- nrow(iran_df) - initial_iran - period_iran * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_iran <- list()
plot_data_iran <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_iran + (i-1) * period_iran
  
  training_set <- iran_df[1:end_index, ]
  testing_set <- iran_df[(end_index + 1):(end_index + horizon_iran), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_iran[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_iran[[i]] <- data.frame(
    date = testing_set$ds,
    actual_iran = actuals,
    forecasted_iran = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_iran <- bind_rows(cv_results_iran)
print(cv_results_df_iran)

average_metrics_iran <- cv_results_df_iran %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_iran)

## combining all folds into one dataframe for plotting  ----
plot_df_iran <- bind_rows(plot_data_iran, .id = "fold")

## plotting the actual vs forecasted values  ----
ggplot(plot_df_iran, aes(x = date)) +
  geom_line(aes(y = actual_iran, color = "Actual")) +
  geom_line(aes(y = forecasted_iran, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "iran: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# mexico ----

# loading data
load("data/preprocessed/univariate/not_split/mexico.rda")

# preparing data 
mexico_df <- mexico %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_mexico <- floor(nrow(mexico_df) * 0.8)
period_mexico <- floor((nrow(mexico_df) - initial_mexico) / 5)
horizon_mexico <- nrow(mexico_df) - initial_mexico - period_mexico * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_mexico <- list()
plot_data_mexico <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_mexico + (i-1) * period_mexico
  
  training_set <- mexico_df[1:end_index, ]
  testing_set <- mexico_df[(end_index + 1):(end_index + horizon_mexico), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_mexico[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_mexico[[i]] <- data.frame(
    date = testing_set$ds,
    actual_mexico = actuals,
    forecasted_mexico = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_mexico <- bind_rows(cv_results_mexico)
print(cv_results_df_mexico)

average_metrics_mexico <- cv_results_df_mexico %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_mexico)

# combining all folds into one dataframe for plotting  ----
plot_df_mexico <- bind_rows(plot_data_mexico, .id = "fold")

# plotting the actual vs forecasted values  ----
ggplot(plot_df_mexico, aes(x = date)) +
  geom_line(aes(y = actual_mexico, color = "Actual")) +
  geom_line(aes(y = forecasted_mexico, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "mexico: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# peru ----

# loading data
load("data/preprocessed/univariate/not_split/peru.rda")

# preparing data 
peru_df <- peru %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_peru <- floor(nrow(peru_df) * 0.8)
period_peru <- floor((nrow(peru_df) - initial_peru) / 5)
horizon_peru <- nrow(peru_df) - initial_peru - period_peru * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_peru <- list()
plot_data_peru <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_peru + (i-1) * period_peru
  
  training_set <- peru_df[1:end_index, ]
  testing_set <- peru_df[(end_index + 1):(end_index + horizon_peru), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_peru[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_peru[[i]] <- data.frame(
    date = testing_set$ds,
    actual_peru = actuals,
    forecasted_peru = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_peru <- bind_rows(cv_results_peru)
print(cv_results_df_peru)

average_metrics_peru <- cv_results_df_peru %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_peru)

# combining all folds into one dataframe for plotting  ----
plot_df_peru <- bind_rows(plot_data_peru, .id = "fold")

# plotting the actual vs forecasted values  ----
ggplot(plot_df_peru, aes(x = date)) +
  geom_line(aes(y = actual_peru, color = "Actual")) +
  geom_line(aes(y = forecasted_peru, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "peru: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# russia ----

# loading data
load("data/preprocessed/univariate/not_split/russia.rda")

# preparing data 
russia_df <- russia %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_russia <- floor(nrow(russia_df) * 0.8)
period_russia <- floor((nrow(russia_df) - initial_russia) / 5)
horizon_russia <- nrow(russia_df) - initial_russia - period_russia * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_russia <- list()
plot_data_russia <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_russia + (i-1) * period_russia
  
  training_set <- russia_df[1:end_index, ]
  testing_set <- russia_df[(end_index + 1):(end_index + horizon_russia), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_russia[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_russia[[i]] <- data.frame(
    date = testing_set$ds,
    actual_russia = actuals,
    forecasted_russia = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_russia <- bind_rows(cv_results_russia)
print(cv_results_df_russia)

average_metrics_russia <- cv_results_df_russia %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_russia)

# combining all folds into one dataframe for plotting  ----
plot_df_russia <- bind_rows(plot_data_russia, .id = "fold")

# plotting the actual vs forecasted values  ----
ggplot(plot_df_russia, aes(x = date)) +
  geom_line(aes(y = actual_russia, color = "Actual")) +
  geom_line(aes(y = forecasted_russia, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "russia: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# saudi ----

# loading data
load("data/preprocessed/univariate/not_split/saudi.rda")

# preparing data 
saudi_df <- saudi %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_saudi <- floor(nrow(saudi_df) * 0.8)
period_saudi <- floor((nrow(saudi_df) - initial_saudi) / 5)
horizon_saudi <- nrow(saudi_df) - initial_saudi - period_saudi * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_saudi <- list()
plot_data_saudi <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_saudi + (i-1) * period_saudi
  
  training_set <- saudi_df[1:end_index, ]
  testing_set <- saudi_df[(end_index + 1):(end_index + horizon_saudi), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_saudi[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_saudi[[i]] <- data.frame(
    date = testing_set$ds,
    actual_saudi = actuals,
    forecasted_saudi = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_saudi <- bind_rows(cv_results_saudi)
print(cv_results_df_saudi)

average_metrics_saudi <- cv_results_df_saudi %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_saudi)

# combining all folds into one dataframe for plotting  ----
plot_df_saudi <- bind_rows(plot_data_saudi, .id = "fold")

# plotting the actual vs forecasted values  ----
ggplot(plot_df_saudi, aes(x = date)) +
  geom_line(aes(y = actual_saudi, color = "Actual")) +
  geom_line(aes(y = forecasted_saudi, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "saudi: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# turkey ----

# loading data
load("data/preprocessed/univariate/not_split/turkey.rda")

# preparing data 
turkey_df <- turkey %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_turkey <- floor(nrow(turkey_df) * 0.8)
period_turkey <- floor((nrow(turkey_df) - initial_turkey) / 5)
horizon_turkey <- nrow(turkey_df) - initial_turkey - period_turkey * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_turkey <- list()
plot_data_turkey <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_turkey + (i-1) * period_turkey
  
  training_set <- turkey_df[1:end_index, ]
  testing_set <- turkey_df[(end_index + 1):(end_index + horizon_turkey), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_turkey[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_turkey[[i]] <- data.frame(
    date = testing_set$ds,
    actual_turkey = actuals,
    forecasted_turkey = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_turkey <- bind_rows(cv_results_turkey)
print(cv_results_df_turkey)

average_metrics_turkey <- cv_results_df_turkey %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_turkey)

# combining all folds into one dataframe for plotting  ----
plot_df_turkey <- bind_rows(plot_data_turkey, .id = "fold")

# plotting the actual vs forecasted values  ----
ggplot(plot_df_turkey, aes(x = date)) +
  geom_line(aes(y = actual_turkey, color = "Actual")) +
  geom_line(aes(y = forecasted_turkey, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "turkey: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# us ----

# loading data
load("data/preprocessed/univariate/not_split/us.rda")

# preparing data 
us_df <- us %>% 
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  select(-date)  # Assuming 'date' is no longer needed.

# defining cross-validation settings 
initial_us <- floor(nrow(us_df) * 0.8)
period_us <- floor((nrow(us_df) - initial_us) / 5)
horizon_us <- nrow(us_df) - initial_us - period_us * 4

# performing 5-fold cross-validation and storing results for plotting 
cv_results_us <- list()
plot_data_us <- list()

for (i in 1:5) {
  start_index <- 1
  end_index <- initial_us + (i-1) * period_us
  
  training_set <- us_df[1:end_index, ]
  testing_set <- us_df[(end_index + 1):(end_index + horizon_us), ]
  
  model <- prophet(training_set)
  future <- make_future_dataframe(model, periods = nrow(testing_set))
  forecast <- predict(model, future)
  
  actuals <- testing_set$y
  predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
  
  naive_forecast <- rep(tail(training_set$y, 1), nrow(testing_set))
  
  # Calculating accuracy metrics for cross-validation
  cv_results_us[[i]] <- data.frame(
    RMSE = sqrt(mean((actuals - predictions)^2)),
    MSE = mean((actuals - predictions)^2),
    MAE = mean(abs(actuals - predictions)),
    MASE = mean(abs(actuals - predictions)) / mean(abs(diff(training_set$y))),
    fold = i
  )
  
  # Storing actual and forecasted values for plotting
  plot_data_us[[i]] <- data.frame(
    date = testing_set$ds,
    actual_us = actuals,
    forecasted_us = predictions
  )
}

# reviewing and averaging the cross-validation results 
cv_results_df_us <- bind_rows(cv_results_us)
print(cv_results_df_us)

average_metrics_us <- cv_results_df_us %>% summarize(across(c(RMSE, MSE, MAE, MASE), mean))
print(average_metrics_us)

# combining all folds into one dataframe for plotting  ----
plot_df_us <- bind_rows(plot_data_us, .id = "fold")

# plotting the actual vs forecasted values  ----
ggplot(plot_df_us, aes(x = date)) +
  geom_line(aes(y = actual_us, color = "Actual")) +
  geom_line(aes(y = forecasted_us, color = "Forecasted"), linetype = "dashed") +
  labs(x = "Date", y = "Value", title = "us: Actual vs Forecasted") +
  scale_color_manual("", 
                     breaks = c("Actual", "Forecasted"),
                     values = c("Actual" = "blue", "Forecasted" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



# displaying models averages ----

country_datasets <- list(
  bolivia = bolivia_df,
  brazil = brazil_df,
  colombia = colombia_df,
  iran = iran_df,
  mexico = mexico_df,
  peru = peru_df,
  russia = russia_df,
  saudi = saudi_df,
  turkey = turkey_df,
  us = us_df
)

# performing cross-validation and metrics collection
perform_cv_and_metrics <- function(country_name, df) {
  initial <- floor(nrow(df) * 0.8)
  period <- floor((nrow(df) - initial) / 5)
  horizon <- nrow(df) - initial - period * 4
  
  cv_results <- list()
  
  for (i in 1:5) {
    start_index <- 1
    end_index <- initial + (i-1) * period
    
    training_set <- df[1:end_index, ]
    testing_set <- df[(end_index + 1):(end_index + horizon), ]
    
    model <- prophet(training_set)
    future <- make_future_dataframe(model, periods = nrow(testing_set))
    forecast <- predict(model, future)
    
    actuals <- testing_set$y
    predictions <- forecast$yhat[(nrow(future)-nrow(testing_set)+1):nrow(future)]
    
    cv_results[[i]] <- data.frame(
      Country = country_name,
      Model_Type = "Univariate Prophet",
      RMSE = sqrt(mean((actuals - predictions)^2)),
      MSE = mean((actuals - predictions)^2),
      MAE = mean(abs(actuals - predictions)),
      MASE = mean(abs(actuals - predictions)) / mean(abs(diff(na.omit(training_set$y)))),
      Fold = i
    )
  }
  
  cv_results_df <- bind_rows(cv_results)
  average_metrics <- cv_results_df %>% 
    group_by(Country, Model_Type) %>%
    summarize(across(c(RMSE, MSE, MAE, MASE), mean), .groups = 'drop')
  
  return(list(cv_results_df = cv_results_df, average_metrics = average_metrics))
}

# initializing an empty list to store the average metrics dataframes
average_metrics_all_countries <- list()

# iterating over the country datasets and store the average metrics in the list
for(country_name in names(country_datasets)) {
  df <- country_datasets[[country_name]]
  results <- perform_cv_and_metrics(tolower(country_name), df)
  average_metrics_all_countries[[country_name]] <- results$average_metrics
}

# combining all average metrics into a single dataframe
univar_prophet_combined_average_metrics <- bind_rows(average_metrics_all_countries, .id = "Country")

# ensuring naming consistency
univar_prophet_combined_average_metrics$Country <- tolower(univar_prophet_combined_average_metrics$Country)



# saving files ----
save(univar_prophet_combined_average_metrics, file = "data_frames/univar_prophet_combined_average_metrics_maria.rda")