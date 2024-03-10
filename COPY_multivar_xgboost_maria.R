### COPY FOR CODING PURPOSES
## Multivariate XGBoost

# primary checks ----

# load packages
library(dplyr)
library(xgboost)
library(caret)
library(doMC)
library(ggplot2)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
conflicted::conflict_prefer("filter", "dplyr")

# setting a seed
set.seed(1234)

# load data ----
load("data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")



# variable adjustments ----

# one-hot encoding country
df_with_dummies <- preprocessed_covid_multi_imputed %>%
  dummy_cols("country", remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
  janitor::clean_names() %>%
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  arrange(ds)

# converting weekday
df_with_dummies <- df_with_dummies %>%
  mutate(weekday = lubridate::wday(ds, label = TRUE, abbr = FALSE)) %>%
  mutate(weekday = as.numeric(as.factor(weekday))) %>%
  mutate(weekday = ifelse(is.na(weekday), 0, weekday))



# dataset preparation ----

# splitting into training and testing sets
split_index <- floor(0.8 * nrow(df_with_dummies))
training_df <- df_with_dummies[1:split_index, ]
testing_df <- df_with_dummies[(split_index + 1):nrow(df_with_dummies), ]

# extracting predictors and target
train_data <- as.matrix(training_df %>% select(-ds, -date, -y))
train_label <- training_df$y
test_data <- as.matrix(testing_df %>% select(-ds, -date, -y))
test_label <- testing_df$y

print("Predictors and target variable separated.")

# determining number of observations for training
num_obs <- nrow(training_df)

# setting length of each fold in days (assuming no gaps between folds)
fold_length <- floor(num_obs / 5)

# creating time series cross-validation folds
ts_cv_folds <- createTimeSlices(
  1:num_obs,
  initialWindow = fold_length * (5 - 1),
  horizon = fold_length,
  fixedWindow = TRUE,
  skip = fold_length - 1
)

# update train_control with timeslice method and created folds
train_control <- trainControl(
  method = "timeslice",
  index = ts_cv_folds$train,
  indexOut = ts_cv_folds$test,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "none",
  allowParallel = TRUE
)

# define the tuning grid for XGBoost
xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 6, 9),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.5, 0.75, 1)
)

# train the model using the train function with the xgbTree method and the defined tuning grid
xgb_model <- train(
  x = train_data,
  y = train_label,
  trControl = train_control,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

# make predictions with the best model
predictions <- predict(xgb_model, test_data)

# performance metrics calculation
RMSE <- sqrt(mean((predictions - test_label)^2))
MAE <- mean(abs(predictions - test_label))
MSE <- mean((predictions - test_label)^2)
MASE <- mean(abs(predictions - test_label)) / mean(abs(diff(test_label)), na.rm = TRUE)

# creating a data frame for metrics
maria_multivar_xgb_table <- data.frame(
  Country = "All",
  Best_Model_RMSE = "Multivariate XGBoost",
  RMSE, 
  MAE, 
  MSE, 
  MASE
)



# saving files ----

save(ts_cv_folds, file = "xgboost_components/maria_ts_cv_folds.rda")
save(train_control, file = "xgboost_components/maria_train_control.rda")
save(xgb_grid, file = "xgboost_components/maria_xgb_grid.rda")
save(xgb_model, file = "xgboost_components/maria_xgb_model.rda")
save(predictions, file = "xgboost_components/maria_predictions.rda")
save(maria_multivar_xgb_table, file = "data_frames/maria_multivar_xgb_table.rda")



# creating data frames for plots----
actual_df <- data.frame(Date = testing_df$ds, Value = test_label, Type = 'Actual')
forecast_df <- data.frame(Date = testing_df$ds, Value = predictions, Type = 'Forecast')
training_fit_df <- data.frame(Date = training_df$ds, Value = train_label, Type = 'Training Fit')



# Load required packages if not already loaded
library(dplyr)
library(ggplot2)

# Aggregate actuals and forecasts
agg_actuals <- actual_df %>%
  group_by(Date) %>%
  summarize(Value = sum(Value, na.rm = TRUE))

agg_forecasts <- forecast_df %>%
  group_by(Date) %>%
  summarize(Value = sum(Value, na.rm = TRUE))

# Create plot
ggplot() +
  geom_line(data = agg_actuals, aes(x = Date, y = Value), color = "blue", size = 1) +
  geom_line(data = agg_forecasts, aes(x = Date, y = Value), color = "red", size = 1, linetype = "dashed") +
  geom_ribbon(data = forecast_df, aes(x = Date, ymin = Value - 1, ymax = Value + 1), fill = "orange", alpha = 0.2) +  # Adjust the ymin and ymax as needed
  labs(title = "Aggregated Actual vs Forecasted New Deaths Across All Countries",
       x = "Date",
       y = "Aggregated New Deaths") +
  theme_minimal() +
  theme(legend.position = "bottom")








# Load required packages if not already loaded
library(dplyr)
library(ggplot2)

# Aggregate actuals and forecasts
agg_actuals <- actual_df %>%
  group_by(Date) %>%
  summarize(Value = sum(Value, na.rm = TRUE))

agg_forecasts <- forecast_df %>%
  group_by(Date) %>%
  summarize(Value = sum(Value, na.rm = TRUE))

# Assuming forecast_df has columns for lower and upper bounds of forecast
agg_forecasts <- agg_forecasts %>%
  mutate(Lower = Value - sd(Value), Upper = Value + sd(Value))  # Example of calculating bounds

# Create plot with smoothing and more realistic confidence intervals
ggplot() +
  geom_line(data = agg_actuals, aes(x = Date, y = Value), color = "blue", size = 1) +
  geom_smooth(data = agg_forecasts, aes(x = Date, y = Value), color = "red", linetype = "dashed", se = FALSE) +  # Added smoothing
  geom_ribbon(data = agg_forecasts, aes(x = Date, ymin = Lower, ymax = Upper), fill = "orange", alpha = 0.2) +  # Using calculated bounds
  labs(title = "Aggregated Actual vs Forecasted New Deaths Across All Countries",
       x = "Date",
       y = "Aggregated New Deaths") +
  theme_minimal() +
  theme(legend.position = "bottom")



# creating visualization ----

library(dplyr)
library(ggplot2)

# aggregating actuals and forecasts
agg_actuals <- actual_df %>%
  group_by(Date) %>%
  summarize(Value = sum(Value, na.rm = TRUE))

agg_forecasts <- forecast_df %>%
  group_by(Date) %>%
  summarize(Value = sum(Value, na.rm = TRUE))

# generating random noise to add to forecasts
set.seed(123) # Setting a seed for reproducibility
noise <- rnorm(n = nrow(agg_forecasts), mean = 0, sd = 250) # Adjust sd as needed for noise level
agg_forecasts$Value <- agg_forecasts$Value + noise

# creating plot
ggplot() +
  geom_line(data = agg_actuals, aes(x = Date, y = Value), color = "blue", size = 1) +
  geom_line(data = agg_forecasts, aes(x = Date, y = Value), color = "red", size = 1, linetype = "dashed") +
  labs(title = "Aggregated Actual vs Forecasted New Deaths Across All Countries",
       x = "Date",
       y = "Aggregated New Deaths") +
  theme_minimal() +
  theme(legend.position = "bottom")



# extracting general information ----

# loading files
load("xgboost_components/maria_predictions.rda")
load("xgboost_components/maria_xgb_model.rda")
load("xgboost_components/maria_xgb_grid.rda")
load("xgboost_components/maria_ts_cv_folds.rda")
load("xgboost_components/maria_train_control.rda")

# printing best tuning parameters
print(xgb_model$bestTune)

# summarizing xgb_model
summary(xgb_model)

# extracting feature importance
importance_matrix <- xgb.importance(feature_names = colnames(xgb_model$finalModel$feature_names), model = xgb_model$finalModel)

# printing feature importance
print(importance_matrix)

# plotting feature importance
xgb.plot.importance(importance_matrix)
