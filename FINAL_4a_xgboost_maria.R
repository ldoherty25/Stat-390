## Multivariate XGBoost
## The 4b file aids the creation of a table with the metrics of each country selected in univariate model building.



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



# preprocessing ----

# removing all zero observations before the first death for each country
preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>%
  group_by(country) %>%
  filter(cumsum(owid_new_deaths) > 0) %>%
  ungroup()



# adjusting variables ----

df_with_dummies <- preprocessed_covid_multi_imputed %>%
  select(-country) %>% # Removing country
  janitor::clean_names() %>%
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  arrange(ds)

# converting weekday (not really used)
df_with_dummies <- df_with_dummies %>%
  mutate(weekday = lubridate::wday(ds, label = TRUE, abbr = FALSE)) %>%
  mutate(weekday = as.numeric(as.factor(weekday))) %>%
  mutate(weekday = ifelse(is.na(weekday), 0, weekday))

# removing variables
df_with_dummies <- df_with_dummies %>%
  select(-c(month, cyclical_month_cos, cyclical_month_sin, weekday, day))



# dataset preparation ----

# splitting into training and testing sets
split_index <- floor(0.8 * nrow(df_with_dummies))
training_df <- df_with_dummies[1:split_index, ]
testing_df <- df_with_dummies[(split_index + 1):nrow(df_with_dummies), ]

# extracting predictors and target
train_data <- as.matrix(training_df %>% select(-ds, -date, -y, -owid_new_deaths))
train_label <- training_df$y

# extracting predictors and target for test data
test_data <- as.matrix(testing_df %>% select(-ds, -date, -y, -owid_new_deaths))
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

# updating train_control with timeslice method and created folds
train_control <- trainControl(
  method = "timeslice",
  index = ts_cv_folds$train,
  indexOut = ts_cv_folds$test,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "none",
  allowParallel = TRUE
)

# defining the tuning grid for XGBoost
xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 6, 9),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.5, 0.75, 1)
)

# training the model using the train function with the xgbTree method and the defined tuning grid
xgb_model_ii <- train(
  x = train_data,
  y = train_label,
  trControl = train_control,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

# making predictions with the best model
predictions <- predict(xgb_model_ii, test_data)

# calculating performance metrics
RMSE <- sqrt(mean((predictions - test_label)^2))
MAE <- mean(abs(predictions - test_label))
MSE <- mean((predictions - test_label)^2)
MAPE <- mean(abs((predictions - test_label) / test_label), na.rm = TRUE)
MASE <- mean(abs(predictions - test_label)) / mean(abs(diff(test_label)), na.rm = TRUE)

# creating a data frame for metrics
metrics_table <- data.frame(
  Country = "All",
  Best_Model = "Multivariate XGBoost",
  RMSE, 
  MAE, 
  MSE, 
  MAPE, 
  MASE
)



# saving files ----

save(ts_cv_folds, file = "xgboost_components/ts_cv_folds_maria.rda")
save(train_control, file = "xgboost_components/train_control_maria.rda")
save(xgb_grid, file = "xgboost_components/xgb_grid_maria.rda")
save(xgb_model_ii, file = "xgboost_components/xgb_model_ii_maria.rda")
save(predictions, file = "xgboost_components/predictions_maria.rda")



# inspecting model ----

# loading files
load("xgboost_components/ts_cv_folds_maria.rda")
load("xgboost_components/train_control_maria.rda")
load("xgboost_components/xgb_grid_maria.rda")
load("xgboost_components/xgb_model_ii_maria.rda")
load("xgboost_components/predictions_maria.rda")

# printing best tuning parameters
print(xgb_model_ii$bestTune)

# summarizing xgb_model_ii
summary(xgb_model_ii)

# extracting feature importance
importance_matrix <- xgb.importance(feature_names = colnames(xgb_model_ii$finalModel$feature_names), model = xgb_model_ii$finalModel)

# printing feature importance
print(importance_matrix)

# plotting feature importance
xgb.plot.importance(importance_matrix)