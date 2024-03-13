## Multivariate Prophet

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
library(fastDummies)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)

#loading the univariate datasets
load("data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")


############################################################################################################################################################################################################################################################

df_dummy <- preprocessed_covid_multi_imputed %>%
  dummy_cols("country", remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
  janitor::clean_names() %>%
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  arrange(ds)

df_dummy <- df_dummy %>%
  mutate(weekday = lubridate::wday(ds, label = TRUE, abbr = FALSE)) %>%
  mutate(weekday = as.numeric(as.factor(weekday))) %>%
  mutate(weekday = ifelse(is.na(weekday), 0, weekday))

# split data into train and test sets
split_index <- floor(0.8 * nrow(df_dummy))
train <- df_dummy[1:split_index, ]
test <- df_dummy[(split_index + 1):nrow(df_dummy), ]


train_data <- as.matrix(train %>% select(-ds, -date, -y))
train_label <- train$y
test_data <- as.matrix(test %>% select(-ds, -date, -y))
test_label <- test$y

fold_len <- floor(nrow(train) / 5)

ts_cv_folds <- createTimeSlices(
  1:nrow(train),
  initialWindow = fold_len * (5 - 1),
  horizon = fold_len,
  fixedWindow = TRUE,
  skip = fold_len - 1
)

train_control <- trainControl(
  method = "timeslice",
  index = ts_cv_folds$train,
  indexOut = ts_cv_folds$test,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "none",
  allowParallel = TRUE
)

xgb_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 6, 9),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.5, 0.7, 1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.5, 0.75, 1)
)


xgb_model <- train(
  x = train_data,
  y = train_label,
  trControl = train_control,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

predictions <- predict(xgb_model, test_data)

# performance metrics
rmse <- sqrt(mean((predictions - test_label)^2))
mae <- mean(abs(predictions - test_label))
mse <- mean((predictions - test_label)^2)
mape <- mean(abs((test_label - predictions) / test_label)) * 100
mase <- mean(abs(diff(test_label)) / mean(abs(predictions - test_label)))

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MSE:", mse, "\n")
cat("MAPE:", mape, "\n")
cat("MASE:", mase, "\n")







