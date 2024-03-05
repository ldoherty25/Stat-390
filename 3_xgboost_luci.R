## XG Boost (Univariate)

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
library(caret)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
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


##########################################################################################################################################################################################################################################################


## Bolivia model

n_distinct(bolivia$date) * .8 # corresponds to 9/17/20


# Split data 

bolivia_train <- bolivia %>% 
  filter(date <= "2020-09-17")
bolivia_test <- bolivia %>% 
  filter(date > "2020-09-17")


bolivia_outcome <- bolivia_train$owid_new_deaths

bolivia_xgtrain <- as.matrix(bolivia_train %>% select(- date))
bolivia_xgtest <- as.matrix(bolivia_test %>% select(- date))


bolivia_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

bolivia_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
bolivia_model_xgb <- caret::train(
  x = bolivia_xgtrain,
  y = bolivia_outcome,
  trControl = bolivia_xgb_train_control,
  tuneGrid = bolivia_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

bolivia_best_params <- bolivia_model_xgb$bestTune

bolivia_fitted <- bolivia_model_xgb %>%
  stats::predict(bolivia_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

bolivia_preds <- predict(bolivia_model_xgb, bolivia_xgtest)

bolivia_metrics <- data.frame(
  RMSE = sqrt(mean((bolivia_preds - bolivia_test$owid_new_deaths)^2)),
  MAE = mean(abs(bolivia_preds - bolivia_test$owid_new_deaths)),
  MSE = mean((bolivia_preds - bolivia_test$owid_new_deaths)^2),
  MAPE = mean(abs((bolivia_preds - bolivia_test$owid_new_deaths) / bolivia_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(bolivia_test$owid_new_deaths)) / mean(abs(bolivia_preds - bolivia_test$owid_new_deaths))))


cat("RMSE:", mean(bolivia_metrics$RMSE))
cat("MAE:", mean(bolivia_metrics$MAE))
cat("MSE:", mean(bolivia_metrics$MSE))
cat("MAPE:", mean(bolivia_metrics$MAPE))
cat("MASE:", mean(bolivia_metrics$MASE))



plot(bolivia$date, bolivia$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "Bolivia XGBoost") + 
  lines(bolivia_test$date, bolivia_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2), 
         lwd = c(1, 2),
         cex = 0.7)


##########################################################################################################################################################################################################################################################


## brazil model

n_distinct(brazil$date) * .8 # corresponds to 9/17/20


# Split data 

brazil_train <- brazil %>% 
  filter(date <= "2020-09-17")
brazil_test <- brazil %>% 
  filter(date > "2020-09-17")


brazil_outcome <- brazil_train$owid_new_deaths

brazil_xgtrain <- as.matrix(brazil_train %>% select(- date))
brazil_xgtest <- as.matrix(brazil_test %>% select(- date))


brazil_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

brazil_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
brazil_model_xgb <- caret::train(
  x = brazil_xgtrain,
  y = brazil_outcome,
  trControl = brazil_xgb_train_control,
  tuneGrid = brazil_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

brazil_best_params <- brazil_model_xgb$bestTune

brazil_fitted <- brazil_model_xgb %>%
  stats::predict(brazil_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

brazil_preds <- predict(brazil_model_xgb, brazil_xgtest)

brazil_metrics <- data.frame(
  RMSE = sqrt(mean((brazil_preds - brazil_test$owid_new_deaths)^2)),
  MAE = mean(abs(brazil_preds - brazil_test$owid_new_deaths)),
  MSE = mean((brazil_preds - brazil_test$owid_new_deaths)^2),
  MAPE = mean(abs((brazil_preds - brazil_test$owid_new_deaths) / brazil_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(brazil_test$owid_new_deaths)) / mean(abs(brazil_preds - brazil_test$owid_new_deaths))))


cat("RMSE:", mean(brazil_metrics$RMSE))
cat("MAE:", mean(brazil_metrics$MAE))
cat("MSE:", mean(brazil_metrics$MSE))
cat("MAPE:", mean(brazil_metrics$MAPE))
cat("MASE:", mean(brazil_metrics$MASE))



plot(brazil$date, brazil$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "brazil XGBoost") + 
  lines(brazil_test$date, brazil_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2), 
         lwd = c(1, 2),
         cex = 0.7)


############################################################################################################################################################################################################################


## colombia model

n_distinct(colombia$date) * .8 # corresponds to 9/17/20


# Split data 

colombia_train <- colombia %>% 
  filter(date <= "2020-09-17")
colombia_test <- colombia %>% 
  filter(date > "2020-09-17")


colombia_outcome <- colombia_train$owid_new_deaths

colombia_xgtrain <- as.matrix(colombia_train %>% select(- date))
colombia_xgtest <- as.matrix(colombia_test %>% select(- date))


colombia_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

colombia_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
colombia_model_xgb <- caret::train(
  x = colombia_xgtrain,
  y = colombia_outcome,
  trControl = colombia_xgb_train_control,
  tuneGrid = colombia_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

colombia_best_params <- colombia_model_xgb$bestTune

colombia_fitted <- colombia_model_xgb %>%
  stats::predict(colombia_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

colombia_preds <- predict(colombia_model_xgb, colombia_xgtest)

colombia_metrics <- data.frame(
  RMSE = sqrt(mean((colombia_preds - colombia_test$owid_new_deaths)^2)),
  MAE = mean(abs(colombia_preds - colombia_test$owid_new_deaths)),
  MSE = mean((colombia_preds - colombia_test$owid_new_deaths)^2),
  MAPE = mean(abs((colombia_preds - colombia_test$owid_new_deaths) / colombia_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(colombia_test$owid_new_deaths)) / mean(abs(colombia_preds - colombia_test$owid_new_deaths))))


cat("RMSE:", mean(colombia_metrics$RMSE))
cat("MAE:", mean(colombia_metrics$MAE))
cat("MSE:", mean(colombia_metrics$MSE))
cat("MAPE:", mean(colombia_metrics$MAPE))
cat("MASE:", mean(colombia_metrics$MASE))



plot(colombia$date, colombia$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "colombia XGBoost") + 
  lines(colombia_test$date, colombia_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2), 
         lwd = c(1, 2),
         cex = 0.7)


##########################################################################################################################################################################################################################################################


## iran model

n_distinct(iran$date) * .8 # corresponds to 9/17/20


# Split data 

iran_train <- iran %>% 
  filter(date <= "2020-09-17")
iran_test <- iran %>% 
  filter(date > "2020-09-17")


iran_outcome <- iran_train$owid_new_deaths

iran_xgtrain <- as.matrix(iran_train %>% select(- date))
iran_xgtest <- as.matrix(iran_test %>% select(- date))


iran_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

iran_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
iran_model_xgb <- caret::train(
  x = iran_xgtrain,
  y = iran_outcome,
  trControl = iran_xgb_train_control,
  tuneGrid = iran_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

iran_best_params <- iran_model_xgb$bestTune

iran_fitted <- iran_model_xgb %>%
  stats::predict(iran_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

iran_preds <- predict(iran_model_xgb, iran_xgtest)

iran_metrics <- data.frame(
  RMSE = sqrt(mean((iran_preds - iran_test$owid_new_deaths)^2)),
  MAE = mean(abs(iran_preds - iran_test$owid_new_deaths)),
  MSE = mean((iran_preds - iran_test$owid_new_deaths)^2),
  MAPE = mean(abs((iran_preds - iran_test$owid_new_deaths) / iran_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(iran_test$owid_new_deaths)) / mean(abs(iran_preds - iran_test$owid_new_deaths))))


cat("RMSE:", mean(iran_metrics$RMSE))
cat("MAE:", mean(iran_metrics$MAE))
cat("MSE:", mean(iran_metrics$MSE))
cat("MAPE:", mean(iran_metrics$MAPE))
cat("MASE:", mean(iran_metrics$MASE))



plot(iran$date, iran$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "iran XGBoost") + 
  lines(iran_test$date, iran_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2), 
         lwd = c(1, 2),
         cex = 0.7)


##########################################################################################################################################################################################################################################################


## mexico model

n_distinct(mexico$date) * .8 # corresponds to 9/17/20


# Split data 

mexico_train <- mexico %>% 
  filter(date <= "2020-09-17")
mexico_test <- mexico %>% 
  filter(date > "2020-09-17")


mexico_outcome <- mexico_train$owid_new_deaths

mexico_xgtrain <- as.matrix(mexico_train %>% select(- date))
mexico_xgtest <- as.matrix(mexico_test %>% select(- date))


mexico_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

mexico_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
mexico_model_xgb <- caret::train(
  x = mexico_xgtrain,
  y = mexico_outcome,
  trControl = mexico_xgb_train_control,
  tuneGrid = mexico_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

mexico_best_params <- mexico_model_xgb$bestTune

mexico_fitted <- mexico_model_xgb %>%
  stats::predict(mexico_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

mexico_preds <- predict(mexico_model_xgb, mexico_xgtest)

mexico_metrics <- data.frame(
  RMSE = sqrt(mean((mexico_preds - mexico_test$owid_new_deaths)^2)),
  MAE = mean(abs(mexico_preds - mexico_test$owid_new_deaths)),
  MSE = mean((mexico_preds - mexico_test$owid_new_deaths)^2),
  MAPE = mean(abs((mexico_preds - mexico_test$owid_new_deaths) / mexico_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(mexico_test$owid_new_deaths)) / mean(abs(mexico_preds - mexico_test$owid_new_deaths))))


cat("RMSE:", mean(mexico_metrics$RMSE))
cat("MAE:", mean(mexico_metrics$MAE))
cat("MSE:", mean(mexico_metrics$MSE))
cat("MAPE:", mean(mexico_metrics$MAPE))
cat("MASE:", mean(mexico_metrics$MASE))



plot(mexico$date, mexico$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "mexico XGBoost") + 
  lines(mexico_test$date, mexico_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)

##########################################################################################################################################################################################################################################################


## peru model

n_distinct(peru$date) * .8 # corresponds to 9/17/20


# Split data 

peru_train <- peru %>% 
  filter(date <= "2020-09-17")
peru_test <- peru %>% 
  filter(date > "2020-09-17")


peru_outcome <- peru_train$owid_new_deaths

peru_xgtrain <- as.matrix(peru_train %>% select(- date))
peru_xgtest <- as.matrix(peru_test %>% select(- date))


peru_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

peru_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
peru_model_xgb <- caret::train(
  x = peru_xgtrain,
  y = peru_outcome,
  trControl = peru_xgb_train_control,
  tuneGrid = peru_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

peru_best_params <- peru_model_xgb$bestTune

peru_fitted <- peru_model_xgb %>%
  stats::predict(peru_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

peru_preds <- predict(peru_model_xgb, peru_xgtest)

peru_metrics <- data.frame(
  RMSE = sqrt(mean((peru_preds - peru_test$owid_new_deaths)^2)),
  MAE = mean(abs(peru_preds - peru_test$owid_new_deaths)),
  MSE = mean((peru_preds - peru_test$owid_new_deaths)^2),
  MAPE = mean(abs((peru_preds - peru_test$owid_new_deaths) / peru_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(peru_test$owid_new_deaths)) / mean(abs(peru_preds - peru_test$owid_new_deaths))))


cat("RMSE:", mean(peru_metrics$RMSE))
cat("MAE:", mean(peru_metrics$MAE))
cat("MSE:", mean(peru_metrics$MSE))
cat("MAPE:", mean(peru_metrics$MAPE))
cat("MASE:", mean(peru_metrics$MASE))



plot(peru$date, peru$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "peru XGBoost") + 
  lines(peru_test$date, peru_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


##########################################################################################################################################################################################################################################################


## russia model

n_distinct(russia$date) * .8 # corresponds to 9/17/20


# Split data 

russia_train <- russia %>% 
  filter(date <= "2020-09-17")
russia_test <- russia %>% 
  filter(date > "2020-09-17")


russia_outcome <- russia_train$owid_new_deaths

russia_xgtrain <- as.matrix(russia_train %>% select(- date))
russia_xgtest <- as.matrix(russia_test %>% select(- date))


russia_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

russia_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
russia_model_xgb <- caret::train(
  x = russia_xgtrain,
  y = russia_outcome,
  trControl = russia_xgb_train_control,
  tuneGrid = russia_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

russia_best_params <- russia_model_xgb$bestTune

russia_fitted <- russia_model_xgb %>%
  stats::predict(russia_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

russia_preds <- predict(russia_model_xgb, russia_xgtest)

russia_metrics <- data.frame(
  RMSE = sqrt(mean((russia_preds - russia_test$owid_new_deaths)^2)),
  MAE = mean(abs(russia_preds - russia_test$owid_new_deaths)),
  MSE = mean((russia_preds - russia_test$owid_new_deaths)^2),
  MAPE = mean(abs((russia_preds - russia_test$owid_new_deaths) / russia_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(russia_test$owid_new_deaths)) / mean(abs(russia_preds - russia_test$owid_new_deaths))))


cat("RMSE:", mean(russia_metrics$RMSE))
cat("MAE:", mean(russia_metrics$MAE))
cat("MSE:", mean(russia_metrics$MSE))
cat("MAPE:", mean(russia_metrics$MAPE))
cat("MASE:", mean(russia_metrics$MASE))




plot(russia$date, russia$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "russia XGBoost") + 
  lines(russia_test$date, russia_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


##########################################################################################################################################################################################################################################################


## saudi model

n_distinct(saudi$date) * .8 # corresponds to 9/17/20


# Split data 

saudi_train <- saudi %>% 
  filter(date <= "2020-09-17")
saudi_test <- saudi %>% 
  filter(date > "2020-09-17")


saudi_outcome <- saudi_train$owid_new_deaths

saudi_xgtrain <- as.matrix(saudi_train %>% select(- date))
saudi_xgtest <- as.matrix(saudi_test %>% select(- date))


saudi_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

saudi_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
saudi_model_xgb <- caret::train(
  x = saudi_xgtrain,
  y = saudi_outcome,
  trControl = saudi_xgb_train_control,
  tuneGrid = saudi_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

saudi_best_params <- saudi_model_xgb$bestTune

saudi_fitted <- saudi_model_xgb %>%
  stats::predict(saudi_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

saudi_preds <- predict(saudi_model_xgb, saudi_xgtest)

saudi_metrics <- data.frame(
  RMSE = sqrt(mean((saudi_preds - saudi_test$owid_new_deaths)^2)),
  MAE = mean(abs(saudi_preds - saudi_test$owid_new_deaths)),
  MSE = mean((saudi_preds - saudi_test$owid_new_deaths)^2),
  MAPE = mean(abs((saudi_preds - saudi_test$owid_new_deaths) / saudi_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(saudi_test$owid_new_deaths)) / mean(abs(saudi_preds - saudi_test$owid_new_deaths))))


cat("RMSE:", mean(saudi_metrics$RMSE))
cat("MAE:", mean(saudi_metrics$MAE))
cat("MSE:", mean(saudi_metrics$MSE))
cat("MAPE:", mean(saudi_metrics$MAPE))
cat("MASE:", mean(saudi_metrics$MASE))



plot(saudi$date, saudi$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "saudi XGBoost") + 
  lines(saudi_test$date, saudi_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


##########################################################################################################################################################################################################################################################


## turkey model

n_distinct(turkey$date) * .8 # corresponds to 9/17/20


# Split data 

turkey_train <- turkey %>% 
  filter(date <= "2020-09-17")
turkey_test <- turkey %>% 
  filter(date > "2020-09-17")


turkey_outcome <- turkey_train$owid_new_deaths

turkey_xgtrain <- as.matrix(turkey_train %>% select(- date))
turkey_xgtest <- as.matrix(turkey_test %>% select(- date))


turkey_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

turkey_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
turkey_model_xgb <- caret::train(
  x = turkey_xgtrain,
  y = turkey_outcome,
  trControl = turkey_xgb_train_control,
  tuneGrid = turkey_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

turkey_best_params <- turkey_model_xgb$bestTune

turkey_fitted <- turkey_model_xgb %>%
  stats::predict(turkey_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

turkey_preds <- predict(turkey_model_xgb, turkey_xgtest)

turkey_metrics <- data.frame(
  RMSE = sqrt(mean((turkey_preds - turkey_test$owid_new_deaths)^2)),
  MAE = mean(abs(turkey_preds - turkey_test$owid_new_deaths)),
  MSE = mean((turkey_preds - turkey_test$owid_new_deaths)^2),
  MAPE = mean(abs((turkey_preds - turkey_test$owid_new_deaths) / turkey_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(turkey_test$owid_new_deaths)) / mean(abs(turkey_preds - turkey_test$owid_new_deaths))))


cat("RMSE:", mean(turkey_metrics$RMSE))
cat("MAE:", mean(turkey_metrics$MAE))
cat("MSE:", mean(turkey_metrics$MSE))
cat("MAPE:", mean(turkey_metrics$MAPE))
cat("MASE:", mean(turkey_metrics$MASE))



plot(turkey$date, turkey$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "turkey XGBoost") + 
  lines(turkey_test$date, turkey_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


##########################################################################################################################################################################################################################################################


## us model

n_distinct(us$date) * .8 # corresponds to 9/17/20


# Split data 

us_train <- us %>% 
  filter(date <= "2020-09-17")
us_test <- us %>% 
  filter(date > "2020-09-17")


us_outcome <- us_train$owid_new_deaths

us_xgtrain <- as.matrix(us_train %>% select(- date))
us_xgtest <- as.matrix(us_test %>% select(- date))


us_xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = FALSE
)

us_xgb_grid <- expand.grid(
  list(
    nrounds = seq(100, 200),
    max_depth = c(6, 15, 20), 
    colsample_bytree = 1, 
    eta = 0.5,
    gamma = 0,
    min_child_weight = 1,  
    subsample = 1)
)

#Building the model
us_model_xgb <- caret::train(
  x = us_xgtrain,
  y = us_outcome,
  trControl = us_xgb_train_control,
  tuneGrid = us_xgb_grid,
  method = "xgbTree",
  nthread = 10
)

us_best_params <- us_model_xgb$bestTune

us_fitted <- us_model_xgb %>%
  stats::predict(us_xgtrain) %>%
  stats::ts(start = c(2013,1),frequency = 12)

us_preds <- predict(us_model_xgb, us_xgtest)

us_metrics <- data.frame(
  RMSE = sqrt(mean((us_preds - us_test$owid_new_deaths)^2)),
  MAE = mean(abs(us_preds - us_test$owid_new_deaths)),
  MSE = mean((us_preds - us_test$owid_new_deaths)^2),
  MAPE = mean(abs((us_preds - us_test$owid_new_deaths) / us_test$owid_new_deaths)) * 100,
  MASE = mean(abs(diff(us_test$owid_new_deaths)) / mean(abs(us_preds - us_test$owid_new_deaths))))


cat("RMSE:", mean(us_metrics$RMSE))
cat("MAE:", mean(us_metrics$MAE))
cat("MSE:", mean(us_metrics$MSE))
cat("MAPE:", mean(us_metrics$MAPE))
cat("MASE:", mean(us_metrics$MASE))



plot(us$date, us$owid_new_deaths, type = "l", col = "black", lwd = 1, xlab = "Date", 
     ylab = "New Deaths", main = "us XGBoost") + 
  lines(us_test$date, us_preds, col = "blue", lty = 2, lwd = 2) + 
  legend("topright", 
         legend = c("Actual", "Forecast"), 
         col = c("black", "blue"), 
         lty = c(1, 2, 1), 
         lwd = c(1, 2, 1),
         cex = 0.7)


###########################################################################################################################

metric_dfs <- list(bolivia_metrics, brazil_metrics, colombia_metrics, iran_metrics, russia_metrics, turkey_metrics, 
                   us_metrics, peru_metrics, mexico_metrics, saudi_metrics)

country_names <- c("Bolivia", "Brazil", "Colombia", "Iran", "Russia", "Turkey", "US", "Peru", "Mexico", "Saudi Arabia")

all_dfs <- bind_rows(metric_dfs, .id = "Country")

all_dfs$Country <- country_names

print(all_dfs)

save(all_dfs, file = "data_frames/luci_xgboost_final_metrics_df.rda")
