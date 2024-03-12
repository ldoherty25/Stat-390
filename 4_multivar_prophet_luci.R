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


multivar <- preprocessed_covid_multi_imputed %>% 
  dummy_cols("country", remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
  janitor::clean_names() %>% 
  rename(ds = date, y = owid_new_deaths) %>% 
  arrange(ds)

multivar <- multivar %>% 
  mutate(weekday = lubridate::wday(ds, label = TRUE, abbr = FALSE)) %>% 
  mutate(weekday = as.numeric(as.factor(weekday))) %>% 
  mutate(weekday = ifelse(is.na(weekday), 0, weekday))

split_index = floor(0.8 * nrow(multivar))
train <- multivar[1:split_index, ]
test <- multivar[(split_index + 1):nrow(multivar), ]


model <- prophet(
  growth = "linear",
  n.changepoints = 40, 
  changepoint.range = 0.80, 
  yearly.seasonality = FALSE, 
  weekly.seasonality = TRUE, 
  daily.seasonality = FALSE, 
  seasonality.mode = "additive", 
  seasonality.prior.scale = 10, 
  holidays.prior.scale = 10, 
  changepoint.prior.scale = 0.05, 
  interval.width = 0.80, 
  uncertainty.samples = 1000, 
)


predictors <- setdiff(names(train), c("ds", "y", "owid_new_deaths", "date"))

for (predictor in predictors){
  model <- add_regressor(model, predictor)
}

model <- fit.prophet(model, df = train)

## Add cross validation

split_index2 <- floor(0.9 * nrow(train))
train2 <- train[1:split_index2, ]
test2 <- train[(split_index2 + 1):nrow(train), ]


model2 <- prophet(
    growth = "linear",
    n.changepoints = 40, 
    changepoint.range = 0.80, 
    yearly.seasonality = FALSE, 
    weekly.seasonality = TRUE, 
    daily.seasonality = FALSE, 
    seasonality.mode = "additive", 
    seasonality.prior.scale = 10, 
    holidays.prior.scale = 10, 
    changepoint.prior.scale = 0.05, 
    interval.width = 0.80, 
    uncertainty.samples = 1000, 
)

for (predictor in predictors){
  model2 <- add_regressor(model2, predictor)
}

model2 <- fit.prophet(model2, df = train2)

future <- make_future_dataframe(model2, periods = nrow(test2), freq = 'day')

test2_subset <- test2[1:nrow(future), ]

future <- cbind(future, test2_subset[, predictors])

future <- future[!is.na(future$owid_new_cases), ]

forecast <- predict(model2, future)

actual <- test2$y
forecast_values <- forecast$yhat

multivar_metrics <- data.frame(
  RMSE = sqrt(mean((forecast$yhat - test2$y)^2)),
  MAE = mean(abs(forecast$yhat - test2$y)),
  MSE = mean((forecast$yhat - test2$y)^2),
  MAPE = mean(abs((test2$y - forecast$yhat) / test2$y)) * 100,
  MASE = mean(abs(diff(test2$y)) / mean(abs(forecast$yhat - test2$y))))


cat("RMSE:", mean(multivar_metrics$RMSE))
cat("MAE:", mean(multivar_metrics$MAE))
cat("MSE:", mean(multivar_metrics$MSE))
cat("MAPE:", mean(multivar_metrics$MAPE))
cat("MASE:", mean(multivar_metrics$MASE))

print(multivar_metrics)

plot_data <- data.frame(
  ds = future$ds,
  actual = actual,
  forecast = forecast_values
)

# Plot actual vs forecasted values
ggplot(plot_data, aes(x = ds)) +
  geom_line(aes(y = actual, color = "Actual"), size = 1.2) +
  geom_line(aes(y = forecast, color = "Forecast"), size = 1.2, linetype = "dashed") +
  labs(
    title = "Actual vs Forecasted COVID-19 Deaths",
    x = "Date",
    y = "Number of Deaths"
  ) +
  scale_color_manual(values = c("Actual" = "blue", "Forecast" = "red")) +
  theme_minimal()

save(multivar_metrics, file = "data_frames/luci_multivar_prophet_final_metrics_df.rda")


