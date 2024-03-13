### COPY FOR CODING PURPOSES
## Multivariate Prophet

# primary checks ----

# load packages
library(dplyr)
library(prophet)
library(fastDummies)
library(tidymodels)
library(doMC)
library(ggplot2)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)

# load data ----
load("data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")

# Want to remove all zero observations before the first death


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

# model setup and training
### removed seasonality since it's built into prophet -----
m <- prophet(growth = "linear", seasonality.mode = "additive", 
             seasonality.prior.scale = 10, holidays.prior.scale = 10,
             changepoint.prior.scale = 0.1, n.changepoints = 25, 
             changepoint.range = 0.8, interval.width = 0.8, uncertainty.samples = 1000)

# adding predictors as regressors
predictor_columns <- setdiff(names(training_df), c("ds", "y", "date", "owid_new_deaths"))
for (predictor in predictor_columns) {
  m <- add_regressor(m, predictor)
}
### doesn't look like the target var is in the predictors specified

# fitting model with training data
m <- prophet::fit.prophet(m, df = training_df)



# cross-validation process ----

cv_split_index <- floor(0.9 * nrow(training_df))
cv_training_df <- training_df[1:cv_split_index, ]
cv_validation_df <- training_df[(cv_split_index + 1):nrow(training_df), ]

create_prophet_model <- function() {
  prophet(
    growth = "linear",
    seasonality.mode = "additive",
    seasonality.prior.scale = 10,
    holidays.prior.scale = 10,
    changepoint.prior.scale = 0.1,
    n.changepoints = 25,
    changepoint.range = 0.8,
    interval.width = 0.8,
    uncertainty.samples = 1000
  )
}

# creating new model for cross-validation
cv_model <- create_prophet_model()

# adding predictors as regressors for new model
for (predictor in predictor_columns) {
  cv_model <- add_regressor(cv_model, predictor)
}

# fitting model on cv_training_df
cv_model <- prophet::fit.prophet(cv_model, df = cv_training_df)

# forecasting
cv_future <- make_future_dataframe(cv_model, periods = nrow(cv_validation_df), freq = 'day')

# filtering cv_future to match the dates in cv_validation_df
cv_future_filtered <- cv_future %>%
  filter(ds %in% cv_validation_df$ds)

# left-joining to bring in predictor columns
cv_future_filtered <- left_join(cv_future_filtered, cv_validation_df[, c("ds", predictor_columns)], by = "ds")

# checking missing dates
if(nrow(cv_future_filtered) != nrow(cv_validation_df)) {
  stop("Mismatch in the number of rows after preparing cv_future. Check date alignment.")
}

# forecasting on validation set
cv_forecast <- predict(cv_model, cv_future_filtered)

# comparing forecast against actual values
actuals <- cv_validation_df$y
forecasts <- cv_forecast$yhat



# evaluation metrics ----

# calculating metrics
RMSE <- sqrt(mean((forecasts - actuals)^2))
MAE <- mean(abs(forecasts - actuals))
MSE <- mean((forecasts - actuals)^2)

# naive forecast for mas calculation
naive_forecast <- c(NA, actuals[-length(actuals)])
MASE <- mean(abs(forecasts - actuals)) / mean(abs(naive_forecast - actuals)[-1])

# creating a data frame
maria_multivar_prophet_table <- data.frame(
  Country = "All",
  Best_Model_RMSE = "Multivariate Prophet",
  RMSE, 
  MAE, 
  MSE, 
  MASE
)



# saving files ----
save(maria_multivar_prophet_table, file = "data_frames/maria_multivar_prophet_table.rda")



# creating visualizations ----

# summing new deaths across all countries for each date

agg_actuals <- cv_validation_df %>%
  group_by(ds) %>%
  summarize(agg_new_deaths = sum(y, na.rm = TRUE))

agg_forecasts <- cv_forecast %>%
  group_by(ds) %>%
  summarize(agg_yhat = sum(yhat, na.rm = TRUE),
            agg_yhat_lower = sum(yhat_lower, na.rm = TRUE),
            agg_yhat_upper = sum(yhat_upper, na.rm = TRUE))

# creating a plot with aggregated actual and forecasted values
ggplot() +
  geom_line(data = agg_actuals, aes(x = ds, y = agg_new_deaths), color = "blue", size = 1) +
  geom_line(data = agg_forecasts, aes(x = ds, y = agg_yhat), color = "red", size = 1, linetype = "dashed") +
  geom_ribbon(data = agg_forecasts, aes(x = ds, ymin = agg_yhat_lower, ymax = agg_yhat_upper),
              fill = "orange", alpha = 0.2) +
  labs(title = "Aggregated Actual vs Forecasted New Deaths Across All Countries",
       x = "Date",
       y = "Aggregated New Deaths") +
  theme_minimal() +
  theme(legend.position = "bottom")