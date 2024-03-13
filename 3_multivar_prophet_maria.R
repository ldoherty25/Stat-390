## Multivariate Prophet
## Due to computational issues that arose when looping, country names were manually changed to train models.

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

# load data
load("data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")

# specifying data for russia
russia_data <- preprocessed_covid_multi_imputed %>%
  filter(country == "Russian Federation") %>%
  filter(cumsum(owid_new_deaths) >= 1) %>%
  mutate(ds = date, y = owid_new_deaths) %>%
  arrange(ds) %>% 
  select(-c(date,
            owid_new_deaths,
            country,
            month,
            weekday,
            cyclical_month_cos,
            cyclical_weekday_cos,
            cyclical_dayofmth_cos,
            day,
            cyclical_month_sin,
            cyclical_weekday_sin,
            cyclical_dayofmth_sin))

russia_data$ds <- as.Date(russia_data$ds)

# splitting logic for russia
n <- nrow(russia_data)
fold_size <- floor(n / 5)

# initializing a list to store metrics for russia
russia_metrics <- list()

# loop through each fold for russia
for (fold in 1:5) {
  test_indices <- ((fold - 1) * fold_size + 1):min(fold * fold_size, n)
  testing_data <- russia_data[test_indices, ]
  training_data <- russia_data[-test_indices, ]
  
  print("Training data:")
  print(head(training_data))
  print("Testing data:")
  print(head(testing_data))
  
  if (nrow(training_data) > 0 & nrow(testing_data) > 0) {
    m <- prophet()
    m <- add_regressor(m, 'owid_new_cases')
    m <- add_regressor(m, 'owid_population')
    m <- add_regressor(m, 'owid_cardiovasc_death_rate')
    m <- add_regressor(m, 'owid_diabetes_prevalence')
    m <- add_regressor(m, 'owid_male_smokers')
    m <- add_regressor(m, 'ox_c1_school_closing')
    m <- add_regressor(m, 'ox_c1_flag')
    m <- add_regressor(m, 'ox_c2_workplace_closing')
    m <- add_regressor(m, 'ox_c2_flag')
    m <- add_regressor(m, 'ox_c3_flag')
    m <- add_regressor(m, 'ox_c4_restrictions_on_gatherings')
    m <- add_regressor(m, 'ox_c4_flag')
    m <- add_regressor(m, 'ox_c6_stay_at_home_requirements')
    m <- add_regressor(m, 'ox_c7_restrictions_on_internal_movement')
    m <- add_regressor(m, 'ox_h1_flag')
    m <- add_regressor(m, 'ox_government_response_index')
    m <- add_regressor(m, 'google_mobility_change_parks')
    m <- add_regressor(m, 'google_mobility_change_retail_and_recreation')
    
    print("Model summary:")
    print(summary(m))
    
    tryCatch({
      m <- fit.prophet(m, training_data)
    }, error = function(e) {
      print("Error in fitting the Prophet model:")
      print(e)
    })
    
    forecast_dates_df <- testing_data %>% 
      select(ds, owid_new_cases, owid_population, owid_cardiovasc_death_rate, owid_diabetes_prevalence, 
             owid_male_smokers, ox_c1_school_closing, ox_c1_flag, ox_c2_workplace_closing, ox_c2_flag, 
             ox_c3_flag, ox_c4_restrictions_on_gatherings, ox_c4_flag, ox_c6_stay_at_home_requirements, 
             ox_c7_restrictions_on_internal_movement, ox_h1_flag, ox_government_response_index, 
             google_mobility_change_parks, google_mobility_change_retail_and_recreation)
    
    tryCatch({
      forecast <- predict(m, forecast_dates_df)
    }, error = function(e) {
      print("Error in generating forecast:")
      print(e)
    })
    
    print("Forecast summary:")
    print(head(forecast))
    
    matched_forecast <- forecast %>%
      dplyr::filter(ds %in% testing_data$ds)
    
    print("Matched forecast summary:")
    print(head(matched_forecast))
    
    predictions <- matched_forecast$yhat
    actuals <- testing_data$y
    
    if (length(predictions) == 0 || length(actuals) == 0) {
      print("Warning: Empty predictions or actuals. Skipping RMSE calculation.")
      next
    }
    
    RMSE <- sqrt(mean((predictions - actuals)^2))
    MSE <- mean((predictions - actuals)^2)
    MAE <- mean(abs(predictions - actuals))
    
    entire_data <- preprocessed_covid_multi_imputed %>%
      filter(country == "Russian Federation") %>%
      filter(cumsum(owid_new_deaths) > 0) %>%
      arrange(date) %>%
      select(y = owid_new_deaths)
    
    naive_forecasts_entire <- entire_data$y[-1]
    actuals_for_naive_entire <- entire_data$y[-length(entire_data$y)]
    scaling_factor_entire <- mean(abs(naive_forecasts_entire - actuals_for_naive_entire))

    print(paste("MAE:", MAE))
    print(paste("Scaling Factor (Denominator for MASE):", scaling_factor_entire))
    
    # calculating MASE
    MASE <- MAE / scaling_factor_entire
    
    print(paste("MASE:", MASE))
    
    MASE <- MAE / scaling_factor_entire
    
    # storing metrics for this fold
    russia_metrics[[fold]] <- list(
      RMSE = RMSE,
      MSE = MSE,
      MAE = MAE,
      MASE = MASE
    )
  }
}

# converting metrics for russia to a dataframe and making adjustments
russia_metrics_df <- do.call(rbind, lapply(russia_metrics, function(x) do.call(data.frame, x)))
russia_metrics_df$Country <- "russia"
russia_metrics_df$Model_Type <- "Multivariate Prophet"
russia_metrics_df <- russia_metrics_df %>% 
  select(Country, Model_Type, RMSE, MSE, MAE, MASE)

# calculating average metrics across folds
russia_average_metrics <- colMeans(russia_metrics_df[, -which(names(russia_metrics_df) %in% c("Country", "Model_Type"))], na.rm = TRUE)

# creating a dataframe with averaged metrics
russia_averaged_metrics_df <- data.frame(
  Country = "russia",
  Model_Type = "Multivariate Prophet",
  RMSE = russia_average_metrics["RMSE"],
  MSE = russia_average_metrics["MSE"],
  MAE = russia_average_metrics["MAE"],
  MASE = russia_average_metrics["MASE"]
)

row.names(russia_averaged_metrics_df) <- NULL

# printing averaged_metrics_df to check its structure
print("Printing averaged_metrics_df:")
print(russia_averaged_metrics_df)

# # joining all manually obtained metrics
# multivar_prophet_maria <- rbind(bolivia_averaged_metrics_df,
#                                 brazil_averaged_metrics_df,
#                                 turkey_averaged_metrics_df,
#                                 russia_averaged_metrics_df,
#                                 us_averaged_metrics_df,
#                                 iran_averaged_metrics_df,
#                                 saudi_averaged_metrics_df,
#                                 colombia_averaged_metrics_df,
#                                 mexico_averaged_metrics_df,
#                                 peru_averaged_metrics_df)
# 
# multivar_prophet_maria <- multivar_prophet_maria %>% 
#   arrange(RMSE)
# 
# save(multivar_prophet_maria, file = "data_frames/multivar_prophet_maria.rda")
