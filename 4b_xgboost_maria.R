
# Load the model
load("xgboost_components/xgb_model_ii_maria.rda")

# Load the testing dataset
load("data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")

# Select only the data for Bolivia after applying the same preprocessing steps as done for the entire dataset
bolivia_testing_df <- preprocessed_covid_multi_imputed %>%
  filter(country == "Bolivia") %>%
  janitor::clean_names() %>%
  mutate(ds = as.Date(date), y = owid_new_deaths) %>%
  arrange(ds) %>%
  select(-country, -date) # Keep 'ds' for now to ensure we have a date column for arranging

# Check for character columns and convert them to factors then to numeric
bolivia_test_data <- bolivia_testing_df %>%
  mutate_if(is.character, as.factor) %>% # Convert character columns to factors
  mutate_if(is.factor, as.numeric) %>% # Then convert factors to numeric
  select(-ds, -y) # Finally, exclude the date and target variable

# Ensure the data is all numeric now
if (any(sapply(bolivia_test_data, is.character))) {
  stop("Data still contains non-numeric columns.")
}

# Convert to matrix for prediction
bolivia_test_data_matrix <- as.matrix(bolivia_test_data)

# Make predictions for Bolivia
bolivia_predictions <- predict(xgb_model_ii, bolivia_test_data_matrix)

# Extract the actual values for 'y' from the Bolivian testing dataframe
bolivia_actuals <- bolivia_testing_df$y

# Calculate performance metrics for Bolivia
bolivia_RMSE <- sqrt(mean((bolivia_predictions - bolivia_actuals)^2))
bolivia_MAE <- mean(abs(bolivia_predictions - bolivia_actuals))
bolivia_MSE <- mean((bolivia_predictions - bolivia_actuals)^2)
bolivia_MAPE <- mean(abs((bolivia_predictions - bolivia_actuals) / bolivia_actuals), na.rm = TRUE)
bolivia_MASE <- mean(abs(bolivia_predictions - bolivia_actuals)) / mean(abs(diff(na.omit(bolivia_actuals))), na.rm = TRUE)

# Create a data frame for the metrics for Bolivia
bolivia_metrics_table <- data.frame(
  Country = "Bolivia",
  Best_Model = "Multivariate XGBoost",
  RMSE = bolivia_RMSE, 
  MAE = bolivia_MAE, 
  MSE = bolivia_MSE, 
  MAPE = bolivia_MAPE, 
  MASE = bolivia_MASE
)

# Print the metrics for Bolivia
print(bolivia_metrics_table)
