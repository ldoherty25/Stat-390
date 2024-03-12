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
