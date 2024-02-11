## A PLACE FOR ALL OUR COMMENTED OUT CODE

## constructing ACF and PACF visualizations for each country ----
# 
# # defining a list of selected countries
# selected_countries <- c("China", "Japan", "France", "Iran, Islamic Rep.",
#                         "Italy", "United States", "Switzerland",
#                         "United Kingdom", "Netherlands", "Germany")
# 
# # creating a grouped dataset summing each day's new deaths
# uni_grouped_covid <- preprocessed_covid_multi %>%
#   filter(country %in% selected_countries) %>%
#   group_by(date) %>%
#   summarize(total_new_deaths = sum(owid_new_deaths, na.rm = TRUE)) %>%
#   ungroup() %>%
#   complete(date = seq(min(date), max(date), by = "day"), fill = list(total_new_deaths = 0))
# 
# # calculating ACF and PACF
# # (how the values time series are correlated with their own lagged values)
# acf_vals <- acf(uni_grouped_covid$total_new_deaths, plot = FALSE)
# # (controlling for other lags)
# pacf_vals <- pacf(uni_grouped_covid$total_new_deaths, plot = FALSE)
# 
# # calculating standard error
# # (if the ACF >> se -> true pattern in data)
# n <- sum(!is.na(uni_grouped_covid$total_new_deaths))
# se <- 1 / sqrt(n)
# 
# # producing the desired plots
# 
# # (strong + autocorrelation at all lags / decreasing trend as lags increase)
# acf_plot <- ggplot(data.frame(Lag = 1:(length(acf_vals$acf)-1), ACF = acf_vals$acf[-1]), aes(x = Lag, y = ACF)) +
#   geom_bar(stat = "identity", fill = "grey") +
#   geom_hline(yintercept = 0) +
#   geom_hline(yintercept = c(1.96, -1.96) * se, color = "blue") +
#   theme_minimal() +
#   labs(title = "Autocorrelation Function (ACF)", x = "Lags", y = "ACF")
# 
# # (no significant partial autocorrelation)
# pacf_plot <- ggplot(data.frame(Lag = 1:(length(pacf_vals$acf)-1), PACF = pacf_vals$acf[-1]), aes(x = Lag, y = PACF)) +
#   geom_bar(stat = "identity", fill="grey") +
#   geom_hline(yintercept = 0) +
#   geom_hline(yintercept = c(1.96, -1.96) * se, color = "blue") +
#   theme_minimal() +
#   labs(title = "Partial Autocorrelation Function (PACF)", x = "Lags", y = "PACF")


## data decomposition (sufficient since trends very similar) ----

# # generating plot components
# time_series <- ts(uni_grouped_covid$total_new_deaths)
# trend <- ma(time_series, order = 14)
# 
# # calculating residuals
# residuals <- time_series - trend
# 
# # joining plot components (needs saving)
# par(mfrow = c(3, 1))
# plot(time_series, main = "Observed", xlab = "", ylab = "New Deaths", col = "black", xaxt='n')
# plot(trend, main = "Trend", xlab = "", ylab = "Trend", col = "blue", xaxt='n')
# # (what remains after the trend is removed)
# plot(residuals, main = "Residuals", xlab = "Time", ylab = "Residuals", col = "red")
# par(mfrow = c(1, 1))


## checking if the series is stationary ----

# # creating differenced time series
# # (difference between it and the value)
# time_series_diff <- diff(time_series, differences = 1)
# 
# # using time_series_diff as the differenced time series object
# adf_test_result_diff <- tseries::adf.test(time_series_diff, alternative = "stationary")
# 
# # checking the test statistic and p-value ("p-value smaller than printed p-value" above)
# cat("ADF Test Statistic:", adf_test_result_diff$statistic, "\n")
# cat("ADF Test p-value:", adf_test_result_diff$p.value, "\n")
# 
# # plotting the differenced series (needs saving)
# plot(time_series_diff, main = "Differenced Time Series")



# # Prophet
# 
# # selecting appropriate variables
# selected_variables <- c("date", "country", "owid_new_deaths")
# 
# # Filter the dataset
# prophet_dataset <- preprocessed_covid_multi %>%
#   filter(country %in% selected_countries) %>%
#   select(all_of(selected_variables))
# 
# # loop through countries to create dataset
# for (country_name in selected_countries) {
#   
#   # filter by country
#   country_data <- prophet_dataset %>%
#     filter(country == country_name, owid_new_deaths > 0) %>%
#     select(date, owid_new_deaths)
#   
#   # determining splitting index
#   split_index <- floor(nrow(country_data) * 0.8)
#   
#   # splitting into training and testing
#   train_data <- country_data[1:split_index, ]
#   test_data <- country_data[(split_index + 1):nrow(country_data), ]
#   
#   # writing files
#   write.csv(train_data, file.path("data/preprocessed/univariate/prophet/", paste0(country_name, "_train.csv")), row.names = FALSE)
#   write.csv(test_data, file.path("data/preprocessed/univariate/prophet/", paste0(country_name, "_test.csv")), row.names = FALSE)
# }


#univariate
#uni_grouped_covid_lag <- uni_grouped_covid %>%
#  mutate(lagged_nd_1 = dplyr::lag(total_new_deaths, n=1),
#         lagged_nd_2 = dplyr::lag(total_new_deaths, n=2),
#        lagged_nd_7 = dplyr::lag(total_new_deaths, n=7))




## ROLLING WINDOW STATISTICS

#rolling averages
#convert the time series data to a zoo object
# Convert data to a zoo object
#time_series_zoo <- zoo(uni_grouped_covid$total_new_deaths, order.by = uni_grouped_covid$date)
# Calculate a 30-day rolling mean
#rolling_fast_mean <- rollapply(time_series_zoo, width = 30, FUN = mean, align = "right", fill = NA)
#rolling_slow_mean <- rollapply(time_series_zoo, width = 90, FUN = mean, align = "right", fill = NA)
#plotting the rolling means
# Plot the original time series data and rolling mean
#plot(time_series_zoo, type = "l", col = "blue", ylab = "Total new deaths", main = "Rolling Mean 30 Days")
#lines(rolling_fast_mean, col = "red", lwd = 2)
#legend("topright", legend = c("Original Data", "Rolling Mean"), col = c("blue", "red"), lty = 1:1, cex = 0.8)

#plot(time_series_zoo, type = "l", col = "blue", ylab = "Total new deaths", main = "Rolling Mean 90 Days")
#lines(rolling_slow_mean, col = "red", lwd = 2)
#legend("topright", legend = c("Original Data", "Rolling Mean"), col = c("blue", "red"), lty = 1:1, cex = 0.8)

# rolling sd
#rolling_fast_sd <- rollapply(time_series_zoo, width = 30, FUN = sd, align = "right", fill = NA)
#rolling_slow_sd <- rollapply(time_series_zoo, width = 90, FUN = sd, align = "right", fill = NA)

#plotting the rolling sd's
# Plot the original time series data and rolling mean
#plot(time_series_zoo, type = "l", col = "blue", ylab = "Total new deaths", main = "Rolling Standard Deviation 30 Days")
#lines(rolling_fast_sd, col = "red", lwd = 2)
#legend("topright", legend = c("Original Data", "Rolling SD"), col = c("blue", "red"), lty = 1:1, cex = 0.8)

#plot(time_series_zoo, type = "l", col = "blue", ylab = "Total new deaths", main = "Rolling Standard Deviation 90 Days")
#lines(rolling_slow_sd, col = "red", lwd = 2)
#legend("topright", legend = c("Original Data", "Rolling SD"), col = c("blue", "red"), lty = 1:1, cex = 0.8)

## TIME BASED FEATURES ---

#extract_season <- function(date) {
#  month_val <- month(date)
# if (month_val %in% c(3, 4, 5)) {
#   return("Spring")
# } else if (month_val %in% c(6, 7, 8)) {
#   return("Summer")
# } else if (month_val %in% c(9, 10, 11)) {
#  return("Fall")
# } else {
#   return("Winter")
# }
#}

#univariate
#uni_time_eng <- uni_grouped_covid %>%
#  mutate(month = month(date),
#         day = mday(date),
#         weekday = weekdays(date))

#uni_season <- sapply(uni_grouped_covid, extract_season(uni_grouped_covid$date))


#ggplot(uni_time_eng, mapping = aes(x = weekday)) + 
#  geom_bar()

#ggplot(uni_time_eng, mapping = aes(x = month)) +
#  geom_bar()

## SEASONALITY---

#plotting
#seasonplot(time_series, year.labels = TRUE, year.labs.left = TRUE, main = "Seasonal new death plot", 
# xlab = "Year", ylab = "New_death")
#Error in seasonplot(time_series, year.labels = TRUE, year.labs.left = TRUE,  : 
#Data are not seasonal


# Prophet (3050)
#dim(prophet_dataset)

# Install and load necessary packages
install.packages("tidyverse")
library(tidyverse)

# Sample data frame with a column representing months
df <- data.frame(Month = c("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"))

# Function to perform cyclical encoding and create separate columns
cyclical_encode <- function(x, period) {
  sin_val <- sin(2 * pi * x / period)
  cos_val <- cos(2 * pi * x / period)
  return(data.frame(sin_val = sin_val, cos_val = cos_val))
}

cyclical_encode_sin <- function(x, time_period) {
  sin_val <- sin(2 * pi * x / time_period)
  return(data.frame(sin_val = sin_val))
}

# Apply cyclical encoding and create separate columns
df <- df %>%
  rowwise() %>%
  mutate(cyclical_encoding = list(cyclical_encode_sin(as.integer(factor(Month, levels = month.name)), time_period = 12))) #%>%
  #bind_cols(., unnest(cols = df$cyclical_encoding))

# Remove the temporary column
df <- df %>% select(-cyclical_encoding)

# Print the result
print(df)


#multivariate
# holidays <- as.Date(c("2020-01-01", "2023-04-12", "2020-12-24", "2020-12-25", "2020-05-23", "2020-05-04", "2020-07-30",
#                       "2020-07-31", "2020-11-14", "2020-02-14", "2020-05-05", "2020-12-10", "2020-12-18",
#                       "2020-10-31", "2020-12-21", "2020-11-01", "2020-11-02", "2020-11-26", "2020-03-19"))
##Cyclical encoding
#Months_df <- data.frame(Month = c("January", "February", "March", "April", "May", "June", "July", 
#                                  "August", "September", "October", "November", "December"))
# 
# # defining the function for cyclical encoding (sin)
# cyclical_encode_sin <- function(x, time_period) {
#   sin_val <- sin(2 * pi * x / time_period)
#   return(sin_val)
# }



#list(cyclical_encode_sin(as.integer(factor(multi_time_eng$Month, levels = month.name)), 
#    time_period = 12))) #%>%
# bind_cols(., unnest(cols = multi_time_eng$cyclical_month)) #%>%
# rename(cyclical_month_sin = )
# mutate(cyclical_weekday = list(cyclical_encode(as.integer(factor(weekday, levels = weekday.name)),
#time_period = 7))) %>%
#bind_cols(., unnest(multi_time_eng$cyclical_weekday))

# #plotting season and new_deaths
# ggplot(multi_time_eng, mapping = aes(x = season, y = owid_new_deaths))+
#   geom_boxplot()
# ggplot(multi_time_eng, mapping = aes(x = season, y = owid_new_deaths))+
#   geom_violin()
# ggplot(multi_time_eng, mapping = aes(x = season))+
#   geom_bar() +
#   labs(title = "Seasonal Count Plot")
# ggplot(multi_time_eng, mapping = aes(x = owid_new_deaths))+
#   geom_histogram()+
#   facet_wrap(~season)



# ggplot(preprocessed_covid_multi_imputed, mapping = aes(x = weekday, y = owid_new_deaths))+
#   geom_violin()
# ggplot(preprocessed_covid_multi_imputed, mapping = aes(x = owid_new_deaths))+
#   geom_histogram()+
#   facet_wrap(~weekday)


# #plotting IsHoliday with new deaths
# ggplot(multi_time_eng, mapping = aes(x = IsHoliday, y = owid_new_deaths))+
#   geom_boxplot()
# ggplot(multi_time_eng, mapping = aes(x = IsHoliday, y = owid_new_deaths))+
#   geom_violin()
# ggplot(multi_time_eng, mapping = aes(x = owid_new_deaths))+
#   geom_histogram()+
#   facet_wrap(~IsHoliday)
# ggplot(multi_time_eng, mapping = aes(x = IsHoliday))+
#   geom_bar() +
#   labs(title = "Holiday Count Plot")

# preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>%
#   mutate(capacity_to_case = owid_hospital_beds_per_thousand / rollmean(averaged_confirmed_cases, 7, fill = NA, align = 'right')) %>% 
#   mutate(vulnerability_index = (owid_aged_65_older + owid_aged_70_older + owid_diabetes_prevalence + owid_cardiovasc_death_rate) / 4) %>% 
#   mutate(policy_stringency_index = (ox_c1_school_closing + ox_c2_workplace_closing + ox_c3_cancel_public_events + ox_c4_restrictions_on_gatherings + ox_c6_stay_at_home_requirements + ox_c7_restrictions_on_internal_movement + ox_c8_international_travel_controls) / 7,
#          policy_population_index = policy_stringency_index * owid_population_density) %>% 
#   mutate_if(is.numeric, ~replace(., is.infinite(.) | is.nan(.), NA))


# season = case_when(
#   month %in% c(3, 4, 5) ~ "Spring",
#   month %in% c(6, 7, 8) ~ "Summer",
#   month %in% c(9, 10, 11) ~ "Fall",
#   TRUE ~ "Winter"))