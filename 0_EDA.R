## EDA

# primary checks ----

## load packages ----
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

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)


## load data ----

# read and clean
covid <- read_csv('data/raw/data.csv') %>% 
  janitor::clean_names()

# data quality assurance
skimr::skim_without_charts(covid)



# working through multivariate dataset, i ----

# calculate average of identical variables
mutated_covid <- covid %>%
  mutate(average_confirmed = rowMeans(select(., jhu_confirmed, owid_total_cases, ox_confirmed_cases), na.rm = TRUE),
         average_cummulative_deaths = rowMeans(select(., jhu_deaths, owid_total_deaths, ox_confirmed_deaths), na.rm = TRUE),
         average_stringency_index = rowMeans(select(., owid_stringency_index, ox_stringency_index), na.rm = TRUE)) %>%
  select(-jhu_confirmed, -owid_total_cases, -ox_confirmed_cases, 
         -jhu_deaths, -owid_total_deaths, -ox_confirmed_deaths, 
         -owid_stringency_index, -ox_stringency_index) %>% 
  mutate(
    average_confirmed = replace(average_confirmed, is.nan(average_confirmed), NA),
    average_cummulative_deaths = replace(average_cummulative_deaths, is.nan(average_cummulative_deaths), NA),
    average_stringency_index = replace(average_stringency_index, is.nan(average_stringency_index), NA)
    )

# missingness per variable
prop_non_missing <- mutated_covid %>% 
  summarise(across(everything(), ~sum(!is.na(.)) / n())) %>% 
  pivot_longer(cols = everything(), names_to = "variable", values_to = "prop_non_missing")

# finding variables with more than 50% missingness
columns_to_remove <- prop_non_missing %>% 
  filter(prop_non_missing < 0.5) %>% 
  pull(variable)

# removing unwanted columns
covid_cleaned <- mutated_covid %>% 
  select(-all_of(columns_to_remove))

# selecting only non-redundant variables; new: removed average_cummulative_deaths
preprocessed_covid_multi <- covid_cleaned %>%
  select(country, date, average_confirmed, # average_cummulative_deaths,
         owid_new_cases, owid_new_deaths, average_stringency_index,
         owid_population, owid_population_density, owid_median_age,
         owid_aged_65_older, owid_aged_70_older, owid_gdp_per_capita, owid_cardiovasc_death_rate,
         owid_diabetes_prevalence, owid_female_smokers, owid_male_smokers, owid_hospital_beds_per_thousand, 
         owid_life_expectancy, ox_c1_school_closing, ox_c1_flag,
         ox_c2_workplace_closing, ox_c2_flag, ox_c3_cancel_public_events,
         ox_c3_flag, ox_c4_restrictions_on_gatherings, ox_c4_flag, ox_c6_stay_at_home_requirements,
         ox_c7_restrictions_on_internal_movement,
         ox_c8_international_travel_controls, ox_e1_income_support,
         ox_e2_debt_or_contract_relief, ox_e3_fiscal_measures, ox_e4_international_support,
         ox_h1_public_information_campaigns, ox_h1_flag, ox_h2_testing_policy,
         ox_h3_contact_tracing, ox_h4_emergency_investment_in_healthcare,
         ox_h5_investment_in_vaccines, ox_government_response_index,
         ox_containment_health_index, ox_economic_support_index,
         marioli_effective_reproduction_rate, marioli_ci_65_u, marioli_ci_65_l,
         marioli_ci_95_u, marioli_ci_95_l,
         google_mobility_change_grocery_and_pharmacy,
         google_mobility_change_parks, google_mobility_change_transit_stations,
         google_mobility_change_retail_and_recreation, google_mobility_change_residential,
         google_mobility_change_workplaces, sdsn_effective_reproduction_rate_smoothed)

# skim after clearing issues
skimr::skim_without_charts(preprocessed_covid_multi)


## attending to unexpected negative values ----

# looking at suspicious negative values
negative_values_dataset <- preprocessed_covid_multi %>%
  filter(owid_new_cases < 0 | 
           owid_new_deaths < 0 | 
           ox_e3_fiscal_measures < 0 | 
           ox_e4_international_support < 0) %>% 
  select(country,owid_new_cases, owid_new_deaths, ox_e3_fiscal_measures, ox_e4_international_support, marioli_effective_reproduction_rate)

# reverting to absolute values where necessary
preprocessed_covid_multi <- preprocessed_covid_multi %>%
  mutate(owid_new_cases = abs(owid_new_cases),
         owid_new_deaths = abs(owid_new_deaths),
         ox_e3_fiscal_measures = abs(ox_e3_fiscal_measures),
         ox_e4_international_support = abs(ox_e4_international_support))


## assessing final missingness ----

# inspecting missingness again
missing_prop_covid <- preprocessed_covid_multi %>% 
  naniar::miss_var_summary() %>% 
  filter(pct_miss >= 0) %>% 
  DT::datatable()

# creating graph
missing_graph <- preprocessed_covid_multi %>%
  naniar::gg_miss_var() +
  labs(title = "Missing Data")


## (preliminary) correlation matrix ----

# filter out numerical data
numerical_data <- preprocessed_covid_multi %>% select_if(is.numeric)

# create a correlation matrix
correlation_matrix <- cor(numerical_data, use = "complete.obs")

# establish threshold to reduce dimensions
# (drop if the absolute value of correlation coefficient is under 0.5)
correlation_matrix[abs(correlation_matrix) < 0.5] <- NA

# adapt to ggplot2
# (convert wide-format data to long-format data)
melted_corr_matrix <- melt(correlation_matrix, na.rm = TRUE)

# produce heatmap
correlation_graph <- ggplot(melted_corr_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = '', title = 'Correlation Matrix Heatmap')


## target variable consideration ----

# graphing distribution with square root transformation
tv_distribution_log <- preprocessed_covid_multi %>%
  filter(!is.na(owid_new_deaths)) %>%
  ggplot(aes(x = owid_new_deaths)) +
  geom_histogram(bins = 30, fill = "red", color = "black") +
  scale_x_sqrt(breaks = pretty_breaks(n = 5)) +
  labs(title = "Distribution of Target Variable",
       x = "New Deaths (sqrt)",
       y = "Count") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))


## miscellaneous ----

# determining beginning and start dates of data collection
preprocessed_covid_multi$date <- as.Date(preprocessed_covid_multi$date)
first_date <- min(preprocessed_covid_multi$date, na.rm = TRUE)
last_date <- max(preprocessed_covid_multi$date, na.rm = TRUE)



# working through univariate dataset, i ----

# selecting countries for univariate models
uni_countries <- preprocessed_covid_multi %>%
  group_by(country) %>%
  summarize(
    completeness = sum(!is.na(owid_new_deaths)) / n(),
    earliest_death = ifelse(any(owid_new_deaths > 0),
                            as.Date(min(date[owid_new_deaths > 0],
                                        na.rm = TRUE)),
                            NA_real_),
    total_deaths = sum(owid_new_deaths, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  filter(total_deaths >= 1000, !is.na(earliest_death)) %>%
  arrange(desc(completeness), earliest_death, desc(total_deaths)) %>%
  slice_head(n = 10)


# creating a dataset for each selected country

china <- preprocessed_covid_multi %>%
  filter(country == "China", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

japan <- preprocessed_covid_multi %>%
  filter(country == "Japan", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

france <- preprocessed_covid_multi %>%
  filter(country == "France", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

iran <- preprocessed_covid_multi %>%
  filter(country == "Iran, Islamic Rep.", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

italy <- preprocessed_covid_multi %>%
  filter(country == "Italy", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

us <- preprocessed_covid_multi %>%
  filter(country == "United States", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

switzerland <- preprocessed_covid_multi %>%
  filter(country == "Switzerland", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

uk <- preprocessed_covid_multi %>%
  filter(country == "United Kingdom", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

netherlands <- preprocessed_covid_multi %>%
  filter(country == "Netherlands", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)

germany <- preprocessed_covid_multi %>%
  filter(country == "Germany", owid_new_deaths > 0) %>% 
  select(date, owid_new_deaths)


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

# specifying countries
countries <- c("China", "Japan", "France", "Iran, Islamic Rep.", "Italy", "United States", "Switzerland", "United Kingdom", "Netherlands", "Germany")

# creating a function to generate ACF and PACF visualizations
generate_acf_pacf_plots <- function(country_name) {
  country_data <- preprocessed_covid_multi %>%
    filter(country == country_name, owid_new_deaths > 0) %>%
    arrange(date) %>%  # Sort data by date
    select(date, owid_new_deaths)
  
  if (nrow(country_data) < 2) {
    cat("Insufficient data for", country_name, "\n")
    return(NULL)
  }
  
  acf_plot <- acf(country_data$owid_new_deaths, main = paste("ACF - ", country_name), ylim = c(-1,1), mar = c(4, 4, 2, 1))
  pacf_plot <- pacf(country_data$owid_new_deaths, main = paste("PACF - ", country_name), ylim = c(-1,1), mar = c(4, 4, 2, 1))
  
  return(list(acf_plot = acf_plot, pacf_plot = pacf_plot, country_name = country_name))
}

# generating plots for each country
plots_list <- lapply(countries, generate_acf_pacf_plots)

# determining layout
par(mfrow = c(5, 4), mar = c(4, 4, 2, 1))

# including country names
for (i in 1:10) {
  if (!is.null(plots_list[[i]])) {
    plot(plots_list[[i]]$acf_plot)
    title(main = plots_list[[i]]$country_name, line = -1, cex.main = 0.8)
    plot(plots_list[[i]]$pacf_plot)
    title(main = plots_list[[i]]$country_name, line = -1, cex.main = 0.8)
  }
}

# resetting plot layout
par(mfrow = c(1, 1))


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

# looping through selected countries
for (country_name in countries) {
  
  # filtering only selected country
  country_data <- preprocessed_covid_multi %>%
    filter(country == country_name, !is.na(owid_new_deaths))

  # creating differenced time series
  time_series_diff <- diff(country_data$owid_new_deaths, differences = 1)
  
  # performing ADF test
  adf_test_result_diff <- tseries::adf.test(time_series_diff, alternative = "stationary")
  
  # printing results
  cat("Country:", country_name, "\n")
  cat("ADF Test Statistic:", adf_test_result_diff$statistic, "\n")
  cat("ADF Test p-value:", adf_test_result_diff$p.value, "\n")
}


## creating separate dataset for ARIMA and Univariate Prophet models ----

# creating storage list
country_datasets <- list()

# determining split ratio
split_ratio <- 0.8

# creaing datasets for each country
for (country_name in countries) {
  # Filter the data for the current country
  country_data <- preprocessed_covid_multi %>%
    filter(country == country_name, owid_new_deaths > 0) %>%
    select(date, owid_new_deaths)
  
  # calculating index for splitting data
  split_index <- floor(nrow(country_data) * split_ratio)
  
  # splitting into training and testing sets
  train_data <- country_data[1:split_index, ]
  test_data <- country_data[(split_index + 1):nrow(country_data), ]
  
  # storing datasets in list
  country_datasets[[country_name]] <- list(train_data = train_data, test_data = test_data)
  
  # writing csv files
  write.csv(train_data, file.path("data/preprocessed/univariate/arima/split_datasets", paste0(country_name, "_train.csv")), row.names = FALSE)
  write.csv(test_data, file.path("data/preprocessed/univariate/arima/split_datasets", paste0(country_name, "_test.csv")), row.names = FALSE)
}


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



# working through multivariate dataset, ii ----

# removing quasi-constant features
preprocessed_covid_multi <- preprocessed_covid_multi %>%
  select_if(~ n_distinct(.) > 0.95)


## removing low-correlation with target variable features ----

# extracting numeric data
numeric_data <- preprocessed_covid_multi %>% select_if(is.numeric)

# listing calculated correlation with target variable
corr_target <- cor(numeric_data, use = "complete.obs")[, "owid_new_deaths"]

# removing target variable from the list
corr_target <- corr_target[-which(names(corr_target) == "owid_new_deaths")]

# removing target variables with correlation between -0.1 and 0.1
low_vars <- names(which(abs(corr_target) < 0.1))

# preprocessed data after removing low correlations
preprocessed_covid_multi <- preprocessed_covid_multi %>% select(-all_of(low_vars))


## imputing ----

# imputing with linear interpolation
preprocessed_covid_multi_imputed <- na_interpolation(preprocessed_covid_multi)




# CHECKED ISSUE WITH AGGREGATING DATA UNTIL HERE ----




## CREATING LAGS

#univariate
uni_grouped_covid_lag <- uni_grouped_covid %>%
  mutate(lagged_nd_1 = dplyr::lag(total_new_deaths, n=1),
         lagged_nd_2 = dplyr::lag(total_new_deaths, n=2),
         lagged_nd_7 = dplyr::lag(total_new_deaths, n=7))

#multivariate
preprocessed_covid_multi_lag <- preprocessed_covid_multi_imputed %>%
  mutate(lagged_nd_1 = dplyr::lag(owid_new_deaths, n=1),
         lagged_nd_2 = dplyr::lag(owid_new_deaths, n=2),
         lagged_nd_7 = dplyr::lag(owid_new_deaths, n=7))
  
## ROLLING WINDOW STATISTICS

#rolling averages
#convert the time series data to a zoo object
# Convert data to a zoo object
time_series_zoo <- zoo(uni_grouped_covid$total_new_deaths, order.by = uni_grouped_covid$date)
# Calculate a 30-day rolling mean
rolling_fast_mean <- rollapply(time_series_zoo, width = 30, FUN = mean, align = "right", fill = NA)
rolling_slow_mean <- rollapply(time_series_zoo, width = 90, FUN = mean, align = "right", fill = NA)
#plotting the rolling means
# Plot the original time series data and rolling mean
plot(time_series_zoo, type = "l", col = "blue", ylab = "Total new deaths", main = "Rolling Mean 30 Days")
lines(rolling_fast_mean, col = "red", lwd = 2)
legend("topright", legend = c("Original Data", "Rolling Mean"), col = c("blue", "red"), lty = 1:1, cex = 0.8)

plot(time_series_zoo, type = "l", col = "blue", ylab = "Total new deaths", main = "Rolling Mean 90 Days")
lines(rolling_slow_mean, col = "red", lwd = 2)
legend("topright", legend = c("Original Data", "Rolling Mean"), col = c("blue", "red"), lty = 1:1, cex = 0.8)

# rolling sd
rolling_fast_sd <- rollapply(time_series_zoo, width = 30, FUN = sd, align = "right", fill = NA)
rolling_slow_sd <- rollapply(time_series_zoo, width = 90, FUN = sd, align = "right", fill = NA)

#plotting the rolling sd's
# Plot the original time series data and rolling mean
plot(time_series_zoo, type = "l", col = "blue", ylab = "Total new deaths", main = "Rolling Standard Deviation 30 Days")
lines(rolling_fast_sd, col = "red", lwd = 2)
legend("topright", legend = c("Original Data", "Rolling SD"), col = c("blue", "red"), lty = 1:1, cex = 0.8)

plot(time_series_zoo, type = "l", col = "blue", ylab = "Total new deaths", main = "Rolling Standard Deviation 90 Days")
lines(rolling_slow_sd, col = "red", lwd = 2)
legend("topright", legend = c("Original Data", "Rolling SD"), col = c("blue", "red"), lty = 1:1, cex = 0.8)

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
uni_time_eng <- uni_grouped_covid %>%
  mutate(month = month(date),
         day = mday(date),
         weekday = weekdays(date))

#uni_season <- sapply(uni_grouped_covid, extract_season(uni_grouped_covid$date))

#multivariate
multi_time_eng <- preprocessed_covid_multi_imputed %>%
  mutate(month = month(date),
         day = mday(date),
         weekday = weekdays(date))

ggplot(uni_time_eng, mapping = aes(x = weekday)) + 
  geom_bar()

ggplot(uni_time_eng, mapping = aes(x = month)) +
  geom_bar()

## SEASONALITY---

#plotting
#seasonplot(time_series, year.labels = TRUE, year.labs.left = TRUE, main = "Seasonal new death plot", 
          # xlab = "Year", ylab = "New_death")
#Error in seasonplot(time_series, year.labels = TRUE, year.labs.left = TRUE,  : 
#Data are not seasonal


## custom features ----

preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>%
  mutate(capacity_to_case = owid_hospital_beds_per_thousand / rollmean(average_confirmed, 7, fill = NA, align = 'right')) %>% 
  mutate(vulnerability_index = (owid_aged_65_older + owid_aged_70_older + owid_diabetes_prevalence + owid_cardiovasc_death_rate) / 4) %>% 
  mutate(policy_stringency_index = (ox_c1_school_closing + ox_c2_workplace_closing + ox_c3_cancel_public_events + ox_c4_restrictions_on_gatherings + ox_c6_stay_at_home_requirements + ox_c7_restrictions_on_internal_movement + ox_c8_international_travel_controls) / 7,
         policy_population_index = policy_stringency_index * owid_population_density) %>% 
  mutate_if(is.numeric, ~replace(., is.infinite(.) | is.nan(.), NA))


## feature selection ----

# training an rf model
rf_model <- randomForest(owid_new_deaths ~ ., data = preprocessed_covid_multi_imputed, importance = TRUE, na.action = na.omit)

# computing importance scores
importance_scores <- importance(rf_model)

# converting to data frame
importance_df <- as.data.frame(importance_scores)

# make sure there are no duplicate names
names(importance_df) <- make.unique(names(importance_df))

# incorporating variable name column
importance_df$Variable <- rownames(importance_df)

# melting to long format
importance_long <- melt(importance_df, id.vars = "Variable")

# producing plot
importance <- ggplot(importance_long, aes(x = reorder(Variable, value), y = value)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip the axes to make the plot horizontal
  theme_minimal() +
  labs(x = "Feature", y = "Importance", title = "Feature Importance from Random Forest Model") +
  theme(plot.title = element_text(hjust = 0.5))


# final dimensions ----

# ARIMA (2128)
observations_table <- data.frame(
  Country = c("China", "Japan", "France", "Iran", "Italy", "US", "Switzerland", "UK", "Netherlands", "Germany"),
  Observations = c(nrow(china), nrow(japan), nrow(france), nrow(iran), nrow(italy), nrow(us), nrow(switzerland), nrow(uk), nrow(netherlands), nrow(germany))
) %>% 
  DT::datatable()

# Prophet (3050)
dim(prophet_dataset)

# multivariate (64675 x 57)
dim(preprocessed_covid_multi_imputed)



# saving files ----
save(preprocessed_covid_multi_imputed, file = "data/preprocessed/multivariate/preprocessed_covid_multi_imputed")
save(missing_prop_covid, file = "visuals/missing_prop_covid.rda")
save(correlation_graph, file = "visuals/correlation_graph.rda")
save(missing_graph, file = "visuals/missing_graph.rda")
save(tv_distribution_log, file = "visuals/tv_distribution_log.rda")
save(china, file = "data/preprocessed/univariate/arima/china.rda")
save(japan, file = "data/preprocessed/univariate/arima/japan.rda")
save(france, file = "data/preprocessed/univariate/arima/france.rda")
save(iran, file = "data/preprocessed/univariate/arima/iran.rda")
save(italy, file = "data/preprocessed/univariate/arima/italy.rda")
save(us, file = "data/preprocessed/univariate/arima/us.rda")
save(switzerland, file = "data/preprocessed/univariate/arima/switzerland.rda")
save(uk, file = "data/preprocessed/univariate/arima/uk.rda")
save(netherlands, file = "data/preprocessed/univariate/arima/netherlands.rda")
save(germany, file = "data/preprocessed/univariate/arima/germany.rda")
save(acf_plot, file = "visuals/acf_plot.rda")
save(pacf_plot, file = "visuals/pacf_plot.rda")
save(importance, file = "visuals/importance.rda")