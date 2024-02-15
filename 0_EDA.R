## EDA

# primary checks ----

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

## preliminary cleaning ----

# calculate average of identical variables
mutated_covid <- covid %>%
  mutate(averaged_confirmed_cases = rowMeans(select(., jhu_confirmed, owid_total_cases, ox_confirmed_cases), na.rm = TRUE),
         averaged_cummulative_deaths = rowMeans(select(., jhu_deaths, owid_total_deaths, ox_confirmed_deaths), na.rm = TRUE),
         averaged_stringency_index = rowMeans(select(., owid_stringency_index, ox_stringency_index), na.rm = TRUE)) %>%
  select(-jhu_confirmed, -owid_total_cases, -ox_confirmed_cases, 
         -jhu_deaths, -owid_total_deaths, -ox_confirmed_deaths, 
         -owid_stringency_index, -ox_stringency_index) %>% 
  mutate(
    averaged_confirmed_cases = replace(averaged_confirmed_cases, is.nan(averaged_confirmed_cases), NA),
    averaged_cummulative_deaths = replace(averaged_cummulative_deaths, is.nan(averaged_cummulative_deaths), NA),
    averaged_stringency_index = replace(averaged_stringency_index, is.nan(averaged_stringency_index), NA)
    )

# missingness per variable
prop_non_missing <- mutated_covid %>% 
  summarise(across(everything(), ~sum(!is.na(.)) / n())) %>% 
  pivot_longer(cols = everything(), names_to = "variable", values_to = "prop_non_missing")

# finding variables with more than 40% missingness
columns_to_remove <- prop_non_missing %>% 
  filter(prop_non_missing < 0.4) %>% 
  pull(variable)

# removing unwanted columns
covid_cleaned <- mutated_covid %>% 
  select(-all_of(columns_to_remove))

# selecting only non-redundant variables
preprocessed_covid_multi <- covid_cleaned %>%
  select(country, date, averaged_confirmed_cases,
         owid_new_cases, owid_new_deaths, averaged_stringency_index,
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


## removing outliers ----

# creating a function to remove outliers within a group
remove_outliers <- function(df, var_name) {
  Q1 <- quantile(df[[var_name]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[var_name]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  df <- df %>% filter(df[[var_name]] >= lower_bound & df[[var_name]] <= upper_bound)
  return(df)
}

# applying to each country
preprocessed_covid_multi <- preprocessed_covid_multi %>%
  group_by(country) %>%
  do(remove_outliers(., 'owid_new_deaths')) %>%
  ungroup()


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



# working through univariate dataset, i ----

## creating country datasets ----

# selecting countries for univariate models
uni_countries <- preprocessed_covid_multi %>%
  group_by(country) %>%
  summarize(
    completeness = sum(!is.na(owid_new_deaths)) / n(),
    date_diff_std = ifelse(sum(owid_new_deaths > 0) > 1,
                           sd(diff(date[owid_new_deaths > 0])),
                           NA_real_),
    earliest_death = ifelse(any(owid_new_deaths > 0),
                            as.Date(min(date[owid_new_deaths > 0],
                                        na.rm = TRUE)),
                            NA_real_),
    total_deaths = sum(owid_new_deaths, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  filter(total_deaths >= 1000, !is.na(earliest_death), !is.na(date_diff_std)) %>%
  arrange(desc(completeness), date_diff_std, earliest_death, desc(total_deaths)) %>%
  slice_head(n = 10)


# creating a dataset for each selected country

# brazil

# filtering country
brazil <- preprocessed_covid_multi %>% 
  filter(country == "Brazil") %>%
  select(date, owid_new_deaths)

# finding nonzero index
brazil_first_nonzero_index <- which.max(brazil$owid_new_deaths != 0)

# keeping 0 values only after the first death
brazil <- brazil %>%
  filter(row_number() >= brazil_first_nonzero_index)

# turkey
turkey <- preprocessed_covid_multi %>% 
  filter(country == "Turkey") %>%
  select(date, owid_new_deaths)
turkey_first_nonzero_index <- which.max(turkey$owid_new_deaths != 0)
turkey <- turkey %>%
  filter(row_number() >= turkey_first_nonzero_index)

# russia
russia <- preprocessed_covid_multi %>% 
  filter(country == "Russian Federation") %>%
  select(date, owid_new_deaths)
russia_first_nonzero_index <- which.max(russia$owid_new_deaths != 0)
russia <- russia %>%
  filter(row_number() >= russia_first_nonzero_index)

# united states
us <- preprocessed_covid_multi %>% 
  filter(country == "United States") %>%
  select(date, owid_new_deaths)
us_first_nonzero_index <- which.max(us$owid_new_deaths != 0)
us <- us %>%
  filter(row_number() >= us_first_nonzero_index)

# iran
iran <- preprocessed_covid_multi %>% 
  filter(country == "Iran, Islamic Rep.") %>%
  select(date, owid_new_deaths)
iran_first_nonzero_index <- which.max(iran$owid_new_deaths != 0)
iran <- iran %>%
  filter(row_number() >= iran_first_nonzero_index)

# saudi arabia
saudi <- preprocessed_covid_multi %>% 
  filter(country == "Saudi Arabia") %>%
  select(date, owid_new_deaths)
saudi_first_nonzero_index <- which.max(saudi$owid_new_deaths != 0)
saudi <- saudi %>%
  filter(row_number() >= saudi_first_nonzero_index)

# colombia
colombia <- preprocessed_covid_multi %>% 
  filter(country == "Colombia") %>%
  select(date, owid_new_deaths)
colombia_first_nonzero_index <- which.max(colombia$owid_new_deaths != 0)
colombia <- colombia %>%
  filter(row_number() >= colombia_first_nonzero_index)

# mexico
mexico <- preprocessed_covid_multi %>% 
  filter(country == "Mexico") %>%
  select(date, owid_new_deaths)
mexico_first_nonzero_index <- which.max(mexico$owid_new_deaths != 0)
mexico <- mexico %>%
  filter(row_number() >= mexico_first_nonzero_index)

# bolivia
bolivia <- preprocessed_covid_multi %>% 
  filter(country == "Bolivia") %>%
  select(date, owid_new_deaths)
bolivia_first_nonzero_index <- which.max(bolivia$owid_new_deaths != 0)
bolivia <- bolivia %>%
  filter(row_number() >= bolivia_first_nonzero_index)

# peru
peru <- preprocessed_covid_multi %>% 
  filter(country == "Peru") %>%
  select(date, owid_new_deaths)
peru_first_nonzero_index <- which.max(peru$owid_new_deaths != 0)
peru <- peru %>%
  filter(row_number() >= peru_first_nonzero_index)


## exploring target distribution for each country (bar and plot) ----

brazil_plot <- ggplot(data = brazil, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
brazil_line <- ggplot(data = brazil, aes(x = date, y = owid_new_deaths)) +
  geom_line()

turkey_plot <- ggplot(data = turkey, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
turkey_line <- ggplot(data = turkey, aes(x = date, y = owid_new_deaths)) +
  geom_line()

russia_plot <- ggplot(data = russia, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
russia_line <- ggplot(data = russia, aes(x = date, y = owid_new_deaths)) +
  geom_line()

us_plot <- ggplot(data = us, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
us_line <- ggplot(data = us, aes(x = date, y = owid_new_deaths)) +
  geom_line()

iran_plot <- ggplot(data = iran, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
iran_line <- ggplot(data = iran, aes(x = date, y = owid_new_deaths)) +
  geom_line()

saudi_plot <- ggplot(data = saudi, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
saudi_line <- ggplot(data = saudi, aes(x = date, y = owid_new_deaths)) +
  geom_line()

colombia_plot <- ggplot(data = colombia, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
colombia_line <- ggplot(data = colombia, aes(x = date, y = owid_new_deaths)) +
  geom_line()

mexico_plot <- ggplot(data = mexico, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
mexico_line <- ggplot(data = mexico, aes(x = date, y = owid_new_deaths)) +
  geom_line()

bolivia_plot <- ggplot(data = bolivia, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
bolivia_line <- ggplot(data = bolivia, aes(x = date, y = owid_new_deaths)) +
  geom_line()

peru_plot <- ggplot(data = peru, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
peru_line <- ggplot(data = peru, aes(x = date, y = owid_new_deaths)) +
  geom_line()

# adding titles to bar plots
brazil_plot <- brazil_plot + ggtitle("Brazil")
turkey_plot <- turkey_plot + ggtitle("Turkey")
russia_plot <- russia_plot + ggtitle("Russia")
us_plot <- us_plot + ggtitle("U.S.")
iran_plot <- iran_plot + ggtitle("Iran")
saudi_plot <- saudi_plot + ggtitle("Saudi Arabia")
colombia_plot <- colombia_plot + ggtitle("Colombia")
mexico_plot <- mexico_plot + ggtitle("Mexico")
bolivia_plot <- bolivia_plot + ggtitle("Bolivia")
peru_plot <- peru_plot + ggtitle("Peru")

# adding titles to line plots
brazil_line <- brazil_line + ggtitle("Brazil")
turkey_line <- turkey_line + ggtitle("Turkey")
russia_line <- russia_line + ggtitle("Russia")
us_line <- us_line + ggtitle("U.S.")
iran_line <- iran_line + ggtitle("Iran")
saudi_line <- saudi_line + ggtitle("Saudi Arabia")
colombia_line <- colombia_line + ggtitle("Colombia")
mexico_line <- mexico_line + ggtitle("Mexico")
bolivia_line <- bolivia_line + ggtitle("Bolivia")
peru_line <- peru_line + ggtitle("Peru")

# combining bar plots
bar_plots_combined <- brazil_plot + turkey_plot + russia_plot + us_plot + iran_plot + 
  saudi_plot + colombia_plot + mexico_plot + bolivia_plot + peru_plot +
  plot_layout(ncol = 2)

# combining line plots
line_plots_combined <- brazil_line + turkey_line + russia_line + us_line + iran_line + 
  saudi_line + colombia_line + mexico_line + bolivia_line + peru_line +
  plot_layout(ncol = 2)


## assessing seasonality, i ---

# assuming weekly seasonality (as observed below in ACF graphs)
brazil_ts_data <- ts(brazil$owid_new_deaths, frequency = 7)
turkey_ts_data <- ts(turkey$owid_new_deaths, frequency = 7)
russia_ts_data <- ts(russia$owid_new_deaths, frequency = 7)
us_ts_data <- ts(us$owid_new_deaths, frequency = 7)
iran_ts_data <- ts(iran$owid_new_deaths, frequency = 7)
saudi_ts_data <- ts(saudi$owid_new_deaths, frequency = 7)
colombia_ts_data <- ts(colombia$owid_new_deaths, frequency = 7)
mexico_ts_data <- ts(mexico$owid_new_deaths, frequency = 7)
bolivia_ts_data <- ts(bolivia$owid_new_deaths, frequency = 7)
peru_ts_data <- ts(peru$owid_new_deaths, frequency = 7)

# using isSeasonal()
seastests::isSeasonal(brazil_ts_data)
seastests::isSeasonal(turkey_ts_data) # FALSE
seastests::isSeasonal(russia_ts_data)
seastests::isSeasonal(us_ts_data)
seastests::isSeasonal(iran_ts_data) # FALSE
seastests::isSeasonal(saudi_ts_data) # FALSE
seastests::isSeasonal(colombia_ts_data) # FALSE
seastests::isSeasonal(mexico_ts_data)
seastests::isSeasonal(bolivia_ts_data) # FALSE
seastests::isSeasonal(peru_ts_data) # FALSE

# multiplicative decomposition (increasing variance over time)
brazil_decomposed_data <- decompose(brazil_ts_data, type = "multiplicative")
turkey_decomposed_data <- decompose(turkey_ts_data, type = "multiplicative")
russia_decomposed_data <- decompose(russia_ts_data, type = "multiplicative")
us_decomposed_data <- decompose(us_ts_data, type = "multiplicative")
iran_decomposed_data <- decompose(iran_ts_data, type = "multiplicative")
saudi_decomposed_data <- decompose(saudi_ts_data, type = "multiplicative")
colombia_decomposed_data <- decompose(colombia_ts_data, type = "multiplicative")
mexico_decomposed_data <- decompose(mexico_ts_data, type = "multiplicative")
bolivia_decomposed_data <- decompose(bolivia_ts_data, type = "multiplicative")
peru_decomposed_data <- decompose(peru_ts_data, type = "multiplicative")

# plotting decomposed data
plot(brazil_decomposed_data)
plot(turkey_decomposed_data)
plot(russia_decomposed_data)
plot(us_decomposed_data)
plot(iran_decomposed_data)
plot(saudi_decomposed_data)
plot(colombia_decomposed_data)
plot(mexico_decomposed_data)
plot(bolivia_decomposed_data)
plot(peru_decomposed_data)

# ACF and PACF assessment

# specifying countries
countries <- c("Brazil", "Turkey", "Russian Federation", "United States","Iran, Islamic Rep.",
               "Saudi Arabia", "Colombia", "Mexico", "Bolivia", "Peru")

# creating a function to generate ACF and PACF visualizations
generate_acf_pacf_plots <- function(country_name) {
  country_data <- preprocessed_covid_multi %>%
    filter(country == country_name, owid_new_deaths > 0) %>%
    arrange(date) %>%
    select(date, owid_new_deaths)
  if (nrow(country_data) < 2) {
    cat("Insufficient data for", country_name, "\n")
    return(NULL)
  }
  acf_plot <- acf(country_data$owid_new_deaths, main = paste(country_name), ylim = c(-1,1), mar = c(5, 4, 4, 2) + 0.1)
  pacf_plot <- pacf(country_data$owid_new_deaths, main = paste(country_name), ylim = c(-1,1), mar = c(5, 4, 4, 2) + 0.1)
  return(list(acf_plot = acf_plot, pacf_plot = pacf_plot, country_name = country_name))
}

# generating plots for each country
plots_list <- lapply(countries, generate_acf_pacf_plots)

# determining layout and margins
par(mfrow = c(5, 4), mar = c(5, 2, 2, 2) + 0.1)

# determining plot preferences (ISSUE HERE)
for (i in 1:length(plots_list)) {
  if (!is.null(plots_list[[i]])) {
    country_data <- preprocessed_covid_multi %>%
      filter(country == plots_list[[i]]$country_name, owid_new_deaths > 0) %>%
      arrange(date) %>%
      select(date, owid_new_deaths)
    n <- length(country_data$owid_new_deaths)
    se <- 1/sqrt(n)
    plot(plots_list[[i]]$acf_plot$acf, type = "h", main = "", xlab = "Lag", ylab = "ACF",
         ylim = plots_list[[i]]$acf_plot$ylim, xlim = plots_list[[i]]$acf_plot$xlim)
    title(main = paste(plots_list[[i]]$country_name), line = 1, cex.main = 0.8)
    abline(h = 0, col = "blue")
    abline(h = c(se, -se), col = "red", lty = 2)
    plot(plots_list[[i]]$pacf_plot$acf, type = "h", main = "", xlab = "Lag", ylab = "PACF",
         ylim = plots_list[[i]]$pacf_plot$ylim, xlim = plots_list[[i]]$pacf_plot$xlim)
    title(main = paste(plots_list[[i]]$country_name), line = 1, cex.main = 0.8)
    abline(h = 0, col = "blue")
    abline(h = c(se, -se), col = "red", lty = 2)
  }
}


## assessing stationarity ----

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

# differencing countries manually

brazil <- brazil %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()

turkey <- turkey %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()

russia <- russia %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()

us <- us %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()

iran <- iran %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>%
  na.omit()

saudi <- saudi %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()

colombia <- colombia %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()

mexico <- mexico %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()

peru <- peru %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()

bolivia <- bolivia %>%
  mutate(owid_new_deaths = c(NA, diff(owid_new_deaths, differences = 1))) %>% 
  na.omit()



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

# performing imputation using Kalman smoothing to estimate missing values
preprocessed_covid_multi_imputed <- na_kalman(preprocessed_covid_multi)



## creating lagged variable ----

preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>%
  mutate(lagged_nd_7 = dplyr::lag(owid_new_deaths, n=7))


## temporal features ----

# adding date features
preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>%
  mutate(month = month(date),
         day = mday(date),
         weekday = weekdays(date, abbreviate = FALSE))

# cyclical encoding
preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>%
  mutate(cyclical_month_sin = sin(2 * pi * month / 12),
         cyclical_month_cos = cos(2 * pi * month / 12),
         cyclical_weekday_sin = sin(2 * pi * month / 7),
         cyclical_weekday_cos = cos(2 * pi * month / 7),
         cyclical_dayofmth_sin = sin(2 * pi * month / 31),
         cyclical_dayofmth_cos = cos(2 * pi * month / 31))

# plotting weekday and target variable
ggplot(preprocessed_covid_multi_imputed, mapping = aes(x = weekday, y = owid_new_deaths))+
  geom_boxplot()

## assessing weekday ----

# grouping
deaths_by_weekday <- preprocessed_covid_multi_imputed %>%
  group_by(weekday) %>%
  summarize(total_deaths = sum(owid_new_deaths))

# total deaths calculation and rounding to whole number
total_deaths <- round(sum(deaths_by_weekday$total_deaths))

# rounding proportion to three decimal points and displaying table
deaths_by_weekday <- deaths_by_weekday %>%
  mutate(proportion = round(total_deaths / sum(total_deaths), 3),
         total_deaths = as.integer(total_deaths)) %>% 
  DT::datatable()

# testing significance

# creating table of observed frequencies
observed <- table(preprocessed_covid_multi_imputed$weekday)

# performing chi-squared test
chi_squared_test <- chisq.test(observed)

# printing
print(chi_squared_test)


## (preliminary) general correlation matrix ----

# filter out numerical data
numerical_data <- preprocessed_covid_multi_imputed %>% select_if(is.numeric)

# create a correlation matrix
correlation_matrix <- cor(numerical_data, use = "complete.obs")

# establish threshold to reduce dimensions
# (drop if the absolute value of correlation coefficient is under 0.5)
correlation_matrix[abs(correlation_matrix) < 0.5] <- NA

# adapt to ggplot2
# (convert wide-format data to long-format data)
melted_corr_matrix <- melt(correlation_matrix, na.rm = TRUE)

# produce heatmap
correlation_graph_i <- ggplot(melted_corr_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = '', title = 'Correlation Matrix Heatmap - General')


## custom features ----

# creating custom domain knowledge features
preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>%
  mutate(cases_per_population_cf = owid_new_cases / owid_population,
         deaths_per_population_cf = owid_new_deaths / owid_population,
         policy_response_impact_cf = ((ox_c1_school_closing + ox_c2_workplace_closing + ox_c4_restrictions_on_gatherings +
                                        ox_c6_stay_at_home_requirements + ox_c7_restrictions_on_internal_movement) / 5),
         vulnerability_index_cf = ((owid_diabetes_prevalence + owid_male_smokers)/2)*ox_containment_health_index)

cf_preprocessed_multi <- preprocessed_covid_multi_imputed %>% 
  select(cases_per_population_cf, deaths_per_population_cf, policy_response_impact_cf, vulnerability_index_cf)


## (updated) general correlation matrix ----

# filter out numerical data
numerical_data <- preprocessed_covid_multi_imputed %>% select_if(is.numeric)

# create a correlation matrix
correlation_matrix <- cor(numerical_data, use = "complete.obs")

# establish threshold to reduce dimensions
# (drop if the absolute value of correlation coefficient is under 0.5)
correlation_matrix[abs(correlation_matrix) < 0.5] <- NA

# adapt to ggplot2
# (convert wide-format data to long-format data)
melted_corr_matrix <- melt(correlation_matrix, na.rm = TRUE)

# produce heatmap
correlation_graph_ii <- ggplot(melted_corr_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = '', title = 'Correlation Matrix Heatmap - General')


## specific correlation matrices ----

# cases and outcomes

# filter out numerical data
cases_and_outcomes <- preprocessed_covid_multi_imputed %>% select_if(is.numeric) %>% 
  select(owid_new_deaths, averaged_confirmed_cases, owid_new_cases, owid_population, owid_male_smokers)

# create a correlation matrix
co_correlation_matrix <- cor(cases_and_outcomes, use = "complete.obs")

# adapt to ggplot2
# (convert wide-format data to long-format data)
co_melted_corr_matrix <- melt(co_correlation_matrix, na.rm = TRUE)

# produce heatmap
co_correlation_graph <- ggplot(co_melted_corr_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = '', title = 'Correlation Matrix Heatmap - Cases and Outcomes (+ Demographics)')

# policy measures

# filter out numerical data
policy_measures <- preprocessed_covid_multi_imputed %>% select_if(is.numeric) %>% 
  select(owid_new_deaths, ox_c1_school_closing, ox_c1_flag, ox_c2_workplace_closing, ox_c2_flag,
         ox_c3_flag, ox_c4_restrictions_on_gatherings, ox_c4_flag, ox_c6_stay_at_home_requirements,
         ox_c7_restrictions_on_internal_movement, ox_h1_flag,
         ox_containment_health_index)

# create a correlation matrix
pm_correlation_matrix <- cor(policy_measures, use = "complete.obs")

# adapt to ggplot2
# (convert wide-format data to long-format data)
pm_melted_corr_matrix <- melt(pm_correlation_matrix, na.rm = TRUE)

# produce heatmap
pm_correlation_graph <- ggplot(pm_melted_corr_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = '', title = 'Correlation Matrix Heatmap - Policy Measures')

# custom features

# filter out numerical data
custom_features <- preprocessed_covid_multi_imputed %>% select_if(is.numeric) %>% 
  select(owid_new_deaths, cases_per_population_cf, deaths_per_population_cf, policy_response_impact_cf, vulnerability_index_cf)

# create a correlation matrix
cf_correlation_matrix <- cor(custom_features, use = "complete.obs")

# adapt to ggplot2
# (convert wide-format data to long-format data)
cf_melted_corr_matrix <- melt(cf_correlation_matrix, na.rm = TRUE)

# produce heatmap
cf_correlation_graph <- ggplot(cf_melted_corr_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = '', title = 'Correlation Matrix Heatmap - Custom Features')

## custom feature relationships ----

# correlation between custom features and target variable
custom_features_target_corr <- preprocessed_covid_multi_imputed %>%
  select(owid_new_deaths, ends_with("_cf")) %>%
  cor(use = "complete.obs") %>%
  as.data.frame() %>%
  tibble::rownames_to_column("feature") %>% 
  select(feature, owid_new_deaths) %>% 
  arrange(desc(owid_new_deaths)) %>%
  DT::datatable()

# defining plots

plot1 <- ggplot(preprocessed_covid_multi_imputed, aes(x = cases_per_population_cf, y = owid_new_deaths)) +
  geom_jitter(alpha = 0.5, width = 0.02, height = 0.02) +
  geom_smooth(method = "lm", formula = y ~ exp(x), se = FALSE, color = "red") +
  scale_x_continuous(trans = 'log10', labels = scales::scientific) +
  labs(x = "Cases per Population", y = "New Deaths")

plot2 <- ggplot(preprocessed_covid_multi_imputed, aes(x = deaths_per_population_cf, y = owid_new_deaths)) +
  geom_jitter(alpha = 0.5, width = 0.3, height = 0) +
  geom_smooth(method = "lm", formula = y ~ exp(x), se = FALSE, color = "red") +
  scale_x_log10() +
  labs(x = "Deaths per Population (log scale)", y = "New Deaths")

plot3 <- ggplot(preprocessed_covid_multi_imputed, aes(x = policy_response_impact_cf, y = owid_new_deaths)) +
  geom_jitter(alpha = 0.5, width = 0.1, height = 0) +
  geom_smooth(method = "lm", formula = y ~ exp(x), se = FALSE, color = "red") +
  labs(x = "Policy Response Impact", y = "New Deaths")

plot4 <- ggplot(preprocessed_covid_multi_imputed, aes(x = vulnerability_index_cf, y = owid_new_deaths)) +
  geom_jitter(alpha = 0.5, width = 0.1, height = 0) +
  geom_smooth(method = "lm", formula = y ~ exp(x), se = FALSE, color = "red") +
  labs(x = "Vulnerability Index", y = "New Deaths")

# arranging plots in a single visualization
combined_plot_ii <- grid.arrange(plot1, plot2, plot3, plot4, ncol = 1)


<<<<<<< HEAD
## feature selection ----

preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>% 
  select(-lagged_nd_7, -deaths_per_population_cf, -averaged_confirmed_cases)


# ## feature selection ----
# 
# preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>% 
#   select(-lagged_nd_7, -deaths_per_population_cf, -averaged_confirmed_cases)
#
# # training a random forest model
# rf_model <- randomForest(owid_new_deaths ~ ., data = preprocessed_covid_multi_imputed, importance = TRUE, na.action = na.omit)
# 
# # computing importance scores
# importance_scores <- importance(rf_model)
# 
# # converting to data frame
# importance_df <- as.data.frame(importance_scores)
# 
# # make sure there are no duplicate names
# names(importance_df) <- make.unique(names(importance_df))
# 
# # incorporating variable name column
# importance_df$Variable <- rownames(importance_df)
# 
# # melting to long format
# importance_long <- melt(importance_df, id.vars = "Variable")
# 
# # producing plot
# importance <- ggplot(importance_long, aes(x = reorder(Variable, value), y = value)) +
#   geom_bar(stat = "identity") +
#   coord_flip() +  # Flip the axes to make the plot horizontal
#   theme_minimal() +
#   labs(x = "Feature", y = "Importance", title = "Feature Importance from Random Forest Model") +
#   theme(plot.title = element_text(hjust = 0.5))


# final dimensions ----

# univariate (2256 x 2)
observations_table <- data.frame(
  Country = c("Brazil", "Turkey", "Russian Federation", "U.S.", "Iran, Islamic Rep.", "Saudi Arabia", "Colombia", "Mexico", "Bolivia", "Peru"),
  Observations = c(nrow(brazil), nrow(turkey), nrow(russia), nrow(us), nrow(iran), nrow(saudi), nrow(colombia), nrow(mexico), nrow(bolivia), nrow(peru))
) %>% 
  DT::datatable()

# multivariate (48212 x 30)
dim(preprocessed_covid_multi_imputed)



# saving files ----
save(preprocessed_covid_multi_imputed, file = "data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")
save(brazil, file = "data/preprocessed/univariate/not_split/univariate_brazil.rda")
save(turkey, file = "data/preprocessed/univariate/not_split/univariate_turkey.rda")
save(russia, file = "data/preprocessed/univariate/not_split/univariate_russia.rda")
save(us, file = "data/preprocessed/univariate/not_split/univariate_us.rda")
save(iran, file = "data/preprocessed/univariate/not_split/univariate_iran.rda")
save(saudi, file = "data/preprocessed/univariate/not_split/univariate_saudi.rda")
save(colombia, file = "data/preprocessed/univariate/not_split/univariate_colombia.rda")
save(mexico, file = "data/preprocessed/univariate/not_split/univariate_mexico.rda")
save(bolivia, file = "data/preprocessed/univariate/not_split/univariate_bolivia.rda")
save(peru, file = "data/preprocessed/univariate/not_split/univariate_peru.rda")
# save(importance, file = "visuals/importance.rda")