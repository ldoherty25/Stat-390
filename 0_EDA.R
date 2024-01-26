## EDA


# load packages ----
library(tidyverse)
library(tidymodels)
library(reshape2)
library(lubridate)
library(forecast)
library(modelr)

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


## basic inspection ----

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

# selecting only non-redundant variables
preprocessed_covid_multi <- covid_cleaned %>%
  select(country, date, average_confirmed, average_cummulative_deaths,
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


## correlation matrix ----

# filter out numerical data
numerical_data <- preprocessed_covid_multi %>% select_if(is.numeric)

# create a correlation matrix
correlation_matrix <- cor(numerical_data, use = "complete.obs")

# establish threshold to reduce dimensions
correlation_matrix[abs(correlation_matrix) < 0.5] <- NA

# adapt to ggplot2
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


## univariate models preprocessing ----

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
  select(date, owid_new_cases)

japan <- preprocessed_covid_multi %>%
  filter(country == "Japan", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)

france <- preprocessed_covid_multi %>%
  filter(country == "France", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)

iran <- preprocessed_covid_multi %>%
  filter(country == "Iran, Islamic Rep.", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)

italy <- preprocessed_covid_multi %>%
  filter(country == "Italy", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)

us <- preprocessed_covid_multi %>%
  filter(country == "United States", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)

switzerland <- preprocessed_covid_multi %>%
  filter(country == "Switzerland", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)

uk <- preprocessed_covid_multi %>%
  filter(country == "United Kingdom", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)

netherlands <- preprocessed_covid_multi %>%
  filter(country == "Netherlands", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)

germany <- preprocessed_covid_multi %>%
  filter(country == "Germany", owid_new_deaths > 0) %>% 
  select(date, owid_new_cases)


## constructing ACF and PACF visualizations for selected countries (aggregated) ----

# defining a list of selected countries
selected_countries <- c("China", "Japan", "France", "Iran, Islamic Rep.", 
                        "Italy", "United States", "Switzerland", 
                        "United Kingdom", "Netherlands", "Germany")

# creating a grouped dataset summing each day's new deaths
uni_grouped_covid <- preprocessed_covid_multi %>%
  filter(country %in% selected_countries) %>%
  group_by(date) %>%
  summarize(total_new_deaths = sum(owid_new_deaths, na.rm = TRUE)) %>%
  ungroup() %>%
  complete(date = seq(min(date), max(date), by = "day"), fill = list(total_new_deaths = 0))

# calculating ACF and PACF
acf_vals <- acf(uni_grouped_covid$total_new_deaths, plot = FALSE)
pacf_vals <- pacf(uni_grouped_covid$total_new_deaths, plot = FALSE)

# calculating bounds based on non-NA values
n <- sum(!is.na(uni_grouped_covid$total_new_deaths))
se <- 1 / sqrt(n)

# producing the desired plots

acf_plot <- ggplot(data.frame(Lag = 1:(length(acf_vals$acf)-1), ACF = acf_vals$acf[-1]), aes(x = Lag, y = ACF)) +
  geom_bar(stat = "identity", fill = "grey") +
  geom_hline(yintercept = 0) +
  geom_hline(yintercept = c(1.96, -1.96) * se, color = "blue") +
  theme_minimal() +
  labs(title = "Autocorrelation Function (ACF)", x = "Lags", y = "ACF")

pacf_plot <- ggplot(data.frame(Lag = 1:(length(pacf_vals$acf)-1), PACF = pacf_vals$acf[-1]), aes(x = Lag, y = PACF)) +
  geom_bar(stat = "identity", fill="grey") +
  geom_hline(yintercept = 0) +
  geom_hline(yintercept = c(1.96, -1.96) * se, color = "blue") +
  theme_minimal() +
  labs(title = "Partial Autocorrelation Function (PACF)", x = "Lags", y = "PACF")


## data decomposition ----

# generating plot components
time_series <- ts(uni_grouped_covid$total_new_deaths)
trend <- ma(time_series, order = 14)

# calculating residuals
residuals <- time_series - trend

# joining plot components (needs saving)
par(mfrow = c(3, 1))
plot(time_series, main = "Observed", xlab = "", ylab = "New Deaths", col = "black", xaxt='n')
plot(trend, main = "Trend", xlab = "", ylab = "Trend", col = "blue", xaxt='n')
plot(residuals, main = "Residuals", xlab = "Time", ylab = "Residuals", col = "red")
par(mfrow = c(1, 1))


## save files ----
save(preprocessed_covid_multi, file = "data/preprocessed/multivariate/preprocessed_covid_multi.rda")
save(missing_prop_covid, file = "visuals/missing_prop_covid.rda")
save(correlation_graph, file = "visuals/correlation_graph.rda")
save(missing_graph, file = "visuals/missing_graph.rda")
save(tv_distribution_log, file = "visuals/tv_distribution_log.rda")
save(china, file = "data/preprocessed/univariate/china.rda")
save(japan, file = "data/preprocessed/univariate/japan.rda")
save(france, file = "data/preprocessed/univariate/france.rda")
save(iran, file = "data/preprocessed/univariate/iran.rda")
save(italy, file = "data/preprocessed/univariate/italy.rda")
save(us, file = "data/preprocessed/univariate/us.rda")
save(switzerland, file = "data/preprocessed/univariate/switzerland.rda")
save(uk, file = "data/preprocessed/univariate/uk.rda")
save(netherlands, file = "data/preprocessed/univariate/netherlands.rda")
save(germany, file = "data/preprocessed/univariate/germany.rda")
save(acf_plot, file = "visuals/acf_plot.rda")
save(pacf_plot, file = "visuals/pacf_plot.rda")