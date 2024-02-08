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

china_check <- preprocessed_covid_multi %>%
  filter(country == "China") %>% 
  select(date, owid_new_deaths)
china_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", 
                      "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", 
                      "2020-01-10")
china <- china_check %>%
  filter(!(date %in% china_first_zero_dates))

japan_check <- preprocessed_covid_multi %>%
  filter(country == "Japan") %>% 
  select(date, owid_new_deaths)
japan_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", 
                            "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", 
                            "2020-01-10", "2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14",
                            "2020-01-15", "2020-01-16", "2020-01-17", "2020-01-18", "2020-01-19", 
                            "2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24", 
                            "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", 
                            "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", 
                            "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08",
                            "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12")

japan <- japan_check %>%
  filter(!(date %in% japan_first_zero_dates))

france_check <- preprocessed_covid_multi %>%
  filter(country == "France") %>% 
  select(date, owid_new_deaths)
france_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", 
                             "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10",
                             "2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16",
                             "2020-01-17", "2020-01-18", "2020-01-19", "2020-01-20", "2020-01-21", "2020-01-22",
                             "2020-01-23", "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28",
                             "2020-01-29", "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03",
                             "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09",
                             "2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14")

france <- france_check %>%
  filter(!(date %in% france_first_zero_dates))

iran_check <- preprocessed_covid_multi %>%
  filter(country == "Iran, Islamic Rep.") %>% 
  select(date, owid_new_deaths)
iran_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
                           "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11", 
                           "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17", 
                           "2020-01-18", "2020-01-19", "2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23",
                           "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29",
                           "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04",
                           "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10", 
                           "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16", 
                           "2020-02-17", "2020-02-18", "2020-02-19")
iran <- iran_check %>%
  filter(!(date %in% iran_first_zero_dates))

italy_check <- preprocessed_covid_multi %>%
  filter(country == "Italy") %>% 
  select(date, owid_new_deaths)
italy_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
                            "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11",
                            "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17",
                            "2020-01-18", "2020-01-19", "2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23",
                            "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29",
                            "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04",
                            "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10",
                            "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16",
                            "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22")
italy <- italy_check %>%
  filter(!(date %in% italy_first_zero_dates))

us_check <- preprocessed_covid_multi %>%
  filter(country == "United States") %>% 
  select(date, owid_new_deaths)
us_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
                         "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11",
                         "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17",
                         "2020-01-18", "2020-01-19", "2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23",
                         "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29",
                         "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04",
                         "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10",
                         "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16",
                         "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22",
                         "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28",
                         "2020-02-29")
us <- us_check %>%
  filter(!(date %in% us_first_zero_dates))

switzerland_check <- preprocessed_covid_multi %>%
  filter(country == "Switzerland") %>% 
  select(date, owid_new_deaths)
swiss_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
                            "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11",
                            "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17",
                            "2020-01-18", "2020-01-19", "2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23",
                            "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29",
                            "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04",
                            "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10",
                            "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16",
                            "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22",
                            "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28",
                            "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05")
switzerland <- switzerland_check %>%
  filter(!(date %in% swiss_first_zero_dates))
  

uk_check <- preprocessed_covid_multi %>%
  filter(country == "United Kingdom") %>% 
  select(date, owid_new_deaths)
uk_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
                         "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11",
                         "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17",
                         "2020-01-18", "2020-01-19", "2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", 
                         "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29", 
                         "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04",
                         "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10",
                         "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16",
                         "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22",
                         "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28",
                         "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05", 
                         "2020-03-06")
uk <- uk_check %>%
  filter(!(date %in% uk_first_zero_dates))

netherlands_check <- preprocessed_covid_multi %>%
  filter(country == "Netherlands") %>% 
  select(date, owid_new_deaths)
netherlands_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", 
                                  "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09",
                                  "2020-01-10", "2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14",
                                  "2020-01-15", "2020-01-16", "2020-01-17", "2020-01-18", "2020-01-19",
                                  "2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23", "2020-01-24",
                                  "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29",
                                  "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03",
                                  "2020-02-04", "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08",
                                  "2020-02-09", "2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13",
                                  "2020-02-14", "2020-02-15", "2020-02-16", "2020-02-17", "2020-02-18",
                                  "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22", "2020-02-23",
                                  "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28",
                                  "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04",
                                  "2020-03-05", "2020-03-06")
netherlands <- netherlands_check %>%
  filter(!(date %in% netherlands_first_zero_dates))

germany_check <- preprocessed_covid_multi %>%
  filter(country == "Germany") %>% 
  select(date, owid_new_deaths)
germany_first_zero_dates <- c("2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
                              "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-11",
                              "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17",
                              "2020-01-18", "2020-01-19", "2020-01-20", "2020-01-21", "2020-01-22", "2020-01-23",
                              "2020-01-24", "2020-01-25", "2020-01-26", "2020-01-27", "2020-01-28", "2020-01-29",
                              "2020-01-30", "2020-01-31", "2020-02-01", "2020-02-02", "2020-02-03", "2020-02-04",
                              "2020-02-05", "2020-02-06", "2020-02-07", "2020-02-08", "2020-02-09", "2020-02-10", 
                              "2020-02-11", "2020-02-12", "2020-02-13", "2020-02-14", "2020-02-15", "2020-02-16",
                              "2020-02-17", "2020-02-18", "2020-02-19", "2020-02-20", "2020-02-21", "2020-02-22",
                              "2020-02-23", "2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27", "2020-02-28", 
                              "2020-02-29", "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05",
                              "2020-03-06", "2020-03-07", "2020-03-08", "2020-03-09")
germany <- germany_check %>%
  filter(!(date %in% germany_first_zero_dates))

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

#multivariate
preprocessed_covid_multi_lag <- preprocessed_covid_multi_imputed %>%
  mutate(lagged_nd_1 = dplyr::lag(owid_new_deaths, n=1),
         lagged_nd_2 = dplyr::lag(owid_new_deaths, n=2),
         lagged_nd_7 = dplyr::lag(owid_new_deaths, n=7))
  
#multivariate
holidays <- as.Date(c("2020-01-01", "2023-04-12", "2020-12-24", "2020-12-25", "2020-05-23", "2020-05-04", "2020-07-30",
                      "2020-07-31", "2020-11-14", "2020-02-14", "2020-05-05", "2020-12-10", "2020-12-18",
                      "2020-10-31", "2020-12-21", "2020-11-01", "2020-11-02", "2020-11-26", "2020-03-19"))
  
multi_time_eng <- preprocessed_covid_multi_imputed %>%
  mutate(month = month(date),
         day = mday(date),
         weekday = weekdays(date, abbreviate = FALSE),
         season = case_when(
           month %in% c(3, 4, 5) ~ "Spring",
           month %in% c(6, 7, 8) ~ "Summer",
           month %in% c(9, 10, 11) ~ "Fall",
           TRUE ~ "Winter"),
         IsHoliday = date %in% holidays)

#plotting season and new_deaths
ggplot(multi_time_eng, mapping = aes(x = season, y = owid_new_deaths))+
  geom_boxplot()
ggplot(multi_time_eng, mapping = aes(x = season, y = owid_new_deaths))+
  geom_violin()
ggplot(multi_time_eng, mapping = aes(x = season))+
  geom_bar() +
  labs(title = "Seasonal Count Plot")
ggplot(multi_time_eng, mapping = aes(x = owid_new_deaths))+
  geom_histogram()+
  facet_wrap(~season)

#plotting weekday and new_deaths
ggplot(multi_time_eng, mapping = aes(x = weekday, y = owid_new_deaths))+
  geom_boxplot()
ggplot(multi_time_eng, mapping = aes(x = weekday, y = owid_new_deaths))+
  geom_violin()
ggplot(multi_time_eng, mapping = aes(x = owid_new_deaths))+
  geom_histogram()+
  facet_wrap(~weekday)

#plotting IsHoliday with new deaths
ggplot(multi_time_eng, mapping = aes(x = IsHoliday, y = owid_new_deaths))+
  geom_boxplot()
ggplot(multi_time_eng, mapping = aes(x = IsHoliday, y = owid_new_deaths))+
  geom_violin()
ggplot(multi_time_eng, mapping = aes(x = owid_new_deaths))+
  geom_histogram()+
  facet_wrap(~IsHoliday)
ggplot(multi_time_eng, mapping = aes(x = IsHoliday))+
  geom_bar() +
  labs(title = "Holiday Count Plot")

## custom features ----

preprocessed_covid_multi_imputed <- preprocessed_covid_multi_imputed %>%
  mutate(capacity_to_case = owid_hospital_beds_per_thousand / rollmean(average_confirmed, 7, fill = NA, align = 'right')) %>% 
  mutate(vulnerability_index = (owid_aged_65_older + owid_aged_70_older + owid_diabetes_prevalence + owid_cardiovasc_death_rate) / 4) %>% 
  mutate(policy_stringency_index = (ox_c1_school_closing + ox_c2_workplace_closing + ox_c3_cancel_public_events + ox_c4_restrictions_on_gatherings + ox_c6_stay_at_home_requirements + ox_c7_restrictions_on_internal_movement + ox_c8_international_travel_controls) / 7,
         policy_population_index = policy_stringency_index * owid_population_density) %>% 
  mutate_if(is.numeric, ~replace(., is.infinite(.) | is.nan(.), NA))

### COMMENTED OUT UNI_GROUPED UNTIL HERE

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