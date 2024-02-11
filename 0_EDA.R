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

# finding variables with more than 50% missingness
columns_to_remove <- prop_non_missing %>% 
  filter(prop_non_missing < 0.4) %>% 
  pull(variable)

# removing unwanted columns
covid_cleaned <- mutated_covid %>% 
  select(-all_of(columns_to_remove))

# selecting only non-redundant variables; new: removed averaged_cummulative_deaths
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


## exploring target distribution for each country (bar and plot) ----

china_plot <- ggplot(data = china, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
china_line <- ggplot(data = china, aes(x = date, y = owid_new_deaths)) +
  geom_line()

japan_plot <- ggplot(data = japan, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
japan_line <- ggplot(data = japan, aes(x = date, y = owid_new_deaths)) +
  geom_line()

france_plot <- ggplot(data = france, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
france_line <- ggplot(data = france, aes(x = date, y = owid_new_deaths)) +
  geom_line()

iran_plot <- ggplot(data = iran, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
iran_line <- ggplot(data = iran, aes(x = date, y = owid_new_deaths)) +
  geom_line()

italy_plot <- ggplot(data = italy, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
italy_line <- ggplot(data = italy, aes(x = date, y = owid_new_deaths)) +
  geom_line()

us_plot <- ggplot(data = us, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
us_line <- ggplot(data = us, aes(x = date, y = owid_new_deaths)) +
  geom_line()

switzerland_plot <- ggplot(data = switzerland, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
switzerland_line <- ggplot(data = switzerland, aes(x = date, y = owid_new_deaths)) +
  geom_line()

uk_plot <- ggplot(data = uk, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
uk_line <- ggplot(data = uk, aes(x = date, y = owid_new_deaths)) +
  geom_line()

netherlands_plot <- ggplot(data = netherlands, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
netherlands_line <- ggplot(data = netherlands, aes(x = date, y = owid_new_deaths)) +
  geom_line()

germany_plot <- ggplot(data = germany, aes(x = date, y = owid_new_deaths)) +
  geom_bar(stat = "identity")
germany_line <- ggplot(data = germany, aes(x = date, y = owid_new_deaths)) +
  geom_line()

# adding titles to bar plots
china_plot <- china_plot + ggtitle("China")
japan_plot <- japan_plot + ggtitle("Japan")
france_plot <- france_plot + ggtitle("France")
iran_plot <- iran_plot + ggtitle("Iran")
italy_plot <- italy_plot + ggtitle("Italy")
us_plot <- us_plot + ggtitle("U.S.")
switzerland_plot <- switzerland_plot + ggtitle("Switzerland")
uk_plot <- uk_plot + ggtitle("U.K.")
netherlands_plot <- netherlands_plot + ggtitle("Netherlands")
germany_plot <- germany_plot + ggtitle("Germany")

# adding titles to line plots
china_line <- china_line + ggtitle("China")
japan_line <- japan_line + ggtitle("Japan")
france_line <- france_line + ggtitle("France")
iran_line <- iran_line + ggtitle("Iran")
italy_line <- italy_line + ggtitle("Italy")
us_line <- us_line + ggtitle("U.S.")
switzerland_line <- switzerland_line + ggtitle("Switzerland")
uk_line <- uk_line + ggtitle("U.K.")
netherlands_line <- netherlands_plot + ggtitle("Netherlands")
germany_line <- germany_line + ggtitle("Germany")

# combining bar plots
bar_plots_combined <- china_plot + japan_plot + france_plot + iran_plot + italy_plot + 
  us_plot + switzerland_plot + uk_plot + netherlands_plot + germany_plot +
  plot_layout(ncol = 2)

# combining line plots
line_plots_combined <- china_line + japan_line + france_line + iran_line + italy_line + 
  us_line + switzerland_line + uk_line + netherlands_line + germany_line +
  plot_layout(ncol = 2)


## assessing seasonality, i ---

# assuming weekly seasonality (as observed below in ACF graphs)
china_ts_data <- ts(china$owid_new_deaths, frequency = 7)
japan_ts_data <- ts(japan$owid_new_deaths, frequency = 7)
france_ts_data <- ts(france$owid_new_deaths, frequency = 7)
iran_ts_data <- ts(iran$owid_new_deaths, frequency = 7)
italy_ts_data <- ts(italy$owid_new_deaths, frequency = 7)
us_ts_data <- ts(us$owid_new_deaths, frequency = 7)
switzerland_ts_data <- ts(switzerland$owid_new_deaths, frequency = 7)
uk_ts_data <- ts(uk$owid_new_deaths, frequency = 7)
netherlands_ts_data <- ts(netherlands$owid_new_deaths, frequency = 7)
germany_ts_data <- ts(germany$owid_new_deaths, frequency = 7)

# using isSeasonal()
seastests::isSeasonal(china_ts_data) # FALSE
seastests::isSeasonal(japan_ts_data)
seastests::isSeasonal(france_ts_data)
seastests::isSeasonal(iran_ts_data) # FALSE
seastests::isSeasonal(italy_ts_data)
seastests::isSeasonal(us_ts_data)
seastests::isSeasonal(switzerland_ts_data) # FALSE
seastests::isSeasonal(uk_ts_data)
seastests::isSeasonal(netherlands_ts_data)
seastests::isSeasonal(germany_ts_data)

# multiplicative decomposition (increasing variance over time)
china_decomposed_data <- decompose(china_ts_data, type = "multiplicative")
japan_decomposed_data <- decompose(japan_ts_data, type = "multiplicative")
france_decomposed_data <- decompose(france_ts_data, type = "multiplicative")
iran_decomposed_data <- decompose(iran_ts_data, type = "multiplicative")
italy_decomposed_data <- decompose(italy_ts_data, type = "multiplicative")
us_decomposed_data <- decompose(us_ts_data, type = "multiplicative")
switzerland_decomposed_data <- decompose(switzerland_ts_data, type = "multiplicative")
uk_decomposed_data <- decompose(uk_ts_data, type = "multiplicative")
netherlands_decomposed_data <- decompose(netherlands_ts_data, type = "multiplicative")
germany_decomposed_data <- decompose(germany_ts_data, type = "multiplicative")

# plotting decomposed data
plot(china_decomposed_data)
plot(japan_decomposed_data)
plot(france_decomposed_data)
plot(iran_decomposed_data)
plot(italy_decomposed_data)
plot(us_decomposed_data)
plot(switzerland_decomposed_data)
plot(uk_decomposed_data)
plot(netherlands_decomposed_data)
plot(germany_decomposed_data)


## assessing stationarity ----

# specifying countries
countries <- c("China", "Japan", "France", "Iran, Islamic Rep.", "Italy", "United States", "Switzerland", "United Kingdom", "Netherlands", "Germany")

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


## assessing seasonality, ii ---

# ACF and PACF assessment

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

# determining plot preferences
for (i in 1:length(plots_list)) {
  if (!is.null(plots_list[[i]])) {
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


## creating univariate dataset ----

# creating storage list
country_datasets <- list()

# determining split ratio
split_ratio <- 0.8

# creating datasets for each country
for (country_name in countries) {
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
  write.csv(train_data, file.path("data/preprocessed/univariate/split/train", paste0(tolower(gsub("\\s+", "", country_name)), "_train.csv")), row.names = FALSE)
  write.csv(test_data, file.path("data/preprocessed/univariate/split/test", paste0(tolower(gsub("\\s+", "", country_name)), "_test.csv")), row.names = FALSE)
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
         policy_response_impact_cf = (ox_c1_school_closing + ox_c2_workplace_closing + ox_c4_restrictions_on_gatherings +
                                        ox_c6_stay_at_home_requirements + ox_c7_restrictions_on_internal_movement) / 5)

cf_preprocessed_multi <- preprocessed_covid_multi_imputed %>% 
  select(cases_per_population_cf, deaths_per_population_cf, policy_response_impact_cf)


## (updated) general correlation matrix ----

# filter out numerical data
numerical_data <- preprocessed_covid_multi_imputed %>% select_if(is.numeric)

# create a correlation matrix
correlation_matrix <- cor(numerical_data, use = "complete.obs")

# adapt to ggplot2
# (convert wide-format data to long-format data)
melted_corr_matrix <- melt(correlation_matrix, na.rm = TRUE)

# produce heatmap
correlation_graph_ii <- ggplot(melted_corr_matrix, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
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
  select(owid_new_deaths, cases_per_population_cf, deaths_per_population_cf, policy_response_impact_cf)

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

# Define the plots
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

# Arrange plots in a single visualization
combined_plot <- grid.arrange(plot1, plot2, plot3, ncol = 1)

# Print the combined plot
print(combined_plot)


## feature selection ----

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

# univariate (2175 x 2)
observations_table <- data.frame(
  Country = c("China", "Japan", "France", "Iran", "Italy", "US", "Switzerland", "UK", "Netherlands", "Germany"),
  Observations = c(nrow(china), nrow(japan), nrow(france), nrow(iran), nrow(italy), nrow(us), nrow(switzerland), nrow(uk), nrow(netherlands), nrow(germany))
) %>% 
  DT::datatable()

# multivariate (48212 x 32)
dim(preprocessed_covid_multi_imputed)



# saving files ----
save(preprocessed_covid_multi_imputed, file = "data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")
save(china, file = "data/preprocessed/univariate/not_split/univariate_china.rda")
save(japan, file = "data/preprocessed/univariate/not_split/univariate_japan.rda")
save(france, file = "data/preprocessed/univariate/not_split/univariate_france.rda")
save(iran, file = "data/preprocessed/univariate/not_split/univariate_iran.rda")
save(italy, file = "data/preprocessed/univariate/not_split/univariate_italy.rda")
save(us, file = "data/preprocessed/univariate/not_split/univariate_us.rda")
save(switzerland, file = "data/preprocessed/univariate/not_split/univariate_switzerland.rda")
save(uk, file = "data/preprocessed/univariate/not_split/univariate_uk.rda")
save(netherlands, file = "data/preprocessed/univariate/not_split/univariate_netherlands.rda")
save(germany, file = "data/preprocessed/univariate/not_split/univariate_germany.rda")
# save(importance, file = "visuals/importance.rda")