## EDA


# load packages ----
library(tidyverse)
library(tidymodels)
library(reshape2)

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


## basic EDA ----

# calculate average of identical variables
mutated_covid <- covid %>%
  mutate(average_confirmed = rowMeans(select(., jhu_confirmed, owid_total_cases, ox_confirmed_cases), na.rm = TRUE),
         average_deaths = rowMeans(select(., jhu_deaths, owid_total_deaths, ox_confirmed_deaths), na.rm = TRUE),
         average_stringency_index = rowMeans(select(., owid_stringency_index, ox_stringency_index), na.rm = TRUE)) %>%
  select(-jhu_confirmed, -owid_total_cases, -ox_confirmed_cases, 
         -jhu_deaths, -owid_total_deaths, -ox_confirmed_deaths, 
         -owid_stringency_index, -ox_stringency_index) %>% 
  mutate(
    average_confirmed = replace(average_confirmed, is.nan(average_confirmed), NA),
    average_deaths = replace(average_deaths, is.nan(average_deaths), NA),
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
preprocessed_covid <- covid_cleaned %>%
  select(country, date, average_confirmed, average_deaths,
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
skimr::skim_without_charts(preprocessed_covid)


## assessing final missingness ----

# inspecting missingness again
missing_prop_covid <- preprocessed_covid %>% 
  naniar::miss_var_summary() %>% 
  filter(pct_miss >= 0) %>% 
  DT::datatable()

# create graph
missing_graph <- preprocessed_covid %>%
  naniar::gg_miss_var() +
  labs(title = "Graph 1: Missing Data")


## correlation matrix ----

# filter out numerical data
numerical_data <- preprocessed_covid %>% select_if(is.numeric)

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


## Examining the Target Variable
# Target Variable distribution ----
# t_var <- preprocessed_covid %>% 
#   count(average_deaths) %>% 
#   mutate(proportion = n / sum(n))
# What exactly is the code above trying to achieve?

#Target variable prep
ggplot(data = preprocessed_covid, mapping = aes(x = average_deaths)) +
  geom_histogram(bins= 80)


# alternative target variable consideration ----

# graphing distribution
tv_distribution_log <- preprocessed_covid %>%
  filter(!is.na(average_deaths)) %>%
  ggplot() +
  geom_histogram(aes(x = average_deaths), bins = 50) +
  scale_x_log10() +
  labs(title = "Graph 2: Distribution of Target Variable",
       x = "Average Deaths (log)",
       y = "Count") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))


## save files ----
save(preprocessed_covid, file = "data/preprocessed/preprocessed_covid.rda")
save(missing_prop_covid, file = "visuals/missing_prop_covid.rda")
save(correlation_graph, file = "visuals/correlation_graph.rda")
save(missing_graph, file = "visuals/missing_graph.rda")
save(tv_distribution_log, file = "visuals/tv_distribution_log.rda")
