## EDA

# load packages ----
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(reshape2)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)


## load data ----

# read and clean
covid <- read_csv('data/owid-covid-data.csv') %>% 
  janitor::clean_names()

# data quality assurance
skimr::skim_without_charts(covid)


## basic eda ----

# inspecting missingness
missing_prop <- covid %>% 
  naniar::miss_var_summary() %>% 
  filter(pct_miss > 0) %>% 
  DT::datatable()

# data completion rate by continent
completion_by_continent <- covid %>%
  group_by(continent) %>%
  summarise(across(everything(), ~ sum(!is.na(.))/n(), .names = "completion_rate_{.col}"))

# overall completion rate by continent
overall_completion <- completion_by_continent %>%
  rowwise() %>%
  mutate(overall_completion_rate = mean(c_across(starts_with("completion_rate_")), na.rm = TRUE)) %>%
  ungroup() %>%
  select(continent, overall_completion_rate)

# keeping only observations in europe and with less than 65% missingness
eu_covid <- covid %>% 
  filter(continent == "Europe") %>% 
  select(-continent,
         -handwashing_facilities,
         -excess_mortality_cumulative_absolute,
         -excess_mortality_cumulative,
         -excess_mortality,
         -excess_mortality_cumulative_per_million,
         -weekly_icu_admissions,
         -weekly_icu_admissions_per_million,
         -total_boosters,
         -total_boosters_per_hundred,
         -weekly_hosp_admissions,
         -weekly_hosp_admissions_per_million,
         -new_vaccinations,
         -people_fully_vaccinated,
         -people_fully_vaccinated_per_hundred,
         -people_vaccinated,
         -people_vaccinated_per_hundred)

# inspecting missingness again
missing_prop_eu <- eu_covid %>% 
  naniar::miss_var_summary() %>% 
  filter(pct_miss >= 0) %>% 
  DT::datatable()

# create graph
missing_graph <- eu_covid %>%
  naniar::gg_miss_var() +
  labs(title = "Graph 1: Missing Data")

# checking final dimensions
dim(eu_covid)


## correlation matrix ----

# filter out numerical data
numerical_data <- eu_covid %>% select_if(is.numeric)

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

## save files ----
save(correlation_graph, file = "visuals/correlation_graph.rda")
save(missing_graph, file = "visuals/missing_graph.rda")