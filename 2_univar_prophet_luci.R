## Univariate Prophet

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
library(timetk)
library(e1071)
library(prophet)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# set a seed
set.seed(1234)

#loading the univariate datasets
load("data/preprocessed/univariate/not_split/bolivia.rda")
load("data/preprocessed/univariate/not_split/brazil.rda")
load("data/preprocessed/univariate/not_split/colombia.rda")
load("data/preprocessed/univariate/not_split/iran.rda")
load("data/preprocessed/univariate/not_split/mexico.rda")
load("data/preprocessed/univariate/not_split/peru.rda")
load("data/preprocessed/univariate/not_split/russia.rda")
load("data/preprocessed/univariate/not_split/saudi.rda")
load("data/preprocessed/univariate/not_split/turkey.rda")
load("data/preprocessed/univariate/not_split/us.rda")

####################################################################################################################################################################################################################################################################  

## Bolivia model

bolivia_total_days <- nrow(bolivia)
bolivia_train_days <- ceiling(0.9 * bolivia_total_days)
bolivia_test_days <- ceiling((bolivia_total_days - bolivia_train_days))

# creating folds

bolivia_folds <- time_series_cv(
  bolivia,
  date_var = date,
  initial = bolivia_train_days,
  assess = bolivia_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

bolivia_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model fit


##################################################################################################################################################################################################################################################################


## Brazil Model

brazil_total_days <- nrow(brazil)
brazil_train_days <- ceiling(0.9 * brazil_total_days)
brazil_test_days <- ceiling((brazil_total_days - brazil_train_days))

# creating folds

brazil_folds <- time_series_cv(
  brazil,
  date_var = date,
  initial = brazil_train_days,
  assess = brazil_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

brazil_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model

##################################################################################################################################################################################################################################################################


## Colombia

colombia_total_days <- nrow(colombia)
colombia_train_days <- ceiling(0.9 * colombia_total_days)
colombia_test_days <- ceiling((colombia_total_days - colombia_train_days))

# creating folds

colombia_folds <- time_series_cv(
  colombia,
  date_var = date,
  initial = colombia_train_days,
  assess = colombia_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

colombia_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model



##################################################################################################################################################################################################################################################################


## Iran

iran_total_days <- nrow(iran)
iran_train_days <- ceiling(0.9 * iran_total_days)
iran_test_days <- ceiling((iran_total_days - iran_train_days))

# creating folds

iran_folds <- time_series_cv(
  iran,
  date_var = date,
  initial = iran_train_days,
  assess = iran_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

iran_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model



##################################################################################################################################################################################################################################################################


## Mexico

mexico_total_days <- nrow(mexico)
mexico_train_days <- ceiling(0.9 * mexico_total_days)
mexico_test_days <- ceiling((mexico_total_days - mexico_train_days))

# creating folds

mexico_folds <- time_series_cv(
  mexico,
  date_var = date,
  initial = mexico_train_days,
  assess = mexico_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

mexico_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model



##################################################################################################################################################################################################################################################################


## Peru

peru_total_days <- nrow(peru)
peru_train_days <- ceiling(0.9 * peru_total_days)
peru_test_days <- ceiling((peru_total_days - peru_train_days))

# creating folds

peru_folds <- time_series_cv(
  peru,
  date_var = date,
  initial = peru_train_days,
  assess = peru_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

peru_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model



##################################################################################################################################################################################################################################################################

## Russia

russia_total_days <- nrow(russia)
russia_train_days <- ceiling(0.9 * russia_total_days)
russia_test_days <- ceiling((russia_total_days - russia_train_days))

# creating folds

russia_folds <- time_series_cv(
  russia,
  date_var = date,
  initial = russia_train_days,
  assess = russia_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

russia_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model



##################################################################################################################################################################################################################################################################


## Saudi Arabia

saudi_total_days <- nrow(saudi)
saudi_train_days <- ceiling(0.9 * saudi_total_days)
saudi_test_days <- ceiling((saudi_total_days - saudi_train_days))

# creating folds

saudi_folds <- time_series_cv(
  saudi,
  date_var = date,
  initial = saudi_train_days,
  assess = saudi_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

saudi_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model



##################################################################################################################################################################################################################################################################

## Turkey

turkey_total_days <- nrow(turkey)
turkey_train_days <- ceiling(0.9 * turkey_total_days)
turkey_test_days <- ceiling((turkey_total_days - turkey_train_days))

# creating folds

turkey_folds <- time_series_cv(
  turkey,
  date_var = date,
  initial = turkey_train_days,
  assess = turkey_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

turkey_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model



##################################################################################################################################################################################################################################################################

## US

us_total_days <- nrow(us)
us_train_days <- ceiling(0.9 * us_total_days)
us_test_days <- ceiling((us_total_days - us_train_days))

# creating folds

us_folds <- time_series_cv(
  us,
  date_var = date,
  initial = us_train_days,
  assess = us_test_days,
  fold = 1,
  slice_limit = 1
)

# filtering by slice

us_folds %>% 
  tk_time_series_cv_plan() %>% 
  filter(.id == "Slice2") %>% 
  nrow()

# prophet model



##################################################################################################################################################################################################################################################################
