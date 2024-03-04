#### MEENA PROPHET MULTIVARIATE #### ------

#calling the packages
library(tidyverse)
library(prophet)
library(forecast)
library(lubridate)
library(modelr)
library(purrr)
library(zoo)
library(randomForest)
library(caret)
library(imputeTS)
library(doMC)
library(patchwork)
library(seastests)
library(gridExtra)
library(timetk)
library(tidymodels)

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

#setting the seed
set.seed(1234)

load("data/preprocessed/multivariate/preprocessed_covid_multi_imputed.rda")

####################### changing variable names

multivar_covid <- preprocessed_covid_multi_imputed %>%
  rename(ds = date, )