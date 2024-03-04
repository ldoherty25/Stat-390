## XG Boost (Univariate)

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

# set up parallel processing 
registerDoMC(cores = 8)

# handling common conflicts
tidymodels_prefer()

# setting a seed
set.seed(1234)

#loading the univariate datasets
load("data/preprocessed/multivariate/not_split/preprocessed_covid_multi_imputed.rda")


############################################################################################################################################################################################################################################################


multivar <- preprocessed_covid_multi_imputed %>% 
  rename(ds = date, y = owid_new_deaths)









