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


##########################################################################################################################################################################################################################################################


## Bolivia model










