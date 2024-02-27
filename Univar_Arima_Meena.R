## UNIVARIATE ARIMA MEENA

## uploading packages
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

#setting the seed
set.seed(1234)

#loading the univariate datasets
load("data/preprocessed/univariate/bolivia.rda")
load("data/preprocessed/univariate/brazil.rda")
load("data/preprocessed/univariate/colombia.rda")
load("data/preprocessed/univariate/iran.rda")
load("data/preprocessed/univariate/mexico.rda")
load("data/preprocessed/univariate/peru.rda")
load("data/preprocessed/univariate/russia.rda")
load("data/preprocessed/univariate/saudi.rda")
load("data/preprocessed/univariate/turkey.rda")
load("data/preprocessed/univariate/us.rda")