library(tidyverse)
library(tibble)
library(explore)
library(dataMaid)
library(summarytools)
# Set path for Java Path because i use h2o Library
Sys.setenv(JAVA_HOME="D:\\ZZZ Program Files\\Java")
library(rJava)
# Set path for Work space
setwd("C:/Users/Hendrich/Desktop/BTL_Software_Technology/Analysis_Problem_By_R")

library(keras)
library(dplyr)
require(foreign)
require(farff)
library(ggplot2)
library(tidyr)
library(tibble)

# Reading data set
df = read.csv("Breast_cancer_data.csv")

# EDA 
makeDataReport(df,
               render = FALSE,
               file = "EDA.Rmd",
               replace = TRUE)

names(df) = c("mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness",
             "diagnosis")
df$diagnosis = df$diagnosis %>% recode_factor(.,`1` = 1 , `0` = 0)
Hmisc::describe(df)

View(df)
library(caret)
set.seed(123)  
idTrain = caret::createDataPartition(y = df$diagnosis,p = 499/569,list = FALSE)
trainset = df[idTrain,]
testset = df[-idTrain,]
View(testset)

library(h2o)
h2o.init(nthreads = -1, max_mem_size = "2g")

wtrain = as.h2o(trainset)
wtest = as.h2o(testset)
reponse = "diagnosis"
feature = setdiff(colnames(wtrain),reponse)
rfmod1 = h2o.randomForest(x = feature,
                          y = reponse,
                          training_frame = wtrain, nfolds = 12,
                          fold_assignment = "AUTO",
                          ntrees = 100, max_depth = 50, sample_rate = 0.5,
                          mtries = 4,balance_classes = TRUE,
                          stopping_metric = "logloss",
                          stopping_tolerance = 0.001,
                          stopping_rounds = 3,
                          keep_cross_validation_models = TRUE,
                          keep_cross_validation_predictions = TRUE,
                          keep_cross_validation_fold_assignment = TRUE,
                          seed = 12345)
h2o.performance(rfmod1,wtrain)
h2o.performance(rfmod1,wtest)

# Explaination For Result

library(lime)
model_type.H2OModel <- function(x, ...) "classification"

predict_model.H2OModel <- function(x, newdata, type, ...){
  pred <- h2o.predict(x, as.h2o(newdata))
  return(as.data.frame(pred[,-1]))
}

set.seed(12345)

explain <- lime(trainset[,])