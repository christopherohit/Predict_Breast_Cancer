library(ISLR)
library(tidyverse)
library(ggplot2)
library(tibble)
library(tidyr)
library(dataMaid)
library(Amelia)
library(klaR)
library(needs)
needs(readr,
      dplyr,
      ggplot2,
      corrplot,
      gridExtra,
      pROC,
      MASS,
      caTools,
      caret,
      caretEnsemble,
      doMC)


df = read.csv("data.csv")
set.seed(456)

#Drop columns which unnecessary
df = subset(df,select = -c(X,id))

print(head(df,4))

makeDataReport(df,output = "EDA.Rmd",
               render = TRUE,
               replace = TRUE,
               )

str(df)

df$diagnosis <- as.factor(df$diagnosis)
summary(df)
df[,33] <- NULL

prop.table(table(df$diagnosis))
corr_mat <- cor(df[,3:ncol(df)])
corrplot(corr_mat,order = "hclust",t1.cex = 1 , addrect = 8)

set.seed(1234)


data_index <- createDataPartition(df$diagnosis, p = 0.7 , list = FALSE)
train_df <- df[data_index, - 1]
test_df <- df[-data_index, -1]

# Principal Component Analysis

pca_res <- prcomp(df[,3:ncol(df)], center = TRUE , scale = TRUE)
plot(pca_res , type = "b")

# LDA 

lda_res <- lda(diagnosis~.,df,center= TRUE,scale = TRUE)
lda_df <- predict(lda_res,df)$x %>% as.data.frame() %>% cbind(diagnosis = df$diagnosis)
lda_res

ggplot(lda_df,aes(x= LD1, y = 0, col= diagnosis)) + geom_point(alpha = 0.5)
ggplot(lda_df, aes(x = LD1, fill= diagnosis)) + geom_density(alpha = 0.5)

train_data <- lda_df[data_index,]
test_data <- lda_df[-data_index,]

fit_control <- trainControl(method = "cv",
                            number = 5,
                            preProcOptions = list(thresh = 0.99),
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)
# Neural Network NNET

model_nnet <- train(diagnosis~.,
                    train_df,
                    methods = "nnet",
                    metric = "ROC",
                    preProcess = c('center', 'scale'),
                    trace= FALSE,
                    tuneLength = 10,
                    trControl = fit_control)

pred_nnet <- predict(model_nnet, test_df) # Check probality in Test data
pred_train <- predict(model_nnet, train_df) # Check probality in Train data
cm_nnet <- confusionMatrix(pred_nnet,test_df$diagnosis, positive = "M")
cm_nnet_train <- confusionMatrix(pred_train ,train_df$diagnosis, positive = "M" )
cm_nnet_train
cm_nnet

# K- Nearest Neighbor

model_knn <- train(diagnosis~.,
                   train_df,
                   method = "knn",
                   metric = "ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength = 10,
                   trControl = fit_control)

pred_knn <- predict(model_knn, train_df) # Check probality in train data
pred_knn_tesr <- predict(model_knn , test_df) # check probality in test data
cm_knn_train <- confusionMatrix(pred_knn,train_df$diagnosis,positive = "M")
cm_knn_test <- confusionMatrix(pred_knn_tesr, test_df$diagnosis, positive = "M")
cm_knn_train
cm_knn_test

# Naive Bayes
model_nb <- train(diagnosis~.,
                  train_df,
                  method = "nb",
                  metric = "ROC",
                  preProcess = c('center', 'scale'),
                  trace= FALSE,
                  trControl = fit_control)
pred_nb_train <- predict(model_nb , train_df) # Check train
pred_nb_test <- predict(model_nb, test_df) # Check test
cm_nb_train <- confusionMatrix(pred_nb_train , train_df$diagnosis , positive = "M")
cm_nb_test <- confusionMatrix(pred_nb_test , test_df$diagnosis , positive =  "M")
cm_nb_train
cm_nb_test
