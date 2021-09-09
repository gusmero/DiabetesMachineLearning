#########################
##LOADING LIBRARIES#####
#######################

setwd("~/")

install.packages("e1071")
install.packages("ROCR")
install.packages("corrplot")
install.packages("caret")
install.packages("plotly")
install.packages("ggplot2")
install.packages("rpart")
install.packages("rattle")
install.packages("rpart.plot")
install.packages("RColorBrewer")
install.packages("pROC")

library(e1071)
library(ROCR)
library(corrplot)
library(caret)
library(plotly)
library(ggplot2)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(pROC)
library(neuralnet)
library(kernlab)





######################
##LOADING DATASET####
#####################

options(max.print = 9999)
dataset_originale = read.csv("diabetes.csv", header = TRUE, sep = ",", col.names=c("Pregnant","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Diabetes"))
#dataset_originale$Diabetes = as.factor(dataset_originale$Diabetes)
dataset_originale$ID <- seq.int(nrow(dataset_originale))
#informazione minimo dell'oggetto
str(dataset_originale)
nrow(data.frame(dataset_originale))

#diminuiamo il numero di variabili PERCHè?
#Togliamo i record che contengono degli zeri
dataset_zeri = dataset_originale[,c("Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","ID")]
row_sub = apply(dataset_zeri, 1, function(row) all(row !=0 ))
dataset_zeri = dataset_zeri[row_sub,]
str(dataset_zeri)
nrow(data.frame(dataset_zeri))

#dataset 
dataset = merge(dataset_originale, dataset_zeri, by.dataset_originale="ID", by.dataset_zeri="ID", all.dataset_originale=TRUE)
dataset$ID = NULL
str(dataset)
nrow(data.frame(dataset))
dataset_without_outcomes = dataset[,1:8]
outcomes = dataset[,9]
#dataset$Diabetes= as.factor(dataset$Diabetes)
#rende la variabile diabete un factor
dataset$Diabetes = factor(dataset$Diabetes, levels=c(0,1), labels=c("no", "yes"))


#DIVISIONE INIZIALE DEL DATASET E DEL TRAINING SET
set.seed(1000)
ind = sample(2, nrow(dataset), replace = TRUE, prob=c(0.7, 0.3))
trainset = dataset[ind == 1,]
testset = dataset[ind == 2,]

# TRAINING
#create the traincontrol parameter
control = trainControl(method = "repeatedcv", number = 10,repeats = 10,
                       classProbs = TRUE, summaryFunction = twoClassSummary)









str(trainset)






#########
####SVM
############

# PRIMI GRAFICI
#rapporto tra una singola variabile e l'outcomes
counts = table(dataset$Diabetes, dataset$Glucose)
barplot(counts, col=c("darkblue","red"),ylab="Soggetti" , xlab="Glucose", legend = c("Negativi", "Positivi"), main = "Diabete per livello di glucosio nel sangue")
#rapporto tra una singola variabile e l'outcomes
counts = table(dataset$Diabetes, dataset$Glucose)
barplot(counts, col=c("darkblue","red"),ylab="Soggetti" , xlab="Glucose", legend = c("Negativi", "Positivi"), main = "Diabete per livello di glucosio nel sangue")
# Differenza tra daset originale e filtrato
plot(dataset_originale, col = dataset$Diabetes)
plot(dataset, col = dataset$Diabetes)
#Matrice correlazione variabili dataset
pairs(dataset[,1:8], panel = panel.smooth, col = dataset$Diabetes)


#TIPOLOGIE DI TRAIN
#tuneGrid()
# #traing the SVM
# svm.model= train(Diabetes ~ ., data = trainset,preProc = c("center", "scale") , 
#                  tuneLength = 9 , method = "svmRadial", metric =
#                    "ROC", trControl = control)

#traing the SVM
svm.model= train(Diabetes ~ ., data = trainset, method = "svmRadial", metric =
                   "ROC", trControl = control)
# trellis.par.set(caretTheme())
# plot(svm.model)

# 
plot(svm.model)
# svm.model$bestTune



svm.subset = subset(trainset, select=c("BMI", "Insulin",
                                        "Diabetes"))
plot(x=trainset$BMI,y= trainset$Insulin,
     col=trainset$Diabetes, pch=19)
#Then, we can mark the support vectors with blue circles: 
#points(svm.subset[svm.model$index,c(1,2)],col="blue",cex=2) mancano gli indici dei vettori di supporto
summary(svm.model)


#testing SVM 
#svm.probs ritorna i valori di probabilità per classe
svm.probs = predict(svm.model, testset[,! names(testset) %in% c("Diabetes")],
                    type = "prob")
#svm.raw ritorna i valori predetti a classi
svm.raw = predict(svm.model, testset[,! names(testset) %in% c("Diabetes")],
                  type = "raw")
# Look of the performance for both "positive" and "negative" class
#Compute all parameter
svm.result.class1 = confusionMatrix(svm.raw, testset[,c("Diabetes")],positive="yes", mode = "everything") 
svm.result.class2 = confusionMatrix(svm.raw, testset[,c("Diabetes")],positive="no", mode = "everything") 


#Receiver Operating Characteristic (ROC)

#svm roc and auc
svm.ROC = roc(response =testset[,c("Diabetes")], predictor =svm.probs$yes,
              levels = levels(testset[,c("Diabetes")]))
plot(svm.ROC,type="S", col="green")
svm.pred.prob = svm.probs
svm.pred.to.roc = svm.pred.prob[, 2] 
svm.pred.rocr = ROCR::prediction(svm.pred.to.roc, testset[,c("Diabetes")])
svm.perf.rocr = performance(svm.pred.rocr, measure = "auc", x.measure = "cutoff")
svm.perf.tpr.rocr = performance(svm.pred.rocr, "tpr","fpr")
plot(svm.perf.tpr.rocr, colorize=T,main=paste("AUC:",(svm.perf.rocr@y.values))) 

abline(a=0, b=1)

opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]],
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}

print(opt.cut(svm.perf.tpr.rocr, svm.pred.rocr))

svm.acc.perf = performance(svm.pred.rocr, measure = "acc")
plot(svm.acc.perf)

ind = which.max( slot(svm.acc.perf, "y.values")[[1]] )
acc = slot(svm.acc.perf, "y.values")[[1]][ind]
cutoff = slot(svm.acc.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))








#training the DECISION TREE
rpart.model= train(Diabetes ~ ., data = trainset, method = "rpart", metric = "ROC",
                   trControl = control)
plot(rpart.model)
summary(rpart.model)
#tree plot
fancyRpartPlot(rpart.model$finalModel)

#testing DECISION TREE
rpart.probs = predict(rpart.model, testset[,! names(testset) %in% c("Diabetes")],
                      type = "prob")
rpart.raw = predict(rpart.model, testset[,! names(testset) %in% c("Diabetes")],
                      type = "raw")
# Look of the performance for both "positive" and "negative" class
result.rpart.class1 = confusionMatrix(rpart.raw, testset[,c("Diabetes")],positive="yes") 
result.rpart.class2 = confusionMatrix(rpart.raw, testset[,c("Diabetes")],mode = "everything") 



#decision tree roc
rpart.ROC = roc(response = testset[,c("Diabetes")], predictor =rpart.probs$yes,
                levels = levels(testset[,c("Diabetes")]))
plot(rpart.ROC, add=TRUE, col="blue")










#COMPARING BETWEEN MODELS

#These functions provide methods for collection, analyzing and visualizing 
#a set of resampling results from a common data set, si possono aggiungere 
#altri modelli e fare il confronto su due o piu modelli
cv.values = resamples(list(svm=svm.model, rpart = rpart.model))
summary(cv.values)
#Use dotplot to plot the results in the ROC metric
dotplot(cv.values, metric = "ROC") 
# Plots a series of vertical box-and-whisker plots where the individual boxplots 
# represent the data subdivided by the value of some factor. Optionally the y-axis
# may be scaled logarithmically. A variety of other plot options are available,
# see Details and Note below
bwplot(cv.values, layout = c(3, 1)) 
#Draw Conditional Scatter Plot Matrices and Parallel Coordinate Plots
splom(cv.values,metric="ROC")
#tempistiche
cv.values$timings


