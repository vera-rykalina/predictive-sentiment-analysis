# R version 3.6.2 (2019-12-12)                                                  # # Date August 2020
# Full dataset

#### Load Packages  #####

pacman:: p_load("doParallel", "rstudioapi", "readr","dplyr", "tidyr", "ggplot2", "plotly", "GGally","data.table", "caret", "randomForest","hablar", "wesanderson", "RColorBrewer", "ggsci", "gplots","ggpubr", "C50", "kknn", "rminer", "plyr", "e1071", "caTools", "gbm")


#### Setting Working Directory ####
current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)
getwd() # resulted in /Users/vera/Desktop/DA_Online/Big Data/Task3


#### Loading datasets ####

# Reading Small Matrix (train data) for iPhone ####
raw.iphone <-readRDS(file = "raw.iphone.rds")
head(raw.iphone)
str(raw.iphone) # overall sentiment toward the device on a scale of 0-5 (last column)
summary(raw.iphone)
dim(raw.iphone) # 12973 rows x 59 columns
colnames(raw.iphone)

# iPhonesentiment as a factor
raw.iphone$iphonesentiment <- as.factor(raw.iphone$iphonesentiment)

# Reading Small Matrix (train data) for Galaxy ####
raw.galaxy <-readRDS(file = "raw.galaxy.rds")
head(raw.galaxy)
str(raw.galaxy) # overall sentiment toward the device on a scale of 0-5 (last column)
summary(raw.galaxy)
dim(raw.galaxy) # 12973 rows x 59 columns
colnames(raw.galaxy)

# Galaxysentiment as a factor
raw.galaxy$galaxysentiment <- as.factor(raw.galaxy$galaxysentiment)


# iPhone
intrain1  <- createDataPartition(
  y = raw.iphone$iphonesentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

str(intrain1)

raw.iphonetrain <- raw.iphone[intrain1,]
dim(raw.iphonetrain)
str(raw.iphonetrain)

raw.iphonetest <- raw.iphone[-intrain1,]
dim(raw.iphonetest)
str(raw.iphonetest)


# Galaxy
intrain2  <- createDataPartition(
  y = raw.galaxy$galaxysentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

str(intrain2)

raw.galaxytrain <- raw.galaxy[intrain2,]
dim(raw.galaxytrain)
str(raw.galaxytrain)

raw.galaxytest <- raw.galaxy[-intrain2,]
dim(raw.galaxytest)
str(raw.galaxytest)


#### Core Selection ####
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores.
cl <- makeCluster(4)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

set.seed(1234)
#### C5.0 ####

C50trctrl <- trainControl(method = "repeatedcv", 
                          number = 10, 
                          repeats = 2)


C50grid <- expand.grid(.model="tree",.trials = c(1:10),.winnow = FALSE)

# iPhone
C50model1 <- train(iphonesentiment~.,
                   data = raw.iphonetrain,
                   method = "C5.0",
                   metric = "Accuracy",
                   tuneGrid = C50grid,
                   trControl = C50trctrl)


C50model1
plot(C50model1)
varImp(C50model1)
varImp(C50model1$finalModel, scale=FALSE)
predC50model1 <- predict(C50model1, raw.iphonetest)
C50model1metrics <- postResample(predC50model1, raw.iphonetest$iphonesentiment)
C50model1metrics

cmC50iphone <- confusionMatrix(predC50model1, raw.iphonetest$iphonesentiment) 
cmC50iphone

## Save a model to a file
saveRDS(C50model1, file = "C50model1.rds")
# C50model1 <- readRDS(file = "C50model1.rds")

# Galaxy
C50model2 <- train(galaxysentiment~.,
                   data = raw.galaxytrain,
                   method = "C5.0",
                   metric = "Accuracy",
                   tuneGrid = C50grid,
                   trControl = C50trctrl)


C50model2
plot(C50model2)
varImp(C50model2)
varImp(C50model2$finalModel, scale=FALSE)
predC50model2 <- predict(C50model2, raw.galaxytest)
C50model2metrics <- postResample(predC50model2, raw.galaxytest$galaxysentiment)
C50model2metrics

cmC50galaxy <- confusionMatrix(predC50model2, raw.galaxytest$galaxysentiment) 
cmC50galaxy

## Save a model to a file
saveRDS(C50model2, file = "C50model2.rds")
# C50model2 <- readRDS(file = "C50model2.rds")



#### Random Forest ####
RFtrctrl <- trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 2)

RFgrid <- expand.grid(mtry=c(1:5))

# iPhone
RFmodel1 <- train(iphonesentiment ~ ., 
                  raw.iphonetrain,
                  method = "rf",
                  trControl = RFtrctrl,
                  tuneGrid = RFgrid,
                  tuneLenght = 2)

RFmodel1
plot(RFmodel1)
varImp(RFmodel1)
predRFmodel1 <- predict(RFmodel1, raw.iphonetest)
RFmodel1metrics <- postResample(predRFmodel1, raw.iphonetest$iphonesentiment)
RFmodel1metrics

cmRFiphone <- confusionMatrix(predRFmodel1, raw.iphonetest$iphonesentiment) 
cmRFiphone

# Save a model to a file
saveRDS(RFmodel1, file = "RFmodel1.rds")
# RFmodel1 <- readRDS(file = "RFmodel1.rds")

# Galaxy
RFmodel2 <- train(galaxysentiment ~ ., 
                  raw.galaxytrain,
                  method = "rf",
                  trControl = RFtrctrl,
                  tuneGrid = RFgrid,
                  tuneLenght = 2)

RFmodel2
plot(RFmodel2)
varImp(RFmodel2)
predRFmodel2 <- predict(RFmodel2, raw.galaxytest)
RFmodel2metrics <- postResample(predRFmodel2, raw.galaxytest$galaxysentiment)
RFmodel2metrics

cmRFgalaxy <- confusionMatrix(predRFmodel2, raw.galaxytest$galaxysentiment) 
cmRFgalaxy

# Save a model to a file
saveRDS(RFmodel2, file = "RFmodel2.rds")
# RFmodel2 <- readRDS(file = "RFmodel2.rds")



#### SVM ####

## SVM Train Control
SVMcontrol <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2,
                           preProc = "center")

SVMmodel1 <- train(iphonesentiment ~ ., 
                   raw.iphonetrain,
                   method = "svmLinear",
                   trControl = SVMcontrol,
                   tuneLenght = 2)

SVMmodel1
plot(SVMmodel1)
predSVMmodel1 <- predict(SVMmodel1, raw.iphonetest)
SVMmodel1metrics <- postResample(predSVMmodel1, raw.iphonetest$iphonesentiment)
SVMmodel1metrics

cmSVMiphone <- confusionMatrix(predSVMmodel1, raw.iphonetest$iphonesentiment) 
cmSVMiphone

# Save a model to a file
saveRDS(SVMmodel1, file = "SVMmodel1.rds")
# SVMmodel1<- readRDS(file = "SVMmodel1.rds")

# Galaxy
SVMmodel2 <- train(galaxysentiment ~ ., 
                   raw.galaxytrain,
                   method = "svmLinear",
                   trControl = SVMcontrol,
                   tuneLenght = 2)

SVMmodel2
plot(SVMmodel2)
predSVMmodel2 <- predict(SVMmodel2, raw.galaxytest)
SVMmodel2metrics <- postResample(predSVMmodel2, raw.galaxytest$galaxysentiment)
SVMmodel2metrics

cmSVMgalaxy <- confusionMatrix(predSVMmodel2, raw.galaxytest$galaxysentiment) 
cmSVMgalaxy

# Save a model to a file
saveRDS(SVMmodel2, file = "SVMmodel2.rds")
# SVMmodel2<- readRDS(file = "SVMmodel2.rds")



#### kNN ####

## kNN Train Control
kNNcontrol <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2,
                           preProc = c("center", "range"))

# iPhone
kNNmodel1 <- train(iphonesentiment ~ ., 
                   raw.iphonetrain,
                   method = "kknn",
                   trControl = kNNcontrol)

kNNmodel1
plot(kNNmodel1)
predkNNmodel1 <- predict(kNNmodel1, raw.iphonetest)
kNNmodel1metrics <- postResample(predkNNmodel1, raw.iphonetest$iphonesentiment)
kNNmodel1metrics

cmkNNiphone <- confusionMatrix(predkNNmodel1, raw.iphonetest$iphonesentiment) 
cmkNNiphone

# Save a model to a file
saveRDS(kNNmodel1, file = "kNNmodel1.rds")
# kNNmodel1<- readRDS(file = "kNNmodel1.rds")

# Galaxy
kNNmodel2 <- train(galaxysentiment ~ ., 
                   raw.galaxytrain,
                   method = "kknn",
                   trControl = kNNcontrol)

kNNmodel2
plot(kNNmodel2)
predkNNmodel2 <- predict(kNNmodel2, raw.galaxytest)
kNNmodel2metrics <- postResample(predkNNmodel2, raw.galaxytest$galaxysentiment)
kNNmodel2metrics

cmkNNgalaxy <- confusionMatrix(predkNNmodel2, raw.galaxytest$galaxysentiment) 
cmkNNgalaxy

# Save a model to a file
saveRDS(kNNmodel2, file = "kNNmodel2.rds")
# kNNmodel12 <- readRDS(file = "kNNmodel2.rds")


#### XGB ####

## XGB Train Control
XGBcontrol <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2)
# xgbGrid
tune.grid.xgb <- expand.grid(eta=0.1,
                             nrounds=100,
                             max_depth=6,
                             min_child_weight =2.0,
                             colsample_bytree = 0.5,
                             gamma=0,
                             subsample=1)

## iPhone
XGBmodel1 <- train(iphonesentiment ~ ., 
                   raw.iphonetrain,
                   method = "xgbTree",
                   tuneGrid = tune.grid.xgb,
                   trControl = XGBcontrol)

XGBmodel1 
predXGBmodel1  <- predict(XGBmodel1 , raw.iphonetest)
XGBmodel1metrics <- postResample(predXGBmodel1 , raw.iphonetest$iphonesentiment)
XGBmodel1metrics
varImp(XGBmodel1)
cmXGBiphone <- confusionMatrix(predXGBmodel1, raw.iphonetest$iphonesentiment) 
cmXGBiphone


#saveRDS(XGBmodel1, file = "XGBmodel1.rds")
XGBmodel1 <-readRDS(file = "XGBmodel1.rds")


## Galaxy
XGBmodel2 <- train(galaxysentiment ~ ., 
                   raw.galaxytrain,
                   method = "xgbTree",
                   tuneGrid = tune.grid.xgb,
                   trControl = XGBcontrol)

XGBmodel2 
predXGBmodel2  <- predict(XGBmodel2 , raw.galaxytest)
XGBmodel2metrics <- postResample(predXGBmodel2 , raw.galaxytest$galaxysentiment)
XGBmodel2metrics
varImp(XGBmodel2)
cmXGBgalaxy <- confusionMatrix(predXGBmodel2, raw.galaxytest$galaxysentiment) 
cmXGBgalaxy


saveRDS(XGBmodel2, file = "XGBmodel2.rds")
XGBmodel2 <-readRDS(file = "XGBmodel2.rds")


## Stop Cluster
stopCluster(cl)

#### Comparison of model performance
# iPhone
ClassModelsiPhone <- resamples(list(C50 = C50model1, RF= RFmodel1, SVM=SVMmodel1, KNN = kNNmodel1, XGB=XGBmodel1))

summary(ClassModelsiPhone) 
bwplot(ClassModelsiPhone, metric = "Accuracy")
densityplot(ClassModelsiPhone, metric = "Accuracy")

# Galaxy
ClassModelsGalaxy <- resamples(list(C50 = C50model2, RF= RFmodel2, KNN = kNNmodel2, XGB=XGBmodel2))

summary(ClassModelsGalaxy) 
bwplot(ClassModelsGalaxy, metric = "Accuracy")
densityplot(ClassModelsGalaxy, metric = "Accuracy")


## Creating data frames for performance and accuracy metrics
AccuracyMetrics <- data.frame(C50model1metrics, 
                              C50model2metrics,
                              RFmodel1metrics,
                              RFmodel2metrics,
                              SVMmodel1metrics,
                              SVMmodel2metrics,
                              kNNmodel1metrics,
                              kNNmodel2metrics,
                              XGBmodel1metrics,
                              XGBmodel2metrics)

## Transposing the data frame
AccuracyMetrics <- data.frame(t(AccuracyMetrics))

## Naming the rows
AccuracyMetrics$Algorithms <- rownames(AccuracyMetrics)

## Reordering the columns
AccuracyMetrics <- AccuracyMetrics[, c(3,1,2)]

## Arranging by Accuracy and Kappa to see the best models
AccuracyMetrics %>% 
  arrange(desc(Accuracy))

