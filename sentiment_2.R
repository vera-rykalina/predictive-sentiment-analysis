#### Load Packages  #####

pacman:: p_load("doParallel", "rstudioapi", "readr","dplyr", "tidyr", "ggplot2", "plotly", "GGally","data.table", "caret", "randomForest","hablar", "wesanderson", "RColorBrewer", "ggsci", "gplots","ggpubr", "C50", "class", "rminer","kernlab", "plyr", "xgboost", "gbm")


#### Setting Working Directory ####
current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)
getwd() # resulted in /Users/vera/Desktop/DA_Online/Big Data/Task3

#### Read dataframes ####
nzv.iphone<-readRDS(file = "nzv.iphone.rds")
nzv.galaxy<-readRDS(file = "nzv.galaxy.rds")
str(nzv.iphone)
rfe.iphone<-readRDS(file = "rfe.iphone.rds")
rfe.galaxy<-readRDS(file = "rfe.galaxy.rds")
str(rfe.iphone)
cor.iphone<-readRDS(file = "cor.iphone.rds")
cor.galaxy<-readRDS(file = "cor.galaxy.rds")

# iPhone NVZ
nzv.intrain1  <- createDataPartition(
  y = nzv.iphone$iphonesentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

nzv.iphonetrain <- nzv.iphone[nzv.intrain1,]
nzv.iphonetest <- nzv.iphone[-nzv.intrain1,]

# iPhone RFE
rfe.intrain1  <- createDataPartition(
  y = rfe.iphone$iphonesentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

rfe.iphonetrain <- rfe.iphone[rfe.intrain1,]
rfe.iphonetest <- rfe.iphone[-rfe.intrain1,]

# iPhone COR
cor.intrain1  <- createDataPartition(
  y = cor.iphone$iphonesentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

cor.iphonetrain <- cor.iphone[cor.intrain1,]
cor.iphonetest <- cor.iphone[-cor.intrain1,]

#### Core Selection ####
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores.
cl <- makeCluster(4)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

## Set seed
set.seed(1234)


#### XGB ####
## Train Control
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

## iPhone NZV
nzvXGB1 <- train(iphonesentiment ~ ., 
                   nzv.iphonetrain,
                   method = "xgbTree",
                   tuneGrid = tune.grid.xgb,
                   trControl = XGBcontrol)

nzvXGB1
pred.nzvXGB1  <- predict(nzvXGB1 , nzv.iphonetest)
nzvXGB1metrics <- postResample(pred.nzvXGB1 , nzv.iphonetest$iphonesentiment)
nzvXGB1metrics
varImp(nzvXGB1)
NZVcmXGBiphone <- confusionMatrix(pred.nzvXGB1, nzv.iphonetest$iphonesentiment) 
NZVcmXGBiphone

saveRDS(nzvXGB1, file = "nzvXGB1.rds")


## iPhone RFE
rfeXGB1 <- train(iphonesentiment ~ ., 
                 rfe.iphonetrain,
                 method = "xgbTree",
                 tuneGrid = tune.grid.xgb,
                 trControl = XGBcontrol)

rfeXGB1
pred.rfeXGB1  <- predict(rfeXGB1 , rfe.iphonetest)
rfeXGB1metrics <- postResample(pred.rfeXGB1 , rfe.iphonetest$iphonesentiment)
rfeXGB1metrics
varImp(rfeXGB1)
RFEcmXGBiphone <- confusionMatrix(pred.rfeXGB1, rfe.iphonetest$iphonesentiment) 
RFEcmXGBiphone

saveRDS(rfeXGB1, file = "rfeXGB1.rds")

## iPhone correlation
corXGB1 <- train(iphonesentiment ~ ., 
                 cor.iphonetrain,
                 method = "xgbTree",
                 tuneGrid = tune.grid.xgb,
                 trControl = XGBcontrol)

corXGB1
pred.corXGB1  <- predict(corXGB1 , cor.iphonetest)
corXGB1metrics <- postResample(pred.corXGB1 , cor.iphonetest$iphonesentiment)
corXGB1metrics
varImp(corXGB1)
CORcmXGBiphone <- confusionMatrix(pred.corXGB1, cor.iphonetest$iphonesentiment) 
CORcmXGBiphone

saveRDS(corXGB1, file = "corXGB1.rds")

####################################################################
# iPhone NVZ
nzv.intrain2  <- createDataPartition(
  y = nzv.galaxy$galaxysentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

nzv.galaxytrain <- nzv.galaxy[nzv.intrain2,]
nzv.galaxytest <- nzv.galaxy[-nzv.intrain2,]

# iPhone RFE
rfe.intrain2  <- createDataPartition(
  y = rfe.galaxy$galaxysentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

rfe.galaxytrain <- rfe.galaxy[rfe.intrain2,]
rfe.galaxytest <- rfe.galaxy[-rfe.intrain2,]

# iPhone COR
cor.intrain2  <- createDataPartition(
  y = cor.galaxy$galaxysentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

cor.galaxytrain <- cor.galaxy[cor.intrain2,]
cor.galaxytest <- cor.galaxy[-cor.intrain2,]

#### Core Selection ####
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores.
cl <- makeCluster(4)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

## Set seed
set.seed(1234)


#### XGB ####
## Train Control
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

## iPhone NZV
nzvXGB2 <- train(galaxysentiment ~ ., 
                 nzv.galaxytrain,
                 method = "xgbTree",
                 tuneGrid = tune.grid.xgb,
                 trControl = XGBcontrol)

nzvXGB2
pred.nzvXGB2  <- predict(nzvXGB2 , nzv.galaxytest)
nzvXGB2metrics <- postResample(pred.nzvXGB2 , nzv.galaxytest$galaxysentiment)
nzvXGB2metrics
varImp(nzvXGB2)
NZVcmXGBgalaxy <- confusionMatrix(pred.nzvXGB2, nzv.galaxytest$galaxysentiment) 
NZVcmXGBgalaxy

saveRDS(nzvXGB2, file = "nzvXGB2.rds")


## iPhone RFE
rfeXGB2 <- train(galaxysentiment ~ ., 
                 rfe.galaxytrain,
                 method = "xgbTree",
                 tuneGrid = tune.grid.xgb,
                 trControl = XGBcontrol)

rfeXGB2
pred.rfeXGB2  <- predict(rfeXGB2 , rfe.galaxytest)
rfeXGB2metrics <- postResample(pred.rfeXGB2 , rfe.galaxytest$galaxysentiment)
rfeXGB2metrics
varImp(rfeXGB2)
RFEcmXGBgalaxy <- confusionMatrix(pred.rfeXGB2, rfe.galaxytest$galaxysentiment) 
RFEcmXGBgalaxy

saveRDS(rfeXGB2, file = "rfeXGB2.rds")

## iPhone correlation
corXGB2 <- train(galaxysentiment ~ ., 
                 cor.galaxytrain,
                 method = "xgbTree",
                 tuneGrid = tune.grid.xgb,
                 trControl = XGBcontrol)

corXGB2
pred.corXGB2  <- predict(corXGB2 , cor.galaxytest)
corXGB2metrics <- postResample(pred.corXGB2 , cor.galaxytest$galaxysentiment)
corXGB2metrics
varImp(corXGB2)
CORcmXGBgalaxy <- confusionMatrix(pred.corXGB2, cor.galaxytest$galaxysentiment) 
CORcmXGBgalaxy

saveRDS(corXGB2, file = "corXGB2.rds")

# Stop Cluster
stopCluster(cl)

#### Comparison of model performance
# iPhone
ClassModelsiPhone1 <- resamples(list(RFE1 = rfeXGB1, COR1=corXGB1, NZV1=nzvXGB1))

summary(ClassModelsiPhone1) 
bwplot(ClassModelsiPhone1, metric = "Accuracy")
densityplot(ClassModelsiPhone1, metric = "Accuracy")

# Galaxy
ClassModelsGalaxy2<- resamples(list(RFE2 = rfeXGB2, COR2=corXGB2, NZV2=nzvXGB2))

summary(ClassModelsGalaxy2) 
bwplot(ClassModelsGalaxy2, metric = "Accuracy")
densityplot(ClassModelsGalaxy2, metric = "Accuracy")


