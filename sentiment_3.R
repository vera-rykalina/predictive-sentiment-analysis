#### Load Packages  #####

pacman:: p_load("doParallel", "rstudioapi", "readr","dplyr", "tidyr", "ggplot2", "plotly", "GGally","data.table", "caret", "randomForest","hablar", "wesanderson", "RColorBrewer", "ggsci", "gplots","ggpubr", "C50", "class", "rminer","kernlab", "plyr", "xgboost", "gbm")


#### Setting Working Directory ####
current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)
getwd() # resulted in /Users/vera/Desktop/DA_Online/Big Data/Task3

rfe.iphone<-readRDS(file = "rfe.iphone.rds")
rfe.galaxy<-readRDS(file = "rfe.galaxy.rds")
str(rfe.iphone)
str(rfe.galaxy)

## Recode sentiment to combine factor levels
rfe.iphone$iphonesentiment <- recode(rfe.iphone$iphonesentiment, 
                                     "0" = "N", 
                                     "1" = "N", 
                                     "2" = "N", 
                                     "3" = "P", 
                                     "4" = "P", 
                                     "5" = "P") 

rfe.galaxy$galaxysentiment <- recode(rfe.galaxy$galaxysentiment, 
                                     "0" = "N", 
                                     "1" = "N", 
                                     "2" = "N", 
                                     "3" = "P", 
                                     "4" = "P", 
                                     "5" = "P") 

## Change dependent variables' data type
str(rfe.iphone)
str(rfe.galaxy)

# iPhone RFE
rc.rfe.intrain1  <- createDataPartition(
  y = rfe.iphone$iphonesentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

rc.rfe.iphonetrain <- rfe.iphone[rc.rfe.intrain1,]
rc.rfe.iphonetest <- rfe.iphone[-rc.rfe.intrain1,]


# Galaxy RFE
rc.rfe.intrain2  <- createDataPartition(
  y = rfe.galaxy$galaxysentiment,
  p = .7,
  list = FALSE, # avoids returns data as list 
  times = 1) # can create multiple splits at once

rc.rfe.galaxytrain <- rfe.galaxy[rc.rfe.intrain2,]
rc.rfe.galaxytest <- rfe.galaxy[-rc.rfe.intrain2,]

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
tune.grid.xgb1 <- expand.grid(eta=0.1,
                             nrounds=80,
                             max_depth=7,
                             min_child_weight =3.0,
                             colsample_bytree = 0.9,
                             gamma=0,
                             subsample=1)

## iPhone RFE Recoded
rc.rfeXGB1 <- train(iphonesentiment ~ ., 
                 rc.rfe.iphonetrain,
                 method = "xgbTree",
                 tuneGrid = tune.grid.xgb1,
                 trControl = XGBcontrol)

rc.rfeXGB1
rc.rfeXGB1$bestTune
pred.rc.rfeXGB1  <- predict(rc.rfeXGB1 , rc.rfe.iphonetest)
rc.rfeXGB1metrics <- postResample(pred.rc.rfeXGB1 , rc.rfe.iphonetest$iphonesentiment)
rc.rfeXGB1metrics
varImp(rc.rfeXGB1)
rc.RFEcmXGBiphone <- confusionMatrix(pred.rc.rfeXGB1, rc.rfe.iphonetest$iphonesentiment) 
rc.RFEcmXGBiphone

saveRDS(rc.rfeXGB1, file = "rc.rfeXGB1.rds")


## Galaxy RFE Recoded
## Set seed
set.seed(1234)
# xgbGrid
tune.grid.xgb2 <- expand.grid(eta=0.1,
                              nrounds=80,
                              max_depth=7,
                              min_child_weight =3.5,
                              colsample_bytree = 0.9,
                              gamma=0,
                              subsample=1)

rc.rfeXGB2 <- train(galaxysentiment ~ ., 
                    rc.rfe.galaxytrain,
                    method = "xgbTree",
                    tuneGrid = tune.grid.xgb2,
                    trControl = XGBcontrol)

rc.rfeXGB2
rc.rfeXGB2$bestTune
pred.rc.rfeXGB2  <- predict(rc.rfeXGB2 , rc.rfe.galaxytest)
rc.rfeXGB2metrics <- postResample(pred.rc.rfeXGB1 , rc.rfe.galaxytest$galaxysentiment)
rc.rfeXGB2metrics
varImp(rc.rfeXGB2)
rc.RFEcmXGBgalaxy <- confusionMatrix(pred.rc.rfeXGB2, rc.rfe.galaxytest$galaxysentiment) 
rc.RFEcmXGBgalaxy

saveRDS(rc.rfeXGB2, file = "rc.rfeXGB2.rds")



# Stop Cluster
stopCluster(cl)

#### Comparison iPhone/Galaxy

ClassModelFinal <- resamples(list(iPhone = rc.rfeXGB1, Galaxy=rc.rfeXGB2))

summary(ClassModelFinal) 
bwplot(ClassModelFinal, metric = "Accuracy")
densityplot(ClassModelFinal, metric = "Accuracy")



features1 <-colnames(rfe.iphone) # iPhone
features1 <-features1[1:18]

features2 <-colnames(rfe.galaxy) # iPhone
features2 <-features2[1:25]

raw.large.matrix <- read.csv("LargeMatrix.csv",TRUE,sep =",")
dim(raw.large.matrix)
colnames(raw.large.matrix)
str(raw.large.matrix)

large.iphone <- raw.large.matrix[features1]
str(large.iphone)

large.galaxy <- raw.large.matrix[features2]
str(large.galaxy)

#### Predictions ####
## Best performing models for both data is XGB
# iphone predictions
iphone.predictions <- predict(rc.rfeXGB1, newdata = large.iphone)
summary(iphone.predictions)
length(iphone.predictions)

# probability comparison
prop.table(table(iphone.predictions))
prop.table(table(rfe.iphone$iphonesentiment))


# galaxy predictions
galaxy.predictions <- predict(rc.rfeXGB2, newdata = large.galaxy)
summary(galaxy.predictions)

# probability comparison
prop.table(table(galaxy.predictions))
prop.table(table(rfe.galaxy$galaxysentiment))


sentiment <- raw.large.matrix
str(sentiment)
sentiment$iphone.sentiment <-iphone.predictions
sentiment$galaxy.sentiment <-galaxy.predictions

is.data.frame(sentiment)
dim(sentiment)
sentiment$iphone.sentiment <- ifelse(sentiment$iphone.sentiment=="P", "Positive", "Negative")
sentiment$galaxy.sentiment <- ifelse(sentiment$galaxy.sentiment=="P", "Positive", "Negative")
colnames(sentiment)
head(sentiment)

sentiment.long <- sentiment %>%
  gather(Sentiment, Review, iphone.sentiment, galaxy.sentiment)

sentiment.long$Sentiment <- ifelse(sentiment.long$Sentiment=="iphone.sentiment", "iPhone", "Galaxy")

head(sentiment.long)

final.plot <- sentiment.long %>% 
  group_by(Sentiment) %>%
  ggplot() +
  geom_bar(aes(x = Sentiment, fill=Review)) +
  labs(y="Counts", x="Model") +
  scale_fill_manual(values = pal) +
  #coord_flip() +
  theme_minimal()+
  theme(text=element_text(size=16, color = "black"))+ 
  theme(axis.text.x = element_text(color="black",size=12),axis.text.y = element_text(color="black",size=12)) +
  theme(legend.position="top", legend.title=element_blank())

# https://chrisalbon.com/
# www.geeksforgeeks.org
