## Sentiment Analysis towards smartphone 

# Objective: perform a sentiment analysis for mobile devices by counting words associated with sentiment appearing within relevant documents on the web. Then, leverage this data to look for patterns in the documents that enable us to label each of these documents with a value that represents the level of positive or negative sentiment toward each of the selected devices.

# Data: a cloud computing platform provided by Amazon Web Services (AWS) is used to conduct the analysis; the data sets analyzed come from Common Crawl.

# R version 3.6.2 (2019-12-12)                                                  # # Date August 2020


#### Load Packages  #####

pacman:: p_load("doParallel", "rstudioapi", "readr","dplyr", "tidyr", "ggplot2", "plotly", "GGally","data.table", "caret", "randomForest","hablar", "wesanderson", "RColorBrewer", "ggsci", "gplots","ggpubr", "C50", "class", "rminer","kernlab", "plyr", "xgboost", "gbm", "corrplot")


#### Setting Working Directory ####
current_working_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(current_working_dir)
getwd() # resulted in /Users/vera/Desktop/DA_Online/Big Data/Task3


#### Loading datasets ####

# Reading Small Matrix (train data) for iPhone ####
raw.iphone <- read.csv("iphone_smallmatrix_labeled_8d.csv",TRUE,sep =",")
head(raw.iphone)
str(raw.iphone) # overall sentiment toward the device on a scale of 0-5 (last column)
summary(raw.iphone)
dim(raw.iphone) # 12973 rows x 59 columns
colnames(raw.iphone)

# Reading Small Matrix (train data) for Galaxy ####
raw.galaxy <- read.csv("galaxy_smallmatrix_labeled_8d.csv",TRUE,sep =",")
head(raw.galaxy)
str(raw.galaxy) # overall sentiment toward the device on a scale of 0-5 (last column)
summary(raw.galaxy)
dim(raw.galaxy) # 12973 rows x 59 columns
colnames(raw.galaxy)



# Save an object to a file
saveRDS(raw.iphone, file = "raw.iphone.rds")
getwd()
ls()

# Save an object to a file
saveRDS(raw_galaxy, file = "raw.galaxy.rds")
getwd()
ls()

#### Data exploration ####
## iPhone dataset

# Add review rating
iphone.matrix <- mutate(raw.iphone, review = ifelse(iphonesentiment ==0, "Very negative", ifelse(iphonesentiment ==1, "Negative", ifelse(iphonesentiment == 2, "Somewhat negative", ifelse(iphonesentiment==3, "Somewhat positive", ifelse(iphonesentiment==4, "Positive", "Very positive"))))))

# Check counts
table(iphone.matrix$iphonesentiment)

# Visualize counts in relation to ratings 
pal <- wes_palette(6, name = "Cavalcanti1", type = "continuous")
iphone.matrix.bar <- iphone.matrix %>% 
  #filter(iphone != 0) %>% 
  group_by(iphonesentiment) %>%
  ggplot(aes(x = factor(review, levels = c("Very negative", "Negative", "Somewhat negative", "Somewhat positive", "Positive", "Very positive")))) +
  geom_bar(aes(fill = factor(review, levels = c("Very negative", "Negative", "Somewhat negative", "Somewhat positive", "Positive", "Very positive")))) + labs(y="Counts", x="Review categories") +
  scale_y_continuous(limits = c(0, 8500)) +
  scale_fill_manual(values = pal) +
  coord_flip() +
  theme_minimal()+
  theme(text=element_text(size=16, color = "black"))+ 
  theme(axis.text.x = element_text(color="black",size=12),axis.text.y = element_text(color="black",size=12)) +
  theme(legend.position="none", legend.title=element_blank())

## Galaxy dataset
# Add review rating
galaxy.matrix <- mutate(raw.galaxy, review = ifelse(galaxysentiment ==0, "Very negative", ifelse(galaxysentiment ==1, "Negative", ifelse(galaxysentiment == 2, "Somewhat negative", ifelse(galaxysentiment==3, "Somewhat positive", ifelse(galaxysentiment==4, "Positive", "Very positive"))))))

# Check counts
table(galaxy.matrix$galaxysentiment)

# Visualize counts in relation to ratings 
pal <- wes_palette(6, name = "Cavalcanti1", type = "continuous")
galaxy.matrix.bar <- galaxy.matrix %>% 
  #filter(samsunggalaxy!= 0) %>% 
  group_by(galaxysentiment) %>%
  arrange(galaxysentiment) %>%
  ggplot(aes(x = factor(review, levels = c("Very negative", "Negative", "Somewhat negative", "Somewhat positive", "Positive", "Very positive")))) +
  geom_bar(aes(fill = factor(review, levels = c("Very negative", "Negative", "Somewhat negative", "Somewhat positive", "Positive", "Very positive")))) + 
  labs(y="Counts", x="Review categories") +
  scale_y_continuous(limits = c(0, 8500)) +
  scale_fill_manual(values = pal) +
  coord_flip() +
  theme_minimal()+
  theme(text=element_text(size=16, color = "black"))+ 
  theme(axis.text.x = element_text(color="black",size=12),axis.text.y = element_text(color="black",size=12)) +
  theme(legend.position="none", legend.title=element_blank()) 
  

## Detect NA's
any(is.na(iphone.matrix)) # Result = 0
any(is.na(galaxy.matrix)) # Result = 0


#### Preparation of datasets ####
#### Raw dataset ####
# Save an object to a file
saveRDS(raw.iphone, file = "raw.iphone.rds")
getwd()
ls()

# Save an object to a file
saveRDS(raw_galaxy, file = "raw.galaxy.rds")
getwd()
ls()

#### Finding Near Zero Variances ####
nzv.metrics.iphone <- nearZeroVar(raw.iphone, saveMetrics = T)
nzv.metrics.iphone # FALSE 12
nzv.iphone <- nearZeroVar(raw.galaxy, saveMetrics = FALSE) 
count(nzv.iphone) #  result 47


nzv.metrics.galaxy <- nearZeroVar(raw.galaxy, saveMetrics = T)
nzv.metrics.galaxy # FALSE 12
nzv.galaxy <- nearZeroVar(raw.galaxy, saveMetrics = FALSE) 
count(nzv.galaxy) # result 47


# Creating a new data set without near zero variance
nzv.iphone <- raw.iphone[,-nzv.iphone]
str(nzv.iphone)
head(nzv.iphone)

# Creating a new data set without near zero variance
nzv.galaxy <- raw.galaxy[,-nzv.galaxy]
str(nzv.galaxy)
head(nzv.galaxy)

# Save an object to a file
saveRDS(nzv.iphone, file = "nzv.iphone.rds")
saveRDS(nzv.galaxy, file = "nzv.galaxy.rds")


#### Recursive Feature Elimination (RFE) ####
## Parallel computing
# Find how many cores are on your machine
detectCores() # Result = 8
## Create cluster with desired number of cores.
cl <- makeCluster(4)
## Register cluster
registerDoParallel(cl)
## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

## Set seed
set.seed(1234)

## Let's sample the data before using RFE

iphoneSample <- raw.iphone[sample(1:nrow(raw.iphone), 1000, replace=FALSE),]
galaxySample <- raw.galaxy[sample(1:nrow(raw.galaxy), 1000, replace=FALSE),]

## Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

## Use rfe and omit the response variable (attribute 59 iphonesentiment & galaxysentiment) 
rfeResults1 <- rfe(iphoneSample[,1:58], 
                   iphoneSample$iphonesentiment, 
                   sizes = (1:58), 
                   rfeControl = ctrl)

rfeResults2 <- rfe(galaxySample[,1:58], 
                   galaxySample$galaxysentiment, 
                   sizes = (1:58), 
                   rfeControl = ctrl)
## Stop Cluster
stopCluster(cl)

## Get results
rfeResults1
predictors(rfeResults1)

rfeResults2
predictors(rfeResults2)

## Plot results
plot(rfeResults1, type=c("g", "o")) # iphone
plot(rfeResults2, type=c("g", "o")) # galaxy

## Create new data set with rfe recommended features
rfe.iphone <- raw.iphone[,predictors(rfeResults1)]
rfe.galaxy<- raw.galaxy[,predictors(rfeResults2)]

## Add the dependent variable to iphoneRFE & galaxyRFE
rfe.iphone$iphonesentiment <- raw.iphone$iphonesentiment
rfe.galaxy$galaxysentiment <- raw.galaxy$galaxysentiment

## Review outcome
str(rfe.iphone)
str(rfe.galaxy)

# Changing type of variables
str(rfe.iphone)
summary(rfe.iphone)

str(rfe.galaxy)
summary(rfe.galaxy)

rfe.iphone$iphonesentiment <- as.factor(rfe.iphone$iphonesentiment)
rfe.galaxy$galaxysentiment <- as.factor(rfe.galaxy$galaxysentiment)

# Save an object to a file
saveRDS(rfe.iphone, file = "rfe.iphone.rds")
saveRDS(rfe.galaxy, file = "rfe.galaxy.rds")


## Increase max print 
options(max.print = 1000000)

## Check correlations
# iPhone
cor(raw.iphone)
ggcorr(raw.iphone)
corrplot(cor(raw.iphone))
correlationMatrix1 <- cor(raw.iphone)
print(correlationMatrix1)
# find attributes that are highly corrected (ideally > 0.75)
highlyCorrelated1 <- findCorrelation(correlationMatrix1, cutoff=0.5)
# print indexes of highly correlated attributes
count(print(highlyCorrelated1))
cor.iphone <- raw.iphone[,-highlyCorrelated1]
dim(cor.iphone)

# Galaxy
cor(raw.galaxy)
ggcorr(raw.galaxy)
corrplot(cor(raw.galaxy))
correlationMatrix2 <- cor(raw.galaxy)
print(correlationMatrix2)
# find attributes that are highly corrected (ideally > 0.75)
highlyCorrelated2 <- findCorrelation(correlationMatrix2, cutoff=0.5)
# print indexes of highly correlated attributes
count(print(highlyCorrelated2))
cor.galaxy <- raw.galaxy[,-highlyCorrelated2]
dim(cor.galaxy)



cor.iphone$iphonesentiment <- as.factor(cor.iphone$iphonesentiment)
cor.galaxy$galaxysentiment <- as.factor(cor.galaxy$galaxysentiment)

# Save an object to a file
saveRDS(cor.iphone, file = "cor.iphone.rds")
saveRDS(cor.galaxy, file = "cor.galaxy.rds")




# Removing rows with no iphone and galaxy observations
# Column 1-5 represents the number of instances that type of phone mentioned in a webpage

iphone.df <- iphone.matrix %>%
  filter(iphone != 0) %>% 
  select(starts_with("ios"), starts_with("iphone"))

galaxy.df <- galaxy.matrix %>%
  filter(samsunggalaxy != 0) %>% 
  select(starts_with("google"), starts_with("samsung"), starts_with("galaxy"))

## Dependent variable visualizations
plot_ly(iphone.df, x= ~iphone.df$iphonesentiment, type='histogram')
plot_ly(galaxy.df, x= ~galaxy.df$galaxysentiment, type='histogram')

