rm(list = ls())

#Importing the libraries
library(ggplot2)
library(caret)
library(DMwR)
library(rpart)
library(randomForest)
library(inTrees)
#install.packages("inTrees")

setwd("C:/Users/vishal/")
# loading datasets
df = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
test = read.csv("test.csv")

# Structure of data
str(df)
str(test)
head(df,5)
head(test,5)
summary(df)
summary(test)

#Converting the data into numeric
df$fare_amount = as.numeric(as.character(df$fare_amount))

#Discovering the missing values
missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
# fare_amount 25
# pickup_datetime 0
# pickup_longitude 0
# pickup_latitude 0
# dropoff_longitude 0
# dropoff_latitude 0
# passenger_count 55


#Filling na values
df$passenger_count[is.na(df$passenger_count)] <- median(df$passenger_count, na.rm=TRUE)
df=df[complete.cases(df), ]


#Removing the  Outliers and plotting them
df[which(df$fare_amount < 1 ),]
df=df[-which(df$fare_amount < 1 ),]

df[which(df$passenger_count <1 ),]
df[which(df$passenger_count >6 ),]
df=df[-which(df$passenger_count <1 ),]
df=df[-which(df$passenger_count >6 ),]

pl1 = ggplot(df,aes(x = factor(passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

#Removing outliers using boxplot
vals = df[,"fare_amount"] %in% boxplot.stats(df[,"fare_amount"])$out
df[which(vals),"fare_amount"] = NA
df=df[complete.cases(df), ]


#Removing outliers of longitude and latitude using the test data to find the range
min_long = -74.263242
min_lat = 40.573143
max_long = -72.986532
max_lat = 41.709555

df=df[which((df$pickup_longitude >= min_long) & (df$pickup_longitude <= max_long )& (df$dropoff_longitude >= min_long) & (df$dropoff_longitude <= max_long )),]



###Feature Extraction####

df['longitude_diff']=abs(df$pickup_longitude-df$dropoff_longitude)
df['latitude_diff']=abs(df$pickup_latitude-df$dropoff_latitude)

df = df[which(df$longitude_diff > 0.0) & (df$longitude_diff < 5.0) & (df$latitude_diff > 0.0) & (df$latitude_diff < 5.0),]


#Splitting the date_time into components
df$pickup_date = as.Date(as.character(df$pickup_datetime))
df$pickup_weekday = as.factor(format(df$pickup_date,"%u"))# Monday = 1
df$pickup_mnth = as.factor(format(df$pickup_date,"%m"))
df$pickup_yr = as.factor(format(df$pickup_date,"%Y"))
pickup_time = strptime(df$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df$pickup_hour = as.factor(format(pickup_time,"%H"))




###Calculating the Haversine distance
deg_to_rad = function(deg){
  (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
  #long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  #long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}

df$dist = haversine(df$pickup_longitude,df$pickup_latitude,df$dropoff_longitude,df$dropoff_latitude)
df=df[complete.cases(df), ]
df = subset(df,select = -c(pickup_datetime))


###Splitting the dataset into train and validation#####

set.seed(1000)
tr.idx = createDataPartition(df$fare_amount,p=0.80,list = FALSE) # 80% in trainin and 20% in Validation Datasets
train_data = df[tr.idx,]
test_data = df[-tr.idx,]


par('mar')
par(mar=c(1,1,1,1))


##Applying the models
################Linear Regression###############

lm_model = lm(fare_amount ~.,data=train_data)

summary(lm_model)
str(train_data)
plot(lm_model$fitted.values,rstandard(lm_model),main = "Residual plot",
     xlab = "Predicted values of fare_amount",
     ylab = "standardized residuals")


lm_predictions = predict(lm_model,test_data[,2:14])

qplot(x = test_data[,1], y = lm_predictions, data = test_data, color = I("blue"), geom = "point")
d=test_data[,1]-lm_predictions
regr.eval(test_data[,1],lm_predictions)
#mae          mse         rmse         mape 
#2.5253237  23.1446664   4.8108904   0.3339379 



############Decision Tree###########
Dt_model = rpart(fare_amount ~ ., data = train_data, method = "anova")

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data[,2:14])

qplot(x = test_data[,1], y = predictions_DT, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],predictions_DT)
#mae          mse          rmse        mape 
#1.6768773 5.4294679 2.3301219 0.1991693 





#############Random forest#####################
rf_model = randomForest(fare_amount ~.,data=train_data)

summary(rf_model)

rf_predictions = predict(rf_model,test_data[,2:14])

qplot(x = test_data[,1], y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],rf_predictions)
#mae       mse      rmse      mape 
#1.3980513 3.8907723 1.9725041 0.1716425 