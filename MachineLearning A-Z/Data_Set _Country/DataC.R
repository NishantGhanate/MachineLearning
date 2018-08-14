# Data Preprocessing 

# Importing the datasetfile , akeep R and csv in same file 
#cat("\014")  clear console

dataset = read.csv('Data.csv')
print(getwd()) # Where does the code think it is?

#selecting Age coulmn in dataset ave= average fucntion 
dataset$Age = ifelse(is.na(dataset$Age),
                    ave(dataset$Age,FUN = function(x) mean(x,na.rm = TRUE) ),
                    dataset$Age)
    
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary,FUN = function(x) mean(x,na.rm = TRUE) ),
                     dataset$Salary)

#encoding country column to int labels
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No','Yes'),
                         labels = c(0,1))

#Splitting data to train and test 
#install.packages('caTools') # enter ctrl + enter
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set = subset(dataset ,split ==TRUE)
test_set = subset(dataset ,split ==FALSE)

#Feautre Scaling 
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])


