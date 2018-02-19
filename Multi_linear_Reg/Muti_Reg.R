
# Importing the dataset
dataset = read.csv('50_Startups.csv')

# label Encoding 
dataset$State = factor(dataset$State,
                         levels = c('New York','California','Florida'),
                         labels = c(1,2,3))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#regressor = lm(formula = Profit ~ R.D.Spend + Adminstration + Marketting.Spend + State)
regressor = lm(formula = Profit ~ .,
               data = training_set) # select all independet variable
#summary(regressor) in console

y_pred = predict(regressor, newdata = test_set)
#y_pred in console
