#Splitting data to train and test 
#setwd("G:/ProgramData/Anaconda_Projects/Salary_Prediction")
#install.packages('caTools') # enter ctrl + enter


dataset = read.csv('Salary_Data.csv')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3) #2/3 of value go to test 
training_set = subset(dataset ,split ==TRUE)
test_set = subset(dataset ,split ==FALSE)

#Feautre Scaling 
#training_set[,2:3] = scale(training_set[,2:3])
#test_set[,2:3] = scale(test_set[,2:3])

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
Y_pred = predict(regressor,newdata = test_set)

# Run only at a time 
#plotting graph 
library(ggplot2)
ggplot()+
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor,newdata = training_set)),
            colour = 'blue' ) +
  ggtitle('Salary vs Experience (Training set)')+
  xlab('Years of Experience')+
  ylab('Salary')

library(ggplot2)
ggplot()+
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor,newdata = training_set)),
            colour = 'blue' ) +
  ggtitle('Salary vs Experience (Testing set)')+
  xlab('Years of Experience')+
  ylab('Salary')
