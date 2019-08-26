## ----------------------------
## This file returns the key predictors of outcome using:
#ridge
#lasso
#elasticnet
#randomforest
## ----------------------------

## ----------------------------
## Load packages
## ----------------------------
rm(list=ls(all=TRUE))
vec.pac= c("foreign", "quantreg", "gbm", "glmnet",
           "MASS", "rpart", "doParallel", "sandwich", "randomForest",
           "nnet", "matrixStats", "xtable", "readstata13", "car", "lfe", "doParallel",
           "caret", "foreach", "multcomp","cowplot")

lapply(vec.pac, require, character.only = TRUE) 

ptm <- proc.time()
set.seed(123);

## ----------------------------
## Load data
## ----------------------------
load("../../DATA/fdat.rda") 
nrow(fdat1)


## --------------------------------------
## Data prep
# all variables as numeric
# y should be centered
# x should be standardised
# x that are categorical as dummies
## --------------------------------------

y.var <- ifelse(fdat1$you_buy == "no", 1, 0)

y.var <- scale(y.var, center = TRUE, scale = FALSE)

x.var <- model.matrix(~ group 
                      , fdat1)[,-1]

x.var <- scale(x.var, center = FALSE, scale = TRUE)

nrow(y.var)
nrow(x.var)


## --------------------------------------
## Lasso #ridge=0; lasso=1; elasticnet=0.5
## --------------------------------------
alpha_input <- 0.2

# Perform 10-fold cross-validation to select lambda 
lambdas_to_try <- 10^seq(-3, 5, length.out = 100)

# Setting alpha = 0 implements ridge regression
cv <- cv.glmnet(x.var, y.var, alpha = 0, lambda = lambdas_to_try, family="binomial",
                      standardize = FALSE, nfolds = 10) 

# Plot cross-validation results
plot(cv)

# Best cross-validated lambda
lambda_cv <- cv$lambda.min

# Fit final model
model_cv <- glmnet(x.var, y.var, alpha = alpha_input, lambda = lambda_cv, 
                   standardize = FALSE)
summary(model_cv)

# get its sum of squared residuals and multiple R-squared
y_hat_cv <- predict(model_cv, x.var)
ssr_cv <- t(y.var - y_hat_cv) %*% (y.var - y_hat_cv)
rsq_cv <- cor(y.var, y_hat_cv)^2

##Feature
varImp(model_cv,lambda_cv)
rsq_cv 
coef(model_cv)



## ----------------------------
###RANDOM FOREST
## # Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
## ----------------------------
set.seed(100)
train <- sample(nrow(fdat1), 0.7*nrow(fdat1), replace = FALSE)
TrainSet <- fdat1[train,]
ValidSet <- fdat1[-train,]
nrow(TrainSet)
nrow(ValidSet)


mm <- model.matrix(~ group 
                   , TrainSet)



## ----------------------------
## # Create a Random Forest model with default parameters
## ----------------------------

model2 <- randomForest(mm, y=as.factor(TrainSet$y.var), data = TrainSet, 
                       ntree = 300, mtry = 10, importance = TRUE)

model2

importance    <- importance(model2)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

#Create a rank variable based on importance
library(magrittr)
library(dplyr)

rankImportance <- varImportance %>% mutate(Rank = paste0('#',dense_rank(desc(Importance))))

df <-rankImportance[order(rankImportance$Importance),]
df





