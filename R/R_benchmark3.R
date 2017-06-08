rm(list = ls()); gc()
library(caret)
library(data.table)
f = list.files("./data/", full.names = T)
train.full = fread(f[3])
test = fread(f[2])
submit = fread(f[1])

# Encoding ----------------------------------------------------------------
test$y = NA
all = rbind(train.full, test)
setDF(all)
cat.feat = paste0("X", c(0:6,8))
num.feat = names(all)[!names(all) %in% c(cat.feat, 'ID', 'y')]
dummies <- dummyVars( ~ ., data = all[, cat.feat])
dummies = predict(dummies, newdata = all)


# Median
setDT(all)
all[, medX0 := median(y, na.rm = TRUE), by = X0]
all[, medX0Idx := medX0/median(y, na.rm = TRUE)]
all[, medX1 := median(y, na.rm = TRUE), by = X1]
all[, medX1Idx := medX1/median(y, na.rm = TRUE)]
all[, medX2 := median(y, na.rm = TRUE), by = X2]
all[, medX2Idx := medX2/median(y, na.rm = TRUE)]
all[, medX3 := median(y, na.rm = TRUE), by = X3]
all[, medX3Idx := medX3/median(y, na.rm = TRUE)]
all[, medX4 := median(y, na.rm = TRUE), by = X4]
all[, medX4Idx := medX4/median(y, na.rm = TRUE)]
all[, medX5 := median(y, na.rm = TRUE), by = X5]
all[, medX5Idx := medX5/median(y, na.rm = TRUE)]
all[, medX6 := median(y, na.rm = TRUE), by = X6]
all[, medX6Idx := medX6/median(y, na.rm = TRUE)]
all[, medX8 := median(y, na.rm = TRUE), by = X8]
all[, medX8Idx := medX8/median(y, na.rm = TRUE)]
setDF(all)

# Dummy
all[, cat.feat] = NULL
all = cbind(all, dummies)
cat.feat = names(all)[!names(all) %in% c(num.feat, 'ID', 'y')]
all.feat = names(all)[!names(all) %in% c('ID', 'y')]

# Interaction terms

# Cnt
all$Cnt = rowSums(all[, num.feat])

# tsne
# library(tsne)
# all.feat = names(all)[!names(all) %in% c('ID', 'y', 'Cnt')]
# tsne_out = tsne(all[, all.feat], k = 3)
# all$tsneX = tsne_out[, 1]
# all$tsneY = tsne_out[, 2]
# all$tsneZ = tsne_out[, 3]

# pca
library(caret)
preProcValues <- preProcess(all[, num.feat], method = c("pca"))
trainTransformed <- predict(preProcValues, all[, all.feat])
# ica
preProcValues <- preProcess(all[, num.feat], method = c("ica"), n.comp = 60)
trainTransformed.ica <- predict(preProcValues, all[, num.feat])


# Combine -----------------------------------------------------------------
all = cbind(all, trainTransformed, trainTransformed.ica)
# all$Inter.Cnt = inter.data$Inter.Cnt
save(all, file = "../train.RData")
load(file = "./train.RData")
# drop duplication
dim(all)
train <- all[, !duplicated(t(all))]
dim(train)



# Modeling ----------------------------------------------------------------
test.full = train[is.na(train$y), ]
train.full = train[!is.na(train$y), ]
predictors =colnames(train.full)[!colnames(train.full) %in% c('y')]
response = 'y'

# xgboost -----------------------------------------------------------------
library(xgboost)
r2squared_xgb_feval <- function(pred, dtrain) {
    pred <- as.numeric(pred)
    actual <- as.numeric(getinfo(dtrain, "label"))
    return (list(metric = 'r2', value = 1 - (sum((actual-pred)^2) / sum((actual-mean(actual))^2))))
}
param <- list(
    max_depth = 5, #6
    eta = 0.01,
    nthread = 7,
    objective = "reg:linear",
    eval_metric=r2squared_xgb_feval,
    booster = "gbtree",
    gamma = 0.1,
    min_child_weight = 10,
    subsample = 0.9,
    colsample_bytree = 0.9,
    lambda = 0.001,
    alpha = 100,
    seed = 1989
)


library(xgboost)
dtest <- xgb.DMatrix(data.matrix(test.full[, predictors]), label = test.full[, response])

for(i in 1:5){
    print(i)
    trainBC = train.full
    dtrain <- xgb.DMatrix(data.matrix(trainBC[, predictors]), label = trainBC[, response])
    xgbFit = xgb.cv(data = dtrain, nrounds = 1000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 20, verbose = 1, maximize =T)
    best_scr = paste0(round(tail(xgbFit$evaluation_log$test_r2_mean, 1),5), "_", round(tail(xgbFit$evaluation_log$test_r2_std, 1),5))
    xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration,print.every.n = 100, verbose = 1, maximize =T)
    
    pred = predict(xgbFit, dtest, xgbFit$bestInd)
    submit = data.frame(ID = test.full$ID, y = pred)
    write.csv(submit, file = paste0("./prediction/xgb_benchmark3_",best_scr,"_",i,".csv"), row.names = F)
    
}
























































# GBM
fitControl <- trainControl(## 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated ten times
    repeats = 10)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

nrow(gbmGrid)
set.seed(825)
gbmFit2 <- train(y ~ ., data = train.full, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = TRUE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid)
gbmFit2

ggplot(gbmFit2)  


# SVM
set.seed(825)
svmFit <- train(Class ~ ., data = training, 
                method = "svmRadial", 
                trControl = fitControl, 
                preProc = c("center", "scale"),
                tuneLength = 8,
                metric = "ROC")
svmFit       


# RDA
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           search = "random")

set.seed(825)
rda_fit <- train(Class ~ ., data = training, 
                 method = "rda",
                 metric = "ROC",
                 tuneLength = 30,
                 trControl = fitControl)
rda_fit



# Compare
resamps <- resamples(list(GBM = gbmFit3,
                          SVM = svmFit,
                          RDA = rdaFit))
resamps
summary(resamps)