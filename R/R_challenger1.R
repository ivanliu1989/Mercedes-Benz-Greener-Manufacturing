rm(list = ls()); gc()
library(caret)
library(data.table)
load(file = './data/featureSelection_20170706.RData')
load(file = "../data20170706.RData")
LBScore = fread("./data/LBScore.csv", data.table = F)
for(id in LBScore$ID){
    print(id)
    print(paste0(all[all$ID == id, 'y'], " to ", LBScore[LBScore$ID == id, 'y']))
    all[all$ID == id, 'y'] = LBScore[LBScore$ID == id, 'y']
}




# Modeling ----------------------------------------------------------------
test.full = all[is.na(all$y), ]
train.full = all[!is.na(all$y), ] #  & all$y < 200
predictors = final.feature
response = 'y'

# xgboost -----------------------------------------------------------------
library(xgboost)
r2squared_xgb_feval <- function(pred, dtrain) {
    pred <- as.numeric(pred)
    actual <- as.numeric(getinfo(dtrain, "label"))
    r2 = 1 - (sum((actual-pred)^2) / sum((actual-mean(actual))^2))
    return (list(metric = 'r2', value = r2))
}
param <- list(
    max_depth = 2, 
    eta = 0.005,
    nthread = 7,
    objective = "reg:linear",
    eval_metric=r2squared_xgb_feval,
    eval_metric='rmse',
    booster = "gbtree",
    gamma = 0.001,
    min_child_weight = 1,
    subsample = 0.92,
    colsample_bytree = 0.9,
    lambda = 0.0001,
    base_score = mean(train.full$y),
    alpha = 10
)

dtrain <- xgb.DMatrix(data.matrix(train.full[, predictors]), label = train.full[, response])
xgbFit = xgb.cv(data = dtrain, nrounds = 15000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 100, verbose = 1, maximize =T, prediction = T)
r2 = 1 - (sum((train.full[, response]-xgbFit$pred)^2) / sum((train.full[, response]-mean(train.full[, response]))^2))
print(r2)
xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration, print_every_n = 100, verbose = 1, maximize =T)
var.imp = xgb.importance(colnames(dtrain), model = xgbFit)



dtest <- xgb.DMatrix(data.matrix(test.full[, predictors]), label = train.full[, response])