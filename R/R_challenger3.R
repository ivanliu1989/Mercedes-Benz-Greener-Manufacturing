rm(list = ls()); gc()
library(caret)
library(data.table)
load(file = './data/featureSelection_20170709.RData')
load(file = "../Common Data/data20170709.RData")
load(file = "./data/testIDs.RData")
setDT(all)
final.feature = final.feature[-486]

# Data scale --------------------------------------------------------------
all[,xgbStack := NULL]

head(all)
all[, IDScale := ID]
setDF(all)
for(i in 3:ncol(all)){
    all[,i] = (all[, i] - min(all[,i]))/(max(all[,i]) - min(all[, i]))
}
final.feature = c(final.feature, 'IDScale', 'y')

setDF(all)
all = all[, final.feature]

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
    eta = 0.8,
    nthread = 4,
    objective = "reg:linear",
    eval_metric=r2squared_xgb_feval,
    eval_metric='rmse',
    booster = "gblinear",
    gamma = 0.001,
    min_child_weight = 1,
    subsample = 0.3,
    colsample_bytree = 0.3,
    lambda = 0.0001,
    base_score = mean(train.full$y),
    alpha = 10
)

dtrain <- xgb.DMatrix(data.matrix(train.full[, predictors]), label = train.full[, response])
xgbFit = xgb.cv(data = dtrain, nrounds = 15000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 10, verbose = 1, maximize =T, prediction = T)
r2 = 1 - (sum((train.full[, response]-xgbFit$pred)^2) / sum((train.full[, response]-mean(train.full[, response]))^2))
print(r2)

xgbStacking = xgbFit$pred

# Submit ------------------------------------------------------------------
xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration, print_every_n = 100, verbose = 1, maximize =T)
var.imp = xgb.importance(colnames(dtrain), model = xgbFit)

setDF(test.raw)
dtest <- xgb.DMatrix(data.matrix(test.raw[, predictors]), label = test.raw[, response])
pred = predict(xgbFit, dtest, xgbFit$bestInd)
submit = data.frame(ID = test.raw$ID, y = pred)


# Stacking ----------------------------------------------------------------
xgbStacking = c(xgbStacking, pred)
xgbStacking = data.frame(ID = c(train.full$ID, test.raw$ID), xgbStack = xgbStacking)
setDT(xgbStacking)
xgbLinear = xgbStacking[!duplicated(xgbStacking$ID), ]

dim(all)
setDT(all)
all = merge(all, xgbLinear, by = 'ID', all.x = T)


colnames(all) = c(colnames(all)[1:487], 'xgbLinear1','xgbLinear2','xgbLinear3','xgbLinear4','xgbLinear5')
all[, xgbLinearSD := sd(c(xgbLinear1,xgbLinear2,xgbLinear3,xgbLinear4,xgbLinear5)), by = ID]



# Modeling ----------------------------------------------------------------
test.raw = all[ID %in% test.ID, ]
setDF(all)
test.full = all[is.na(all$y), ]
train.full = all[!is.na(all$y), ] #  & all$y < 200
predictors = colnames(all)[!colnames(all)%in%c('ID', 'y', 'xgbLinear1','xgbLinear2','xgbLinear3','xgbLinear4','xgbLinear5')]
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
    nthread = 4,
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
xgbFit = xgb.cv(data = dtrain, nrounds = 15000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 10, verbose = 1, maximize =T, prediction = T)
r2 = 1 - (sum((train.full[, response]-xgbFit$pred)^2) / sum((train.full[, response]-mean(train.full[, response]))^2))
print(r2)

# Submit ------------------------------------------------------------------
xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration, print_every_n = 100, verbose = 1, maximize =T)
var.imp = xgb.importance(colnames(dtrain), model = xgbFit)

setDF(test.raw)
dtest <- xgb.DMatrix(data.matrix(test.raw[, predictors]), label = test.raw[, response])
pred = predict(xgbFit, dtest, xgbFit$bestInd)
submit = data.frame(ID = test.raw$ID, y = pred)
