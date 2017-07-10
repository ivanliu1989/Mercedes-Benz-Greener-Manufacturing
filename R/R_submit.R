rm(list = ls()); gc()
library(caret)
library(data.table)
load(file = './data/featureSelection_20170709.RData')
load(file = "../Common Data/data20170709.RData")
load(file = "./data/testIDs.RData")
setDT(all)
final.feature = final.feature[-486]
final.feature = colnames(all)[!colnames(all) %in% c('xgbStack', 'y')]

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
setDT(all)

# Modeling ----------------------------------------------------------------
test.raw = all[ID %in% test.ID, ]
setDF(all)
test.full = all[is.na(all$y), ]
train.full = all[!is.na(all$y), ] #  & all$y < 200
predictors = final.feature[!final.feature%in% c('ID', 'y')]
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
    gamma = 0.01,
    min_child_weight = 0,
    subsample = 0.9,
    colsample_bytree = 0.7,
    lambda = 0.0001,
    base_score = mean(train.full$y),
    alpha = 10
)

submit = data.frame(ID = test.raw$ID, y = NA)
submit_train = data.frame(ID = train.full$ID, y = NA)
scores = c()
 
for(i in 1:10){
    set.seed(9*i)
    print(i)
    dtrain <- xgb.DMatrix(data.matrix(train.full[, predictors]), label = train.full[, response])
    xgbFit = xgb.cv(data = dtrain, nrounds = 15000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 100, verbose = 1, maximize =T, prediction = T)
    r2 = 1 - (sum((train.full[, response]-xgbFit$pred)^2) / sum((train.full[, response]-mean(train.full[, response]))^2))
    print(r2)
    
    xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration, print_every_n = 100, verbose = 1, maximize =T)
    var.imp = xgb.importance(colnames(dtrain), model = xgbFit)
    
    
    # Feature select
    predictors2 = var.imp$Feature
    
    dtrain <- xgb.DMatrix(data.matrix(train.full[, predictors2]), label = train.full[, response])
    xgbFit = xgb.cv(data = dtrain, nrounds = 15000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 100, verbose = 1, maximize =T, prediction = T)
    r2 = 1 - (sum((train.full[, response]-xgbFit$pred)^2) / sum((train.full[, response]-mean(train.full[, response]))^2))
    print(r2)
    submit_train[, i] = xgbFit$pred
    
    xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration, print_every_n = 100, verbose = 1, maximize =T)
    var.imp = xgb.importance(colnames(dtrain), model = xgbFit)
    
    setDF(test.raw)
    dtest <- xgb.DMatrix(data.matrix(test.raw[, predictors2]), label = test.raw[, response])
    pred = predict(xgbFit, dtest, xgbFit$bestInd)
    
    submit[, i] = pred
    scores[i] = r2
    
}


submit2 = submit
submit2_train = submit_train

scores2 = (scores - min(scores))/(max(scores) - min(scores))
scores2 = scores2/sum(scores2)

for(i in 1:length(scores2)){
    submit2[, i] = submit[, i] * scores2[i]    
    submit2_train[, i] = submit_train[, i] * scores2[i]    
}

submit = data.frame(ID = test.raw$ID, y = rowSums(submit2))
submit_train = data.frame(ID = train.full$ID, y = rowSums(submit2_train))

write.csv(submit, file = paste0("./fnl_submit/ivan_xgb_05731_test.csv"), row.names = F)
write.csv(submit_train, file = paste0("./fnl_submit/ivan_xgb_05731_train.csv"), row.names = F)
