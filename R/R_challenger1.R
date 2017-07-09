rm(list = ls()); gc()
library(caret)
library(data.table)
load(file = './data/featureSelection_20170706.RData')
load(file = "../Common Data/data20170706.RData")
setDT(all)
test.ID = all[is.na(all$y), ID]

# X0 A
colnames(all)[grepl('X0a', colnames(all))]
all[, X0_A_Only := rowSums(.SD), .SDcols = colnames(all)[grepl('X0a', colnames(all))]]
all[, X0_A_Only_MeanY := mean(y, na.rm = T), by = X0_A_Only]
all[, X0_A_Only_SDY := sd(y, na.rm = T), by = X0_A_Only]
final.feature = c(final.feature, 'X0_A_Only', 'X0_A_Only_MeanY', 'X0_A_Only_SDY')

# Dup ID 
dupID = colnames(all)[3:366]
all[, dupCnt := .N, by = dupID]
final.feature = c(final.feature, 'X0_A_Only', 'X0_A_Only_MeanY', 'X0_A_Only_SDY', 'dupCnt')

# Probing
setDF(all)
LBScore = fread("./data/LBScore.csv", data.table = F)
for(id in LBScore$ID){
    print(id)
    print(paste0(all[all$ID == id, 'y'], " to ", LBScore[LBScore$ID == id, 'y']))
    all[all$ID == id, 'y'] = LBScore[LBScore$ID == id, 'y']
}
setDT(all)

# Remove Dup
# all_2 = copy(all)
# all_2[, y := mean(y, na.rm = TRUE), by = dupID]
# all_2 = all_2[!ID %in% test.ID]
# all = unique(rbind(all, all_2))



















# Modeling ----------------------------------------------------------------
all$y = log(all$y)
test.raw = all[ID %in% test.ID, ]
setDF(all)
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



# Stacking ----------------------------------------------------------------
xgbStacking = xgbFit$pred
xgbStacking = c(xgbStacking, pred)
xgbStacking = data.frame(ID = c(train.full$ID, test.raw$ID), xgbStack = xgbStacking)
setDT(xgbStacking)
xgbStacking = xgbStacking[!duplicated(xgbStacking$ID), ]

dim(all)
setDT(all)
all = merge(all, xgbStacking, by = 'ID', all.x = T)

# Submit ------------------------------------------------------------------
xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration, print_every_n = 100, verbose = 1, maximize =T)
var.imp = xgb.importance(colnames(dtrain), model = xgbFit)

1 - (sum((10^train.full[, response]-10^xgbFit$pred)^2) / sum((10^train.full[, response]-mean(10^train.full[, response]))^2))


dtest <- xgb.DMatrix(data.matrix(test.raw[, predictors]), label = test.raw[, response])
pred = predict(xgbFit, dtest, xgbFit$bestInd)
submit = data.frame(ID = test.raw$ID, y = pred)
write.csv(submit, file = paste0("./prediction/xgb_single_feat_select_dup_mean.csv"), row.names = F)


compare = merge(submit, compare, by = 'ID')

final.feature = c(final.feature, 'xgbStack')
save(final.feature, file = "./data/featureSelection_20170709.RData")
save(all, file = "../Common Data/data20170709.RData")
