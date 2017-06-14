# just train.data[, X0_EMBED := mean(y), by = X0]
#  f_weight * global mean + (1 - f_weight) * group 
#  xgb splits
# closest ID
# closest factor
# discard them.
# There are at least 4 peaks, eyeballed to be grouped around 75, 90, 100, and 110 seconds
# There is one data point at around 265 s
# Clustering based on ID

rm(list = ls()); gc()
library(caret)
library(data.table)
f = list.files("./data/", full.names = T)
train.full = fread(f[3])
test = fread(f[2])
submit = fread(f[1])


# Remove useless columns --------------------------------------------------
setDF(train.full)
setDF(test)
toRM = c()
for(i in 1:ncol(train.full)){
    if(length(unique(train.full[,i]))==1){
        print(names(train.full)[i])
        toRM = c(toRM, names(train.full)[i])
    }
}
train.full[, toRM] = NULL
adj.test = test[, toRM]
test[, toRM] = NULL
### Mannually add time for those columns!!!!!!!!!!!!!!!!!!!!



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
all[, medX0 := median(log(y), na.rm = TRUE), by = X0]
all[, medX1 := median(log(y), na.rm = TRUE), by = X1]
all[, medX2 := median(log(y), na.rm = TRUE), by = X2]
all[, medX3 := median(log(y), na.rm = TRUE), by = X3]
all[, medX4 := median(log(y), na.rm = TRUE), by = X4]
all[, medX5 := median(log(y), na.rm = TRUE), by = X5]
all[, medX6 := median(log(y), na.rm = TRUE), by = X6]
all[, medX8 := median(log(y), na.rm = TRUE), by = X8]

# mean
all[, meanX0 := mean(log(y), na.rm = TRUE), by = X0]
all[, meanX1 := mean(log(y), na.rm = TRUE), by = X1]
all[, meanX2 := mean(log(y), na.rm = TRUE), by = X2]
all[, meanX3 := mean(log(y), na.rm = TRUE), by = X3]
all[, meanX4 := mean(log(y), na.rm = TRUE), by = X4]
all[, meanX5 := mean(log(y), na.rm = TRUE), by = X5]
all[, meanX6 := mean(log(y), na.rm = TRUE), by = X6]
all[, meanX8 := mean(log(y), na.rm = TRUE), by = X8]

# min
all[, minX0 := min(log(y), na.rm = TRUE), by = X0]
all[, minX1 := min(log(y), na.rm = TRUE), by = X1]
all[, minX2 := min(log(y), na.rm = TRUE), by = X2]
all[, minX3 := min(log(y), na.rm = TRUE), by = X3]
all[, minX4 := min(log(y), na.rm = TRUE), by = X4]
all[, minX5 := min(log(y), na.rm = TRUE), by = X5]
all[, minX6 := min(log(y), na.rm = TRUE), by = X6]
all[, minX8 := min(log(y), na.rm = TRUE), by = X8]

# max
all[, maxX0 := max(log(y), na.rm = TRUE), by = X0]
all[, maxX1 := max(log(y), na.rm = TRUE), by = X1]
all[, maxX2 := max(log(y), na.rm = TRUE), by = X2]
all[, maxX3 := max(log(y), na.rm = TRUE), by = X3]
all[, maxX4 := max(log(y), na.rm = TRUE), by = X4]
all[, maxX5 := max(log(y), na.rm = TRUE), by = X5]
all[, maxX6 := max(log(y), na.rm = TRUE), by = X6]
all[, maxX8 := max(log(y), na.rm = TRUE), by = X8]

# sd
all[, sdX0 := sd(log(y), na.rm = TRUE), by = X0]
all[, sdX1 := sd(log(y), na.rm = TRUE), by = X1]
all[, sdX2 := sd(log(y), na.rm = TRUE), by = X2]
all[, sdX3 := sd(log(y), na.rm = TRUE), by = X3]
all[, sdX4 := sd(log(y), na.rm = TRUE), by = X4]
all[, sdX5 := sd(log(y), na.rm = TRUE), by = X5]
all[, sdX6 := sd(log(y), na.rm = TRUE), by = X6]
all[, sdX8 := sd(log(y), na.rm = TRUE), by = X8]
setDF(all)

# Fix
for(i in 1:ncol(all)){
    all[, i] = ifelse(is.infinite(all[,i]), NA, all[,i])
}


# Label Encoding ----------------------------------------------------------
letterwrap <- function(n, depth = 1) {
    args <- lapply(1:depth, FUN = function(x) return(LETTERS))
    x <- do.call(expand.grid, args = list(args, stringsAsFactors = F))
    x <- x[, rev(names(x)), drop = F]
    x <- do.call(paste0, x)
    if (n <= length(x)) return(x[1:n])
    return(c(x, letterwrap(n - length(x), depth = depth + 1)))
}

encoding.list = data.frame(from = tolower(letterwrap(120)), to = 1:120)

for(i in cat.feat){
    all[, i] = match(all[,i], encoding.list$from)
}

# Dummy
# all[, cat.feat] = NULL
# all = cbind(all, dummies)
# cat.feat = colnames(dummies)

# Interaction terms

# Cnt
all$Cnt = rowSums(all[, num.feat])

# pca
library(caret)
preProcValues <- preProcess(all[, c(num.feat, cat.feat)], method = c("pca"))
pca.feat <- predict(preProcValues, all[, c(num.feat, cat.feat)])#[, 1:100]
# ica
preProcValues <- preProcess(all[, c(num.feat, cat.feat)], method = c("ica"), n.comp = 60)
ica.feat <- predict(preProcValues, all[, c(num.feat, cat.feat)])
# svd
svd.feat = as.data.frame(svd(all[, c(num.feat, cat.feat)],nv = )$u)[, 1:100]
names(svd.feat) = paste0('SVD', 1:ncol(svd.feat))

# Gaussian Random Projection
# Sparse Random Projection

# tsne
# library(tsne)
# tsne_out = tsne(all[, c(num.feat, cat.feat)], k = 3)
# all$tsneX = tsne_out[, 1]
# all$tsneY = tsne_out[, 2]
# all$tsneZ = tsne_out[, 3]


# Combine -----------------------------------------------------------------
all = cbind(all, pca.feat, ica.feat, svd.feat)
dim(all)
train <- all#[, !duplicated(t(all))]
dim(train)



# Modeling ----------------------------------------------------------------
test.full = train[is.na(train$y), ]
train.full = train[!is.na(train$y), ]
predictors =colnames(train.full)[!colnames(train.full) %in% c('y')]
response = 'y'

# xgboost -----------------------------------------------------------------
library(xgboost)
r2squared_xgb_feval <- function(pred, dtrain) {
    N = round(4209*0.92)
    p = 854 # length(predictors)
    pred <- as.numeric(pred)
    actual <- as.numeric(getinfo(dtrain, "label"))
    r2 = 1 - (sum((actual-pred)^2) / sum((actual-mean(actual))^2))
    adjr2 = 1-((1-r2)*(N-1)/(N-p-1))
    # return (list(metric = 'r2', value = adjr2))
    return (list(metric = 'r2', value = r2))
}
param <- list(
    max_depth = 2, #6
    eta = 0.003,
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
    # base_score = mean(train.full$y),
    alpha = 10
)
# train-r2:0.616698+0.014985	test-r2:0.576366+0.063593

library(xgboost)
dtest <- xgb.DMatrix(data.matrix(test.full[, predictors]), label = test.full[, response])

# for(i in 1:25){
#     set.seed(i)
#     print(i)
#     trainBC = train.full
#     dtrain <- xgb.DMatrix(data.matrix(trainBC[, predictors]), label = trainBC[, response])
#     xgbFit = xgb.cv(data = dtrain, nrounds = 5000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 20, verbose = 1, maximize =T)
#     best_scr = paste0(round(tail(xgbFit$evaluation_log$test_r2_mean, 1),5), "_", round(tail(xgbFit$evaluation_log$test_r2_std, 1),5))
#     # best_scr = paste0(round(tail(xgbFit$evaluation_log$test_rmse_mean, 1),5), "_", round(tail(xgbFit$evaluation_log$test_rmse_std, 1),5))
#     xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration,print.every.n = 100, verbose = 1, maximize =F)
# 
#     pred = predict(xgbFit, dtest, xgbFit$bestInd)
#     submit = data.frame(ID = test.full$ID, y = pred)
#     write.csv(submit, file = paste0("./prediction/xgb_newfeat_",best_scr,"_",i,".csv"), row.names = F)
# 
# }

predictors.tmp = predictors[!grepl('Idx', predictors)]
# predictors.tmp = predictors.tmp[! predictors.tmp %in% num.feat]
trainBC = train.full
dtrain <- xgb.DMatrix(data.matrix(trainBC[, predictors.tmp]), label = trainBC[, response])
xgbFit = xgb.cv(data = dtrain, nrounds = 5000, nfold = 5, param, print_every_n = 100, early_stopping_rounds = 100, verbose = 1, maximize =T)
xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration,print.every.n = 100, verbose = 1, maximize =T)
var.imp = xgb.importance(colnames(dtrain), model = xgbFit)
xgb.plot.importance(var.imp, top_n = 20)


# Submissions -------------------------------------------------------------
files = list.files("./prediction/", full.names = TRUE)
for(i in 1:length(files)){
    submit = fread(files[i])
    if(i==1){
        fnl.submit = submit
    }else{
        fnl.submit$y = fnl.submit$y + submit$y
    }
}
fnl.submit$y = fnl.submit$y/length(files)
fnl.submit[, V1 := NULL]

write.csv(fnl.submit, file = "./submissions/20170613_ordered.csv", row.names = F)









submit_1 = fread('submissions/20170612_bl.csv')
submit_2 = fread('submissions/20170613_ordered.csv')

hist(submit_1$y - submit_2$y, breaks = 100)
plot(submit_1$y, submit_2$y)








































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