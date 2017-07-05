# Outlier
# library(RandPro)
# # GRP
# set.seed(912)
# R = form_gauss_matrix(n_rows = ncol(mdat),n_cols = 20,JLT=F,eps=0.5)
# mdat_grp_20 = data.frame(as.matrix(mdat) %*% R)
# names(m_grp_20) = paste0("GRP",1:20)
# 
# mdat = cbind(mdat,mdat_grp_20)
# 
# # SRP
# set.seed(912)
# R = form_sparse_matrix(n_rows = ncol(mdat),n_cols = 20,JLT=F,eps=0.5)
# mdat_srp_20 = data.frame(as.matrix(mdat) %*% R)
# names(mdat_srp_20) = paste0("SRP",1:20)
# 
# mdat = cbind(mdat,mdat_srp_20)


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


# Combine Sort ------------------------------------------------------------
test$y = NA
all = rbind(train.full, test)
setDT(all)
setorder(all, ID)

# Encoding ----------------------------------------------------------------
setDF(all)
cat.feat = paste0("X", c(0:6,8))
num.feat = names(all)[!names(all) %in% c(cat.feat, 'ID', 'y')]
dummies <- dummyVars( ~ ., data = all[, cat.feat])
dummies = predict(dummies, newdata = all)

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

# Interactions based on correlations --------------------------------------
y = all[!is.na(all$y), 'y']
for(i in num.feat){
    for(j in num.feat){
        corInteract = all[!is.na(all$y), i] + all[!is.na(all$y), j]
        if(length(unique(corInteract)) == 1){
            print("Summing up to 1")
        }else{
            if(abs(cor(y, corInteract)) > (abs(cor(y, all[!is.na(all$y), i])) + abs(cor(y, all[!is.na(all$y), j])))){
                print(paste0(i, "_", j))
                all[, paste0(i, "_", j)] = all[, i] + all[, j]
            }    
        }
    }
}
for(i in cat.feat){
    for(j in cat.feat){
        corInteract = all[!is.na(all$y), i] + all[!is.na(all$y), j]
        if(length(unique(corInteract)) == 1){
            print("Summing up to 1")
        }else{
            if(abs(cor(y, corInteract)) > (abs(cor(y, all[!is.na(all$y), i])) + abs(cor(y, all[!is.na(all$y), j])))){
                print(paste0(i, "_", j))
                all[, paste0(i, "_", j)] = all[, i] + all[, j]
            }    
        }
    }
}



# PCA ICA SVD TSNE --------------------------------------------------------
# Cnt
all$Cnt = rowSums(all[, num.feat])
# pca
library(caret)
preProcValues <- preProcess(all[, c(num.feat, cat.feat)], method = c("center", "scale", "pca"))
pca.feat <- predict(preProcValues, all[, c(num.feat, cat.feat)])[, 1:12]
# ica
preProcValues <- preProcess(all[, c(num.feat, cat.feat)], method = c("center", "scale", "ica"), n.comp = 12)
ica.feat <- predict(preProcValues, all[, c(num.feat, cat.feat)])
# svd
svd.feat = as.data.frame(svd(all[, c(num.feat, cat.feat)],nv = )$u)[, 1:12]
names(svd.feat) = paste0('SVD', 1:ncol(svd.feat))
# tsne
library(tsne)
tsne_out = tsne(all[, c(num.feat, cat.feat)], k = 3)
all$tsneX = tsne_out[, 1]
all$tsneY = tsne_out[, 2]
all$tsneZ = tsne_out[, 3]


# Clustering --------------------------------------------------------------
cl <- kmeans(all[, grepl('tsne', colnames(all))],4)
cl2 <- kmeans(all[, 11:366],4)
cl3 <- kmeans(cbind(pca.feat, ica.feat, svd.feat),4)
kmean_feat = data.frame(kmean_tsne = as.character(cl$cluster),
                        kmean_num = as.character(cl2$cluster),
                        kmean_pca = as.character(cl3$cluster))
dummies_kmean <- dummyVars( ~ ., data = kmean_feat)
dummies_kmean = predict(dummies_kmean, newdata = kmean_feat)
# par(mfcol = c(2,2))
# hist(all[cl$cluster == 1, 'y'], 100)
# hist(all[cl$cluster == 2, 'y'], 100)
# hist(all[cl$cluster == 3, 'y'], 100)
# hist(all[cl$cluster == 4, 'y'], 100)


all = cbind(all, pca.feat, ica.feat, svd.feat, dummies_kmean, dummies)
save(all,toRM, file = "../data20170704.RData")

# ID Moving Average -------------------------------------------------------
# moving average by ID


# Target mean -------------------------------------------------------------
# setDF(all)
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

for(i in 1:ncol(all)){
    all[, i] = ifelse(is.nan(all[,i]), NA, all[,i])
}


# Combine -----------------------------------------------------------------
load("../data20170615.RData")


# Modeling ----------------------------------------------------------------
test.full = all[is.na(all$y), ]
train.full = all[!is.na(all$y), ] #  & all$y < 200
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
# train-r2:0.616698+0.014985	test-r2:0.576366+0.063593

library(xgboost)
r2.df = data.frame(feature = 'ALL', R2 = r2)
for(f in 682:length(predictors)){
    set.seed(1989)
    print(f)
    predictors.tmp = predictors[-f]
    # predictors.tmp = predictors[!grepl('mean', predictors) & !grepl('med', predictors) & !grepl('max', predictors) & !grepl('min', predictors) & !grepl('sd', predictors)]
    trainBC = train.full
    dtrain <- xgb.DMatrix(data.matrix(trainBC[, predictors.tmp]), label = trainBC[, response])
    xgbFit = xgb.cv(data = dtrain, nrounds = 15000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 100, verbose = 1, maximize =T, prediction = T)
    r2 = 1 - (sum((trainBC[, response]-xgbFit$pred)^2) / sum((trainBC[, response]-mean(trainBC[, response]))^2))
    r2.df[f+1, 1] = predictors[f]
    r2.df[f+1, 2] = r2
}
save(r2.df, file = './data/featureSelection.RData')

# 0.5691248 without tgt mean
# 0.573533 with tgt mean




# X0 Distribution (e.g. weights)
# Duplicates
# Add training obs
# Stacking
