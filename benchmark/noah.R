rm(list = ls()); gc();
require(data.table)
require(caret)
require(fastICA)
require(logisticPCA)
require(rARPACK)
require(RandPro)
require(xgboost)
require(Ckmeans.1d.dp)
require(MLmetrics)


# load --------------------------------------------------------------------


dt_train_raw = fread("./data/train.csv")
dt_test_raw = fread("./data/test.csv")


# clean -------------------------------------------------------------------

# remove single value
cols_single = c()
for(col in names(dt_train_raw)){
    
    len_col = length(unique(dt_train_raw[[col]]))
    if(len_col == 1){
        cols_single = c(cols_single, col)
    }
    
}

dt_train_raw = dt_train_raw[, !cols_single, with = F]
dt_test_raw = dt_test_raw[, !cols_single, with = F]

dim(dt_train_raw); dim(dt_test_raw)

cols_cat = names(dt_train_raw)[sapply(dt_train_raw, is.character)]
cols_bin = names(dt_train_raw)[sapply(dt_train_raw, is.integer)]
cols_bin = cols_bin[!cols_bin %in% "ID"]


# ordered label -----------------------------------------------------------

for(col in cols_cat){
    
    values = unique(c(dt_train_raw[[col]], dt_test_raw[[col]]))
    values_sorted = sort(values)
    values_sorted_final = values_sorted[order(nchar(values_sorted))]
    
    dt_dict = data.table(col = values_sorted_final
                         , encode = 1:length(values_sorted_final))
    setnames(dt_dict, names(dt_dict), c(col, paste0("Encode_Label_", col)))
    
    dt_train_raw = merge(dt_train_raw, dt_dict, by = col)
    dt_test_raw = merge(dt_test_raw, dt_dict, by = col)
    
}

dim(dt_train_raw); dim(dt_test_raw)


#  pca --------------------------------------------------------------------

set.seed(888)
dr_pca = preProcess(dt_train_raw[, !c("y", cols_cat), with = F]
                    , method = "pca"
                    , thresh = .95
                    , uniqueCut = 1
                    # , pcaComp = n_comp
)
dt_pca_train = predict(dr_pca, dt_train_raw[, !c("y", cols_cat), with = F])
dt_pca_test = predict(dr_pca, dt_test_raw[, !c(cols_cat), with = F])


# ica ---------------------------------------------------------------------

n_comp = 50

dr_ica_train = fastICA(dt_train_raw[, !c("y", cols_cat), with = F]
                       , n.comp = n_comp
                       , alg.typ = "parallel"
                       , fun = "logcosh"
                       , method = "C"
                       , row.norm = T)

dt_ica_train = as.data.table(dr_ica_train$S)
setnames(dt_ica_train, names(dt_ica_train), paste0("ICA_", 1:n_comp))

dr_ica_test = fastICA(dt_test_raw[, !c(cols_cat), with = F]
                      , n.comp = n_comp
                      , alg.typ = "parallel"
                      , fun = "logcosh"
                      , method = "C"
                      , row.norm = T)

dt_ica_test = as.data.table(dr_ica_test$S)
setnames(dt_ica_test, names(dt_ica_test), paste0("ICA_", 1:n_comp))



# combine -----------------------------------------------------------------

# pca
dt_train_raw = cbind(dt_train_raw, dt_pca_train)
dt_test_raw = cbind(dt_test_raw, dt_pca_test)
# ica
dt_train_raw = cbind(dt_train_raw, dt_ica_train)
dt_test_raw = cbind(dt_test_raw, dt_ica_test)

dim(dt_train_raw); dim(dt_test_raw)


# targetMean --------------------------------------------------------------

for(col in cols_cat){
    
    dt_targetMean_col = dt_train_raw[, c(col, "y"), with = F]
    dt_targetMean_col = dt_targetMean_col[, .(TargetMean = mean(y, na.rm = T)), by = col]
    setnames(dt_targetMean_col, names(dt_targetMean_col), c(col, paste0("Encode_TargetMean_", col)))
    
    dt_train_raw = merge(dt_train_raw, dt_targetMean_col, by = col, all.x = T)
    dt_test_raw = merge(dt_test_raw, dt_targetMean_col, by = col, all.x = T)
    
    # impute
    set(dt_test_raw
        , which(is.na(dt_test_raw[[paste0("Encode_TargetMean_", col)]]))
        , paste0("Encode_TargetMean_", col)
        , mean(dt_train_raw[, y]))
    
}

dim(dt_train_raw); dim(dt_test_raw)

# select features ---------------------------------------------------------

# dt_all = dt_all[, names(dt_all)[!grepl("SVD|ICA|SRP|GRP|PCA|FA|1770_Match_Bin_Sum|1770_Match|1770_Distant", names(dt_all))], with = F]
dt_train_raw = dt_train_raw[, names(dt_train_raw)[!grepl("Encode_TargetMean_", names(dt_train_raw))], with = F]
dim(dt_train_raw)


# metrics -----------------------------------------------------------------

xg_R_squared = function (yhat, dtrain) {
    
    y = getinfo(dtrain, "label")
    err = R2_Score(yhat, y)
    
    return (list(metric = "error", value = err))
}



# X, y --------------------------------------------------------------------

X_train = dt_train_raw[, !c("y", cols_cat), with = F]
y_train = dt_train_raw[, y]

X_test = dt_test_raw[, !cols_cat, with = F]


# xgb.DMatrix -------------------------------------------------------------

dmx_train = xgb.DMatrix(as.matrix(X_train), label = y_train)
dmx_test = xgb.DMatrix(as.matrix(X_test))
ids_test = X_test$ID



# params ------------------------------------------------------------------

params_xgb = list(
    subsample = 0.9
    , colsample_bytree = 0.7 # 0.9
    , eta = 0.005
    , objective = 'reg:linear'
    , max_depth = 2
    , min_child_weight = 0
    , alpha = 1
    , lambda = 2
    , gamma = 20
    , num_parallel_tree = 1
    , booster = "gbtree"
    , base_score = mean(y_train)
)


# xgb.cv ------------------------------------------------------------------

set.seed(888)
cv_xgb = xgb.cv(params_xgb
                , dmx_train
                , nrounds = 10000
                , nfold = 10
                , early.stop.round = 50
                , print.every.n = 50
                , verbose = 1
                # , obj = pseudo_huber
                , feval = xg_R_squared
                , maximize = T)


# model -------------------------------------------------------------------

vec_preds_y = rep(0, length(ids_test))
n = 10
for(i in 1:n){
    
    cat(paste0(i, " --> "))
    model_xgb = xgb.train(params_xgb, dmx_train
                          , nrounds = which.max(cv_xgb$test.error.mean)
                          , feval = xg_R_squared
                          , maximize = T)
    
    preds_y = predict(model_xgb, dmx_test)
    vec_preds_y = vec_preds_y + preds_y / n
    
}


# importance --------------------------------------------------------------


# xgb.plot.importance(xgb.importance(names(X_train), model = model_xgb))


# submit ------------------------------------------------------------------

# preds_y = predict(model_xgb, dmx_test)
dt_submit = data.table(ID = ids_test
                       # , y = preds_y
                       , y = vec_preds_y)
head(dt_submit)
dim(dt_submit)

write.csv(dt_submit, "submission/30_onBeast_basic_PCA_095_ICA_50_TargetMean_Full.csv", row.names = F)
