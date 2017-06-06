rm(list = ls()); gc()
library(caret)
library(data.table)
f = list.files("./data/", full.names = T)
train.full = fread(f[3])
test = fread(f[2])
submit = fread(f[1])


# Encoding ----------------------------------------------------------------
test$y = -1
all = rbind(train.full, test)
setDF(all)
cat.feat = paste0("X", c(0:6,8))
num.feat = names(all)[!names(all) %in% c(cat.feat, 'ID', 'y')]
dummies <- dummyVars( ~ ., data = all[, cat.feat])
dummies = predict(dummies, newdata = all)
all[, cat.feat] = NULL
all = cbind(all, dummies)
cat.feat = names(all)[!names(all) %in% c(num.feat, 'ID', 'y')]
all.feat = names(all)[!names(all) %in% c('ID', 'y')]

# Interaction terms
# inter.feat = intersect(num.feat, names(train.full))
# formula = paste0("~(", paste0(inter.feat, collapse = "+"), ")^2")
# inter.data = model.matrix(~(X10+X11+X12+X13+X14+X15+X16+X17+X18+X19+X20+X21+X22+X23+X24+X26+X27+
#                                 X28+X29+X30+X31+X32+X33+X34+X36+X38+X40+X41+X42+X43+X44+X45+X46+
#                                 X47+X48+X49+X50+X51+X52+X53+X54+X55+X56+X57+X58+X59+X60+X61+X62+
#                                 X63+X64+X65+X66+X67+X68+X69+X70+X71+X73+X74+X75+X77+X78+X79+X80+
#                                 X81+X82+X83+X85+X86+X87+X88+X89+X90+X91+X92+X93+X94+X95+X96+X97+
#                                 X98+X99+X100+X101+X103+X104+X105+X106+X107+X108+X109+X110+X111+X112+
#                                 X114+X115+X116+X117+X118+X120+X122+X123+X124+X125+X126+X127+X128+X129+X130+
#                                 X131+X132+X133+X135+X136+X137+X138+X139+X140+X141+X142+X143+X144+X145+X148+
#                                 X150+X151+X152+X153+X154+X155+X156+X157+X158+X159+X160+X161+X162+X163+X164+
#                                 X165+X166+X167+X168+X169+X170+X171+X173+X174+X175+X176+X177+X178+X179+X180+
#                                 X181+X182+X183+X184+X185+X186+X187+X189+X190+X191+X192+X194+X195+X196+X197+
#                                 X198+X200+X201+X202+X203+X204+X205+X206+X207+X208+X209+X210+X211+X212+X215+
#                                 X217+X218+X219+X220+X221+X223+X224+X225+X228+X229+X230+X231+X232+X233+X234+
#                                 X235+X236+X237+X238+X240+X241+X242+X243+X245+X246+X247+X248+X249+X250+X251+
#                                 X252+X255+X256+X257+X258+X259+X260+X261+X263+X264+X265+X266+X267+X268+X269+
#                                 X270+X271+X272+X273+X274+X275+X276+X277+X278+X280+X281+X282+X283+X284+X285+
#                                 X286+X287+X288+X289+X290+X291+X292+X294+X295+X297+X298+X300+X301+X304+X305+
#                                 X306+X307+X308+X309+X310+X311+X312+X313+X314+X315+X316+X317+X318+X319+X320+
#                                 X321+X322+X323+X325+X327+X328+X329+X331+X332+X333+X334+X335+X336+X337+X338+
#                                 X339+X340+X341+X342+X343+X344+X345+X346+X347+X348+X349+X350+X351+X352+X353+
#                                 X354+X355+X356+X357+X358+X359+X361+X362+X363+X365+X366+X367+X368+X369+X370+
#                                 X371+X372+X373+X374+X375+X376+X377+X378+X379+X380+X383+X384)^2,all)

# Cnt
all$Cnt = rowSums(all[, num.feat])
# inter.data = as.data.frame(inter.data)
# inter.data$Inter.Cnt = rowSums(inter.data)

# tsne
library(tsne)
all.feat = names(all)[!names(all) %in% c('ID', 'y', 'Cnt')]
tsne_out = tsne(all[, all.feat], k = 3)
all$tsneX = tsne_out[, 1]
all$tsneY = tsne_out[, 2]
all$tsneZ = tsne_out[, 3]

# pca
all.feat = names(all)[!names(all) %in% c('ID', 'y', 'Cnt', 'tsneX', 'tsneY', 'tsneZ')]
library(caret)
preProcValues <- preProcess(all[, all.feat], method = c("pca"))
trainTransformed <- predict(preProcValues, all[, all.feat])

# toRm = c()
# for(i in 1:ncol(inter.data)){
#     if(length(unique(inter.data[,i]))==1){
#         print(colnames(inter.data)[i])
#         toRm = c(toRm, colnames(inter.data)[i])
#     }
# }
# inter.data = inter.data[, !colnames(inter.data) %in% toRm]
# preProcValues <- preProcess(inter.data, method = c("center", "scale", "pca"))
# trainTransformed.inter <- predict(preProcValues, inter.data)
# colnames(trainTransformed.inter) = paste0("Inter.", colnames(trainTransformed.inter))

all = cbind(all, trainTransformed)
# all$Inter.Cnt = inter.data$Inter.Cnt
save(all, file = "./train.RData")

# drop duplication
dim(all)
train <- all[, !duplicated(t(all))]
dim(train)


# Cor
# highlyCorDescr <- findCorrelation(train, cutoff = .99999)
# train.subset <- train[,-highlyCorDescr]
# descrCor2 <- cor(filteredDescr)
# summary(descrCor2[upper.tri(descrCor2)])
# train.full$y = train$y
# train.full$ID = train$ID

# Linear
# comboInfo <- findLinearCombos(train)
# train.subset = train.subset[, -comboInfo$remove]
# dim(train.subset)


# Modeling ----------------------------------------------------------------
test.full = train[train$y ==-1, ]
train.full = train[train$y >0, ]

predictors =colnames(train.full)[!colnames(train.full) %in% c('ID','y')]
response = 'y'

idx = sample(1:nrow(train.full), 2000)
trainBC = train.full[idx,]
validBC = train.full[-idx,]
# idx = sample(1:nrow(validBC), 1000)
# testBC = validBC[-idx,]
# validBC = validBC[idx,]

# xgboost -----------------------------------------------------------------
library(xgboost)
r2squared_xgb_feval <- function(pred, dtrain) {
    pred <- as.numeric(pred)
    actual <- as.numeric(getinfo(dtrain, "label"))
    return (list(metric = 'r2', value = 1 - (sum((actual-pred)^2) / sum((actual-mean(actual))^2))))
}
# dall <- xgb.DMatrix(data.matrix(train.full[, predictors]), label = train.full[, response])
dtrain <- xgb.DMatrix(data.matrix(trainBC[, predictors]), label = trainBC[, response])
# dval <- xgb.DMatrix(data.matrix(validBC[, predictors]), label = validBC[, response])
dtest <- xgb.DMatrix(data.matrix(validBC[, predictors]), label = validBC[, response])
# watchlist <- list(eval = dval)

param <- list(
    max_depth = 6,
    eta = 0.01,
    nthread = 7,
    objective = "reg:linear",
    eval_metric=r2squared_xgb_feval,
    booster = "gbtree",
    gamma = 0.1,
    min_child_weight = 10,
    subsample = 0.7,
    colsample_bytree = 0.8,
    lambda = 100,
    alpha = 100, # 1e-5, 1e-2, 0.1, 1, 100
    seed = 1989
) 
xgbFit = xgb.cv(data = dtrain, nrounds = 1000, nfold = 10, param, print_every_n = 100, early_stopping_rounds = 20, verbose = 1, maximize =T)

xgbFit <- xgb.train(param,dtrain,nrounds = xgbFit$best_iteration,print.every.n = 100, verbose = 1, maximize =T)
var.imp = xgb.importance(colnames(dtrain), model = xgbFit)
pred = predict(xgbFit, dtest, xgbFit$bestInd)
regressionEvaluation(testBC$y, pred)
R2 <- 1 - (sum((testBC$y-pred )^2)/sum((testBC$y-mean(testBC$y))^2))
R2




# Blending ----------------------------------------------------------------
param <- list(
    max_depth = 6,
    eta = 0.01,
    nthread = 7,
    objective = "reg:linear",
    eval_metric=r2squared_xgb_feval,
    booster = "gbtree",
    gamma = 0.1,
    min_child_weight = 10,
    subsample = 0.7,
    colsample_bytree = 0.8,
    lambda = 100,
    alpha = 100, # 1e-5, 1e-2, 0.1, 1, 100
    seed = 1989
) 
dtest <- xgb.DMatrix(data.matrix(test.full[, predictors]), label = test.full[, response])

for(i in 1:40){
    print(i)
    idx = sample(1:nrow(train.full), nrow(train.full)/2)
    trainBC = train.full[idx,]
    validBC = train.full[-idx,]
    
    dtrain <- xgb.DMatrix(data.matrix(trainBC[, predictors]), label = trainBC[, response])
    dval <- xgb.DMatrix(data.matrix(validBC[, predictors]), label = validBC[, response])
    watchlist <- list(eval = dval)
    
    xgbFit <- xgb.train(param,dtrain,nrounds = 10000,watchlist,print.every.n = 100, early.stop.round = 20, verbose = 1, maximize =T)
    
    pred = predict(xgbFit, dtest, xgbFit$bestInd)
    submit = data.frame(ID = test.full$ID, y = pred)
    write.csv(submit, file = paste0("./prediction/xgb_benchmark_",xgbFit$best_score,"_",i,".csv"), row.names = F)
    
}




# Submissions -------------------------------------------------------------
files = list.files("./prediction/", full.names = TRUE)
for(i in 1:length(files)){
    submit = fread(files[1])
    if(i==1){
        fnl.submit = submit
    }else{
        fnl.submit$y = fnl.submit$y + submit$y
    }
}
fnl.submit$y = fnl.submit$y/length(files)
fnl.submit[, V1 := NULL]

write.csv(fnl.submit, file = "./submissions/xgb_submit_01.csv", row.names = F)
