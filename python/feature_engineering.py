# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def feat_standardize(x):
    x = x.apply(lambda x: (x-x.min())/(x.max()-x.min()))
    return x
    
# Read data in
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
test['y'] = -1
test_id = test['ID']
test.head()

# 1. Label Encoding for Categorical features
for col in train.columns:
    if train[col].dtype == 'object':
        alist = list(train[col].values) + list(test[col].values)
        lst_one = sorted([x for x in alist if len(x) == 1])
        lst_two = sorted([x for x in alist if len(x) > 1])
        lst_all = lst_one + lst_two
        encoded_dt, mapping_idx = pd.Series(lst_all).factorize()
        train[col] = train[col].apply(lambda x: mapping_idx.get_loc(x))
        test[col] = test[col].apply(lambda x: mapping_idx.get_loc(x))

test.head()

# Combine train & test
all_dt = pd.concat([train, test])
# Standized
cat_col = [col for col in all_dt.columns if col not in ['y'] and all_dt[col].max()>1]
all_dt[cat_col] = feat_standardize(all_dt[cat_col])
all_dt.tail()

# 6. Interaction features
interact_feat = pd.DataFrame()
for i in all_dt.columns:
    for j in all_dt.columns:
        yi = np.abs(np.corrcoef(train[i], train['y'])[0,1])
        yj = np.abs(np.corrcoef(train[j], train['y'])[0,1])
        yij = np.abs(np.corrcoef(train[j]+train[i], train['y'])[0,1])
        if yij > yi + yj:
            print(i+'_'+j+': '+str(yi)+' '+str(yj)+' '+str(yij))
            interact_feat['Itr_'+i+'_'+j] = all_dt[j]+all_dt[i]
interact_feat = feat_standardize(interact_feat)            

# 2. PCA, ICA & SVD
n_comp = 12
random_seed = 624
# 2.1 PCA
pca = PCA(n_components=n_comp, random_state=random_seed)
pca_feat = pca.fit_transform(all_dt.drop(['y'], axis = 1))
# 2.2 ICA
ica = FastICA(n_components=n_comp, random_state=random_seed)
ica_feat = ica.fit_transform(all_dt.drop(['y'], axis = 1))
# 2.3 SVD
svd = TruncatedSVD(n_components=n_comp, random_state=random_seed)
svd_feat = svd.fit_transform(all_dt.drop(['y'], axis = 1))
# 2.4 TSNE
tsne = TSNE(n_components=3, random_state=random_seed, verbose = 1)
tsne_feat = tsne.fit_transform(all_dt.drop(['y'], axis = 1))
# 2.5 KMeans
kmeans = KMeans(n_clusters=4, random_state=random_seed)
kmeans_feat = kmeans.fit_transform(all_dt.drop(['y'], axis = 1))
# 2.6 Logistic PCA

# 3. Random Projection
# 3.1 SRP
srp = SparseRandomProjection(n_components=n_comp, eps=0.1, dense_output = True, random_state=random_seed)
srp_feat = srp.fit_transform(all_dt.drop(['y'], axis = 1))
# 3.2 GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=random_seed)
grp_feat = grp.fit_transform(all_dt.drop(['y'], axis = 1))

# 4. Rename and merge features
# 4.1 PCA
pca_feat_dat = pd.DataFrame(pca_feat)
pca_feat_dat = pca_feat_dat.rename(columns=lambda x: 'pca_'+str(x))
pca_feat_dat  = feat_standardize(pca_feat_dat)
# 4.2 ICA
ica_feat_dat = pd.DataFrame(ica_feat)
ica_feat_dat = ica_feat_dat.rename(columns=lambda x: 'ica_'+str(x))
ica_feat_dat  = feat_standardize(ica_feat_dat)
# 4.3 SVD
svd_feat_dat = pd.DataFrame(svd_feat)
svd_feat_dat = svd_feat_dat.rename(columns=lambda x: 'svd_'+str(x))
svd_feat_dat  = feat_standardize(svd_feat_dat)
# 4.4 SRP
srp_feat_dat = pd.DataFrame(srp_feat)
srp_feat_dat = srp_feat_dat.rename(columns=lambda x: 'srp_'+str(x))
srp_feat_dat  = feat_standardize(srp_feat_dat)
# 4.5 GRP
grp_feat_dat = pd.DataFrame(grp_feat)
grp_feat_dat = grp_feat_dat.rename(columns=lambda x: 'grp_'+str(x))
grp_feat_dat  = feat_standardize(grp_feat_dat)
# 4.6 TSNE
tsne_feat_dat = pd.DataFrame(tsne_feat)
tsne_feat_dat = tsne_feat_dat.rename(columns=lambda x: 'tsne_'+str(x))
tsne_feat_dat  = feat_standardize(tsne_feat_dat)
# 4.7 KMeans
kmeans_feat_dat = pd.DataFrame(kmeans_feat)
kmeans_feat_dat = kmeans_feat_dat.rename(columns=lambda x: 'kmeans_'+str(x))
kmeans_feat_dat = feat_standardize(kmeans_feat_dat)

# Merge
all_dt = pd.concat([all_dt.reset_index(drop=True), pca_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), ica_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), svd_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), srp_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), grp_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), tsne_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), kmeans_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), interact_feat.reset_index(drop=True)], axis=1)
all_dt.head()

# 5. Remove useless columns
toRM_Col = [c for c in train.columns if len(set(train[c]))==1]
all_dt = all_dt.drop(toRM_Col, axis = 1)

# 7. Target Mean
# 7.1 Tgt mean
y_Mean = all_dt['y'].mean()
X0_Mean = all_dt.groupby(['X0'], as_index = False)['y'].mean(); X0_Mean['y'] = X0_Mean['y'].replace([-1],[y_Mean])
X1_Mean = all_dt.groupby(['X1'], as_index = False)['y'].mean(); X1_Mean['y'] = X1_Mean['y'].replace([-1],[y_Mean])
X2_Mean = all_dt.groupby(['X2'], as_index = False)['y'].mean(); X2_Mean['y'] = X2_Mean['y'].replace([-1],[y_Mean])
X3_Mean = all_dt.groupby(['X3'], as_index = False)['y'].mean(); X3_Mean['y'] = X3_Mean['y'].replace([-1],[y_Mean])
X4_Mean = all_dt.groupby(['X4'], as_index = False)['y'].mean(); X4_Mean['y'] = X4_Mean['y'].replace([-1],[y_Mean])
X5_Mean = all_dt.groupby(['X5'], as_index = False)['y'].mean(); X5_Mean['y'] = X5_Mean['y'].replace([-1],[y_Mean])
X6_Mean = all_dt.groupby(['X6'], as_index = False)['y'].mean(); X6_Mean['y'] = X6_Mean['y'].replace([-1],[y_Mean])
X8_Mean = all_dt.groupby(['X8'], as_index = False)['y'].mean(); X8_Mean['y'] = X8_Mean['y'].replace([-1],[y_Mean])
# 7.2 Tgt sd
y_Std = all_dt['y'].std()
X0_Sd = all_dt.groupby(['X0'], as_index = False)['y'].std(); X0_Sd['y'] = X0_Sd['y'].replace([float('nan')],[y_Std])
X1_Sd = all_dt.groupby(['X1'], as_index = False)['y'].std(); X1_Sd['y'] = X1_Sd['y'].replace([float('nan')],[y_Std])
X2_Sd = all_dt.groupby(['X2'], as_index = False)['y'].std(); X2_Sd['y'] = X2_Sd['y'].replace([float('nan')],[y_Std])
X3_Sd = all_dt.groupby(['X3'], as_index = False)['y'].std(); X3_Sd['y'] = X3_Sd['y'].replace([float('nan')],[y_Std])
X4_Sd = all_dt.groupby(['X4'], as_index = False)['y'].std(); X4_Sd['y'] = X4_Sd['y'].replace([float('nan')],[y_Std])
X5_Sd = all_dt.groupby(['X5'], as_index = False)['y'].std(); X5_Sd['y'] = X5_Sd['y'].replace([float('nan')],[y_Std])
X6_Sd = all_dt.groupby(['X6'], as_index = False)['y'].std(); X6_Sd['y'] = X6_Sd['y'].replace([float('nan')],[y_Std])
X8_Sd = all_dt.groupby(['X8'], as_index = False)['y'].std(); X8_Sd['y'] = X8_Sd['y'].replace([float('nan')],[y_Std])
# 7.3 Tgt kurtosis
y_Kurt = pd.DataFrame.kurt(all_dt['y'])
X0_Kurt = all_dt.groupby(['X0'], as_index = False)['y'].apply(pd.DataFrame.kurt); X0_Kurt = X0_Kurt.replace([float('nan')],[y_Kurt])
X1_Kurt = all_dt.groupby(['X1'], as_index = False)['y'].apply(pd.DataFrame.kurt); X1_Kurt = X1_Kurt.replace([float('nan')],[y_Kurt])
X2_Kurt = all_dt.groupby(['X2'], as_index = False)['y'].apply(pd.DataFrame.kurt); X2_Kurt = X2_Kurt.replace([float('nan')],[y_Kurt])
X3_Kurt = all_dt.groupby(['X3'], as_index = False)['y'].apply(pd.DataFrame.kurt); X3_Kurt = X3_Kurt.replace([float('nan')],[y_Kurt])
X4_Kurt = all_dt.groupby(['X4'], as_index = False)['y'].apply(pd.DataFrame.kurt); X4_Kurt = X4_Kurt.replace([float('nan')],[y_Kurt])
X5_Kurt = all_dt.groupby(['X5'], as_index = False)['y'].apply(pd.DataFrame.kurt); X5_Kurt = X5_Kurt.replace([float('nan')],[y_Kurt])
X6_Kurt = all_dt.groupby(['X6'], as_index = False)['y'].apply(pd.DataFrame.kurt); X6_Kurt = X6_Kurt.replace([float('nan')],[y_Kurt])
X8_Kurt = all_dt.groupby(['X8'], as_index = False)['y'].apply(pd.DataFrame.kurt); X8_Kurt = X8_Kurt.replace([float('nan')],[y_Kurt])
# 7.4 Tgt skewness
y_Skew = all_dt['y'].skew()
X0_Skew = all_dt.groupby(['X0'], as_index = False)['y'].skew(); X0_Skew = X0_Skew.replace([float('nan')],[y_Skew])
X1_Skew = all_dt.groupby(['X1'], as_index = False)['y'].skew(); X1_Skew = X1_Skew.replace([float('nan')],[y_Skew])
X2_Skew = all_dt.groupby(['X2'], as_index = False)['y'].skew(); X2_Skew = X2_Skew.replace([float('nan')],[y_Skew])
X3_Skew = all_dt.groupby(['X3'], as_index = False)['y'].skew(); X3_Skew = X3_Skew.replace([float('nan')],[y_Skew])
X4_Skew = all_dt.groupby(['X4'], as_index = False)['y'].skew(); X4_Skew = X4_Skew.replace([float('nan')],[y_Skew])
X5_Skew = all_dt.groupby(['X5'], as_index = False)['y'].skew(); X5_Skew = X5_Skew.replace([float('nan')],[y_Skew])
X6_Skew = all_dt.groupby(['X6'], as_index = False)['y'].skew(); X6_Skew = X6_Skew.replace([float('nan')],[y_Skew])
X8_Skew = all_dt.groupby(['X8'], as_index = False)['y'].skew(); X8_Skew = X8_Skew.replace([float('nan')],[y_Skew])
# 7.5 Merge
tgt_feat_X0 = pd.DataFrame({'X0':X0_Mean['X0'], 'X0_Mean':X0_Mean['y'], 'X0_Sd':X0_Sd['y'], 'X0_Kurt':X0_Kurt, 'X0_Skew':X0_Skew})
tgt_feat_X1 = pd.DataFrame({'X1':X1_Mean['X1'], 'X1_Mean':X1_Mean['y'], 'X1_Sd':X1_Sd['y'], 'X1_Kurt':X1_Kurt, 'X1_Skew':X1_Skew})
tgt_feat_X2 = pd.DataFrame({'X2':X2_Mean['X2'], 'X2_Mean':X2_Mean['y'], 'X2_Sd':X2_Sd['y'], 'X2_Kurt':X2_Kurt, 'X2_Skew':X2_Skew})
tgt_feat_X3 = pd.DataFrame({'X3':X3_Mean['X3'], 'X3_Mean':X3_Mean['y'], 'X3_Sd':X3_Sd['y'], 'X3_Kurt':X3_Kurt, 'X3_Skew':X3_Skew})
tgt_feat_X4 = pd.DataFrame({'X4':X4_Mean['X4'], 'X4_Mean':X4_Mean['y'], 'X4_Sd':X4_Sd['y'], 'X4_Kurt':X4_Kurt, 'X4_Skew':X4_Skew})
tgt_feat_X5 = pd.DataFrame({'X5':X5_Mean['X5'], 'X5_Mean':X5_Mean['y'], 'X5_Sd':X5_Sd['y'], 'X5_Kurt':X5_Kurt, 'X5_Skew':X5_Skew})
tgt_feat_X6 = pd.DataFrame({'X6':X6_Mean['X6'], 'X6_Mean':X6_Mean['y'], 'X6_Sd':X6_Sd['y'], 'X6_Kurt':X6_Kurt, 'X6_Skew':X6_Skew})
tgt_feat_X8 = pd.DataFrame({'X8':X8_Mean['X8'], 'X8_Mean':X8_Mean['y'], 'X8_Sd':X8_Sd['y'], 'X8_Kurt':X8_Kurt, 'X8_Skew':X8_Skew})
tgt_feat_X0.iloc[:,1:] = feat_standardize(tgt_feat_X0.iloc[:,1:])
tgt_feat_X1.iloc[:,1:] = feat_standardize(tgt_feat_X1.iloc[:,1:])
tgt_feat_X2.iloc[:,1:] = feat_standardize(tgt_feat_X2.iloc[:,1:])
tgt_feat_X3.iloc[:,1:] = feat_standardize(tgt_feat_X3.iloc[:,1:])
tgt_feat_X4.iloc[:,1:] = feat_standardize(tgt_feat_X4.iloc[:,1:])
tgt_feat_X5.iloc[:,1:] = feat_standardize(tgt_feat_X5.iloc[:,1:])
tgt_feat_X6.iloc[:,1:] = feat_standardize(tgt_feat_X6.iloc[:,1:])
tgt_feat_X8.iloc[:,1:] = feat_standardize(tgt_feat_X8.iloc[:,1:])
all_dt = pd.merge(all_dt, tgt_feat_X0, on = ['X0'], how = 'left')
all_dt = pd.merge(all_dt, tgt_feat_X1, on = ['X1'], how = 'left')
all_dt = pd.merge(all_dt, tgt_feat_X2, on = ['X2'], how = 'left')
all_dt = pd.merge(all_dt, tgt_feat_X3, on = ['X3'], how = 'left')
all_dt = pd.merge(all_dt, tgt_feat_X4, on = ['X4'], how = 'left')
all_dt = pd.merge(all_dt, tgt_feat_X5, on = ['X5'], how = 'left')
all_dt = pd.merge(all_dt, tgt_feat_X6, on = ['X6'], how = 'left')
all_dt = pd.merge(all_dt, tgt_feat_X8, on = ['X8'], how = 'left')

# 8. Outliers

# 9. Modeling
train = all_dt[all_dt['y'] > 0]
test = all_dt[all_dt['y'] == -1]
y_train = train['y'].values
y_mean = np.mean(y_train)
features = [c for c in train.columns if c != 'y']
#features = list(set(features))
# features = features_score.index
# 10.1 Xgboost
def the_metric(y_pred, y):
    y_true = y.get_label()
    return 'r2_score', r2_score(y_true, y_pred)

xgb_params = {
    'max_depth': 2 # 4
    ,'eta': 0.002 #0.0045,
    ,'objective': 'reg:linear'
    ,'eval_metric': 'rmse'
    ,'booster': 'gbtree'
    ,'gamma': 10
    ,'min_child_weight': 0
    ,'subsample': 1
    ,'colsample_bytree': 0.7
    ,'lambda': 2
    ,'alpha': 1
    ,'base_score': y_mean
}
dtrain = xgb.DMatrix(train[features], y_train)
dtest = xgb.DMatrix(test[features])
# CV
model = xgb.cv(dict(xgb_params), 
               maximize=True,
               feval=the_metric,
               verbose_eval=100, 
               stratified=True, 
               dtrain = dtrain, 
               num_boost_round=15000, 
               early_stopping_rounds=100, 
               nfold=5)

# Model
model = xgb.train(dict(xgb_params), 
               maximize=True,
               feval=the_metric,
               verbose_eval=100, 
               dtrain = dtrain, 
               num_boost_round=model.shape[0])
features_score = pd.Series(model.get_fscore()).sort_values(ascending=False)#[:80]
plt.figure(figsize=(7,10))
sns.barplot(x=features_score.values, y=features_score.index.values, orient='h')

# Predict
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test_id.astype(np.int32), 'y': y_pred})
output.to_csv('./prediction/python_xgb_3.csv', index=False)
