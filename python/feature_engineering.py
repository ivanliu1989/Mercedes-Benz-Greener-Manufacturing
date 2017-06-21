# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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
all_dt[cat_col] = all_dt[cat_col].apply(lambda x: x/(x.max()-x.min()))
all_dt.tail()

# 2. PCA, ICA & SVD
n_comp = 16
random_seed = 430
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
# 4.2 ICA
ica_feat_dat = pd.DataFrame(ica_feat)
ica_feat_dat = ica_feat_dat.rename(columns=lambda x: 'ica_'+str(x))
# 4.3 SVD
svd_feat_dat = pd.DataFrame(svd_feat)
svd_feat_dat = svd_feat_dat.rename(columns=lambda x: 'svd_'+str(x))
# 4.4 SRP
srp_feat_dat = pd.DataFrame(srp_feat)
srp_feat_dat = srp_feat_dat.rename(columns=lambda x: 'srp_'+str(x))
# 4.5 GRP
grp_feat_dat = pd.DataFrame(grp_feat)
grp_feat_dat = grp_feat_dat.rename(columns=lambda x: 'grp_'+str(x))
# 4.6 TSNE
tsne_feat_dat = pd.DataFrame(tsne_feat)
tsne_feat_dat = tsne_feat_dat.rename(columns=lambda x: 'tsne_'+str(x))

# Merge
all_dt = pd.concat([all_dt.reset_index(drop=True), pca_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), ica_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), svd_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), srp_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), grp_feat_dat], axis=1)
all_dt = pd.concat([all_dt.reset_index(drop=True), tsne_feat_dat], axis=1)
all_dt.head()

# 5. Remove useless columns
toRM_Col = [c for c in train.columns if len(set(train[c]))==1]
all_dt = all_dt.drop(toRM_Col, axis = 1)

# 6. Interaction features

# 7. Target Mean

# 8. K means

# 9. Outliers

# 10. Modeling
train = all_dt[all_dt['y'] > 0]
test = all_dt[all_dt['y'] == -1]
y_train = train['y'].values
y_mean = np.mean(y_train)
features = [c for c in train.columns if c != 'y']

# 10.1 Xgboost
def the_metric(y_pred, y):
    y_true = y.get_label()
    return 'r2_score', r2_score(y_true, y_pred)

xgb_params = {
    'max_depth': 2 # 4
    ,'eta': 0.005 #0.0045,
    ,'objective': 'reg:linear'
    ,'eval_metric': 'rmse'
    ,'booster': 'gbtree'
    ,'gamma': 1
    ,'min_child_weight': 0
    ,'subsample': 0.93
    ,'colsample_bytree': 1
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
features_score = pd.Series(model.get_fscore()).sort_values(ascending=False)[:50]
plt.figure(figsize=(7,10))
sns.barplot(x=features_score.values, y=features_score.index.values, orient='h')

# Predict
y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test_id.astype(np.int32), 'y': y_pred})
output.to_csv('./prediction/python_xgb_1.csv', index=False)