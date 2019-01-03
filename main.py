# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 14:52:11 2018

@author: tgill
"""

import numpy as np
import pandas as pd
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn import preprocessing 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from collections import Counter


from wrappers import NeuralNet, NLPNN
from models import siamois_cnn, siamois_char


train = pd.read_csv('train_gru.csv')
test = pd.read_csv('test_gru.csv')


features = ['Overlap_title', 'Common_authors', 'Date_diff', 'Overlap_abstract', 'Tfidf_cosine_abstracts_nolim', 'Tfidf_cosine_titles', 'Tfidf_abstracts_(1,2)',#, 'Tfidf_abstracts_chars_1,4','Tfidf_abstracts_chars_1,5'
         'Target_degree',
       'Target_nh_subgraph_edges', 'Target_nh_subgraph_edges_plus',
       'Source_degree', 'Source_nh_subgraph_edges',
       'Source_nh_subgraph_edges_plus', 'Preferential attachment', 'Target_core', 'Target_clustering', 'Target_pagerank', 'Source_core',
       'Source_clustering', 'Source_pagerank', 'Common_friends',
       'Total_friends', 'Friends_measure', 'Sub_nh_edges', 'Sub_nh_edges_plus',
       'Len_path',
       'Both',
       'Tfidf_abstract_(1,3)',
       'Tfidf_abstract_(1,4)',
       'Tfidf_abstract_(1,5)',
       'Common_authors_prop',
       'Overlap_journal',
       'WMD_abstract',
 #      'WMD_title',
       'Common_title_prop',
 #      'Target_density_nh_sub', 'Source_density_nh_sub','Target_density_nh_sub_plus', 'Source_density_nh_sub_plus',
 #      'LGBM_edges'
         'LGBM_Meta',
         'LGBM_Abstract',
         
 #        'LGBM_Vertex',
 #        'LGBM_Number'
  #       'LGBM_Measures'
        'Target_indegree', 'Source_indegree',
 #       'Target_outdegree', 'Source_outdegree',
        'Target_scc', 'Source_scc',
        'Target_wcc', 'Source_wcc',
        #MISSING
       # 'Friend_measure_st',
       #'Friend_measure_ts',
       #'Scc',
       'Wcc',
       #MISSING
       'Len_path_st',
       'Len_path_ts',
       'NN_char',
       'GRU',
       #'GRU_Siamois',
       #'CNN_Siamois'
       ]#, 'Jaccard']


K = 5
np.random.seed(7)
cv = KFold(n_splits = K, shuffle = True, random_state=1)

X = train[features].values
X_test = test[features].values
y = train['Edge'].values
X=preprocessing.scale(X)
X_test=preprocessing.scale(X_test)

lr = LogisticRegression()
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=256, n_jobs=-1)
et = ExtraTreesClassifier(n_estimators=64, n_jobs=-1)
xgb = XGBClassifier(n_estimators=512, max_depth=6, subsample=0.9,colsample_bytree=0.8)
#cat = CatBoostClassifier()
lgbm = LGBMClassifier(n_estimators=1024, max_depth=4, reg_lambda=1., subsample=0.9, colsample_bytree=0.8)#class_weight='balanced')
lgbm_f = LGBMClassifier(n_estimators=1024, max_depth=4, reg_lambda=1., subsample=0.9)#class_weight='balanced')
opt_lgbm = LGBMClassifier(n_estimators=871, learning_rate=0.07, colsample_bytree=0.6, num_leaves=13, subsamble=0.7, subsample_freq=4, max_bin=151)
opt_lgbm2 = LGBMClassifier(n_estimators=960, learning_rate=0.025, colsample_bytree=0.6, num_leaves=25, subsamble=0.8, subsample_freq=4, max_bin=237)
nn = NeuralNet(nn, batch_size=1024, epochs=40, units=512, dropout=0.4 , layers=3)
nnlp = NLPNN(siamois_seq, epochs=15, batch_size=1024, num_words=15000, maxlen=150, embedding=False, char=False, maxlen_chars=1000, tokenize=False)
classifier = lgbm#Mean([lgbm, xgb])

#classifier=nnlp
##features_nlp = ['Abstract_target', 'Abstract_source']
#features_nlp = ['Token_target_15', 'Token_source_15']
#X = train[features_nlp].values
#X_test = test[features_nlp].values
#X_ = [[ast.literal_eval(x1), ast.literal_eval(x2)] for x1,x2 in X]
#X= np.asarray(X_)
#X_test_ = [[ast.literal_eval(x1), ast.literal_eval(x2)] for x1,x2 in X_test]
#X_test = np.asarray(X_test_)

sumf1=0
pred_test=0
scores=[]
feat_prob = np.empty(X.shape[0])
for i, (idx_train, idx_val) in enumerate(cv.split(train)):
    t=time()
    print("Fold ", i )
    X_train = X[idx_train]
    y_train = y[idx_train]
    X_valid = X[idx_val]
    y_valid = y[idx_val]
    classifier.fit(X_train, y_train)#, eval_set=(X_valid, y_valid))
    pred=classifier.predict_proba(X_valid)
    feat_prob[idx_val] = pred[:,1]
    pred = np.argmax(pred, axis=1)
    #print(classifier.coef_)
    #print(Counter(pred))
    #print(classifier.feature_importances_)
    pred_test_fold = classifier.predict_proba(X_test)
    pred_test+=pred_test_fold
    print(Counter(np.argmax(pred_test_fold, axis=1)))
    score=f1_score(pred,y_valid)
    scores.append(score)
    print(score)
    sumf1 +=score
    print(time()-t)
sumf1 = sumf1/K
print("Total score ")
print(sumf1)
print(scores)

#a=classifier.feature_importances_
#b=a/np.sum(a)
#c= zip(features, b)
#print(list(c))

pred_test = pred_test/K
pred_t = np.argmax(pred_test, axis=1)
pred_test_feat = pred_test[:,1]
#classifier.fit(X, y)
#pred_t = classifier.predict(X_test)


sub = test.copy()
sub['id']=test.index
sub['category'] = pred_t
sub = sub[['id', 'category']]

sub.to_csv('sub.csv', index=False)
