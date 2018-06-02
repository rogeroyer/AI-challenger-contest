#coding=utf-8

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

'''单变量特征选取'''
from sklearn.feature_selection import SelectKBest, chi2
'''去除方差小的特征'''
from sklearn.feature_selection import VarianceThreshold
'''循环特征选取'''
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
'''RFE_CV'''
from sklearn.ensemble import ExtraTreesClassifier


class FeatureSelection(object):
    def __init__(self, feature_num):
        self.feature_num = feature_num
        self.train_test, self.label, self.test = self.read_data()    # features #
        self.feature_name = list(self.train_test.columns)     # feature name #

    def read_data(self):
        test = pd.read_csv(r'test_feature.csv', encoding='utf-8')
        train_test = pd.read_csv(r'train_test_feature.csv', encoding='utf-8')
        train_test = train_test.drop(['feature_1', 'register_days', 'id_card_one', 'id_card_two', 'id_card_three', 'id_card_four', 'id_card_five', 'id_card_six', 'mobile', 'unicom', 'telecom', 'virtual'], axis=1)
        print('读取数据完毕。。。')
        label = train_test[['target']]
        test = test.iloc[:, 1:]
        train_test = train_test.iloc[:, 2:]
        return train_test, label, test

    def variance_threshold(self):
        sel = VarianceThreshold()
        sel.fit_transform(self.train_test)
        feature_var = list(sel.variances_)    # feature variance #
        features = dict(zip(self.feature_name, feature_var))
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[-self.feature_num:]
        # print(features)   # 100 cols #
        return set(features)   # return set type #

    def select_k_best(self):
        ch2 = SelectKBest(chi2, k=self.feature_num)
        ch2.fit(self.train_test, self.label)
        feature_var = list(ch2.scores_)  # feature scores #
        features = dict(zip(self.feature_name, feature_var))
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[-self.feature_num:]
        # print(features)     # 100 cols #
        return set(features)    # return set type #

    def svc_select(self):
        svc = SVC(kernel='rbf', C=1, random_state=2018)    # linear #
        rfe = RFE(estimator=svc, n_features_to_select=self.feature_num, step=1)
        rfe.fit(self.train_test, self.label.ravel())
        print(rfe.ranking_)
        return rfe.ranking_

    def tree_select(self):
        clf = ExtraTreesClassifier(n_estimators=300, max_depth=7, n_jobs=4)
        clf.fit(self.train_test, self.label)
        feature_var = list(clf.feature_importances_)  # feature scores #
        features = dict(zip(self.feature_name, feature_var))
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[-self.feature_num:]
        # print(features)     # 100 cols #
        return set(features)  # return set type #

    def return_feature_set(self, variance_threshold=False, select_k_best=False, svc_select=False, tree_select=False):
        names = set([])
        if variance_threshold is True:
            name_one = self.variance_threshold()
            names = names.union(name_one)
        if select_k_best is True:
            name_two = self.select_k_best()
            names = names.intersection(name_two)
        if svc_select is True:
            name_three = self.svc_select()
            names = names.intersection(name_three)
        if tree_select is True:
            name_four = self.tree_select()
            names = names.intersection(name_four)

        # print(len(names))
        print(names)
        return list(names)


# selection = FeatureSelection(100)
# selection.return_feature_set(variance_threshold=True, select_k_best=True, svc_select=False, tree_select=True)


def train_xgb_module(features_name, store_result=False):
    '''训练模型'''
    train_feature = pd.read_csv(r'train_feature.csv', encoding='utf-8')
    validate_feature = pd.read_csv(r'validate_feature.csv', encoding='utf-8')
    test_feature = pd.read_csv(r'test_feature.csv', encoding='utf-8')
    train_test_feature = pd.read_csv(r'train_test_feature.csv', encoding='utf-8')

    print('读取数据完毕。。。')

    validate_label = validate_feature[['target']]
    train_label = train_feature[['target']]
    train_test_label = train_test_feature[['target']]

    train_feature = train_feature[features_name]
    test_feature = test_feature[features_name]
    validate_feature = validate_feature[features_name]
    train_test_feature = train_test_feature[features_name]

    print('开始训练xgboost模型。。。')
    '''xgboost分类器'''
    num_round = 500    # 迭代次数 #
    params = {
        'booster': 'gbtree',
        'max_depth': 4,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'eta': 0.03,
        'silent': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'min_child_weight': 1,
        'scale_pos_weight': 1,
        'seed': 27,
        'reg_alpha': 0.01
    }
    '''训练集'''
    dtrain = xgb.DMatrix(train_feature, label=train_label)
    validate_feature = xgb.DMatrix(validate_feature)
    module = xgb.train(params, dtrain, num_round)

    if store_result is True:
        '''测试训练集'''
        dtrain_two = xgb.DMatrix(train_test_feature, label=train_test_label)
        test_feature = xgb.DMatrix(test_feature)
        module_two = xgb.train(params, dtrain_two, num_round)

        features = module_two.get_fscore()
        features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[-20:]
        features.reverse()
        print(features)       # 输出特征重要性 #

        result = module_two.predict(test_feature)
        result = pd.DataFrame(result)
        result.columns = ['predicted_score']
        test_list = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_list.csv', low_memory=False)
        sample = test_list[['id']]
        sample['predicted_score'] = [index for index in result['predicted_score']]
        sample.columns = ['ID', 'PROB']
        sample.to_csv(r'xgb_sample.csv', index=None)
        print(sample)
        print('结果已更新。。。')

    print(" Score_offline:", roc_auc_score(validate_label, module.predict(validate_feature)))
    print('特征维数：', len(features_name))


def train_lr_module(features_name, store_result=False):
    train_feature = pd.read_csv(r'train_feature.csv', encoding='utf-8')
    validate_feature = pd.read_csv(r'validate_feature.csv', encoding='utf-8')
    test_feature = pd.read_csv(r'test_feature.csv', encoding='utf-8')
    train_test_feature = pd.read_csv(r'train_test_feature.csv', encoding='utf-8')
    print('读取数据完毕。。。')

    validate_label = validate_feature[['target']]
    train_label = train_feature[['target']]
    train_test_label = train_test_feature[['target']]

    train_feature = train_feature[features_name]
    test_feature = test_feature[features_name]
    validate_feature = validate_feature[features_name]
    train_test_feature = train_test_feature[features_name]

    print('开始训练logisticRegression模型。。。')
    module = LogisticRegression(penalty='l2', solver='sag', max_iter=500, random_state=42, n_jobs=4)  # , solver='sag'
    # module = lgb.LGBMClassifier(
    #     num_leaves=64,  # num_leaves = 2^max_depth * 0.6 #
    #     max_depth=6,
    #     n_estimators=80,
    #     learning_rate=0.1
    # )
    '''训练集'''
    module.fit(train_feature, train_label)

    if store_result is True:
        '''测试训练集'''
        module_two = LogisticRegression(
            penalty='l2',
            solver='sag',
            max_iter=500,
            random_state=42,
            n_jobs=4
        )

        # module_two = lgb.LGBMClassifier(
        #     num_leaves=64,  # num_leaves = 2^max_depth * 0.6 #
        #     max_depth=6,
        #     n_estimators=80,
        #     learning_rate=0.1
        # )
        module_two.fit(train_test_feature, train_test_label)

        result = module_two.predict_proba(test_feature)[:, 1]
        result = pd.DataFrame(result)
        result.columns = ['predicted_score']
        test_list = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_list.csv', low_memory=False)
        sample = test_list[['id']]
        sample['predicted_score'] = [index for index in result['predicted_score']]
        sample.columns = ['ID', 'PROB']
        sample.to_csv(r'lr_sample.csv', index=None)
        # sample.to_csv(r'lgb_sample.csv', index=None)
        print(sample)
        print('结果已更新。。。')

    print(" Score_offline:", roc_auc_score(validate_label, module.predict_proba(validate_feature)[:, 1]))
    print('特征维数：', len(features_name))


