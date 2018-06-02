#coding=utf-8

import time
import xgboost as xgb

'''导入外部文件'''
from version_two.feature_engine import *
from version_two.stacking import *
from version_two.feature_selection import *


'''划分数据集'''
train_target['date'] = [index.replace('-', '') for index in train_target['appl_sbm_tm']]
train_target['date'] = [index.split(' ')[0][0:6] for index in train_target['date']]
'''验证集'''
validate_data = train_target[(train_target['date'] == '201704')][['target', 'id']]
'''训练集'''
train_data = train_target[(train_target['date'] >= '201603') & (train_target['date'] <= '201703')][['target', 'id']]
'''测试集'''
test_data = test_list[['id']]
'''测试训练集'''
train_test_data = train_target[['target', 'id']]


def extract_feature():
    '''credit_info'''
    train_credit_info_feature = extract_credit_info(train_credit_info)
    train_test_feature = train_test_data.merge(train_credit_info_feature, on='id', how='left')    # 训练测试集 #
    train_feature = train_data.merge(train_credit_info_feature, on='id', how='left')
    validate_feature = validate_data.merge(train_credit_info_feature, on='id', how='left')
    test_feature = test_data.merge(extract_credit_info(test_credit_info), on='id', how='left')

    '''order_info'''
    train_order_info_feature = extract_order_info(train_order_info)
    train_feature = train_feature.merge(train_order_info_feature, on='id', how='left')
    train_test_feature = train_test_feature.merge(train_order_info_feature, on='id', how='left')  # 训练测试集 #
    validate_feature = validate_feature.merge(train_order_info_feature, on='id', how='left')
    test_feature = test_feature.merge(extract_order_info(test_order_info), on='id', how='left')

    '''user_info'''
    train_user_info_feature = extract_user_info(train_user_info)
    train_feature = train_feature.merge(train_user_info_feature, on='id', how='left')
    train_test_feature = train_test_feature.merge(train_user_info_feature, on='id', how='left')    # 训练测试集 #
    validate_feature = validate_feature.merge(train_user_info_feature, on='id', how='left')
    test_feature = test_feature.merge(extract_user_info(test_user_info), on='id', how='left')

    '''recieve_addr_info'''
    train_recieve_addr_info_feature = extract_recieve_addr_info(train_recieve_addr_info)
    train_feature = train_feature.merge(train_recieve_addr_info_feature, on='id', how='left')
    train_test_feature = train_test_feature.merge(train_recieve_addr_info_feature, on='id', how='left')  # 训练测试集 #
    validate_feature = validate_feature.merge(train_recieve_addr_info_feature, on='id', how='left')
    test_feature = test_feature.merge(extract_recieve_addr_info(test_recieve_addr_info), on='id', how='left')

    '''bankcard_info'''
    train_bankcard_info_feature = extract_bankcard_info(train_bankcard_info)
    train_feature = train_feature.merge(train_bankcard_info_feature, on='id', how='left')
    train_test_feature = train_test_feature.merge(train_bankcard_info_feature, on='id', how='left')  # 训练测试集 #
    validate_feature = validate_feature.merge(train_bankcard_info_feature, on='id', how='left')
    test_feature = test_feature.merge(extract_bankcard_info(test_bankcard_info), on='id', how='left')

    '''auth_info'''
    train_auth_info_feature = extract_auth_info(train_auth_info)
    train_feature = train_feature.merge(train_auth_info_feature, on='id', how='left').fillna(0)
    train_test_feature = train_test_feature.merge(train_auth_info_feature, on='id', how='left').fillna(0)  # 训练测试集 #
    validate_feature = validate_feature.merge(train_auth_info_feature, on='id', how='left').fillna(0)
    test_feature = test_feature.merge(extract_auth_info(test_auth_info), on='id', how='left').fillna(0)

    '''time relative features one'''
    train_time_feature = extract_time_feature(train_auth_info, train_target)
    train_feature = train_feature.merge(train_time_feature, on='id', how='left').fillna(0)
    train_test_feature = train_test_feature.merge(train_time_feature, on='id', how='left').fillna(0)  # 训练测试集 #
    validate_feature = validate_feature.merge(train_time_feature, on='id', how='left').fillna(0)
    test_feature = test_feature.merge(extract_time_feature(test_auth_info, test_list), on='id', how='left').fillna(0)

    '''time relative features two'''
    train_order_payment_time = extract_order_payment_time(train_order_info, train_target)
    train_feature = train_feature.merge(train_order_payment_time, on='id', how='left').fillna(0)
    train_test_feature = train_test_feature.merge(train_order_payment_time, on='id', how='left').fillna(0)  # 训练测试集 #
    validate_feature = validate_feature.merge(train_order_payment_time, on='id', how='left').fillna(0)
    test_feature = test_feature.merge(extract_order_payment_time(test_order_info, test_list), on='id', how='left').fillna(0)

    print(train_feature.head(5))
    print(validate_feature.head(5))
    print(test_feature.head(5))
    return train_feature, validate_feature, test_feature, train_test_feature


def train_module(store_result=False, store_feature=False, select_feature=False, feature_num='all', one_encode=False):
    '''训练模型'''
    if store_feature is True:
        train_feature, validate_feature, test_feature, train_test_feature = extract_feature()
        ''' 保存特征数据 '''
        train_feature.to_csv(r'train_feature.csv', index=None, encoding='utf-8')
        validate_feature.to_csv(r'validate_feature.csv', index=None, encoding='utf-8')
        test_feature.to_csv(r'test_feature.csv', index=None, encoding='utf-8')
        train_test_feature.to_csv(r'train_test_feature.csv', index=None, encoding='utf-8')
        print('保存数据完毕。。。')

        print('特征提取完毕。。。')
        exit(0)
    else:
        train_feature = pd.read_csv(r'train_feature.csv', encoding='utf-8')
        validate_feature = pd.read_csv(r'validate_feature.csv', encoding='utf-8')
        test_feature = pd.read_csv(r'test_feature.csv', encoding='utf-8')
        train_test_feature = pd.read_csv(r'train_test_feature.csv', encoding='utf-8')
        print('读取数据完毕。。。')

    validate_label = validate_feature[['target']]
    train_label = train_feature[['target']]
    train_test_label = train_test_feature[['target']]

    train_feature = train_feature.iloc[:, 2:]
    test_feature = test_feature.iloc[:, 1:]
    validate_feature = validate_feature.iloc[:, 2:]
    train_test_feature = train_test_feature.iloc[:, 2:]

    train_feature = train_feature.drop(['feature_1', 'register_days'], axis=1)
    test_feature = test_feature.drop(['feature_1', 'register_days'], axis=1)
    validate_feature = validate_feature.drop(['feature_1', 'register_days'], axis=1)
    train_test_feature = train_test_feature.drop(['feature_1', 'register_days'], axis=1)

    if one_encode is True:
        features = list(train_feature.columns)
        continuous_feature = []
        one_hot = []
        for name in features:
            if len(set(train_feature[name])) != 2:
                continuous_feature.append(name)
            else:
                one_hot.append(name)

        feature = continuous_feature + one_hot[:130]
        train_feature = train_feature[feature]
        validate_feature = validate_feature[feature]
        test_feature = test_feature[feature]
        train_test_feature = train_test_feature[feature]

    if select_feature is True:
        print('开始特征选择。。。')
        ch2 = SelectKBest(chi2, k=feature_num)
        train_feature = ch2.fit_transform(train_feature, train_label)
        test_feature = ch2.transform(test_feature)
        validate_feature = ch2.transform(validate_feature)
        train_test_feature = ch2.transform(train_test_feature)
        print('特征选择完毕。。。')
    else:
        feature_num = train_feature.shape[1]

    print('开始训练xgboost模型。。。')
    '''xgboost分类器'''
    num_round = 500    # 迭代次数 #
    params = {
        'booster': 'gbtree',
        'max_depth': 4,
        'colsample_bytree': 0.6,
        'subsample': 0.7,
        'eta': 0.03,
        'silent': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        # 'min_child_weight': 1,
        'scale_pos_weight': 1,
        # 'seed': 27,
        # 'reg_alpha': 0.01
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

        result = module_two.predict(test_feature)
        result = pd.DataFrame(result)
        result.columns = ['predicted_score']
        sample = test_list[['id']]
        sample['predicted_score'] = [index for index in result['predicted_score']]
        sample.columns = ['ID', 'PROB']
        sample.to_csv(r'xgb_sample.csv', index=None)
        print(sample)
        print('结果已更新。。。')

    print(" Score_offline:", roc_auc_score(validate_label, module.predict(validate_feature)))
    print('特征维数：', feature_num)


''' 模型融合 '''
def module_merge_triple(prob_xgb, prob_lr, prob_lgb):
    xgb_sample = pd.read_csv(r'result_xgb.csv', low_memory=False)   # encode:159:0.790297834417
    lr_sample = pd.read_csv(r'lr_sample.csv', low_memory=False)     # Uncode:0.792171452209
    lgb_sample = pd.read_csv(r'xgb_sample_51.csv', low_memory=False)

    xgb_sample.columns = ['ID', 'PROB_xgb']
    lr_sample.columns = ['ID', 'PROB_lr']
    lgb_sample.columns = ['ID', 'PROB_lgb']
    sample = xgb_sample.merge(lr_sample, on='ID', how='left')
    sample = sample.merge(lgb_sample, on='ID', how='left')
    # print(sample)
    sample['PROB'] = sample['PROB_xgb'] * prob_xgb + sample['PROB_lr'] * prob_lr + sample['PROB_lgb'] * prob_lgb
    sample = sample[['ID', 'PROB']]
    print(sample)
    sample.to_csv(r'sample.csv', index=None)
    print('模型已融合。。。')


def module_merge_double(prob_x, prob_l):
    xgb_sample = pd.read_csv(r'result0501_152.csv', low_memory=False)   # encode:159:0.790297834417
    lr_sample = pd.read_csv(r'xgb_sample_51.csv', low_memory=False)     # Uncode:0.792171452209
    sample = xgb_sample.merge(lr_sample, on='ID', how='left')
    sample['PROB'] = sample['PROB_x'] * prob_x + sample['PROB_y'] * prob_l
    sample = sample[['ID', 'PROB']]
    print(sample)
    sample.to_csv(r'sample.csv', index=None)
    print('模型已融合。。。')


def main():
    '''xgboost单模型'''
    train_module(store_result=False, store_feature=True, select_feature=False, feature_num='all', one_encode=False)

    '''LogisticRegression单模型'''
    # train_LR_module(store_result=False, select_feature=True, feature_num=140, OneEncode=False)
    '''线性融合三个sample'''
    # module_merge_triple(prob_xgb=0.4, prob_lr=0.2, prob_lgb=0.4)
    '''现行融合两个sample'''
    # module_merge_double(prob_x=0.5, prob_l=0.5)
    '''Stacking'''
    # # ensemble = Ensemble(5, xgb_module, [xgb_module, lgb_module, lr_module, rf_module, gb_module])
    # ensemble = Ensemble(4, lr_module, [xgb_module, xgb_module, xgb_module, xgb_module])
    # train_test, label, test = ensemble.read_data()
    # result = ensemble.fit_predict(train_test, label, test)
    # print('模型融合完毕。。。')
    # result = pd.DataFrame(result, columns=['PROB'])
    # sample = pd.read_csv(r'lr_sample.csv', low_memory=False)
    # sample['PROB'] = [index for index in result['PROB']]
    # sample.to_csv(r'stacking.csv', index=None)
    # print(sample)
    # print('数据整合完毕。。。')

    '''multiply_feature_selection  xgboost_module'''
    # for index in range(70, 200, 5):
    #     print('want to select ', index, ' features')
    #     selection = FeatureSelection(index)
    #     features_name = selection.return_feature_set(variance_threshold=True, select_k_best=True, svc_select=False, tree_select=True)
    #     train_xgb_module(features_name, store_result=False)

    features_name = ['order_all_is_null', 'feature_1', 'register_days', 'quota', 'quota_surplus',  'all_is_null_y',  'account_grade_is_null', 'all_is_zero', 'account_grade2', 'age_three', 'type_pay_len', 'null_y', '等待付款', 'income1', 'auth_time_is_null', 'record_count', 'qq_bound_is_null', 'card_record_count', 'quota_is_zero', '新疆', '云南', 'account_grade3', '广东', 'card_time_is_null', 'have_credit_card', '充值成功', '已取消', 'credit_count', '在线', '四川', 'wechat_bound_is_null', 'null', 'credit_score_rank', '未抢中', 'null_x', '完成', '天津', 'age_two', 'female', '订单取消', 'quota_rate', '山东', '重庆', 'sts_order_len', 'merriage1', '福建', 'account_grade1', 'phone_count', 'record_is_unique', '上海', 'income3', '湖北', 'phone_is_null', 'time_phone_is_null', 'province_len', 'birthday_is_zero', '混合支付', 'auth_id_card_is_null', 'credit_score', '江西', '货到付款', '吉林', 'credit_score_is_null', '江苏', 'all_not_null', 'sex_secret', '已完成', 'card_category_count', 'card_count_one', '等待收货', '湖南', 'male', 'store_card_count']
    train_xgb_module(features_name, store_result=True)

    # 0.81882083452 seed=27
    # original   ->    0.816853963449
    # colsample_bytree: 0.8   ->  0.818427843445
    # scale_pos_weight: 16   ->   0.82029535496
    # reg_alpha: 0.01  ->   0.820431061402
    # 'quota', 'quota_surplus',  ->   0.820543215061

    '''multiply_feature_selection  LogisticRegression_module'''
    # for index in range(70, 200, 5):
    #     print('want to select ', index, ' features')
    #     selection = FeatureSelection(index)
    #     features_name = selection.return_feature_set(variance_threshold=True, select_k_best=True, svc_select=False, tree_select=True)
    #     # features_name = ['id_card_one', 'id_card_two', 'id_card_three', 'id_card_four', 'id_card_five', 'id_card_six', 'mobile', 'unicom', 'telecom', 'virtual', 'order_all_is_null', 'feature_1', 'record_is_unique', '浙江', '辽宁', 'card_time_is_null', 'income1', 'account_grade2', '黑龙', '江苏', '未抢中', '山东', '内蒙', '上海', '分期付款', '货到付款', 'overdraft', '公司转账', 'null', '订单取消', 'age_two', '充值成功', '在线', '新疆', '完成', 'quota_rate', 'sex_not_male', '湖北', 'quota', 'account_grade_is_null', '安徽', 'card_category_count', 'all_not_null', 'phone_is_null', '河北', 'merriage_is_null', '混合支付', 'quota_surplus_is_null', 'birthday_is_zero', 'income3', '江西', 'store_card_count', 'time_phone_is_null', 'id_card_is_null', 'auth_id_card_is_null', '已取消', '广东', 'record_count', '云南', '等待付款', '已完成', 'card_count_one', 'type_pay_len', 'female', 'sts_order_len', '福建', 'auth_time_is_null', '在线支付', 'null_x', 'income2', 'quota_is_zero', 'credit_score_is_null', 'account_grade3', '四川', '等待审核', '重庆', '河南', 'all_is_null_y', '吉林', '抢票已取消', 'province_len', 'credit_count', 'account_grade1', 'credit_score_rank', 'sts_order_count', '湖南', '充值失败;退款成功', 'wechat_bound_is_null', 'card_record_count', 'male', '邮局汇款', 'merriage1', '山西', 'phone_count', 'sex_secret', '海南', 'merriage2', '等待收货', 'all_is_zero', '天津', 'credit_score', 'age_three', 'null_y', 'qq_bound_is_null', 'have_credit_card', '北京']
    #     # # features_name = ['record_count', 'quota', 'account_grade_is_null', '安徽', '云南', '等待付款', 'credit_count', 'account_grade1', 'credit_score_rank', '已完成', 'record_is_unique', 'card_count_one', 'card_category_count', 'all_not_null', 'sts_order_count', '湖南', '浙江', '充值失败;退款成功', 'wechat_bound_is_null', 'card_record_count', 'phone_is_null', 'type_pay_len', 'female', 'male', '辽宁', 'card_time_is_null', '河北', 'sts_order_len', '福建', 'auth_time_is_null', 'income1', '在线支付', 'merriage1', 'null_x', 'account_grade2', 'income2', 'quota_is_zero', '江苏', 'credit_score_is_null', 'merriage_is_null', '未抢中', 'phone_count', '山东', '上海', 'sex_secret', '货到付款', '北京', 'null', 'account_grade3', '等待收货', 'all_is_zero', '天津', 'credit_score', '四川', '混合支付', 'quota_surplus_is_null', 'birthday_is_zero', '订单取消', 'age_two', 'income3', '江西', 'store_card_count', 'time_phone_is_null', '充值成功', 'id_card_is_null', '在线', '新疆', '重庆', '河南', 'all_is_null_y', '吉林', 'auth_id_card_is_null', '完成', 'age_three', 'null_y', 'quota_rate', 'province_len', 'qq_bound_is_null', 'have_credit_card', '已取消', 'sex_not_male', '湖北', '广东']
    #     train_lr_module(features_name, store_result=False)
    #     # 0.812781111086

    features_name = ['order_all_is_null', 'feature_1', 'record_is_unique', '浙江', '辽宁', 'card_time_is_null', 'income1', 'account_grade2', '黑龙', '江苏', '未抢中', '山东', '内蒙', '上海', '分期付款', '货到付款', 'overdraft', '公司转账', 'null', '订单取消', 'age_two', '充值成功', '在线', '新疆', '完成', 'quota_rate', 'sex_not_male', '湖北', 'quota', 'account_grade_is_null', '安徽', 'card_category_count', 'all_not_null', 'phone_is_null', '河北', 'merriage_is_null', '混合支付', 'quota_surplus_is_null', 'birthday_is_zero', 'income3', '江西', 'store_card_count', 'time_phone_is_null', 'id_card_is_null', 'auth_id_card_is_null', '已取消', '广东', 'record_count', '云南', '等待付款', '已完成', 'card_count_one', 'type_pay_len', 'female', 'sts_order_len', '福建', 'auth_time_is_null', '在线支付', 'null_x', 'income2', 'quota_is_zero', 'credit_score_is_null', 'account_grade3', '四川', '等待审核', '重庆', '河南', 'all_is_null_y', '吉林', '抢票已取消', 'province_len', 'credit_count', 'account_grade1', 'credit_score_rank', 'sts_order_count', '湖南', '充值失败;退款成功', 'wechat_bound_is_null', 'card_record_count', 'male', '邮局汇款', 'merriage1', '山西', 'phone_count', 'sex_secret', '海南', 'merriage2', '等待收货', 'all_is_zero', '天津', 'credit_score', 'age_three', 'null_y', 'qq_bound_is_null', 'have_credit_card', '北京']
    train_lr_module(features_name, store_result=True)

    module_merge_triple(prob_xgb=0.4, prob_lr=0.2, prob_lgb=0.4)

if __name__ == '__main__':
    start_time = time.clock()
    main()
    end_time = time.clock()
    print('程序耗时：', end_time - start_time)

