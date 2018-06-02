#coding=utf-8

import time
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest    # 特征选择 #
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

'''训练集'''
train_auth_info = pd.read_csv('../dataset/ai_risk_train/train_auth_info.csv', low_memory=False)
train_bankcard_info = pd.read_csv('../dataset/ai_risk_train/train_bankcard_info.csv', low_memory=False)
train_credit_info = pd.read_csv('../dataset/ai_risk_train/train_credit_info.csv', low_memory=False)
train_order_info = pd.read_csv('../dataset/ai_risk_train/train_order_info.csv', low_memory=False)
train_recieve_addr_info = pd.read_csv('../dataset/ai_risk_train/train_recieve_addr_info.csv', low_memory=False)
train_user_info = pd.read_csv('../dataset/ai_risk_train/train_user_info.csv', low_memory=False)
train_target = pd.read_csv('../dataset/ai_risk_train/train_target.csv', low_memory=False)

# '''测试集'''
# test_auth_info = pd.read_csv('../dataset/ai_risk_test/test_auth_info.csv', low_memory=False)
# test_bankcard_info = pd.read_csv('../dataset/ai_risk_test/test_bankcard_info.csv', low_memory=False)
# test_credit_info = pd.read_csv('../dataset/ai_risk_test/test_credit_info.csv', low_memory=False)
# test_order_info = pd.read_csv('../dataset/ai_risk_test/test_order_info.csv', low_memory=False)
# test_recieve_addr_info = pd.read_csv('../dataset/ai_risk_test/test_recieve_addr_info.csv', low_memory=False)
# test_user_info = pd.read_csv('../dataset/ai_risk_test/test_user_info.csv', low_memory=False)
# test_list = pd.read_csv('../dataset/ai_risk_test/test_list.csv', low_memory=False)

test_auth_info = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_auth_info.csv', low_memory=False)
test_bankcard_info = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_bankcard_info.csv', low_memory=False)
test_credit_info = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_credit_info.csv', low_memory=False)
test_order_info = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_order_info.csv', low_memory=False)
test_recieve_addr_info = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_recieve_addr_info.csv', low_memory=False)
test_user_info = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_user_info.csv', low_memory=False)
test_list = pd.read_csv('../dataset/AI_Risk_BtestData_V1.0/Btest_list.csv', low_memory=False)

# print(test_auth_info)
# print(test_bankcard_info)
# print(test_credit_info)
# print(test_order_info)
# print(test_recieve_addr_info)
# exit(0)

def cal_auc(list_one, list_two):
    '''计算AUC值'''
    positive = []
    negative = []
    for index in range(len(list_one)):
        if list_one[index] == 1:
            positive.append(index)
        else:
            negative.append(index)
    SUM = 0
    for i in positive:
        for j in negative:
            if list_two[i] > list_two[j]:
                SUM += 1
            elif list_two[i] == list_two[j]:
                SUM += 0.5
            else:
                pass
    return SUM / (len(positive)*len(negative))


def return_set(group):
    return set(group)


def extract_credit_info(credit_info):
    '''提取credit_info表 特征'''
    credit_info['credit_score'] = credit_info['credit_score'].fillna(credit_info['credit_score'].mean())
    credit_info['quota_is_zero'] = [1 if i != 0.0 else 0 for i in credit_info.quota]  # 是否有信用额度 #
    credit_info['overdraft'] = credit_info['overdraft'].fillna(0)
    credit_info['quota'] = credit_info['quota'].fillna(0)
    credit_info['quota_surplus'] = credit_info['quota'] - credit_info['overdraft']
    # credit_info['quota_rate'] = (credit_info['overdraft'] / credit_info['quota']).fillna(0)
    credit_info['quota_rate'] = credit_info[['overdraft', 'quota']].apply(lambda x: x.overdraft / x.quota if x.quota != 0 else 0, axis=1)
    credit_info['credit_score_rank'] = credit_info['credit_score'].rank(method='first', ascending=False)

    credit_info.loc[:, 'all_is_null'] = credit_info[['credit_score', 'overdraft', 'quota']].apply(lambda x: 1 if ((x.credit_score is not np.nan) and (x.overdraft is not np.nan) and (x.quota is not np.nan)) else 0, axis=1)
    credit_info.loc[:, 'all_is_zero'] = credit_info[['credit_score', 'overdraft', 'quota']].apply(lambda x: 1 if ((x.credit_score == 0) and (x.overdraft == 0) and (x.quota == 0)) else 0, axis=1)
    credit_info.loc[:, 'quota_is_zero'] = credit_info[['quota']].apply(lambda x: 1 if x.quota == 0 else 0, axis=1)
    credit_info.loc[:, 'credit_score_is_null'] = credit_info[['credit_score']].apply(lambda x: 1 if x.credit_score == 0 else 0, axis=1)
    credit_info.loc[:, 'quota_surplus_is_null'] = credit_info[['quota_surplus', 'quota']].apply(lambda x: 1 if (x.quota_surplus == 0) and (x.quota != 0) else 0, axis=1)

    '''归一化'''
    credit_info[['credit_score', 'overdraft', 'quota', 'quota_surplus', 'credit_score_rank']] = credit_info[['credit_score', 'overdraft', 'quota', 'quota_surplus', 'credit_score_rank']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return credit_info

# print(extract_credit_info(train_credit_info))
# print(extract_credit_info(test_credit_info))


def extract_user_info(user_info):
    '''提取 user_info表 特征'''
    feature = user_info[['id']]
    feature.loc[:, 'birthday_is_zero'] = user_info[['birthday']].apply(lambda x: 1 if x.birthday == '0000-00-00' else 0, axis=1)
    feature.loc[:, 'sex_not_male'] = user_info[['sex']].apply(lambda x: 1 if x.sex != '女' else 0, axis=1)
    feature.loc[:, 'female'] = user_info[['sex']].apply(lambda x: 1 if x.sex == '男' else 0, axis=1)
    feature.loc[:, 'male'] = user_info[['sex']].apply(lambda x: 1 if x.sex == '女' else 0, axis=1)
    feature.loc[:, 'sex_secret'] = user_info[['sex']].apply(lambda x: 1 if x.sex == '保密' else 0, axis=1)    # 0.69504936432
    ##
    feature.loc[:, 'merriage1'] = user_info[['merriage']].apply(lambda x: 1 if x.merriage == '未婚' else 0, axis=1)
    feature.loc[:, 'merriage2'] = user_info[['merriage']].apply(lambda x: 1 if x.merriage == '已婚' else 0, axis=1)
    feature.loc[:, 'merriage3'] = user_info[['merriage']].apply(lambda x: 1 if x.merriage == '保密' else 0, axis=1)
    feature.loc[:, 'merriage_is_null'] = user_info[['merriage']].apply(lambda x: 1 if x.merriage is np.nan else 0, axis=1)   # 0.700624700466
    ####
    feature.loc[:, 'account_grade1'] = user_info[['account_grade']].apply(lambda x: 1 if x.account_grade == '注册会员' else 0, axis=1)
    feature.loc[:, 'account_grade2'] = user_info[['account_grade']].apply(lambda x: 1 if x.account_grade == '铜牌会员' else 0, axis=1)
    feature.loc[:, 'account_grade3'] = user_info[['account_grade']].apply(lambda x: 1 if x.account_grade == '银牌会员' else 0, axis=1)
    feature.loc[:, 'account_grade4'] = user_info[['account_grade']].apply(lambda x: 1 if x.account_grade == '金牌会员' else 0, axis=1)
    feature.loc[:, 'account_grade5'] = user_info[['account_grade']].apply(lambda x: 1 if x.account_grade == '钻石会员' else 0, axis=1)
    feature.loc[:, 'account_grade_is_null'] = user_info[['account_grade']].apply(lambda x: 1 if x.account_grade is np.nan else 0, axis=1)
    ###
    feature.loc[:, 'qq_bound_is_null'] = user_info[['qq_bound']].apply(lambda x: 1 if x.qq_bound is np.nan else 0, axis=1)
    feature.loc[:, 'wechat_bound_is_null'] = user_info[['wechat_bound']].apply(lambda x: 1 if x.wechat_bound is np.nan else 0, axis=1)
    feature.loc[:, 'degree'] = user_info[['degree']].apply(lambda x: 1 if (x.degree == '硕士') | (x.degree == '其他') | (x.degree == '博士') else 0, axis=1)
    feature.loc[:, 'id_card_is_null'] = user_info[['id_card']].apply(lambda x: 1 if x.id_card is np.nan else 0, axis=1)
    #####
    feature.loc[:, 'income1'] = [1 if index == '4000-5999元' else 0 for index in user_info['income']]
    feature.loc[:, 'income2'] = [1 if index == '8000元以上' else 0 for index in user_info['income']]
    feature.loc[:, 'income3'] = [1 if index == '2000-3999元' else 0 for index in user_info['income']]
    feature.loc[:, 'income4'] = [1 if index == '6000-7999元' else 0 for index in user_info['income']]
    feature.loc[:, 'income5'] = [1 if index == '2000元以下' else 0 for index in user_info['income']]     # 0.775891365882 #

    '''年龄特征'''
    def is_valid_date(strdate):
        '''''判断是否是一个有效的日期字符串'''
        try:
            if ":" in strdate:
                time.strptime(strdate, "%Y-%m-%d %H:%M:%S")
            else:
                time.strptime(strdate, "%Y-%m-%d")
            return True
        except:
            return False

    ####
    user_info['birthday_two'] = user_info[['birthday']].apply(lambda index: is_valid_date(index.birthday), axis=1)
    user_info['birthday'] = user_info[['birthday']].apply(lambda index: 0 if (index.birthday is np.nan) or (index.birthday == '0000-00-00') else index.birthday[0:4], axis=1)
    user_info['age'] = user_info[['birthday', 'birthday_two']].apply(lambda x: 2018 - int(x.birthday) if x.birthday_two is True else 0, axis=1)
    # print(user_info[['birthday_two', 'age']])
    feature.loc[:, 'age_one'] = user_info[['age']].apply(lambda x: 1 if x.age <= 18 and x.age > 0 else 0, axis=1)
    feature.loc[:, 'age_two'] = user_info[['age']].apply(lambda x: 1 if x.age <= 30 and x.age > 18 else 0, axis=1)
    feature.loc[:, 'age_three'] = user_info[['age']].apply(lambda x: 1 if x.age <= 60 and x.age > 30 else 0, axis=1)
    feature.loc[:, 'age_four'] = user_info[['age']].apply(lambda x: 1 if x.age <= 100 and x.age > 60 else 0, axis=1)
    feature.loc[:, 'age_five'] = user_info[['age']].apply(lambda x: 1 if x.age > 100 and x.age == 0 else 0, axis=1)

    return feature

# print(extract_user_info(train_user_info))
# print(extract_user_info(test_user_info))


def extract_recieve_addr_info(recieve_addr_info):
    '''提取 recieve_addr_info表 特征'''
    recieve_addr_info['all_null'] = recieve_addr_info[['addr_id', 'region', 'phone', 'fix_phone', 'receiver_md5']].apply(lambda x: 1 if (x.addr_id is np.nan) and (x.region is np.nan) and (x.phone is np.nan) and (x.fix_phone is np.nan) | (x.receiver_md5 is np.nan) else 0, axis=1)
    feature = recieve_addr_info.drop_duplicates(['id'])[['id']]
    recieve_addr_info['index'] = recieve_addr_info.index
    all_is_null = pd.pivot_table(recieve_addr_info, index='id', values='all_null', aggfunc='min').reset_index()
    addr_id = pd.pivot_table(recieve_addr_info, index='id', values='index', aggfunc='count').reset_index().rename(columns={'index': 'record_count'})
    feature = feature.merge(all_is_null, on='id', how='left')
    feature = feature.merge(addr_id, on='id', how='left')
    province = {'甘肃', '云南', '贵州', '河南', '黑龙', '香港', '北京', '湖南', '江苏', '青海', '宁夏', '内蒙', '浙江', '吉林', '海南', '福建', '重庆', '台湾', '陕西', '湖北', '江西', '辽宁', '山西', '西藏', '广东', '安徽', '四川', '河北', '山东', '上海', '广西', '新疆', '天津', 'null'}

    train_recieve_addr_info['province'] = train_recieve_addr_info[['region']].apply(lambda x: 'null' if x.region is np.nan else x.region[0:2], axis=1)
    city_set = pd.pivot_table(train_recieve_addr_info, index='id', values='province', aggfunc=return_set).reset_index()
    for string in list(province):
        city_set[string] = [1 if string in index else 0 for index in city_set['province']]
    city_set['province'] = city_set[['province']].apply(lambda x: x.province.clear() if 'null' in x.province else x.province, axis=1)
    city_set['province_len'] = [0 if index is None else len(index) for index in city_set['province']]

    feature = feature.merge(city_set.drop(['province'], axis=1), on='id', how='left')
    # print(feature)
    return feature

# extract_recieve_addr_info(train_recieve_addr_info)
# print(extract_recieve_addr_info(train_recieve_addr_info))


def extract_bankcard_info(bankcard_info):
    ''' 提取 bankcard_info表 特征 '''

    def cal_store_card_num(group):
        flag = 0
        for index in group:
            if index == '储蓄卡':
                flag += 1
        return flag

    def if_have_credit_card(group):
        for index in group:
            if index == '信用卡':
                return 1
            else:
                return 0
        return 0

    def list_set(group):
        return len(set(group))

    bankcard_info = bankcard_info.drop_duplicates()
    feature = bankcard_info.drop_duplicates(['id'])[['id']]
    card_record_count = pd.pivot_table(bankcard_info, index='id', values='phone', aggfunc='count').reset_index().rename(columns={'phone': 'card_record_count'})
    phone_count = pd.pivot_table(bankcard_info, index='id', values='phone', aggfunc=list_set).reset_index().rename(columns={'phone': 'phone_count'})
    store_card_count = pd.pivot_table(bankcard_info, index='id', values='card_type', aggfunc=cal_store_card_num).reset_index().rename(columns={'card_type': 'store_card_count'})
    have_credit_card = pd.pivot_table(bankcard_info, index='id', values='card_type', aggfunc=if_have_credit_card).reset_index().rename(columns={'card_type': 'have_credit_card'})
    card_category_count = pd.pivot_table(bankcard_info, index='id', values='card_type', aggfunc=list_set).reset_index().rename(columns={'card_type': 'card_category_count'})

    feature = feature.merge(phone_count, on='id', how='left')
    feature = feature.merge(card_record_count, on='id', how='left')
    feature = feature.merge(store_card_count, on='id', how='left')
    feature = feature.merge(have_credit_card, on='id', how='left')
    feature = feature.merge(card_category_count, on='id', how='left')
    feature['credit_count'] = feature['card_record_count'] - feature['store_card_count']
    feature['card_count_one'] = feature[['card_record_count']].apply(lambda x: 1 if x.card_record_count > 6 else 0, axis=1)
    feature['record_is_unique'] = feature[['card_record_count']].apply(lambda x: 1 if x.card_record_count == 1 else 0, axis=1)
    # print(feature)

    return feature

# extract_bankcard_info(train_bankcard_info)
# print(extract_bankcard_info(test_bankcard_info))


def extract_auth_info(auth_info):
    '''提取 auth_info表 特征'''
    feature = auth_info[['id']]
    feature.loc[:, 'auth_id_card_is_null'] = auth_info[['id_card']].apply(lambda x: 1 if x.id_card is not np.nan else 0, axis=1)
    feature.loc[:, 'auth_time_is_null'] = auth_info[['auth_time']].apply(lambda x: 1 if x.auth_time is not np.nan else 0, axis=1)
    feature.loc[:, 'phone_is_null'] = auth_info[['phone']].apply(lambda x: 1 if x.phone is not np.nan else 0, axis=1)
    feature.loc[:, 'all_is_null'] = auth_info[['id_card', 'auth_time', 'phone']].apply(lambda x: 1 if ((x.id_card is np.nan) and (x.auth_time is np.nan) and (x.phone is np.nan)) else 0, axis=1)
    feature.loc[:, 'all_not_null'] = auth_info[['id_card', 'auth_time', 'phone']].apply(lambda x: 1 if ((x.id_card is not np.nan) and (x.auth_time is not np.nan) and (x.phone is not np.nan)) else 0, axis=1)
    feature.loc[:, 'card_time_is_null'] = auth_info[['id_card', 'auth_time']].apply(lambda x: 1 if ((x.id_card is np.nan) and (x.auth_time is np.nan)) else 0, axis=1)
    feature.loc[:, 'time_phone_is_null'] = auth_info[['auth_time', 'phone']].apply(lambda x: 1 if ((x.phone is np.nan) and (x.auth_time is np.nan)) else 0, axis=1)
    # '''运营商'''
    # auth_info['id_card'] = [int(index[0]) if index is not np.nan else -1 for index in auth_info['id_card']]
    # auth_info['phone'] = [int(index[:3]) if index is not np.nan else -1 for index in auth_info['phone']]
    # mobile = {134, 135, 136, 137, 138, 139, 150, 151, 152, 157, 158, 159, 182, 183, 184, 187, 188, 147, 178}
    # unicom = {130, 131, 132, 155, 156, 185, 186, 145, 176}
    # telecom = {180, 181, 189, 133, 153, 177}
    # virtual = {170}
    # feature.loc[:, 'mobile'] = auth_info[['phone']].apply(lambda x: 1 if x.phone in mobile else 0, axis=1)
    # feature.loc[:, 'unicom'] = auth_info[['phone']].apply(lambda x: 1 if x.phone in unicom else 0, axis=1)
    # feature.loc[:, 'telecom'] = auth_info[['phone']].apply(lambda x: 1 if x.phone in telecom else 0, axis=1)
    # feature.loc[:, 'virtual'] = auth_info[['phone']].apply(lambda x: 1 if x.phone in virtual else 0, axis=1)
    # # 'mobile', 'unicom', 'telecom', 'virtual'
    # #  'id_card_one', 'id_card_two', 'id_card_three', 'id_card_four', 'id_card_five', 'id_card_six'
    # feature.loc[:, 'id_card_one'] = auth_info[['id_card']].apply(lambda x: 1 if x.id_card == 1 else 0, axis=1)
    # feature.loc[:, 'id_card_two'] = auth_info[['id_card']].apply(lambda x: 1 if x.id_card == 2 else 0, axis=1)
    # feature.loc[:, 'id_card_three'] = auth_info[['id_card']].apply(lambda x: 1 if x.id_card == 3 else 0, axis=1)
    # feature.loc[:, 'id_card_four'] = auth_info[['id_card']].apply(lambda x: 1 if x.id_card == 4 else 0, axis=1)
    # feature.loc[:, 'id_card_five'] = auth_info[['id_card']].apply(lambda x: 1 if x.id_card == 5 else 0, axis=1)
    # feature.loc[:, 'id_card_six'] = auth_info[['id_card']].apply(lambda x: 1 if x.id_card == 6 else 0, axis=1)
    # print(feature)
    return feature

# extract_auth_info(train_auth_info)
# print(extract_auth_info(test_auth_info))


def extract_order_info(order_info):
    '''提取 order_info表 特征'''
    def cal_set(group):
        return len(set(group))

    '''求标准差'''
    def cal_std(group):
        return np.std(group)

    feature = order_info.drop_duplicates(['id'])[['id']]
    # amt_order, type_pay, time_order, sts_order, phone, unit_price, no_order_md5, name_rec_md5, product_id_md5
    order_info['order_all_is_null'] = order_info.apply(lambda x: 1 if ((x.amt_order is np.nan) and (x.type_pay is np.nan) and (x.time_order is np.nan) and (x.sts_order is np.nan)) else 0, axis=1)
    order_all_is_null = pd.pivot_table(order_info[['id', 'order_all_is_null']], index='id', values='order_all_is_null', aggfunc='max').reset_index()

    '''均值填充amt_order属性'''
    order_info_amt = order_info[['amt_order']]
    order_info_amt = order_info_amt[order_info_amt['amt_order'].notnull()]
    order_info_amt = order_info_amt[order_info_amt['amt_order'] != 'null']
    order_info_amt['amt_order'] = [float(index) for index in order_info_amt['amt_order']]
    mean = order_info_amt['amt_order'].mean()
    order_info['amt_order'] = order_info['amt_order'].fillna(mean)
    order_info['amt_order'] = [mean if index == 'null' else index for index in order_info['amt_order']]
    order_info['amt_order'] = [float(index) for index in order_info['amt_order']]

    order_info['unit_price'] = order_info[['amt_order', 'unit_price']].apply(lambda x: x.amt_order if np.isnan(x.unit_price) else x.unit_price, axis=1)
    unit_price_mean = pd.pivot_table(order_info[['id', 'unit_price']], index='id', values='unit_price', aggfunc='mean').reset_index().rename(columns={'unit_price': 'unit_price_mean'})
    unit_price_max = pd.pivot_table(order_info[['id', 'unit_price']], index='id', values='unit_price', aggfunc='max').reset_index().rename(columns={'unit_price': 'unit_price_max'})
    unit_price_min = pd.pivot_table(order_info[['id', 'unit_price']], index='id', values='unit_price', aggfunc='min').reset_index().rename(columns={'unit_price': 'unit_price_min'})
    unit_price_std = pd.pivot_table(order_info[['id', 'unit_price']], index='id', values='unit_price', aggfunc=cal_std).reset_index().rename(columns={'unit_price': 'unit_price_std'})

    amt_order_mean = pd.pivot_table(order_info[['id', 'amt_order']], index='id', values='amt_order', aggfunc='mean').reset_index().rename(columns={'amt_order': 'amt_order_mean'})
    amt_order_max = pd.pivot_table(order_info[['id', 'amt_order']], index='id', values='amt_order', aggfunc='max').reset_index().rename(columns={'amt_order': 'amt_order_max'})
    amt_order_min = pd.pivot_table(order_info[['id', 'amt_order']], index='id', values='amt_order', aggfunc='min').reset_index().rename(columns={'amt_order': 'amt_order_min'})
    amt_order_std = pd.pivot_table(order_info[['id', 'amt_order']], index='id', values='amt_order', aggfunc=cal_std).reset_index().rename(columns={'amt_order': 'amt_order_std'})
    type_pay_count = pd.pivot_table(order_info[['id', 'type_pay']], index='id', values='type_pay', aggfunc=cal_set).reset_index().rename(columns={'type_pay': 'type_pay_count'})
    sts_order_count = pd.pivot_table(order_info[['id', 'sts_order']], index='id', values='sts_order', aggfunc=cal_set).reset_index().rename(columns={'sts_order': 'sts_order_count'})
    order_phone_count = pd.pivot_table(order_info[['id', 'phone']], index='id', values='phone', aggfunc=cal_set).reset_index().rename(columns={'phone': 'order_phone_count'})
    name_rec_md5_count = pd.pivot_table(order_info[['id', 'name_rec_md5']], index='id', values='name_rec_md5', aggfunc=cal_set).reset_index().rename(columns={'name_rec_md5': 'name_rec_md5_count'})

    feature = feature.merge(unit_price_mean, on='id', how='left')
    feature = feature.merge(unit_price_max, on='id', how='left')
    feature = feature.merge(unit_price_min, on='id', how='left')
    feature = feature.merge(unit_price_std, on='id', how='left')

    feature = feature.merge(order_all_is_null, on='id', how='left')
    feature = feature.merge(amt_order_mean, on='id', how='left')
    feature = feature.merge(amt_order_max, on='id', how='left')
    feature = feature.merge(amt_order_min, on='id', how='left')
    feature = feature.merge(amt_order_std, on='id', how='left')
    feature = feature.merge(type_pay_count, on='id', how='left')
    feature = feature.merge(sts_order_count, on='id', how='left')
    feature = feature.merge(order_phone_count, on='id', how='left')
    feature = feature.merge(name_rec_md5_count, on='id', how='left')
    '''归一化'''
    feature.iloc[:, 1:] = feature.iloc[:, 1:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))   # 0.791859501859 #
    '''离散化特征'''
    order_info['type_pay'] = order_info[['type_pay']].apply(lambda x: 'null' if x.type_pay is np.nan else x.type_pay, axis=1)
    type_pay = pd.pivot_table(order_info, index='id', values='type_pay', aggfunc=return_set).reset_index()
    # type_pay_category = {'定向京券支付', '白条支付', '分期付款', '积分支付', '在线+限品东券', '定向东券', '东券混合支付', '余额', '京豆东券混合支付', '前台自付', '在线', '在线+东券支付', '上门自提', '公司转账', '在线支付', '在线支付 ', '在线+京豆', '邮局汇款', '货到付款',
    #                      '在线+全品东券', 'null', '京豆支付', '在线预付', '定向京券', '混合支付', '京豆', '在线+定向东券', '京豆混合支付', '在线+东券'}

    type_pay_category = {'定向京券支付', '白条支付', '在线+余额+限品东券', '高校代理-代理支付', '京券全额支付', '分期付款', '积分支付', '在线+限品东券', '定向东券', '东券混合支付', '余额', '京豆东券混合支付', '前台自付', '在线', '在线+东券支付', '上门自提', '公司转账', '在线支付', '在线支付 ', '在线+京豆', '邮局汇款', '在线+全品京券', '货到付款', '分期付款(招行)', '在线+全品东券', '余额+限品东券', '在线+京券支付', '在线+余额', '限品京券', 'null', '京豆支付', '在线预付', '定向京券', '混合支付', '全品京券', '京豆', '在线+定向东券', '京豆混合支付', '在线+限品京券', '高校代理-自己支付', '京券混合支付', '在线+东券'}

    for string in list(type_pay_category):
        type_pay[string] = [1 if string in index else 0 for index in type_pay['type_pay']]

    type_pay['type_pay'] = type_pay[['type_pay']].apply(lambda x: x.type_pay.clear() if 'null' in x.type_pay else x.type_pay, axis=1)
    type_pay['type_pay_len'] = [0 if index is None else len(index) for index in type_pay['type_pay']]
    feature = feature.merge(type_pay.drop(['type_pay'], axis=1), on='id', how='left')

    '''sts_order离散化'''
    order_info['sts_order'] = order_info[['sts_order']].apply(lambda x: 'null' if x.sts_order is np.nan else x.sts_order, axis=1)
    # sts_order_category = set(train_order_info['sts_order'])
    sts_order = pd.pivot_table(order_info, index='id', values='sts_order', aggfunc=return_set).reset_index()
    sts_order_category = {'null', '等待审核', '等待处理', '已退款', '已收货', '购买成功', '付款成功', '失败退款', '已完成', '预订结束', '退款完成', '正在出库', '订单已取消', '充值成功', '商品出库', '下单失败', '请上门自提', '已晒单', '充值失败;退款成功',
                          '退款成功', '未入住', '等待收货', '配送退货', '出票失败', '等待付款确认', '缴费成功', '预约完成', '未抢中', '完成', '已取消', '出票成功', '抢票已取消', '等待付款', '已取消订单', '正在处理', '等待退款', '充值失败', '订单取消'}

    for string in list(sts_order_category):
        sts_order[string] = [1 if string in index else 0 for index in sts_order['sts_order']]

    sts_order['sts_order'] = sts_order[['sts_order']].apply(lambda x: x.sts_order.clear() if 'null' in x.sts_order else x.sts_order, axis=1)
    sts_order['sts_order_len'] = [0 if index is None else len(index) for index in sts_order['sts_order']]
    # print(sts_order)
    feature = feature.merge(sts_order.drop(['sts_order'], axis=1), on='id', how='left')

    # print(feature)
    return feature

# extract_order_info(train_order_info)
# print(extract_order_info(test_order_info))


def extract_time_feature(auth_info, target_list):
    '''提取时间相关特征'''
    feature = target_list[['id']]
    target_list = target_list[['id', 'appl_sbm_tm']].merge(auth_info[['id', 'auth_time']], on='id', how='left')
    target_list.loc[:, 'appl_sbm_tm'] = [index.split(' ')[0] for index in target_list['appl_sbm_tm']]
    target_list['auth_time'] = target_list[['appl_sbm_tm', 'auth_time']].apply(lambda x: x.appl_sbm_tm if x.auth_time == '0000-00-00' else x.auth_time, axis=1)
    target_list['auth_time'] = target_list[['appl_sbm_tm', 'auth_time']].apply(lambda x: x.appl_sbm_tm if x.auth_time is np.nan else x.auth_time, axis=1)
    feature['feature_1'] = target_list[['appl_sbm_tm', 'auth_time']].apply(lambda x: 1 if x.appl_sbm_tm < x.auth_time else 0, axis=1)
    feature['register_days'] = target_list[['appl_sbm_tm', 'auth_time']].apply(lambda x: (datetime(int(x.appl_sbm_tm.split('-')[0]), int(x.appl_sbm_tm.split('-')[1]), int(x.appl_sbm_tm.split('-')[2])) - datetime(int(x.auth_time.split('-')[0]), int(x.auth_time.split('-')[1]), int(x.auth_time.split('-')[2]))).days, axis=1)
    # print(target_list)
    # print(feature)
    return feature

# extract_time_feature(train_auth_info, train_target)
# print(extract_time_feature(test_auth_info, test_list))

def extract_order_payment_time(order_info, target_list):
    str_len = len('2016-01-19 22:38:26')
    feature = target_list[['id']]
    target_list = target_list[['id', 'appl_sbm_tm']].merge(order_info[['id', 'time_order']], on='id', how='left')
    target_list.loc[:, 'appl_sbm_tm'] = [index.split(' ')[0] for index in target_list['appl_sbm_tm']]
    target_list['time_order'] = target_list[['appl_sbm_tm', 'time_order']].apply(lambda x: x.appl_sbm_tm if x.time_order is np.nan else x.time_order, axis=1)
    target_list['time_order'] = target_list[['appl_sbm_tm', 'time_order']].apply(lambda x: x.appl_sbm_tm if len(x.time_order) != str_len else x.time_order, axis=1)
    target_list.loc[:, 'time_order'] = [index.split(' ')[0] for index in target_list['time_order']]
    target_list['days'] = target_list[['appl_sbm_tm', 'time_order']].apply(lambda x: (datetime(int(x.appl_sbm_tm.split('-')[0]), int(x.appl_sbm_tm.split('-')[1]), int(x.appl_sbm_tm.split('-')[2])) - datetime(int(x.time_order.split('-')[0]), int(x.time_order.split('-')[1]), int(x.time_order.split('-')[2]))).days, axis=1)
    print(target_list)
    day_mean = pd.pivot_table(target_list, index='id', values='days', aggfunc='mean').reset_index().rename(columns={'days': 'day_mean'})
    day_max = pd.pivot_table(target_list, index='id', values='days', aggfunc='max').reset_index().rename(columns={'days': 'day_max'})
    day_min = pd.pivot_table(target_list, index='id', values='days', aggfunc='min').reset_index().rename(columns={'days': 'day_min'})
    order_record_count = pd.pivot_table(target_list, index='id', values='days', aggfunc='count').reset_index().rename(columns={'days': 'order_record_count'})
    feature = feature.merge(day_mean, on='id', how='left')
    feature = feature.merge(day_max, on='id', how='left')
    feature = feature.merge(day_min, on='id', how='left')
    feature = feature.merge(order_record_count, on='id', how='left')     # 记录数 #
    feature.loc[:, 'order_record_unique'] = [1 if index == 1 else 0 for index in feature['order_record_count']]     # 记录数是否唯一 #
    print(feature)
    return feature

extract_order_payment_time(train_order_info, train_target)
# print(extract_order_payment_time(test_order_info, test_list))

'''Logistic Regression'''
def train_LR_module(store_result=False, store_feature=False, select_feature=False, feature_num='all', OneEncode=False):
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

    if OneEncode is True:
        features = list(train_feature.columns)
        one_hot = []
        continuous_feature = []
        for name in features:
            if len(set(train_feature[name])) == 2:
                one_hot.append(name)
            else:
                continuous_feature.append(name)

        feature = one_hot[:140] + continuous_feature
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
        module_two = LogisticRegression(penalty='l2', solver='sag', max_iter=500, random_state=42, n_jobs=4)
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
        sample = test_list[['id']]
        sample['predicted_score'] = [index for index in result['predicted_score']]
        sample.columns = ['ID', 'PROB']
        sample.to_csv(r'lr_sample.csv', index=None)
        # sample.to_csv(r'lgb_sample.csv', index=None)
        print(sample)
        print('结果已更新。。。')

    print(" Score_offline:", roc_auc_score(validate_label, module.predict_proba(validate_feature)[:, 1]))
    print('特征维数：', feature_num)



# def module_merge(prob_x, prob_l):
#     xgb_sample = pd.read_csv(r'xgb_sample.csv', low_memory=False)   # encode:159:0.790297834417
#     lr_sample = pd.read_csv(r'lr_sample.csv', low_memory=False)     # Uncode:0.792171452209
#     sample = xgb_sample.merge(lr_sample, on='ID', how='left')
#     sample['PROB'] = sample['PROB_x'] * prob_x + sample['PROB_y'] * prob_l
#     sample = sample[['ID', 'PROB']]
#     print(sample)
#     sample.to_csv(r'sample.csv', index=None)
#     print('模型已融合。。。')



# def module_merge(prob_xgb, prob_lr, prob_lgb):
#     xgb_sample = pd.read_csv(r'xgb_sample.csv', low_memory=False)   # encode:159:0.790297834417
#     lr_sample = pd.read_csv(r'lr_sample.csv', low_memory=False)     # Uncode:0.792171452209
#     lgb_sample = pd.read_csv(r'lgb_sample.csv', low_memory=False)
#
#     xgb_sample.columns = ['ID', 'PROB_xgb']
#     lr_sample.columns = ['ID', 'PROB_lr']
#     lgb_sample.columns = ['ID', 'PROB_lgb']
#     sample = xgb_sample.merge(lr_sample, on='ID', how='left')
#     sample = sample.merge(lgb_sample, on='ID', how='left')
#     # print(sample)
#     sample['PROB'] = sample['PROB_xgb'] * prob_xgb + sample['PROB_lr'] * prob_lr + sample['PROB_lgb'] * prob_lgb
#     sample = sample[['ID', 'PROB']]
#     print(sample)
#     sample.to_csv(r'sample.csv', index=None)
#     print('模型已融合。。。')
