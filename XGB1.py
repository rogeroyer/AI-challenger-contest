# -*- coding: utf-8 -*-
"""
Created on Sat May  19 22:02:28 2018

@author: Frank
"""



import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
from xgboost import plot_importance
import operator
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def setlen(group):
    return len(set(group))


def return_set(group):
    return set(group)


def auth_info(data):
    data['id_card_isnull'] = [1 if type(i) == str else 0 for i in data.id_card]
    data['phone_isnull'] = [1 if type(i) == str else 0 for i in data.phone]
    data['auth_time_isnull'] = [1 if type(i) == str else 0 for i in data.auth_time]
    data['first_bite'] = [i[0] if type(i) == str else '-1' for i in data['id_card']]
    id_card = ['2', '1', '3', '4', '6', '5']
    for i in id_card:
        data[i] = [1 if i == index else 0 for index in data['first_bite']]

    return data[['id_card_isnull', 'phone_isnull', 'auth_time_isnull', 'id']]


def bankcard_info(data):
    data['card_1'] = [1 if i == '储蓄卡' else 0 for i in data['card_type']]
    data['card_2'] = [1 if i == '信用卡' else 0 for i in data['card_type']]
    card_1_cnt = pd.pivot_table(data, index='id', values='card_1', aggfunc='sum').reset_index().rename(columns={'card_1': 'card_1_cnt'})
    data = data.merge(card_1_cnt, on='id', how='left')
    card_2_cnt = pd.pivot_table(data, index='id', values='card_2', aggfunc='sum').reset_index().rename(columns={'card_2': 'card_2_cnt'})
    data = data.merge(card_2_cnt, on='id', how='left')
    bank_cnt = pd.pivot_table(data, index='bank_name', values='tail_num', aggfunc='count').reset_index().rename(columns={'tail_num': 'bank_cnt'})
    id_bank_cnt = pd.pivot_table(data, index='id', values='bank_name', aggfunc='count').reset_index().rename(columns={'bank_name': 'id_bank_cnt'})
    # id_card1_cnt = pd.pivot_table(data, index='id', values='card_1', aggfunc='sum').reset_index().rename(columns={'card_1': 'id_card1_cnt'})
    # id_card2_cnt = pd.pivot_table(data, index='id', values='card_2', aggfunc='sum').reset_index().rename(columns={'card_2': 'id_card2_cnt'})
    id_phone_set = pd.pivot_table(data, index='id', values='phone', aggfunc=setlen).reset_index().rename(columns={'phone': 'id_phone_set'})
    id_card_set = pd.pivot_table(data, index='id', values='card_type', aggfunc=setlen).reset_index().rename(columns={'card_type': 'id_card_set'})
    id_bank_set = pd.pivot_table(data, index='id', values='bank_name', aggfunc=setlen).reset_index().rename(columns={'bank_name': 'id_bank_set'})

    data = data.merge(bank_cnt, on='bank_name', how='left')
    data = data.merge(id_bank_cnt, on='id', how='left')
    data = data.merge(id_phone_set, on='id', how='left')
    data = data.merge(id_card_set, on='id', how='left')  # ?
    data = data.merge(id_bank_set, on='id', how='left')  # ?
    return data[['id', 'card_1_cnt', 'card_2_cnt', 'id_bank_cnt', 'id_phone_set', 'id_card_set', 'id_bank_set']].drop_duplicates(['id'])


def credit_info(data):
    data['q_o'] = data['quota'] - data['overdraft']
    data['quota'] = [1 if i is np.nan else i for i in data['quota']]
    data['overdraft'] = [1 if i is np.nan else i for i in data['overdraft']]
    data['q/o'] = data[['quota', 'overdraft']].apply(lambda x: 0 if x.quota == 0 else x.overdraft/x.quota, axis=1)
    return data.drop_duplicates(['id'])


def order_info(data):
    id_sample = data.drop_duplicates(['id'])[['id']]

    data = data.drop_duplicates()
    order_info_amt = data[['amt_order']]
    order_info_amt = order_info_amt[order_info_amt['amt_order'].notnull()]
    order_info_amt = order_info_amt[order_info_amt['amt_order'] != 'null']
    order_info_amt['amt_order'] = [float(index) for index in order_info_amt['amt_order']]
    mean = order_info_amt['amt_order'].mean()
    data['amt_order'] = data['amt_order'].fillna(mean)
    data['amt_order'] = [mean if index == 'null' else index for index in data['amt_order']]
    data['amt_order'] = [float(index) for index in data['amt_order']]

    data['pay_way_1'] = [1 if i == '在线支付' else 0 for i in data['type_pay']]
    way1_cnt = pd.pivot_table(data, index='id', values='pay_way_1', aggfunc='sum').reset_index().rename(columns={'pay_way_1': 'way1_cnt'})
    id_sample = id_sample.merge(way1_cnt, on='id', how='left')
    data['pay_way_2'] = [1 if i == '货到付款' else 0 for i in data['type_pay']]
    way2_cnt = pd.pivot_table(data, index='id', values='pay_way_2', aggfunc='sum').reset_index().rename(columns={'pay_way_2': 'way2_cnt'})
    id_sample = id_sample.merge(way2_cnt, on='id', how='left')

    '''统计计数特征'''
    # f1 = pd.pivot_table(data[['id', 'amt_order']], index='id', values='amt_order', aggfunc='mean').reset_index().rename(columns={'amt_order': 'id_amt_order_mean'})
    # id_sample = id_sample.merge(f1, on='id', how='left')
    # f2 = pd.pivot_table(data[['id', 'amt_order']], index='id', values='amt_order', aggfunc='max').reset_index().rename(columns={'amt_order': 'id_amt_order_max'})
    # id_sample = id_sample.merge(f2, on='id', how='left')
    # f3 = pd.pivot_table(data[['id', 'amt_order']], index='id', values='amt_order', aggfunc='min').reset_index().rename(columns={'amt_order': 'id_amt_order_min'})
    # id_sample = id_sample.merge(f3, on='id', how='left')
    # f4 = pd.pivot_table(data[['id', 'amt_order']], index='id', values='amt_order', aggfunc='var').reset_index().rename(columns={'amt_order': 'id_amt_order_var'})
    # id_sample = id_sample.merge(f4, on='id', how='left')
    # f5 = pd.pivot_table(data[['id', 'unit_price']], index='id', values='unit_price', aggfunc='mean').reset_index().rename(columns={'unit_price': 'id_unit_price_mean'})
    # id_sample = id_sample.merge(f5, on='id', how='left')
    # f6 = pd.pivot_table(data[['id', 'unit_price']], index='id', values='unit_price', aggfunc='max').reset_index().rename(columns={'unit_price': 'id_unit_price_max'})
    # id_sample = id_sample.merge(f6, on='id', how='left')
    # f7 = pd.pivot_table(data[['id', 'unit_price']], index='id', values='unit_price', aggfunc='min').reset_index().rename(columns={'unit_price': 'id_unit_price_min'})
    # id_sample = id_sample.merge(f7, on='id', how='left')
    # f8 = pd.pivot_table(data[['id', 'unit_price']], index='id', values='unit_price', aggfunc='var').reset_index().rename(columns={'unit_price': 'id_unit_price_var'})
    # id_sample = id_sample.merge(f8, on='id', how='left')

    f9 = pd.pivot_table(data[['id', 'type_pay']], index='id', values='type_pay', aggfunc=setlen).reset_index().rename(columns={'type_pay': 'id_type_pay_set'})
    id_sample = id_sample.merge(f9, on='id', how='left')
    f10 = pd.pivot_table(data[['id', 'sts_order']], index='id', values='sts_order', aggfunc=setlen).reset_index().rename(columns={'sts_order': 'id_sts_order_set'})
    id_sample = id_sample.merge(f10, on='id', how='left')
    f11 = pd.pivot_table(data[['id', 'phone']], index='id', values='phone', aggfunc=setlen).reset_index().rename(columns={'phone': 'id_phone_set'})
    id_sample = id_sample.merge(f11, on='id', how='left')

    '''其他特征'''
    data['sts_order'] = data['sts_order'].fillna('0')
    data['wan_cheng'] = [1 if ('完成' in i) else 0 for i in data['sts_order']]
    wan_cheng_cnt = pd.pivot_table(data, index='id', values='wan_cheng', aggfunc='sum').reset_index().rename(columns={'wan_cheng': 'wan_cheng_cnt'})
    id_sample = id_sample.merge(wan_cheng_cnt, on='id', how='left')
    data['cheng_gong'] = [1 if '成功' in i else 0 for i in data['sts_order']]
    print(data['cheng_gong'])
    cheng_gong_cnt = pd.pivot_table(data, index='id', values='cheng_gong', aggfunc='sum').reset_index().rename(columns={'cheng_gong': 'cheng_gong_cnt'})
    id_sample = id_sample.merge(cheng_gong_cnt, on='id', how='left')
    data['qu_xiao'] = [1 if '取消' in i else 0 for i in data['sts_order']]
    qu_xiao_cnt = pd.pivot_table(data, index='id', values='qu_xiao', aggfunc='sum').reset_index().rename(columns={'qu_xiao': 'qu_xiao_cnt'})
    id_sample = id_sample.merge(qu_xiao_cnt, on='id', how='left')

    '''时间'''
    # year_month = ['1604', '1704', '1504', '1607', '1508', '1505', '1608', '1602', '1701', '1512', '1612', '1506', '1610', '1412', '1603', '00000000', '1601', '1611', '1605', '1606']
    # data['year_month'] = [i[2:4] + i[5: 7] if type(i) == str else '00000000' for i in data['time_order']]
    # for i in year_month:
    #     data[i] = [1 if i == index else 0 for index in data['year_month']]
    # t_1604 = pd.pivot_table(data, index='id', values='1604', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1604, on='id', how='left')
    # t_1704 = pd.pivot_table(data, index='id', values='1704', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1704, on='id', how='left')
    # t_1504 = pd.pivot_table(data, index='id', values='1504', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1504, on='id', how='left')
    # t_1607 = pd.pivot_table(data, index='id', values='1607', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1607, on='id', how='left')
    # t_1508 = pd.pivot_table(data, index='id', values='1508', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1508, on='id', how='left')
    # t_1505 = pd.pivot_table(data, index='id', values='1505', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1505, on='id', how='left')
    # t_1608 = pd.pivot_table(data, index='id', values='1608', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1608, on='id', how='left')
    # t_1602 = pd.pivot_table(data, index='id', values='1602', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1602, on='id', how='left')
    # t_1701 = pd.pivot_table(data, index='id', values='1701', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1701, on='id', how='left')
    # t_1512 = pd.pivot_table(data, index='id', values='1512', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1512, on='id', how='left')
    # t_1612 = pd.pivot_table(data, index='id', values='1612', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1612, on='id', how='left')
    # t_1506 = pd.pivot_table(data, index='id', values='1506', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1506, on='id', how='left')
    # t_1610 = pd.pivot_table(data, index='id', values='1610', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1610, on='id', how='left')
    # t_1412 = pd.pivot_table(data, index='id', values='1412', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1412, on='id', how='left')
    # t_1603 = pd.pivot_table(data, index='id', values='1603', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1603, on='id', how='left')
    # t_0000 = pd.pivot_table(data, index='id', values='00000000', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_0000, on='id', how='left')
    # t_1601 = pd.pivot_table(data, index='id', values='1601', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1601, on='id', how='left')
    # t_1611 = pd.pivot_table(data, index='id', values='1611', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1611, on='id', how='left')
    # t_1605 = pd.pivot_table(data, index='id', values='1605', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1605, on='id', how='left')
    # t_1606 = pd.pivot_table(data, index='id', values='1606', aggfunc='sum').reset_index()
    # id_sample = id_sample.merge(t_1606, on='id', how='left')


    # sts_order = []
    # for outcome in data['sts_order']:
    #     if type(outcome) == str:
    #         if "完成" in outcome:
    #             sts_order.append(1)
    #         else:
    #             sts_order.append(0)
    #     else:
    #         sts_order.append(0)
    # data['is_ok'] = sts_order
    # data['no_ok'] = [1 if i == 0 else 1 for i in sts_order]
    # wancheng_cnt = pd.pivot_table(data[['id', 'is_ok']], index='id', values='is_ok', aggfunc='sum').reset_index().rename(columns={'is_ok': 'wancheng_cnt'})
    # no_ok = pd.pivot_table(data[['id', 'no_ok']], index='id', values='no_ok', aggfunc='sum').reset_index().rename(columns={'is_ok': 'no_ok'})
    # id_sample = id_sample.merge(wancheng_cnt, on='id', how='left')
    # id_sample = id_sample.merge(no_ok, on='id', how='left')

    # type_pay = []
    # for outcome in data['type_pay']:
    #     if type(outcome) == str:
    #         if "在线支付" in outcome:
    #             type_pay.append(1)
    #         else:
    #             type_pay.append(0)
    #     else:
    #         type_pay.append(0)
    # data['zai_xian'] = sts_order
    # data['no_zai_xian'] = [1 if i == 0 else 1 for i in sts_order]
    # zai_xian_cnt = pd.pivot_table(data[['id', 'zai_xian']], index='id', values='zai_xian', aggfunc='sum').reset_index().rename(columns={'is_ok': 'zai_xian_cnt'})
    # no_zai_xian = pd.pivot_table(data[['id', 'no_zai_xian']], index='id', values='no_zai_xian', aggfunc='sum').reset_index().rename(columns={'is_ok': 'no_zai_xian'})
    # id_sample = id_sample.merge(zai_xian_cnt, on='id', how='left')
    # id_sample = id_sample.merge(no_zai_xian, on='id', how='left')

    return id_sample.drop_duplicates(['id'])


def recieve_addr_info(data):
    province = {'甘肃', '云南', '贵州', '河南', '黑龙', '香港', '北京', '湖南', '江苏', '青海', '宁夏', '内蒙', '浙江', '吉林', '海南', '福建', '重庆', '台湾', '陕西', '湖北', '江西', '辽宁', '山西', '西藏', '广东', '安徽', '四川', '河北', '山东', '上海',
                '广西', '新疆', '天津', 'null'}
    data['province'] = data[['region']].apply(lambda x: 'null' if x.region is np.nan else x.region[0:2], axis=1)
    city_set = pd.pivot_table(data, index='id', values='province', aggfunc=return_set).reset_index()
    for string in list(province):
        city_set[string] = [1 if string in index else 0 for index in city_set['province']]
    city_set['province_p'] = city_set[['province']].apply(lambda x: x.province.clear() if 'null' in x.province else x.province, axis=1)
    city_set['province_len'] = [0 if index is None else len(index) for index in city_set['province']]

    data['phone_isnull'] = [0 if type(i) == float else 1 for i in data.phone]
    data['fix_phone_isnull'] = [1 if type(i) == str else 0 for i in data.fix_phone]
    id_phone_set = pd.pivot_table(data[['id', 'phone']], index='id', values='phone', aggfunc=setlen).reset_index().rename(columns={'phone': 'id_phone_set'})
    data = data.merge(id_phone_set, on='id', how='left')
    data = data.merge(city_set, on='id', how='left')

    return data[['id', 'phone_isnull', 'fix_phone_isnull', 'id_phone_set', 'province_len']].drop_duplicates(['id'])


def user_info(data):
    id_sample = data[['id']]
    degree = ['本科', '初中', '中专', '其他', '硕士', '大专', '博士', '高中']
    for index in degree:
        id_sample[index] = [1 if index == string else 0 for string in data['degree']]

    id_sample['sex_isnull'] = [0 if type(index) == float else 1 for index in data['sex']]
    id_sample['sex1'] = [1 if index == '保密' else 0 for index in data['sex']]
    id_sample['sex2'] = [1 if index == '男' else 0 for index in data['sex']]
    id_sample['sex3'] = [1 if index == '女' else 0 for index in data['sex']]

    id_sample['0000-00-00'] = [1 if index == '0000-00-00' else 0 for index in data['birthday']]

    id_sample['merriage1'] = [1 if index == '未婚' else 0 for index in data['merriage']]
    id_sample['merriage2'] = [1 if index == '已婚' else 0 for index in data['merriage']]
    id_sample['merriage3'] = [1 if index == '保密' else 0 for index in data['merriage']]

    id_sample['income_isnull'] = [1 if type(index) == str else 0 for index in data['income']]
    id_sample['income1'] = [1 if index == '4000-5999元' else 0 for index in data['income']]
    id_sample['income2'] = [1 if index == '8000元以上' else 0 for index in data['income']]
    id_sample['income3'] = [1 if index == '2000-3999元' else 0 for index in data['income']]
    id_sample['income4'] = [1 if index == '6000-7999元' else 0 for index in data['income']]
    id_sample['income5'] = [1 if index == '2000元以下' else 0 for index in data['income']]

    id_sample['id_card_isnull'] = [1 if type(index) == str else 0 for index in data['id_card']]

    id_sample['qq_bound_one'] = [1 if index == '已绑定' else 0 for index in data['qq_bound']]
    id_sample['qq_bound_two'] = [1 if index == '未绑定' else 0 for index in data['qq_bound']]

    id_sample['wechat_bound_one'] = [1 if index == '已绑定' else 0 for index in data['wechat_bound']]
    id_sample['wechat_bound_two'] = [1 if index == '未绑定' else 0 for index in data['wechat_bound']]

    id_sample['account_grade_one'] = [1 if index == '注册会员' else 0 for index in data['account_grade']]
    id_sample['account_grade_two'] = [1 if index == '铜牌会员' else 0 for index in data['account_grade']]
    id_sample['account_grade_three'] = [1 if index == '银牌会员' else 0 for index in data['account_grade']]
    id_sample['account_grade_four'] = [1 if index == '金牌会员' else 0 for index in data['account_grade']]
    id_sample['account_grade_five'] = [1 if index == '钻石会员' else 0 for index in data['account_grade']]
    return id_sample.drop_duplicates(['id'])


def days_feature(auth, order, appl):
    # data = auth.merge(order, on='id', how='left')
    data = auth.merge(appl, on='id', how='left')
    data['auth_time'] = data[['appl_sbm_tm', 'auth_time']].apply(lambda x: x.appl_sbm_tm[:10] if x.auth_time == '0000-00-00' else x.auth_time, axis=1)
    data['auth_time'] = data[['appl_sbm_tm', 'auth_time']].apply(lambda x: x.appl_sbm_tm[:10] if x.auth_time is np.nan else x.auth_time, axis=1)
    data['days'] = data[['auth_time', 'appl_sbm_tm']].apply(lambda x: (datetime.datetime.strptime(x.appl_sbm_tm[:10], '%Y-%m-%d') - datetime.datetime.strptime(x.auth_time[:10], '%Y-%m-%d')).days, axis=1)
    data['days_is_neg'] = [1 if i > 0 else 0 for i in data['days']]
    data['auth_year'] = data[['auth_time']].apply(lambda x: int(x.auth_time[:4]), axis=1)
    data['appl_year'] = data[['appl_sbm_tm']].apply(lambda x: int(x.appl_sbm_tm[:4]), axis=1)
    data['years'] = data['appl_year'] - data['auth_year']
    data['years_is_neg'] =data[['years']].apply(lambda x: 1 if x.years > 0 else 0, axis=1)


    # data['auth_time'] = [i if type(i) == str else '0001-01-01' for i in auth['auth_time']]
    # data['auth_time'] = ['0001-01-01' if i == '0000-00-00' else i for i in auth['auth_time']]
    # data['auth_time'] = ['0001-01-01' if i == 0 else i for i in auth['auth_time']]
    # data['time_order'] = [i if type(i) == str else '0001-01-01 00:00:00' for i in appl['time_order']]
    # data['time_order'] = [i if len(i) > 16 else '0001-01-01 00:00:00' for i in appl['time_order']]
    #
    # data['time_days'] = data[['auth_time', 'time_order']].apply(lambda x: abs((datetime.datetime.strptime(x.time_order, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(x.auth_time, '%Y-%m-%d')).days), axis=1)
    # data['time_days'] = [i if ((i < 50000) & (i > 0)) else -1 for i in data['time_days']]
    # time_days_mean = pd.pivot_table(data[['id', 'time_days']], index='id', values='time_days', aggfunc='mean').reset_index().rename(columns={'time_days': 'time_days_mean'})
    # data = data.merge(time_days_mean, on='id', how='left')
    # data['time_days_mean_is_neg'] = [1 if i > 0 else 0 for i in data['time_days']]

    # appl['appl_sbm_tm'] = [i[:-2] for i in appl['appl_sbm_tm']]
    # data = data.merge(appl, on='id', how='left')
    # data['appl_age'] = data[['auth_time', 'appl_sbm_tm']].apply(lambda x: ((datetime.datetime.strptime(x.auth_time, '%Y-%m-%d') - datetime.datetime.strptime(x.appl_sbm_tm, '%Y-%m-%d %H:%M:%S')).days), axis=1)
    # data['appl_neg'] = [1 if i < 0 else 1 for i in data['appl_age']]
    print("OK")
    return data[['id', 'days', 'days_is_neg', 'years', 'years_is_neg']].drop_duplicates(['id'])


def auth_order(auth, order):
    data = auth.merge(order, on='id', how='left')
    data['auth_time'] = [i if type(i) == str else '0001-01-01' for i in data['auth_time']]
    data['auth_time'] = ['0001-01-01' if i == '0000-00-00' else i for i in data['auth_time']]
    data['auth_time'] = ['0001-01-01' if i == 0 else i for i in data['auth_time']]
    data['time_order'] = [i if type(i) == str else '0001-01-01 00:00:00' for i in data['time_order']]
    data['time_order'] = [i if len(i) > 16 else '0001-01-01 00:00:00' for i in data['time_order']]

    data['time_days'] = data[['auth_time', 'time_order']].apply(
        lambda x: abs((datetime.datetime.strptime(x.time_order, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(x.auth_time, '%Y-%m-%d')).days), axis=1)
    data['time_days'] = [i if ((i < 50000) & (i > 0)) else -1 for i in data['time_days']]
    time_days_mean = pd.pivot_table(data[['id', 'time_days']], index='id', values='time_days', aggfunc='mean').reset_index().rename(columns={'time_days': 'time_days_mean'})
    auth = auth.merge(time_days_mean, on='id', how='left')
    auth['time_days_mean_is_neg'] = [1 if i > 0 else 0 for i in auth['time_days_mean']]
    return auth[['id', 'time_days_mean', 'time_days_mean_is_neg']]


def submit():
    '''训练集读取、提特征'''
    train_auth_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_auth_info.csv', low_memory=False)           
    f_train_auth_info = auth_info(train_auth_info)
    train_bankcard_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_bankcard_info.csv', low_memory=False)
    f_train_bankcard_info = bankcard_info(train_bankcard_info)
    train_credit_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_credit_info.csv', low_memory=False)
    f_train_credit_info = credit_info(train_credit_info)
    train_order_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_order_info.csv', low_memory=False)
    f_train_order_info = order_info(train_order_info)
    train_recieve_addr_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_recieve_addr_info.csv', low_memory=False)
    f_train_recieve_addr_info = recieve_addr_info(train_recieve_addr_info)
    train_user_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_user_info.csv', low_memory=False)
    f_train_user_info = user_info(train_user_info)
    train_target = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_target.csv', low_memory=False)
    feature_l = train_target[['id', 'target']]
    f_day_minus = days_feature(train_auth_info[['id', 'auth_time']], train_order_info[['id', 'time_order']], train_target[['id', 'appl_sbm_tm']])
    f_auth_or = auth_order(train_auth_info, train_order_info)
    # print(f_day_minus)

    '''f_merge'''
    feature_l = feature_l.merge(f_train_auth_info, on='id', how='left')
    feature_l = feature_l.merge(f_train_bankcard_info, on='id', how='left')
    feature_l = feature_l.merge(f_train_credit_info, on='id', how='left')
    feature_l = feature_l.merge(f_train_order_info, on='id', how='left')
    feature_l = feature_l.merge(f_train_recieve_addr_info, on='id', how='left')
    feature_l = feature_l.merge(f_train_user_info, on='id', how='left')
    feature_l = feature_l.merge(f_day_minus, on='id', how='left')
    feature_l = feature_l.merge(f_auth_or, on='id', how='left')
    # feature_l.to_csv(r'F:\Python_project\AL\train_data\train_feature.csv', index=False)
    print(feature_l.shape)
    print(feature_l)
    train_f = feature_l.drop('target', axis=1)
    train_l = feature_l[['target']]

    xgb_train = xgb.DMatrix(train_f.values, label=train_l.values)
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 5,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.8,  # 随机采样训练样本
        'colsample_bytree': 0.8,  # 生成树时进行的列采样
        'min_child_weight': 18,
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.03,  # 如同学习率
        'eval_metric': 'logloss'
    }
    module = xgb.train(params, xgb_train, num_boost_round=500)


    '''测试集读取、提特征'''
    test_auth_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_data_Btest_V2.0\Btest_auth_info.csv', low_memory=False)
    f_test_auth_info = auth_info(test_auth_info)
    test_bankcard_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_data_Btest_V2.0\Btest_bankcard_info.csv', low_memory=False)
    f_test_bankcard_info = bankcard_info(test_bankcard_info)
    test_credit_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_data_Btest_V2.0\Btest_credit_info.csv', low_memory=False)
    f_test_credit_info = credit_info(test_credit_info)
    test_order_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_data_Btest_V2.0\Btest_order_info.csv', low_memory=False)
    f_test_order_info = order_info(test_order_info)
    test_recieve_addr_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_data_Btest_V2.0\Btest_recieve_addr_info.csv', low_memory=False)
    f_test_recieve_addr_info = recieve_addr_info(test_recieve_addr_info)
    test_user_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_data_Btest_V2.0\Btest_user_info.csv', low_memory=False)
    f_test_user_info = user_info(test_user_info)
    test_target = pd.read_csv(r'F:\Python_project\AL\AI_Risk_data_Btest_V2.0\Btest_list.csv', low_memory=False)
    test_fl = test_target[['id']]
    t_day_minus = days_feature(test_auth_info[['id', 'auth_time']], test_order_info[['id', 'time_order']], test_target[['id', 'appl_sbm_tm']])
    t_auth_or = auth_order(test_auth_info, test_order_info)

    '''merge'''
    test_fl = test_fl.merge(f_test_auth_info, on='id', how='left')
    test_fl = test_fl.merge(f_test_bankcard_info, on='id', how='left')
    test_fl = test_fl.merge(f_test_credit_info, on='id', how='left')
    test_fl = test_fl.merge(f_test_order_info, on='id', how='left')
    test_fl = test_fl.merge(f_test_recieve_addr_info, on='id', how='left')
    test_fl = test_fl.merge(f_test_user_info, on='id', how='left')
    test_fl = test_fl.merge(t_day_minus, on='id', how='left')
    test_fl = test_fl.merge(t_auth_or, on='id', how='left')


    test_f = test_fl
    test_l = test_fl[['id']]

    xgb_test = xgb.DMatrix(test_f.values)
    result = module.predict(xgb_test)
    test_l['predicted_score'] = result
    test_l.columns = ['ID', 'PROB']
    test_l.to_csv(r'result_xgb.csv', index=None)


def validation():
    '''训练集读取、提特征'''
    train_auth_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_auth_info.csv', low_memory=False)
    f_train_auth_info = auth_info(train_auth_info)
    train_bankcard_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_bankcard_info.csv', low_memory=False)
    f_train_bankcard_info = bankcard_info(train_bankcard_info)
    train_credit_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_credit_info.csv', low_memory=False)
    f_train_credit_info = credit_info(train_credit_info)
    train_order_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_order_info.csv', low_memory=False)
    f_train_order_info = order_info(train_order_info)
    train_recieve_addr_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_recieve_addr_info.csv', low_memory=False)
    f_train_recieve_addr_info = recieve_addr_info(train_recieve_addr_info)
    train_user_info = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_user_info.csv', low_memory=False)
    f_train_user_info = user_info(train_user_info)
    train_target = pd.read_csv(r'F:\Python_project\AL\AI_Risk_Train_V3.0/train_target.csv', low_memory=False)
    feature_l = train_target[['id', 'target']]
    day_minus = days_feature(train_auth_info[['id', 'auth_time']], train_order_info[['id', 'time_order']], train_target[['id', 'appl_sbm_tm']])
    auth_or = auth_order(train_auth_info[['id', 'auth_time']], train_order_info[['id', 'time_order']])

    '''划分验证集'''
    feature_l['date'] = [index.replace('-', '') for index in train_target['appl_sbm_tm']]
    feature_l['date'] = [index.split(' ')[0][0:6] for index in feature_l['date']]
    validation_train = feature_l[feature_l['date'] != '201704'][['target', 'id']]
    validation_test = feature_l[feature_l['date'] == '201704'][['target', 'id']]

    '''validation_train'''
    validation_train = validation_train.merge(f_train_auth_info, on='id', how='left')
    validation_train = validation_train.merge(f_train_bankcard_info, on='id', how='left')
    validation_train = validation_train.merge(f_train_credit_info, on='id', how='left')
    validation_train = validation_train.merge(f_train_order_info, on='id', how='left')
    validation_train = validation_train.merge(f_train_recieve_addr_info, on='id', how='left')
    validation_train = validation_train.merge(f_train_user_info, on='id', how='left')
    validation_train = validation_train.merge(day_minus, on='id', how='left')
    validation_train = validation_train.merge(auth_or, on='id', how='left')

    validation_train_f = validation_train.drop(['target', 'id'], axis=1)
    validation_train_l = validation_train[['target']]
    print(validation_train_f.columns)

    '''validation_test'''
    validation_test = validation_test.merge(f_train_auth_info, on='id', how='left')
    validation_test = validation_test.merge(f_train_bankcard_info, on='id', how='left')
    validation_test = validation_test.merge(f_train_credit_info, on='id', how='left')
    validation_test = validation_test.merge(f_train_order_info, on='id', how='left')
    validation_test = validation_test.merge(f_train_recieve_addr_info, on='id', how='left')
    validation_test = validation_test.merge(f_train_user_info, on='id', how='left')
    validation_test = validation_test.merge(day_minus, on='id', how='left')
    validation_test = validation_test.merge(auth_or, on='id', how='left')
    print(validation_test.shape)

    validation_test_f = validation_test.drop(['target', 'id'], axis=1)
    validation_test_l = validation_test[['target']]

    xgb_train = xgb.DMatrix(validation_train_f, label=validation_train_l)
    xgb_test = xgb.DMatrix(validation_test_f, label=validation_test_l)
    watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 5,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.8,  # 随机采样训练样本
        'colsample_bytree': 0.8,  # 生成树时进行的列采样
        'min_child_weight': 18,
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.03,  # 如同学习率
        'eval_metric': 'auc',
    }
    module = xgb.train(params, xgb_train, num_boost_round=500, evals=watchlist)
    result = module.predict(xgb_test)

    features = module.get_fscore()
    features = list(dict(sorted(features.items(), key=lambda d: d[1])).keys())[-20:]
    features.reverse()
    print(features)

    plot_importance(module)
    plt.show()
    print("auc: ", roc_auc_score(validation_test_l.values, result))


validation()


# auc:  0.809462477973
# submit()


'''one_hot'''
# testdata = pd.DataFrame({'pet': ['chinese', 'english', 'english', 'math'],
#                          'age': [6, 5, 2, 2],
#                          'salary': [7, 5, 2, 5]})
# one_hot = OneHotEncoder(sparse=False).fit_transform(testdata[['age']])
# print(one_hot)