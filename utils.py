# -- coding:utf-8 --
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report


# 常用参数
top3_feats_cols = ['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte']
in_out_time_cols = ['Admission time', 'Admission time']


def is_number(s):
    if s is None:
        s = np.nan

    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def read(path: str, usecols=None, is_ts='infer'):
    """读取处理后的数据
    合并 parquet, csv, excel 三种格式文件的读取函数

    :param path: 文件路径，必须是 parquet 或 csv 或 excel 文件
    :param usecols: 选取的列。与 pandas 接口不同，此处进行了简化，无需写索引列
    :param is_ts: 是否为时间序列。可选值：'infer', True, False
    :return: 读取的 DateFrame 数据
    """
    # 设置索引
    if is_ts == 'infer':
        index_col = [0, 1] if os.path.split(path)[1].startswith('time_series') else [0]
    elif is_ts is True:
        index_col = [0, 1]
    elif is_ts is False:
        index_col = [0]
    else:
        raise Exception('is_ts 参数错误')

    # 读取数据
    if path.endswith('.parquet'):
        data = pd.read_parquet(path)
    elif path.endswith('.csv'):
        try:
            data = pd.read_csv(path, index_col=index_col, encoding='gb18030')
        except UnicodeDecodeError:
            data = pd.read_csv(path, index_col=index_col, encoding='utf-8')
        except:
            raise
    elif path.endswith('.xlsx'):
        data = pd.read_excel(path, index_col=index_col)
    else:
        raise Exception('文件类型错误')

    # 提取指定列
    if usecols is not None:
        data = data[usecols]

    return data


def merge_data_by_sliding_window(data, n_days=1, dropna=True, subset=None, time_form='diff'):
    """滑窗合并数据

    :param data: 时间序列数据，一级行索引为 PATIENT_ID, 二级行索引为 RE_DATE
    :param n_days: 窗口长度
    :param dropna: 滑窗合并后还缺失的是否删掉
    :param subset: pd.DataFrame().dropna() 参数                                                   Note: 新参数!
    :param time_form: 返回数据的时间索引，'diff' or 'timestamp'
    :return: 合并后的数据，一级行索引为 PATIENT_ID, 二级行索引为 t_diff or RE_DATE, 取决于"time_form"
    """
    #根据PATIENT_ID排序
    data = data.reset_index(level=1)
    # dt.normalize() 取出院时间的天数
    # 距离出院时长        Note: 去掉了出院时间和检测时间的时分秒，因为我觉得以 00:00:00 为分界点更合适
    t_diff = data['Discharge time'].dt.normalize() - data['RE_DATE'].dt.normalize()
    # 滑窗取整的依据。即nn_days天内的会取整成为同一个数值，后面通过groupby方法分组
    data['t_diff'] = t_diff.dt.days.values // n_days * n_days
    #
    data = data.set_index('t_diff', append=True)

    # 滑窗合并。对['PATIENT_ID', 't_diff']groupby，相当于双循环。遍历所有病人与病人的所有窗口
    # 因为之前对data排序，因此每个病人t_diff会是从大到小的排序,ffill()是向上一行插值，因此相当于是向旧日期插值
    # last()是每一组取最后一行，因此即取每个病人对应窗口的最后一次数据，（也一定是最全的）。
    # last(）自带排序。取完last后会按照索引升序排列
    data = (
        data
        .groupby(['PATIENT_ID', 't_diff']).ffill()
        .groupby(['PATIENT_ID', 't_diff']).last()
    )
    # 去掉缺失样本
    if dropna:
        data = data.dropna(subset=subset)         # Note: 这里对缺失值进行了 dropna(), 而不是 fillna(-1)

    # 更新二级索引。（其实timestamp在本论文的中没用到）
    if time_form == 'timestamp':
        data = (
            data
            .reset_index(level=1, drop=True)
            .set_index('RE_DATE', append=True)
        )
    elif time_form == 'diff':
        data = data.drop(columns=['RE_DATE'])

    return data


def score_form(x: np.array):
    """打分表预测
    example: pred, score = score_form(df[['Lactate dehydrogenase','(%)lymphocyte','High sensitivity C-reactive protein']].values)

    :param x: 列顺序：['Lactate dehydrogenase','(%)lymphocyte','High sensitivity C-reactive protein']
    :return: 预测类别及最后得分
    """
    x = x.copy()

    # 乳酸脱氢酶
    x[:, 0] = pd.cut(
        x[:, 0],
        [-2, 107, 159, 210, 262, 313, 365, 416, 467, 519, 570, 622, 673, 724, 776, 827, 1e5],
        labels=list(range(-5, 11))
    )

    # 淋巴细胞(%)
    x[:, 1] = pd.cut(
        x[:, 1],
        [-2, 1.19, 3.12, 5.05, 6.98, 8.91, 10.84, 12.77, 14.7, 16.62, 18.55, 20.48, 22.41, 24.34, 1e5],
        labels=list(range(8, -6, -1))
    )

    # 超敏C反应蛋白
    x[:, 2] = pd.cut(
        x[:, 2],
        [-2, 19.85, 41.2, 62.54, 83.88, 1e5],
        labels=list(range(-1, 4))
    )

    # 统分
    total_score = x.sum(axis=1)

    # 1 分为临界点，大于 1 分死亡，小于 1 分治愈
    pred = (total_score > 1).astype(int)
    return pred, total_score


def decision_tree(x: pd.Series):
    """正文中的决策树
    example: df.apply(decision_tree, axis=1)

    :param x: 单个样本，['Lactate dehydrogenase','(%)lymphocyte','High sensitivity C-reactive protein']
    :return: 0: 治愈, 1: 死亡
    """
    if x['Lactate dehydrogenase'] >= 365:
        return 1

    if x['High sensitivity C-reactive protein'] < 41.2:
        return 0

    if x['(%)lymphocyte'] > 14.7:
        return 0
    else:
        return 1


def get_time_in_advance_of_predict(data):
    """提前预测正确的天数

    :param data: 时间序列数据，一级行索引为 PATIENT_ID, 二级行索引为 t_diff
    :return: pd.Series, index: PATIENT_ID, values: 提前预测正确的天数
    """
    # 由于python的机制，用copy新建一个data，不然会修改原dat
    data = data.copy()
    # 在data 这个dataframe中新建一列right，数值是判定是否正确
    data['right'] = data['pred'] == data['outcome']
    # 新建一个空列表，用于存储提前预测的正确的天数
    time_advance = []
    # data.index.remove_unused_levels().levels[0]表示的是病人id的list，即遍历所有病人
    for id_ in data.index.remove_unused_levels().levels[0]:
        # 因为病人id是一级索引，loc方法取出该病人对应的所有数据（可能有多条）
        d = data.loc[id_]
        # 如果病人只有一条数据单数据
        if len(d) == 1:
            if d.iloc[0]['right']:
                # 将预测对存入time_advance，分别为病人的id，正确的天数，出院的方式
                time_advance.append([id_, d.iloc[0].name, d['outcome'].iat[0]])
            continue

        # 多数据 Step1: 预测错
        if not d.iloc[0]['right']:
            continue

        # 多数据 Step2: 全对
        if d['right'].all():
            # 将预测对存入time_advance，分别为病人的id，正确的天数，出院的方式
            time_advance.append([id_, d.iloc[-1].name, d['outcome'].iat[0]])
            continue

        # 多数据 Step3: 部分对
        for i in range(len(d)):
            if d.iloc[i]['right']:
                continue
            else:
                # 将预测对存入time_advance，分别为病人的id，正确的天数，出院的方式
                time_advance.append([id_, d.iloc[i-1].name, d['outcome'].iat[0]])
                break

    # 将time_advance存成DataFrame
    time_advance = pd.DataFrame(time_advance, columns=['PATIENT_ID', 'time_advance', 'outcome'])
    time_advance = time_advance.set_index('PATIENT_ID')
    return time_advance


class Metrics:
    def __init__(self, report=None, acc=None, f1=None, conf_mat=None):
        self.y_trues  = []
        self.y_preds  = []

        # list or None. 'every': 每折都打印; 'overall': 打印总体的
        if isinstance(report, list):
            self.report = report
        else:
            self.report = [report]

        if isinstance(acc, list):
            self.acc = acc
        else:
            self.acc = [acc]

        if isinstance(f1, list):
            self.f1 = f1
        else:
            self.f1 = [f1]

        if isinstance(conf_mat, list):
            self.conf_mat = conf_mat
        else:
            self.conf_mat = [conf_mat]

    def record(self, y_true, y_pred):
        self.y_trues.append(y_true)
        self.y_preds.append(y_pred)
        return self

    def clear(self):
        self.y_trues = []
        self.y_preds = []
        return self

    def print_metrics(self):
        """打印指标

        :param report:
        :param acc:
        :param f1:
        :param conf_mat:
        :return:
        """
        # Loop: 'every'
        acc_values, f1_values = [], []
        single_fold = True if len(self.y_trues) == 1 else False
        for i, (y_true, y_pred) in enumerate(zip(self.y_trues, self.y_preds)):
            assert (y_true.ndim == 1) and (y_pred.ndim == 1)
            print(f'\n======================== 第 {i+1} 折指标 ========================>')

            # Classification_report
            if (self.report is not None) and ('every' in self.report):
                print(classification_report(y_true, y_pred))

            # Accuracy_score
            a_v = accuracy_score(y_true, y_pred)
            acc_values.append(a_v)
            if (self.acc is not None) and ('every' in self.acc):
                print(f"accuracy: {a_v:.05f}")

            # F1_score
            f1_v = f1_score(y_true, y_pred, average='macro')
            f1_values.append(f1_v)
            if (self.f1 is not None) and ('every' in self.f1):
                print(f"F1: {f1_v:.05f}")

            # Confusion_matrix
            if (self.conf_mat is not None) and ('every' in self.conf_mat):
                print(f"混淆矩阵：\n{confusion_matrix(y_true, y_pred)}")

        # 'Overall'
        print('\n======================== 总体指标 ========================>')
        y_true = np.hstack(self.y_trues)
        y_pred = np.hstack(self.y_preds)

        # Classification_report
        if (self.report is not None) and ('overall' in self.report):
            print(classification_report(y_true, y_pred))

        # Accuracy_score
        if (self.acc is not None) and ('overall' in self.acc):
            if single_fold:
                print(f"accuracy：\t{acc_values[0]: .04f}")
            else:
                print(f"accuracy：\t{np.mean(acc_values): .04f} / {'  '.join([str(a_v.round(2)) for a_v in acc_values])}")

        # F1_score
        if (self.f1 is not None) and ('overall' in self.f1):
            if single_fold:
                print(f"F1-score：\t{f1_values[0]: .04f}")
            else:
                print(f"F1 均值：\t{np.mean(f1_values): .04f} / {'  '.join([str(f1_v.round(2)) for f1_v in f1_values])}")

        # Confusion_matrix
        if (self.conf_mat is not None) and ('overall' in self.conf_mat):
            print(f"混淆矩阵：\n{confusion_matrix(y_true, y_pred)}")


def feat_zh2en(data):
    """特征名中文转英文"""
    feats_zh = data.columns

    # 显示哪些列没有中英翻译
    feats_map = pd.read_excel('data/raw_data/特征名_zh2en/特征名_zh2en.xlsx', index_col=0)['en']
    out_of_map = set(feats_zh) - set(feats_map.index)
    print(f"缺少翻译的特征：{out_of_map}")

    # 开始翻译
    feats_map = feats_map.to_dict()
    data = data.rename(columns=feats_map)
    return data


def concat_data(data375: pd.DataFrame, data110: pd.DataFrame):
    """整合 375 + 110
    因为 PATIENT_ID 都从 1 开始，所以整合时需要调整，避免重合

    :param data375:
    :param data110:
    :return:
    """
    data110 = data110.reset_index()
    data110['PATIENT_ID'] += 375
    data110 = data110.set_index(['PATIENT_ID', 'RE_DATE'])
    data = data375.append(data110)
    return data
