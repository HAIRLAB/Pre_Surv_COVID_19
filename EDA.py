# -- coding:utf-8 --
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, precision_score, recall_score

import utils
import warnings

warnings.filterwarnings("ignore")

font = {
    'family': 'Tahoma',
    'size': 20
}



def predicted_time_horizon():
    """
    分布图——提前预测时长,对应正文的Figure 3中的B C

    孙川  2020.04.02
    """
    #读取375和110的病人数据，并限定特征为['乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)', '出院时间', '出院方式']
    data1 = pd.read_parquet('data/time_series_375.parquet')[['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte', 'Discharge time', 'outcome']]
    data2 = pd.read_parquet('data/time_series_test_110.parquet')[['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte', 'Discharge time', 'outcome']]
    #for 循环，分别画正文Figure 3的图B，即375+110=485的图；和图C 110病人的图
    for data in [utils.concat_data(data1, data2), data2]:
        # 将3特征存在缺省的样本删掉
        data = data.dropna(how='all', subset=['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte'])

        # 滑窗合并数据
        data = utils.merge_data_by_sliding_window(data, n_days=1, dropna=True, subset=utils.top3_feats_cols, time_form='diff')
        # 根据一级（PATINET_ID）和二级索引（距离出院的时间）排序
        data = data.sort_index(level=(0, 1))

        # 调用apply方法对每个样本进行判断 即 论文决策树预测
        data['pred'] = data.apply(utils.decision_tree, axis=1)

        # 统计提前预测时长
        time_advance = utils.get_time_in_advance_of_predict(data)['time_advance']

        # Figure
        plt.figure(dpi=200)
        plt.hist(time_advance, bins=100)
        plt.title('Predicted time horizon', fontdict=font)
        plt.xticks(fontsize=font['size'])
        plt.yticks(fontsize=font['size'])
        plt.xlabel('days to outcome', fontdict=font)
        plt.ylabel('Frequency', fontdict=font)
        x_max = plt.gca().get_xlim()[1]
        y_max = plt.gca().get_ylim()[1]
        plt.text(0.65 * x_max, 0.75 * y_max, f"mean {np.mean(time_advance):.2f}\nstd     {np.std(time_advance):.2f}", fontdict=font)
        plt.show()


def decision_tree_top3_feats_predict_result():
    """
    用所有次的 ['乳酸脱氢酶', '超敏C反应蛋白', '淋巴细胞(%)'] 进行预测

    孙川  2020.04.07
    """
    # 决定是否使用每个病人最后一天的数据
    last_sample = False

    data1 = pd.read_parquet('data/time_series_375.parquet')[['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte', 'Discharge time', 'outcome']]
    data2 = pd.read_parquet('data/time_series_test_110.parquet')[['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte', 'Discharge time', 'outcome']]

    # data1是375 data2是110 concat是485
    for data in [data1, data2, utils.concat_data(data1, data2)]:
        # 滑窗合并数据
        data = utils.merge_data_by_sliding_window(data, n_days=1, dropna=True, subset=utils.top3_feats_cols, time_form='diff')

        #是否使用最后一次的数据，因为merge_data_by_sliding_window中last自带升序因此这里取first()
        #groupby后，每个病人的索引是二级索引t_diff，是升序
        if last_sample:
            data = data.groupby('PATIENT_ID').first()

        # 论文决策树预测
        data['pred'] = data.apply(utils.decision_tree, axis=1)

        # 调用自己写的结果统计方式utils.Metrics
        metrics = utils.Metrics(acc='overall', f1='overall', conf_mat='overall',report='overall')
        metrics.record(data['outcome'], data['pred'])
        metrics.print_metrics()


def hospitalization_time():
    """
    统计 375 + 110 住院时长

    孙川 2020.04.21
    """
    data375 = (
        pd.read_excel('data/time_series_375_prerpocess_en.xlsx', index_col=[0, 1])
        .groupby('PATIENT_ID')[['Admission time', 'Discharge time']].last()
    )
    data110 = (
        pd.read_excel('data/time_series_test_110_preprocess_en.xlsx', index_col=[0, 1])
        .groupby('PATIENT_ID')[['Admission time', 'Discharge time']].last()
    )
    data485 = data375.append(data110, ignore_index=True)

    duration = data485['Discharge time'] - data485['Admission time']


    print(f"患者数目：{len(duration)}")
    print(f"最短住院时长：{duration.min()}")
    print(f"最长住院时长：{duration.max()}")
    print(f"住院时长中位数：{duration.median()}")

def show_confusion_matrix(validations, predictions, path='confusion.png'):
    """
    绘制混淆矩阵

    郭裕祺  2020.04.02
    """
    LABELS = ['Survival', 'Death']
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(path, dpi=500, bbox_inches='tight')
    plt.show()


def ceate_feature_map(features):
    """
    可视化单树时，构造特征名对应的fmap

    郭裕祺  2020.04.02
    """
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def compute_f1_all(data, features, days=10):
    """
    计算每个时间段的f1 score

    郭裕祺  2020.04.02
    """
    day_list = list(range(0, days + 1))
    sample_num = []
    survival_num = []
    death_num = []
    f1 = []
    precision = []
    recall = []
    add_before_f1 = []
    for i in range(0, days + 1):
        if i == 0:
            # 有的病人出院或死亡后的几个小时最后一次检测才会出来
            # data_subset第i天的数据
            data_subset = data.loc[data['t_diff'] <= 0].groupby('PATIENT_ID').last()
            data_subset_sum = data.loc[data['t_diff'] <= 0]
        else:
            # data_subset是<=i的数据
            data_subset = data.loc[data['t_diff'] == i].groupby('PATIENT_ID').last()
            data_subset_sum = data.loc[data['t_diff'] <= i]
        # 统计对应子集的结果
        if data_subset.shape[0] > 0:
            sample_num.append(data_subset.shape[0])
            survival_num.append(sum(data_subset['outcome'] == 0))
            death_num.append(sum(data_subset['outcome'] == 1))
            pred = data_subset[features].apply(utils.decision_tree, axis=1)

            f1.append(f1_score(data_subset['outcome'].values, pred, average='macro'))
            precision.append(precision_score(data_subset['outcome'].values, pred))

            recall.append(recall_score(data_subset['outcome'].values, pred))

            add_before_f1.append(f1_score(data_subset_sum['outcome'].values,
                                          data_subset_sum[features].apply(utils.decision_tree, axis=1),
                                          average='macro'))

        else:
            sample_num.append(np.nan)
            survival_num.append(np.nan)
            death_num.append(np.nan)
            f1.append(np.nan)
            precision.append(np.nan)
            recall.append(np.nan)
            add_before_f1.append(np.nan)
    return day_list, f1, precision, recall, sample_num, survival_num, death_num, add_before_f1


def plot_f1_time_single_tree(data, features, path='image/f1_score_time.png'):
    """
    绘制论文中单树在不同时间段数据上的f1 score

    郭裕祺  2020.04.04
    """
    test_model_result = pd.DataFrame()
    test_model_result['day'], test_model_result['f1-score'], test_model_result['precision-score'], \
    test_model_result['recall-score'], test_model_result['sample_num'], test_model_result['survival_num'], \
    test_model_result['death_num'], test_model_result['add_before_f1'] = compute_f1_all(data, features, days=18)
    # 画f1-时间曲线
    fig = plt.figure(figsize=(8, 6))
    plt.tick_params(labelsize=20)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    n1 = ax.bar(test_model_result['day'], test_model_result['sample_num'], label='Death', color='red', alpha=0.5
                , zorder=0)
    n2 = ax.bar(test_model_result['day'], test_model_result['survival_num'], label='Survival', color='lightgreen',
                alpha=1, zorder=5)
    p1 = ax2.plot(test_model_result['day'], test_model_result['f1-score'], marker='o', linestyle='-', color='black',
                  label='f1 score', zorder=10)

    p2 = ax2.plot(test_model_result['day'], test_model_result['add_before_f1'], marker='o', linestyle='-', color='blue',
                  label='cumulative f1 score', zorder=10)

    fig.legend(loc='center left', bbox_to_anchor=(1.1, 0.95), bbox_transform=ax.transAxes, fontsize=14)

    ax.set_xlabel('days to outcome', fontsize=20)
    ax2.set_ylabel('f1-score(macro avg)', fontsize=20)
    ax.set_ylabel('sample_num', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.xticks(list(range(0, 19, 2)))
    plt.savefig(path, dpi=500, bbox_inches='tight')
    plt.show()


def plot_f1_time_single_tree_test_train():
    """
    绘制论文中单树在训练集测试集不同时间段数据上的f1 score
    对应正文figure3 中DE

    郭裕祺  2020.04.04
    """
    features = ['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte']
    data1 = pd.read_parquet('data/time_series_375.parquet')[['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte', 'Discharge time', 'outcome']]
    data2 = pd.read_parquet('data/time_series_test_110.parquet')[['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte', 'Discharge time', 'outcome']]
    data = data1.append(data2)
    # 滑窗合并数据
    data = utils.merge_data_by_sliding_window(data, n_days=1, dropna=True, time_form='diff')
    data = data.sort_index(level=(0, 1), ascending=True)
    data = data.reset_index()
    data = data.dropna(how='all', subset=['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte'])

    data2 = utils.merge_data_by_sliding_window(data2, n_days=1, dropna=True, time_form='diff')
    data2 = data2.sort_index(level=(0, 1), ascending=True)
    data2 = data2.reset_index()
    data2 = data2.dropna(how='all', subset=['Lactate dehydrogenase', 'High sensitivity C-reactive protein', '(%)lymphocyte'])


    # 训练集+验证集+测试集的结果
    plot_f1_time_single_tree(data, features, path='image/f1_time_train_test.png')

    # 测试集结果
    plot_f1_time_single_tree(data2, features, path='image/f1_time_test.png')



if __name__ == '__main__':
    # predicted_time_horizon()
    decision_tree_top3_feats_predict_result()
    # xgb_predict_result()
    # hospitalization_time()
    # plot_f1_time_single_tree_test_train()
