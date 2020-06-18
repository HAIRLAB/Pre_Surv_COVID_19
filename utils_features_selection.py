# -- coding:utf-8 --
import pandas as pd
import numpy as np
import os
from os.path import join as pjoin

# from utils import is_number

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

from mpl_toolkits.mplot3d import Axes3D
import utils
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl

warnings.filterwarnings('ignore')
#%matplotlib inline
sns.set_style("white")

plt.rcParams['font.sans-serif']=['Simhei']
plt.rcParams['axes.unicode_minus']=False


######################
## CV functions
######################
## AUC and f1 score with CV

def StratifiedKFold_func_with_features_sel(x, y,Num_iter=100,score_type = 'auc'):
    # 分层 K 折交叉验证
    acc_v = []
    acc_t = []
    # 每次K折100次！
    for i in range(Num_iter):
        # 每次折是随机的random_state=i
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x,y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]
            #定义模型超参数
            model = xgb.XGBClassifier(max_depth=4,learning_rate=0.2,reg_alpha=1)
            #模型拟合
            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)
            #调用sklearn 的roc_auc_score 与f1_score计算相关指标
            ## 注明L此处用预测的标签值而不是预测概率求的AUC,原因是因为本文着重考虑预测区分生死，运用预测标签相当于在阈值确定为0.5的情况下模型的结果验证，
            ## 其AUC阈值分割点可视为分别在1，0.5，0, 这样更能反应特征的区分性能的差异性，找出能有区分度贡献的特征。
            if score_type == 'auc':
                acc_v.append(roc_auc_score(y_te, pred))
                acc_t.append(roc_auc_score(y_tr, train_pred))
            else:
                acc_v.append(f1_score(y_te, pred))
                acc_t.append(f1_score(y_tr, train_pred))    
    # 返回平均值
    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]

def StratifiedKFold_func(x, y,Num_iter=100,model = xgb.XGBClassifier(max_depth=4,learning_rate=0.2,reg_alpha=1), score_type ='auc'):
    # 模型在循环外的k折
    # 分层 K 折交叉验证
    acc_v = []
    acc_t = []
    
    for i in range(Num_iter):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x,y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]

            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)

            pred_Proba = model.predict_proba(x_te)[:,1]
            train_pred_Proba = model.predict_proba(x_tr)[:,1]

            if score_type == 'auc':
            	acc_v.append(roc_auc_score(y_te, pred_Proba))
            	acc_t.append(roc_auc_score(y_tr, train_pred_Proba))
            else:
            	acc_v.append(f1_score(y_te, pred))
            	acc_t.append(f1_score(y_tr, train_pred))            	

    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]

################################
## Read data functions
###############################
def read_train_data(path_train):
    data_df = pd.read_excel(path_train, encoding='gbk', index_col=[0, 1])  # train_sample_375_v2 train_sample_351_v4
    data_df = data_df.groupby('PATIENT_ID').last()
    # data_df = data_df.iloc[:,1:]
    # data_df = data_df.set_index(['PATIENT_ID'])
    # data_df['年龄'] = data_df['年龄'].apply(lambda x: x.replace('岁', '') if is_number(x.replace('岁', '')) else np.nan).astype(float)
    # data_df['性别'] = data_df['性别'].map({'男': 1, '女': 2})
    # data_df['护理->出院方式'] = data_df['护理->出院方式'].map({'治愈': 0,'好转': 0, '死亡': 1})
    lable = data_df['outcome'].values
    data_df = data_df.drop(['outcome', 'Admission time', 'Discharge time'], axis=1)
    data_df['Type2'] = lable
    data_df = data_df.applymap(lambda x: x.replace('>', '').replace('<', '') if isinstance(x, str) else x)
    data_df = data_df.applymap(lambda x: x if is_number(x) else -1)
    # data_df = data_df.loc[:, data_df.isnull().mean() < 0.2]
    data_df = data_df.astype(float)

    return data_df


def data_preprocess():
    path_train = './data/time_series_375_prerpocess_en.xlsx'  # to_ml
    data_df_unna = read_train_data(path_train)

    # data_pre_df = pd.read_csv('./data/sample29_v3.csv',encoding='gbk')
    data_pre_df = pd.read_excel('./data/time_series_test_110_preprocess_en.xlsx', index_col=[0, 1], encoding='gbk')
    data_pre_df = utils.merge_data_by_sliding_window(data_pre_df, n_days=1, dropna=True, subset=utils.top3_feats_cols,
                                                     time_form='diff')
    data_pre_df = data_pre_df.groupby('PATIENT_ID').first().reset_index()
    data_pre_df = data_pre_df.applymap(lambda x: x.replace('>', '').replace('<', '') if isinstance(x, str) else x)
    data_pre_df = data_pre_df.drop_duplicates()

    return data_df_unna, data_pre_df

### is_number in the read data
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

### Data read and split
def data_read_and_split(is_dropna=False,sub_cols=None):
    # data_df_unna为375数据集，data_pre_df为110数据集
    data_df_unna,data_pre_df = data_preprocess()
    if is_dropna==True:
        data_df_unna = data_df_unna.dropna(subset=sub_cols,how='any')

    # 计算特征的缺失情况
    col_miss_data = col_miss(data_df_unna)
    # 计算特征缺失比例
    col_miss_data['Missing_part'] = col_miss_data['missing_count']/len(data_df_unna)
    # 选择缺失少于0.2的特征
    sel_cols = col_miss_data[col_miss_data['Missing_part']<=0.2]['col']
    # copy函数将选择的特征数据摘出来，不影响原数据的数值
    data_df_sel = data_df_unna[sel_cols].copy()
    # 计算所有特征
    cols = list(data_df_sel.columns)
    # 剔除年龄和性别
    cols.remove('age')
    cols.remove('gender')
    cols.remove('Type2')
    cols.append('Type2')
    # 构造剔除上述特征的dataframe
    data_df_sel2 = data_df_sel[cols]
    # 新建一个dataframe
    data_df_unna = pd.DataFrame()
    # 类似copy方法，新建变量，修改不会影响原数值
    data_df_unna = data_df_sel2

    # 对缺失数值添-1
    data_df_unna = data_df_unna.fillna(-1)

    # 取出特征名，从第一列到倒数第二列
    x_col = cols[:-1]
    #print(x_col)
    # 取出标签名
    y_col = cols[-1]
    #取出375特征数据
    X_data = data_df_unna[x_col]#.values
    #取出375标签数据
    Y_data = data_df_unna[y_col]#.values

    return X_data,Y_data,x_col


## calculate miss values by col
def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col','missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df

######################
## Plot functions
######################
def show_confusion_matrix(validations, predictions):
    LABELS = ['Survival','Death']
    matrix = metrics.confusion_matrix(validations, predictions)
    # plt.figure(dpi=400,figsize=(4.5, 3))
    plt.figure(figsize=(4.5, 3))
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
    plt.show()

def plot_roc(labels, predict_prob,Moodel_name_i,fig,labels_name,k):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    #plt.figure()
    line_list = ['--','-']
    ax = fig.add_subplot(111)
    plt.title('ROC', fontsize=20)
    ax.plot(false_positive_rate, true_positive_rate,line_list[k%2],linewidth=1+(1-k/5),label=Moodel_name_i+' AUC = %0.4f'% roc_auc)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.xlabel('FPR', fontsize=20)
    labels_name.append(Moodel_name_i+' AUC = %0.4f'% roc_auc)
    #plt.show()
    return labels_name


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def plot_decision_boundary(model, x_tr, y_tr):
    """画出决策边界和样本点

    :param model: 输入 XGBoost 模型
    :param x_tr: 训练集样本
    :param y_tr: 训练集标签
    :return: None
    """
    # x_ss = StandardScaler().fit_transform(x_tr)
    # x_2d = PCA(n_components=2).fit_transform(x_ss)

    coord1_min = x_tr[:, 0].min() - 1
    coord1_max = x_tr[:, 0].max() + 1
    coord2_min = x_tr[:, 1].min() - 1
    coord2_max = x_tr[:, 1].max() + 1

    coord1, coord2 = np.meshgrid(
        np.linspace(coord1_min, coord1_max, int((coord1_max - coord1_min) * 30)).reshape(-1, 1),
        np.linspace(coord2_min, coord2_max, int((coord2_max - coord2_min) * 30)).reshape(-1, 1),
    )
    coord = np.c_[coord1.ravel(), coord2.ravel()]

    category = model.predict(coord).reshape(coord1.shape)
    # prob = model.predict_proba(coord)[:, 1]
    # category = (prob > 0.99).astype(int).reshape(coord1.shape)

    dir_save = './decision_boundary'
    os.makedirs(dir_save, exist_ok=True)

    # Figure
    plt.close('all')
    plt.figure(figsize=(7, 7))
    custom_cmap = ListedColormap(['#EF9A9A', '#90CAF9'])
    plt.contourf(coord1, coord2, category, cmap=custom_cmap)
    plt.savefig(pjoin(dir_save, 'decision_boundary1.png'), bbox_inches='tight')
    plt.scatter(x_tr[y_tr == 0, 0], x_tr[y_tr == 0, 1], c='yellow', label='Survival', s=30, alpha=1, edgecolor='k')
    plt.scatter(x_tr[y_tr == 1, 0], x_tr[y_tr == 1, 1], c='palegreen', label='Death', s=30, alpha=1, edgecolor='k')
    plt.ylabel('Lymphocytes (%)')
    plt.xlabel('Lactate dehydrogenase')
    plt.legend()
    # plt.savefig(pjoin(dir_save, 'decision_boundary2.png'), dpi=500, bbox_inches='tight')
    plt.show()

def plot_3D_fig(X_data):
    cols = ['Lactate dehydrogenase','(%)lymphocyte','High sensitivity C-reactive protein']
    X_data = X_data.dropna(subset=cols,how='all')
    col = 'Type2'
    data_df_sel2_0 = X_data[X_data[col]==0]
    data_df_sel2_1 = X_data[X_data[col]==1]
    
    # fig = plt.figure(dpi=400,figsize=(10, 4))
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    i= 2;j= 0;k= 1; # 120 201
    ax.scatter(data_df_sel2_0[cols[i]], data_df_sel2_0[cols[j]], data_df_sel2_0[cols[k]], c=data_df_sel2_0[col],cmap='Blues_r',label='Cured', linewidth=0.5)
    ax.scatter(data_df_sel2_1[cols[i]], data_df_sel2_1[cols[j]], data_df_sel2_1[cols[k]], c=data_df_sel2_1[col], cmap='gist_rainbow_r',label='Death',marker='x', linewidth=0.5)
 
    cols_en = ['Lactate dehydrogenase','Lymphocyte(%)','High-sensitivity C-reactive protein','Type of Survival(0) or Death(1)']
    ax.set_zlabel(cols_en[k])  # 坐标轴
    ax.set_ylabel(cols_en[j])
    ax.set_xlabel(cols_en[i])
    fig.legend(['Survival','Death'],loc='upper center')
    # plt.savefig('./picture_2class/3D_data_'+str(i)+str(j)+str(k)+'_v6.png')
    plt.show()
