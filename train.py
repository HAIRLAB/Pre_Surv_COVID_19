# -- coding:utf-8 --
import pandas as pd
import numpy as np
import os
from os.path import join as pjoin

from utils import is_number

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

warnings.filterwarnings('ignore')
#%matplotlib inline
# sns.set_style("whitegrid") 

plt.rcParams['font.sans-serif']=['Simhei']
plt.rcParams['axes.unicode_minus']=False


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

def devel_100(x, y):
    # 分层 K 折交叉验证
    acc_v = []
    acc_t = []
    
    for i in range(100):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        for tr_idx, te_idx in skf.split(x,y):
            x_tr = x[tr_idx, :]
            y_tr = y[tr_idx]
            x_te = x[te_idx, :]
            y_te = y[te_idx]
            model = xgb.XGBClassifier(max_depth=4
                      ,learning_rate=0.2
                      ,reg_alpha=1)
            model.fit(x_tr, y_tr)
            pred = model.predict(x_te)
            train_pred = model.predict(x_tr)
            acc_v.append(f1_score(y_te, pred))
            acc_t.append(f1_score(y_tr, train_pred))
    return [np.mean(acc_t), np.mean(acc_v), np.std(acc_t), np.std(acc_v)]

def read_train_data(path_train):

    data_df = pd.read_csv(path_train,encoding='gbk') #train_sample_375_v2 train_sample_351_v4
    # data_df = data_df.iloc[:,1:]
    data_df = data_df.set_index(['PATIENT_ID'])
    data_df['年龄'] = data_df['年龄'].apply(lambda x: x.replace('岁', '')
    if is_number(x.replace('岁', '')) else np.nan).astype(float)
    data_df['性别'] = data_df['性别'].map({'男': 1, '女': 2})
    data_df['护理->出院方式'] = data_df['护理->出院方式'].map({'治愈': 0,'好转': 0, '死亡': 1})
    lable = data_df['护理->出院方式'].values
    data_df = data_df.drop(['护理->出院方式', 'VISIT_ID', '首发症状', '临床症状', '临床表现', '流行病学史', '病情'], axis=1)
    data_df['Type2'] = lable
    data_df = data_df.applymap(lambda x: x.replace('>', '').replace('<', '') if isinstance(x, str) else x)
    data_df = data_df.applymap(lambda x: x if is_number(x) else -1)
    # data_df = data_df.loc[:, data_df.isnull().mean() < 0.2]
    data_df = data_df.astype(float)

    return data_df

## calculate miss values by col
def col_miss(train_df):
    col_missing_df = train_df.isnull().sum(axis=0).reset_index()
    col_missing_df.columns = ['col','missing_count']
    col_missing_df = col_missing_df.sort_values(by='missing_count')
    return col_missing_df

def data_preprocess():
    path_train = './data/train_sample_375_v2.csv'
    data_df_unna = read_train_data(path_train)

    data_pre_df = pd.read_csv('./data/sample29_v4.csv',encoding='gbk')
    data_pre_df = data_pre_df.drop_duplicates()
    
    return data_df_unna,data_pre_df

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

    # model.fit(x_2d, y_tr)

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

def plot_fig(X_data):
    cols = ['乳酸脱氢酶','淋巴细胞(%)','超敏C反应蛋白']
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

def main_1():
    data_df_unna,data_pre_df = data_preprocess()
    
    col_miss_data = col_miss(data_df_unna)
    col_miss_data['Missing_part'] = col_miss_data['missing_count']/len(data_df_unna)
    sel_cols = col_miss_data[col_miss_data['Missing_part']<=0.2]['col']
    data_df_sel = data_df_unna[sel_cols].copy()
    
    cols = list(data_df_sel.columns)
    cols.remove('年龄')
    cols.remove('性别')
    cols.remove('Type2')
    cols.append('Type2')
    
    data_df_sel2 = data_df_sel[cols]
    
    data_df_unna = pd.DataFrame()
    data_df_unna = data_df_sel2
    # data_df_unna = data_df_unna.append(data_df_0)
    # data_df_unna = data_df_unna.append(data_df_1)
    # data_df_unna = data_df_unna.append(data_df_2)
    data_df_unna = data_df_unna.fillna(-1)
    # print(data_df_unna.columns)
    
    x_col = cols[:-1]
    y_col = cols[-1]
    X_data = data_df_unna[x_col]#.values
    Y_data = data_df_unna[y_col]#.values

    name_dict = {'乳酸脱氢酶':'Lactate dehydrogenase (LDH)','淋巴细胞(%)':'Lymphocytes(%)','超敏C反应蛋白':'High-sensitivity C-reactive protein (hs-CRP)',
             '钠':'Sodium','氯':'Chlorine','国际标准化比值':'International Normalized Ratio (INR)','嗜酸细胞(#)':'Eosinophils(#)',
             '嗜酸细胞(%)':'Eosinophils(%)','单核细胞(%)':'Monocytes(%)','白蛋白':'Albumin'}
    
    import_feature = pd.DataFrame()
    import_feature['col'] = x_col
    import_feature['xgb'] = 0
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=i)
        model = xgb.XGBClassifier(
                max_depth=4
                ,learning_rate=0.2
                ,reg_lambda=1
                ,n_estimators=150
                ,subsample = 0.9
                ,colsample_bytree = 0.9)
        model.fit(x_train, y_train)
        import_feature['xgb'] = import_feature['xgb']+model.feature_importances_/100
    
    import_feature = import_feature.sort_values(axis=0, ascending=False, by='xgb')
    print('Top 10 features:')
    print(import_feature.head(10))
    # Sort feature importances from GBC model trained earlier
    indices = np.argsort(import_feature['xgb'].values)[::-1]
    Num_f = 10
    indices = indices[:Num_f]
    
    # Visualise these with a barplot
    # plt.subplots(dpi=400,figsize=(12, 10))
    plt.subplots(figsize=(12, 10))
    g = sns.barplot(y=list(name_dict.values())[:Num_f], x = import_feature.iloc[:Num_f]['xgb'].values[indices], orient='h') #import_feature.iloc[:Num_f]['col'].values[indices]
    # g = sns.barplot(y=import_feature.iloc[:Num_f]['col'].values[indices], x = import_feature.iloc[:Num_f]['xgb'].values[indices], orient='h') #import_feature.iloc[:Num_f]['col'].values[indices]
    g.set_xlabel("Relative importance",fontsize=18)
    g.set_ylabel("Features",fontsize=18)
    g.tick_params(labelsize=14)
    sns.despine()  # 去掉边框，默认去掉上边框和右边框
    # plt.savefig('feature_importances_v3.png')
    plt.show()
    # g.set_title("The mean feature importance of XGB models");

    import_feature_cols= import_feature['col'].values[:10]

    # 分割训练集和测试集 测试集占30%
    num_i = 1
    val_score_old = 0
    val_score_new = 0
    while val_score_new >= val_score_old:
        val_score_old = val_score_new

        x_col = import_feature_cols[:num_i]
        print(x_col)
        X_data = data_df_unna[x_col]#.values

        x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3,random_state=7)
        x_train = x_train.values
        x_test = x_test.values

        xlf=xgb.XGBClassifier(max_depth=4
                          ,learning_rate=0.2
                          ,reg_alpha=1)
        
        xlf.fit(x_train,y_train)
        weight_xgb_train = f1_score(y_train,xlf.predict(x_train),average='macro')
        print('Train f1-score is:',weight_xgb_train)
        weight_xgb = f1_score(y_test,xlf.predict(x_test),average='macro')
        print('Val f1-score is:',weight_xgb)

        y_train_xlf = xlf.predict(x_train)
        print('Train confusion matrix:')
        # show_confusion_matrix(y_train, y_train_xlf)
        print(classification_report(y_train, y_train_xlf))

        y_pre_xlf = xlf.predict(x_test)
        print('Validation confusion matrix:')
        # show_confusion_matrix(y_test, y_pre_xlf)
        print(classification_report(y_test, y_pre_xlf))

        data_pre_df = data_pre_df.fillna(-1)
        X_pre = data_pre_df[import_feature_cols[:num_i]].values
        y_pre_xlf = xlf.predict(X_pre)

        val_score_new = devel_100(X_data.values,Y_data.values)[1]

        ## 交叉验证
        print('5-Fold CV:')
        #devel_100(X_data.values,Y_data.values)
        print("Train f1-score is %.4f ; Validation f1-score is %.4f" % (devel_100(X_data.values,Y_data.values)[0],devel_100(X_data.values,Y_data.values)[1]))
        print("Train f1-score-std is %.4f ; Validation f1-score-std is %.4f" % (devel_100(X_data.values,Y_data.values)[2],devel_100(X_data.values,Y_data.values)[3]))

        Tets_Y = data_pre_df.reset_index()[['PATIENT_ID','出院方式']].copy()
        Tets_Y = Tets_Y.rename(columns={'PATIENT_ID': 'ID', '出院方式': 'Y'})
        Tets_Y['Y'] = (Tets_Y['Y'].map({'治愈':0,'好转':0,'死亡':1}))
        Y_true = Tets_Y['Y'].values

        print('Test confusion matrix:')
        # show_confusion_matrix(Y_true,y_pre_xlf)
        print(classification_report(Y_true, y_pre_xlf))
        test_score = f1_score(Y_true,y_pre_xlf,average='macro')
        print('Test f1-score is:',test_score)
        num_i += 1
        
    print('Selected features:',x_col[:-1])
    
    return list(x_col[:-1])
    
def main_2(cols=['乳酸脱氢酶','淋巴细胞(%)','超敏C反应蛋白']):
    data_df_unna,data_pre_df = data_preprocess()

    # cols =  ['乳酸脱氢酶','淋巴细胞(%)','超敏C反应蛋白']
    cols.append('Type2')

    data_df_unna = data_df_unna.reset_index()
    data_df_unna = data_df_unna.drop('PATIENT_ID',axis=1)
    data_df_unna = data_df_unna.fillna(-1)
    
    x_col = cols[:-1]
    y_col = cols[-1]
    X_data = data_df_unna[x_col]#.values
    Y_data = data_df_unna[y_col]#.values
    
    ## Plot
    plot_fig(data_df_unna)
    
    # 分割训练集和测试集 测试集占30%
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3,random_state=7)

    x_train = x_train.values
    x_test = x_test.values

    xlf=xgb.XGBClassifier(max_depth=4
                      ,learning_rate=0.2
                      ,reg_alpha=1)
    xlf.fit(x_train,y_train)
    weight_xgb_train = f1_score(y_train,xlf.predict(x_train),average='macro')
    print('Train:',weight_xgb_train)
    weight_xgb=f1_score(y_test,xlf.predict(x_test),average='macro')
    print('Val:',weight_xgb)

    y_train_xlf = xlf.predict(x_train)
    print('Train confusion matrix:')
    show_confusion_matrix(y_train, y_train_xlf)
    print(classification_report(y_train, y_train_xlf))

    y_pre_xlf = xlf.predict(x_test)
    print('Validation confusion matrix:')
    show_confusion_matrix(y_test, y_pre_xlf)
    print(classification_report(y_test, y_pre_xlf))

    data_pre_df = data_pre_df.fillna(-1)

    X_pre = data_pre_df[cols[:-1]].values

    y_pre_xlf = xlf.predict(X_pre)

    ## 交叉验证
    print('5-Fold CV:')
    #devel_100(X_data.values,Y_data.values)
    print("Train f1-score is %.4f ; Validation f1-score is %.4f" % (devel_100(X_data.values,Y_data.values)[0],devel_100(X_data.values,Y_data.values)[1]))
    print("Train f1-score-std is %.4f ; Validation f1-score-std is %.4f" % (devel_100(X_data.values,Y_data.values)[2],devel_100(X_data.values,Y_data.values)[3]))


    Tets_Y = data_pre_df.reset_index()[['PATIENT_ID','出院方式']].copy()
    Tets_Y = Tets_Y.rename(columns={'PATIENT_ID': 'ID', '出院方式': 'Y'})
    Tets_Y['Y'] = (Tets_Y['Y'].map({'治愈':0,'好转':0,'死亡':1}))
    Y_true = Tets_Y['Y'].values
    print(y_pre_xlf)
    
    print('Test confusion matrix:')
    show_confusion_matrix(Y_true,y_pre_xlf)
    print(classification_report(Y_true, y_pre_xlf))
    test_score = f1_score(Y_true,y_pre_xlf,average='macro')
    print('Test f1-score is:',test_score)
        
def main_3(cols=['乳酸脱氢酶','淋巴细胞(%)','超敏C反应蛋白']):
    data_df_unna,data_pre_df = data_preprocess()
    
    # cols =  ['乳酸脱氢酶','淋巴细胞(%)','超敏C反应蛋白']
    cols.append('Type2')
    
    Tets_Y = data_pre_df.reset_index()[['PATIENT_ID','出院方式']].copy()
    Tets_Y = Tets_Y.rename(columns={'PATIENT_ID': 'ID', '出院方式': 'Y'})
    Tets_Y['Y'] = (Tets_Y['Y'].map({'治愈':0,'好转':0,'死亡':1}))
    y_true = Tets_Y['Y'].values
    
    x_col = cols[:-1]
    y_col = cols[-1]
    x_np = data_df_unna[x_col].values
    y_np = data_df_unna[y_col].values
    
    x_test = data_pre_df[x_col].values

    X_train, X_val, y_train, y_val = train_test_split(x_np, y_np, test_size=0.3, random_state=6)
    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=1,
    )
    model.fit(X_train,y_train)

    #训练集混淆矩阵
    pred_train = model.predict(X_train)
    show_confusion_matrix(y_train, pred_train)
    print(classification_report(y_train, pred_train))

    #验证集混淆矩阵
    pred_val = model.predict(X_val)
    show_confusion_matrix(y_val, pred_val)
    print(classification_report(y_val, pred_val))
    #测试集混淆矩阵
    pred_test = model.predict(x_test)
    show_confusion_matrix(y_true, pred_test)
    print(classification_report(y_true, pred_test))

    #单树可视化
    ceate_feature_map(cols[:-1])
    graph = xgb.to_graphviz(model, fmap='xgb.fmap', num_trees=0, **{'size': str(10)})
    graph.render(filename='single-tree.dot')

if __name__ == '__main__':
    
    ## 特征筛选
    selected_cols = main_1()
    # 特征筛选完成后，筛选特征的预测
    # main_2()
    main_2(selected_cols)
    ## 单树可视化
    # main_3()
    main_3(selected_cols)
    
