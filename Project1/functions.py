# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:48:09 2022

@author: Ja
"""
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def gini_roc(y_test, y_pred_proba, tytul):
    
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    
    plt.plot(fpr,tpr)
    plt.title(tytul)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    roc_auc = metrics.auc(fpr, tpr)
    gini = (2 * roc_auc) - 1

    return gini

def gini_train_val(model, X_train, y_train, X_val, y_val):
    
    y_pred_proba = model.predict_proba(X_train)[::,1]
    gini_train = gini_roc(y_train, y_pred_proba, "ROC Curve for Training Sample")
    print("gini_test: %.4f" % gini_train)
    
    y_pred_proba = model.predict_proba(X_val)[::,1]
    gini_val = gini_roc(y_val, y_pred_proba, "Roc Curve for Validation Sample")
    print("gini_val: %.4f" % gini_val)

    return

def TestTransform(census_df_test):
    census_df_test.drop("fnlwgt", axis=1, inplace=True)
    census_df_test.drop("education", axis=1, inplace=True)
    census_df_test['income_level'] = census_df_test['income_level'].replace(['<=50K','>50K'],[0, 1])
    census_df_test['age'] = pd.qcut(census_df_test['age'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    census_df_test['hours_per_week'] = pd.cut(census_df_test['hours_per_week'], bins=[0, 35, 45, np.inf], labels=[1, 2, 3])
    census_df_test['capital_gain'] = pd.cut(census_df_test['capital_gain'], bins=[-1, 114.0, 3103.0, 5013.0, 7688.0, 15024.0, np.inf], 
                 labels=[0, 1, 2, 3, 4, 5])
    census_df_test['capital_loss'] = pd.cut(census_df_test['capital_loss'], bins=[-1, 155.0, 1619.2, 1887.0, 1902.0, 2002.0, np.inf], 
                 labels=[0, 1, 2, 3, 4, 5])
    
    census_df_test.loc[(census_df_test.native_country != 'United-States') & (census_df_test.native_country != 'United-States'), 'native_country'] = "Other"
    
    country_df_test = pd.get_dummies(census_df_test['native_country'], prefix='country_')
    census_df_test = pd.concat([census_df_test, country_df_test], axis=1)
    census_df_test.drop(['native_country'],  axis=1, inplace = True)
    df = census_df_test
    return df


def WoETransform(census_df):

    
    census_df['workclass'] = census_df['workclass'].replace(['Without-pay', 'Never-worked', '?'] , 0)
    census_df['workclass'] = census_df['workclass'].replace(['Private'] , 1)
    census_df['workclass'] = census_df['workclass'].replace(['Self-emp-not-inc', 'State-gov', 'Local-gov'] , 2)
    census_df['workclass'] = census_df['workclass'].replace(['Federal-gov'] , 3)
    census_df['workclass'] = census_df['workclass'].replace(['Self-emp-inc'] , 4)

    census_df['occupation'] = census_df['occupation'].replace(['Priv-house-serv'] , 1)
    census_df['occupation'] = census_df['occupation'].replace(['Other-service', 'Handlers-cleaners' ] , 2)
    census_df['occupation'] = census_df['occupation'].replace(['?', 'Adm-clerical','Machine-op-inspct', 'Farming-fishing'] , 3)
    census_df['occupation'] = census_df['occupation'].replace(['Transport-moving', 'Craft-repair'] , 4)
    census_df['occupation'] = census_df['occupation'].replace(['Sales', 'Tech-support', 'Protective-serv', 'Armed-Forces'] , 5)
    census_df['occupation'] = census_df['occupation'].replace(['Prof-specialty', 'Exec-managerial'] , 6)

    census_df['relationship'] = census_df['relationship'].replace(['Own-child'] , 1)
    census_df['relationship'] = census_df['relationship'].replace(['Other-relative'] , 2)
    census_df['relationship'] = census_df['relationship'].replace(['Unmarried'] , 3)
    census_df['relationship'] = census_df['relationship'].replace(['Not-in-family'] , 4)
    census_df['relationship'] = census_df['relationship'].replace(['Wife', 'Husband'] , 5)
    
    census_df['marital_status'] = census_df['marital_status'].replace(['Never-married', 'Separated'] , 1)
    census_df['marital_status'] = census_df['marital_status'].replace(['Widowed', 'Married-spouse-absent', 'Divorced'] , 2)
    census_df['marital_status'] = census_df['marital_status'].replace(['Married-AF-spouse', 'Married-civ-spouse'] , 3)
    
    census_df['race'] = census_df['race'].replace(['Other', 'Amer-Indian-Eskimo', 'Black'] , 1)
    census_df['race'] = census_df['race'].replace(['White', 'Asian-Pac-Islander'] , 2)
    
    census_df['education_num'] = census_df['education_num'].replace([3,4,5,6,7,8] , 3)
    census_df['education_num'] = census_df['education_num'].replace([9, 10] , 4)
    census_df['education_num'] = census_df['education_num'].replace([11, 12] , 5)
    census_df['education_num'] = census_df['education_num'].replace([13] , 6)
    census_df['education_num'] = census_df['education_num'].replace([14] , 7)
    census_df['education_num'] = census_df['education_num'].replace([15, 16] , 8)
    
    census_df['age'] = census_df['age'].replace([4, 10, 5] , 4)
    census_df['age'] = census_df['age'].replace([6,7,8,9] , 5)
    
    census_df['capital_loss'] = census_df['capital_loss'].replace([0], 1)
    census_df['capital_loss'] = census_df['capital_loss'].replace([2, 5], 2)
    census_df['capital_loss'] = census_df['capital_loss'].replace([4], 3)
    census_df['capital_loss'] = census_df['capital_loss'].replace([3], 4)

    census_df['capital_gain'] = census_df['capital_gain'].replace([2], 1)
    census_df['capital_gain'] = census_df['capital_gain'].replace([0], 2)
    census_df['capital_gain'] = census_df['capital_gain'].replace([4,5], 4)

    census_df['sex'] = census_df['sex'].replace(['Female'], 0)
    census_df['sex'] = census_df['sex'].replace(['Male'], 1)
    
    return census_df

def gini_roc2(y_test, y_pred_proba, tytul):
    
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    
    plt.plot(fpr,tpr)
    plt.title(tytul)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    roc_auc = metrics.auc(fpr, tpr)
    gini = (2 * roc_auc) - 1

    return gini

def gini_pred(model, X_train, y_train):
    
    y_pred_proba = model.predict_proba(X_train)[::,1]
    gini_train = gini_roc2(y_train, y_pred_proba, "ROC Curve for Training Sample")
    print("gini_test: %.4f" % gini_train)

    return