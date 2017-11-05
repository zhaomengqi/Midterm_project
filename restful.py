from boto.s3.connection import S3Connection
import os
import json
import boto.s3
import sys
import datetime
import seaborn as sns
from boto.s3.key import Key
from pprint import pprint
import pandas as pd
import urllib
import csv
import io
import requests
import time
import json
import datetime
from pprint import pprint
import scipy
import numpy as np
import matplotlib.pyplot as plt

import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
import boto3
from botocore.client import Config
from boto.s3.connection import S3Connection
from sklearn.metrics import *
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import h2o
h2o.init()
import pandas as pd
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import h2o
h2o.init()
import pandas as pd
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator






#create url
from flask import Flask
from flask import jsonify
from flask import request


#Rest API
app =Flask(__name__)
languages=[{'name':'JavaScript'},{'name':'python'},{'name':'Ruby'}]
@app.route('/lang',methods=['GET'])

def returnAll():
    # 2016数据清理和合并

    # 清理和填充数据

    # 1.导入数据
    rawdataspecificrows = pd.read_csv("train_2016_v2.csv")
    prop_df = pd.read_csv("properties_2016.csv")

    # 2.合并表格
    train_df = pd.merge(rawdataspecificrows, prop_df, on='parcelid', how='left')
    print(train_df.shape)

    # 3.分析数据train和properties_2016
    print((rawdataspecificrows['parcelid'].value_counts().reset_index())['parcelid'].value_counts())
    print(prop_df.isnull().sum())

    pd.options.display.max_rows = 65
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ["Count", "Column Type"]
    print(dtype_df)

    missing_df = train_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]
    print(missing_df.ix[missing_df['missing_ratio'] > 0.99])

    # 4.数据清理drop column of missing>0.99 and column which is not folat
    # train_df.drop(['transactiondate','hashottuborspa','hashottuborspa','propertycountylandusecode','propertyzoningdesc','fireplaceflag','taxdelinquencyflag'],axis=1,inplace=True)
    # train_df.drop(['architecturalstyletypeid','basementsqft','buildingclasstypeid','decktypeid','finishedsquarefeet13','finishedsquarefeet6','storytypeid','typeconstructiontypeid','yardbuildingsqft26','fireplaceflag'],axis=1,inplace=True)
    train_df = pd.read_csv("values.csv")
    print(train_df.shape)

    # 5数据转化
    # train_df.to_csv('midterm_2016.csv',mode='a',encoding='utf-8',index=False)

    # 2017数据清理和合并
    train_df = pd.read_csv("value2.csv")
    print(train_df.shape)

    # 2016和2017表格合并
    form2017 = pd.read_csv("midterm_2017.csv")
    form2016 = pd.read_csv("midterm_2016.csv")
    combine_data = pd.merge(form2016, form2017, on='parcelid', suffixes=('_2016', '_2017'))
    train_df.to_csv('conbinedata.csv', mode='a', encoding='utf-8', index=False)

    # upload into S3
    # rawdata.to_csv('wrangleddata.csv', index=False)
    # rawdata.to_csv('wrangleddata.csv', index=False)
    # rawdata.to_csv('wrangleddata.csv', index=False)


    s3 = boto3.resource('s3')
    for bucket in s3.buckets.all():
        print(bucket.name)

    with open('s3.json') as data_file:
        data = json.load(data_file)
    # secret keys

    AWSAccess1 = data["AWSAccess"]
    AWSSecret1 = data["AWSSecret"]

    # Connection variables

    c = boto.connect_s3(AWSAccess1, AWSSecret1)
    conn = S3Connection(AWSAccess1, AWSSecret1)
    bucket = c.get_bucket('zillowdata')
    b = c.get_bucket(bucket, validate=False)

    #
    BUCKET_NAME = 'zillowdata'
    FILE_NAME = 'midterm_2017.csv'
    data = open(FILE_NAME, 'rb')

    s3.Bucket(BUCKET_NAME).put_object(Key=FILE_NAME, Body=data, ACL='public-read')

    print('successfully uploaded to s3')

    rawdataspecificrows = pd.read_csv("midterm_2016.csv")
    # feature_cols=['heatingorsystemtypeid','finishedsquarefeet12','calculatedfinishedsquarefeet','calculatedbathnbr','structuretaxvaluedollarcnt','bedroomcnt','bathroomcnt','fullbathcnt']
    feature_cols = ['heatingorsystemtypeid', 'finishedsquarefeet12']
    X = rawdataspecificrows[feature_cols]
    Y = rawdataspecificrows.logerror

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

    linreg = LinearRegression()
    model = linreg.fit(X_train, Y_train)
    Y_pred = linreg.predict(X_test)

    sum_mean = 0
    for i in range(len(Y_pred)):
        sum_mean += (Y_pred[i] - Y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / 50)
    # calculate RMSE by hand
    print("RMSE by hand:", sum_erro)
    # R-squared score of this model
    train_pred = linreg.predict(X_train)

    r2_score(Y_train, train_pred)
    test_pred = linreg.predict(X_test)
    # Mean absolute percentage error (MAPE)
    print(mean_absolute_error(Y_test, test_pred) * 100)
    # Mean squared error
    print(mean_squared_error(Y_test, test_pred))
    # Median absolute error
    print(median_absolute_error(Y_test, test_pred))

    # TREE
    # Use Random Forest

    # test 8 features



    df = pd.read_csv('midterm_2016.csv')
    df.index = df['parcelid'].tolist()
    # feature_cols=['heatingorsystemtypeid','finishedsquarefeet12','calculatedfinishedsquarefeet','calculatedbathnbr','structuretaxvaluedollarcnt','bedroomcnt','bathroomcnt','fullbathcnt']
    column = ['heatingorsystemtypeid', 'finishedsquarefeet12']
    X = df[column]
    # X =array[:,1]
    Y = df['logerror']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

    # print (Y_test.head(2))

    regressor = RandomForestRegressor()
    model = regressor.fit(X_train, Y_train)
    print(model)

    Y_pred = regressor.predict(X_test)
    # print(accuracy_score(Y_test, Y_pred))


    # make ROC graph


    # use RMES

    sum_mean = 0
    for i in range(len(Y_pred)):
        sum_mean += (Y_pred[i] - Y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / 50)
    # calculate RMSE by hand
    print("RMSE by hand:", sum_erro)

    # R-squared score of this model
    train_pred = regressor.predict(X_train)

    r2_score(Y_train, train_pred)
    # Mean absolute percentage error (MAPE)
    print(mean_absolute_error(Y_test, Y_pred) * 100)
    # Mean squared error
    print(mean_squared_error(Y_test, Y_pred))
    # Median absolute error
    print(median_absolute_error(Y_test, Y_pred))

    # Neural networks
    # Use Neural networks


    # load pima indians dataset
    df = pd.read_csv("midterm_2016.csv")
    # split into input (X) and output (Y) variables
    column = ['heatingorsystemtypeid', 'finishedsquarefeet12', 'calculatedfinishedsquarefeet', 'calculatedbathnbr',
              'structuretaxvaluedollarcnt', 'bedroomcnt', 'bathroomcnt', 'fullbathcnt']
    X = np.array(df[column])
    Y = np.array(df[['logerror']])

    neural_net = MLPRegressor(
        activation='logistic',
        learning_rate_init=0.001,
        solver='sgd',
        learning_rate='invscaling',
        hidden_layer_sizes=(200,),
        verbose=True,
        max_iter=2000,
        tol=1e-6
    )

    # Scaling the data
    max_min_scaler = preprocessing.MinMaxScaler()
    X_scaled = max_min_scaler.fit_transform(X)
    Y_scaled = max_min_scaler.fit_transform(Y)

    neural_net.fit(X_scaled[0:4001], Y_scaled[0:4001].ravel())

    predicted = neural_net.predict(X_scaled[5001:5051])

    # Scale back to actual scale
    max_min_scaler = preprocessing.MinMaxScaler(feature_range=(Y[5001:5051].min(), Y[5001:5051].max()))
    predicted_scaled = max_min_scaler.fit_transform(predicted.reshape(-1, 1))

    print("Root Mean Square Error ", mean_squared_error(Y[5001:5051], predicted_scaled))

    # part3
    # 使用h2o重做的2016年linear分析和结论


    df = pd.read_csv('midterm_2016.csv')

    df.drop(['parcelid', 'airconditioningtypeid'], axis=1, inplace=True)
    df.drop(
        ['buildingqualitytypeid', 'finishedfloor1squarefeet', 'finishedsquarefeet15', 'finishedsquarefeet50', 'fips',
         'fireplacecnt', 'garagecarcnt', 'garagetotalsqft', 'latitude', 'longitude', 'lotsizesquarefeet', 'poolcnt',
         'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid', 'rawcensustractandblock',
         'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt', 'threequarterbathnbr',
         'unitcnt', 'yardbuildingsqft17', 'yearbuilt', 'numberofstories', 'taxvaluedollarcnt', 'assessmentyear',
         'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyyear', 'censustractandblock'], axis=1, inplace=True)

    wine = h2o.H2OFrame(df)

    wine.head(5)

    feature = list(wine.columns)
    feature.remove('logerror')

    wine_split = wine.split_frame(ratios=[0.75])
    wine_train = wine_split[0]
    wine_test = wine_split[1]


    glm_default = H2OGeneralizedLinearEstimator(family='gaussian', model_id='glm_default')

    glm_default.train(x=feature, y='logerror', training_frame=wine_train)
    print(glm_default)
    yhat_test_glm = glm_default.predict(wine_test)
    print(yhat_test_glm)

    # 使用h2o重做的2016年Tree分析和结论


    df = pd.read_csv('midterm_2016.csv')

    df.drop(['parcelid', 'airconditioningtypeid'], axis=1, inplace=True)
    df.drop(
        ['buildingqualitytypeid', 'finishedfloor1squarefeet', 'finishedsquarefeet15', 'finishedsquarefeet50', 'fips',
         'fireplacecnt', 'garagecarcnt', 'garagetotalsqft', 'latitude', 'longitude', 'lotsizesquarefeet', 'poolcnt',
         'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid', 'rawcensustractandblock',
         'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt', 'threequarterbathnbr',
         'unitcnt', 'yardbuildingsqft17', 'yearbuilt', 'numberofstories', 'taxvaluedollarcnt', 'assessmentyear',
         'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyyear', 'censustractandblock'], axis=1, inplace=True)

    wine = h2o.H2OFrame(df)

    wine.head(5)

    feature = list(wine.columns)
    feature.remove('logerror')

    wine_split = wine.split_frame(ratios=[0.75])
    wine_train = wine_split[0]
    wine_test = wine_split[1]

    drf_default = H2ORandomForestEstimator(seed=1234, model_id='drf_default')
    drf_default.train(x=feature, y='logerror', training_frame=wine_train)
    print(drf_default)
    drf_test_glm = drf_default.predict(wine_test)
    print(drf_test_glm)

    feature = list(wine.columns)
    # feature.remove('logerror')
    feature.remove('calculatedfinishedsquarefeet')
    feature.remove('calculatedbathnbr')
    feature.remove('structuretaxvaluedollarcnt')
    feature.remove('bedroomcnt')
    feature.remove('bathroomcnt')
    feature.remove('fullbathcnt')
    wine_split = wine.split_frame(ratios=[0.75])
    wine_train = wine_split[0]
    wine_test = wine_split[1]
    glm_default = H2OGeneralizedLinearEstimator(family='gaussian', model_id='glm_default')
    glm_default.train(x=feature, y='logerror', training_frame=wine_train)

    drf_default = H2ORandomForestEstimator(seed=1234, model_id='drf_default')
    drf_default.train(x=feature, y='logerror', training_frame=wine_train)
    print(drf_default)

    return jsonify({'languaages':languages})


if __name__=='__main__':
    app.run(debug=True, port=8080)

