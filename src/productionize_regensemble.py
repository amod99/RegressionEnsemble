# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:02:31 2019

@author: Amod Agashe

This script executes xgboost model for predicting variable 'y'

"""

# lets import all required modules
import numpy as np
import sys
import xgboost
import pandas as pd
from sklearn.externals import joblib
import pickle
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

base_dir=sys.argv[1]
mxy=839
test_file=sys.argv[2]


def custom_accr(arrdiff,mxy):
    return(len(arrdiff[arrdiff<=3/mxy])*100/len(arrdiff))
    

col_names=open(base_dir+'colnames.pkl','rb')
col_names=pickle.load(col_names)
#print(col_names)

minmaxwts=joblib.load(base_dir+'scaler_model.pkl')
#minmaxwts=pickle.load(minmaxwts)

xgb_model=joblib.load(base_dir+'xgb_model.pkl')
#xgb_model=pickle.load(xgb_model)  

### filter only required features ###
test_data=pd.read_csv(test_file)

test_data_x=test_data[col_names]

test_data_y=test_data['y']/mxy


## fillna ##
test_data_x=test_data_x.fillna(test_data_x.mean())

### apply minmaxscaler ###
test_data_x=pd.DataFrame(minmaxwts.transform(test_data_x),columns=col_names)
start_time=time.time()
test_preds=xgb_model.predict(test_data_x)
end_time=time.time()
total_time=end_time-start_time

accr=custom_accr(np.abs(test_preds-test_data_y),mxy)

print("Rmse for test data: ",np.sqrt(mean_squared_error(test_preds,test_data_y)),\
      "Accuracy for test data: ",accr," Time for execution " , total_time,"seconds")

#write results to txt file
test_preds=test_preds*mxy
with open(base_dir+'predictions_output.txt', 'w') as f:
    for item in test_preds:
        f.write("%s\n" % item)

