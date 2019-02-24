import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import heapq
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
import operator
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle as pkl


class booking_prediction:
    def __init__(self,filename):
        self.file_name=filename
    
    def encoding(self,X):
        # encode labels with value between 0 and n_classes-1.
        le = preprocessing.LabelEncoder()
        # 2/3. FIT AND TRANSFORM
        X_2 = X.apply(le.fit_transform) # use df.apply() to apply le.fit_transform to all columns
        enc = preprocessing.OneHotEncoder()
        enc.fit(X_2)
        onehotlabels = enc.transform(X_2).toarray()
        return onehotlabels

    def preprocessing(self):
        df=pd.read_table(self.file_name,sep='\t') #read the data
        df1, df2 = [x for _, x in df.groupby(df['dim_is_requested']==False)]
        df3=df2.sample(n=60000, random_state=1)
        df=pd.concat([df1, df3])
        Y=df.loc[:, :'dim_is_requested'] # Extracting Y_True Seperately
        df['ds']=pd.to_datetime(df['ds'])
        df['ds_d'] = df['ds'].dt.day_name()
        df['ds'] = pd.DatetimeIndex(df['ds']).month
        df['ds_night'] = pd.DatetimeIndex(df['ds_night']).month
        df.drop(df.columns[[0,3,4]], axis=1, inplace=True)
        def myfillna(series):
            if series.dtype is pd.np.dtype(float):
                return series.fillna(series.mean())
            elif series.dtype is pd.np.dtype(object):
                return series.fillna('.')
            else:
                return series
        if(df.isnull().values.any()):
            df = df.apply(myfillna)
        cols_to_norm = ['m_effective_daily_price','r_kdt_m_effective_daily_price_n100_p50','r_kdt_m_effective_daily_price_available_n100_p50','r_kdt_m_effective_daily_price_booked_n100_p50']
        df[cols_to_norm] = StandardScaler().fit_transform(df[cols_to_norm])
        X_1=df.select_dtypes(include=[object,bool])
        X_2=df.select_dtypes(include=[float,int])
        X_=np.concatenate((self.encoding(X_1), X_2), axis=1)
        le=preprocessing.LabelEncoder()
        Y=le.fit_transform(Y)

        return X_,Y

    def xgboost(self,X_test,y_test):
        estimator= pkl.load(open('xgb_model.pkl','rb'))
        y_pred= estimator.predict(X_test)
        print "Accuracy of xgboost : "+str(accuracy_score(y_test, y_pred))
        print "f1 score of xgboost : "+ str(f1_score(y_test, y_pred, average=None))
    
    def NeuralNetwork(self,n):
        model = Sequential()
        model.add(Dense(100, input_dim=n, kernel_initializer='normal', activation='relu'))
        model.add(Dense(70, kernel_initializer='normal', activation='relu'))
        model.add(Dense(50, kernel_initializer='normal', activation='relu'))
        model.add(Dense(30, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def neural(self,X_test,y_test):
        model = self.NeuralNetwork(X_test.shape[1])
        model.load_weights('weights-NN.hdf5')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        y_pred=model.predict_classes(X_test)
        print "Accuracy of neural network : "+str(accuracy_score(y_test, y_pred))
        print "f1 score of neural network : "+ str(f1_score(y_test, y_pred, average=None))


if __name__ == "__main__":
    booking_prediction=booking_prediction("TH_data_challenge.tsv")
    X,Y=booking_prediction.preprocessing()
    booking_prediction.neural(X,Y)
    booking_prediction.xgboost(X,Y)