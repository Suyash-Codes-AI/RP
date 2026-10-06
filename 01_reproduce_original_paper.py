from pathlib import Path
import time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from boruta import BorutaPy
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

DATA=Path('data/steel_industry.csv'); OUT=Path('results'); SEED=42; TEST_SIZE=8759; CV_FOLDS=5; JOBS=-1

def rmse(y,p): return float(np.sqrt(mean_squared_error(y,p)))
def mape(y,p):
    y=np.asarray(y); p=np.asarray(p); mask=y!=0
    return float(np.mean(np.abs((y[mask]-p[mask])/y[mask]))*100)
def cvpct(y,p): return rmse(y,p)/np.mean(y)*100

def load_prepare():
    df=pd.read_csv(DATA)
    print('\n=== DATASET ==='); print('Shape:',df.shape); print('Missing values:',int(df.isna().sum().sum()))
    req=['date','Usage_kWh','Lagging_Current_Reactive.Power_kVarh','Leading_Current_Reactive_Power_kVarh','CO2(tCO2)','Lagging_Current_Power_Factor','Leading_Current_Power_Factor','NSM','WeekStatus','Day_of_week','Load_Type']
    miss=[c for c in req if c not in df.columns]
    if miss: raise ValueError(f'Missing columns: {miss}')
    if df.isna().sum().sum(): raise ValueError('Dataset contains missing values.')
    df['Timestamp']=pd.to_datetime(df['date'],dayfirst=True,errors='raise'); df=df.sort_values('Timestamp').reset_index(drop=True)
    nsm=df.Timestamp.dt.hour*3600+df.Timestamp.dt.minute*60+df.Timestamp.dt.second
    print('NSM matches timestamp:',bool((nsm==df.NSM).all()))
    ws=np.where(df.Timestamp.dt.dayofweek>=5,'Weekend','Weekday')
    print('WeekStatus matches timestamp:',bool(np.array_equal(ws,df.WeekStatus)))
    print('Day_of_week matches timestamp:',bool(np.array_equal(df.Timestamp.dt.day_name(),df.Day_of_week)))
    return df.rename(columns={'Usage_kWh':'Usage','Lagging_Current_Reactive.Power_kVarh':'LagRP','Leading_Current_Reactive_Power_kVarh':'LeadRP','CO2(tCO2)':'CO2','Lagging_Current_Power_Factor':'LagPF','Leading_Current_Power_Factor':'LeadPF'})

def encode(tr,te):
    nums=['LagRP','LeadRP','CO2','LagPF','LeadPF','NSM']; cats=['WeekStatus','Day_of_week','Load_Type']
    enc=OneHotEncoder(handle_unknown='ignore',drop=None,sparse_output=False)
    prep=ColumnTransformer([('num','passthrough',nums),('cat',enc,cats)])
    a=prep.fit_transform(tr); b=prep.transform(te); names=prep.get_feature_names_out()
    return pd.DataFrame(a,columns=names,index=tr.index),pd.DataFrame(b,columns=names,index=te.index)

def boruta(X,y):
    print('\n=== BORUTA ===')
    est=RandomForestRegressor(n_estimators=300,max_depth=7,random_state=SEED,n_jobs=JOBS)
    b=BorutaPy(est,n_estimators='auto',max_iter=100,perc=100,alpha=.05,two_step=False,random_state=SEED,verbose=2)
    b.fit(X.to_numpy(),np.asarray(y)); r=pd.DataFrame({'feature':X.columns,'confirmed':b.support_,'tentative':b.support_weak_,'rank':b.ranking_})
    OUT.mkdir(exist_ok=True); r.to_csv(OUT/'boruta_results_5models.csv',index=False)
    fs=r.loc[r.confirmed,'feature'].tolist(); print('Confirmed:',len(fs)); print(fs)
    return X[fs] if fs else X

def rfe(X,y):
    print('\n=== RFE / RFECV ===')
    est=RandomForestRegressor(n_estimators=150,random_state=SEED,n_jobs=JOBS)
    cv=KFold(CV_FOLDS,shuffle=True,random_state=SEED)
    f=RFECV(estimator=est,step=1,min_features_to_select=1,scoring='neg_root_mean_squared_error',cv=cv,n_jobs=JOBS)
    f.fit(X,y); selected=X.columns[f.support_].tolist()
    pd.DataFrame({'feature':X.columns,'selected':f.support_,'rank':f.ranking_}).to_csv(OUT/'rfe_results_5models.csv',index=False)
    print('RFE selected:',len(selected)); print(selected); return selected

def train(name,model,Xtr,ytr,Xte,yte):
    print(f'\n=== {name} ==='); t=time.perf_counter(); model.fit(Xtr,ytr); sec=time.perf_counter()-t; p=model.predict(Xte)
    d={'Model':name,'RMSE':rmse(yte,p),'MAE':float(mean_absolute_error(yte,p)),'MAPE_percent':mape(yte,p),'CV_percent':cvpct(yte,p),'Fit_seconds':sec}
    print(f"RMSE={d['RMSE']:.6f} MAE={d['MAE']:.6f} CV={d['CV_percent']:.4f}% time={sec:.2f}s"); return d

def main():
    OUT.mkdir(exist_ok=True); df=load_prepare()
    predictors=['LagRP','LeadRP','CO2','LagPF','LeadPF','NSM','WeekStatus','Day_of_week','Load_Type']; data=df[predictors+['Usage']]
    tr,te=train_test_split(data,test_size=TEST_SIZE,random_state=SEED,shuffle=True); print('\n=== SPLIT ==='); print('Train:',len(tr),'Test:',len(te))
    ytr=tr.Usage.to_numpy(); yte=te.Usage.to_numpy(); Xtr,Xte=encode(tr,te); print('Encoded predictors:',Xtr.shape[1])
    Xtr=boruta(Xtr,ytr); Xte=Xte[Xtr.columns]; selected=rfe(Xtr,ytr); Xtr=Xtr[selected]; Xte=Xte[selected]
    results=[]
    results.append(train('Linear Regression',LinearRegression(),Xtr,ytr,Xte,yte))
    knn=Pipeline([('scaler',StandardScaler()),('model',KNeighborsRegressor(n_neighbors=2,weights='distance',p=2,n_jobs=JOBS))])
    results.append(train('KNN',knn,Xtr,ytr,Xte,yte))
    rf=RandomForestRegressor(n_estimators=300,max_features=min(18,Xtr.shape[1]),random_state=SEED,n_jobs=JOBS)
    results.append(train('Random Forest',rf,Xtr,ytr,Xte,yte))
    xgb=XGBRegressor(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective='reg:squarederror',eval_metric='rmse',random_state=SEED,n_jobs=JOBS)
    results.append(train('XGBoost',xgb,Xtr,ytr,Xte,yte))
    lgb=LGBMRegressor(n_estimators=300,learning_rate=.05,num_leaves=31,max_depth=-1,subsample=.8,colsample_bytree=.8,objective='regression',random_state=SEED,n_jobs=JOBS,verbosity=-1)
    results.append(train('LightGBM',lgb,Xtr,ytr,Xte,yte))
    out=pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True); out.to_csv(OUT/'five_model_comparison.csv',index=False); pd.DataFrame({'feature':selected}).to_csv(OUT/'final_selected_features_5models.csv',index=False)
    print('\n=== FINAL 5-MODEL COMPARISON ==='); print(out.round(6).to_string(index=False)); print('\nBest model:',out.iloc[0].Model); print('Saved:',(OUT/'five_model_comparison.csv').resolve())

if __name__=='__main__': main()
