import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp
import scipy.stats as st
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from scipy.optimize import curve_fit, fsolve, minimize
import time
from datetime import datetime, timedelta
from scipy import integrate, optimize
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

path_dir = 'C:\\Users\\kumapank\\Documents\\COVID-19\\'

# Downlaod COVID Data
def download_COVID_data(dt = '2020-04-27'):
    try:
        df = pd.read_csv(path_dir+'\\input\\time_series_2019 Vertical Data\\Consolidated_COVID_data_'+dt+'.csv')
    except FileNotFoundError:
        col_nm = ['Province/State','Country/Region','Lat','Long','Date','Value']
        conf_df = pd.read_csv('https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv')
        conf_df = conf_df[col_nm].drop(index=0).rename(index=str,columns={'Value':'Confirmed'})

        dth_df = pd.read_csv('https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_narrow.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv')
        dth_df = dth_df[col_nm].drop(index=0).rename(index=str,columns={'Value':'Death'})

        rcv_df = pd.read_csv('https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_narrow.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv')
        rcv_df = rcv_df[col_nm].drop(index=0).rename(index=str,columns={'Value':'Recovered'})

        conf_df.Date = pd.to_datetime(conf_df.Date, format='%Y-%m-%d')
        dth_df.Date = pd.to_datetime(dth_df.Date, format='%Y-%m-%d')
        rcv_df.Date = pd.to_datetime(rcv_df.Date, format='%Y-%m-%d')
        
        df = pd.merge(conf_df,dth_df,left_on=['Province/State','Country/Region','Lat','Long','Date'],right_on=['Province/State','Country/Region','Lat','Long','Date'],how='left')
        df.columns = ['Province/State','Country/Region','Lat','Long','Date','Confirmed','Death']
        df = pd.merge(df,rcv_df,left_on=['Province/State','Country/Region','Lat','Long','Date'],right_on=['Province/State','Country/Region','Lat','Long','Date'],how='left')
        df.columns = ['Province/State','Country/Region','Lat','Long','Date','Confirmed','Death','Recovered']
        df = df.sort_values(by=['Province/State','Country/Region','Date'], ascending=(True,True,True))
        df.Confirmed = df.Confirmed.fillna(0)
        df.Death = df.Death.fillna(0)
        df.Recovered = df.Recovered.fillna(0)
        df.Confirmed = df.Confirmed.astype(int)
        df.Death = df.Death.astype(int)
        df.Recovered = df.Recovered.astype(int)
        df= df.groupby(by=['Country/Region','Date'])['Confirmed','Death','Recovered'].sum().reset_index()
        df.to_csv(path_dir+'\\input\\time_series_2019 Vertical Data\\Consolidated_COVID_data_'+dt+'.csv', index=False)
    return df
    

# SEIR Model Equation
def sir_model(t,initial_val, R_0, T_inc, T_inf):
    
    if callable(R_0):
        R_t = R_0(t)
    else:
        R_t = R_0
    
    alpha = 1/T_inc # rate of expose
    gamma = 1/T_inf # rate of recovery
    beta = R_t*gamma # rate of infection
    
    S, E, I, R= initial_val 
    
    dSdt = -(beta * S * I) 
    dEdt = (beta * S * I) - (alpha * E)
    dIdt = (beta * S * I) - (gamma * I)
    dRdt = gamma * I
    
    return dSdt, dEdt, dIdt, dRdt;
	
# find the best estimate for beta and gamma
def sir_param_estimate(params, data,population):
    a,b,c,k = params
    N = population*k
    n_infected = data['ActiveInfected'].values[0]
    max_days = len(data)

    initial_state = [(N - n_infected)/ N,0, n_infected / N, 0]
    args1 = [a,b,c]
    sol1 = solve_ivp(sir_model, [0, max_days], initial_state, args=args1, t_eval=np.arange(max_days))
    optim_days = min(28, max_days)  # Days to optimise for manually days
    weights = 1 / np.arange(1, optim_days+1)[::-1]  # Recent data is more heavily weighted

    pred_infected = sol1.y[2]*N
    pred_infected = np.clip(pred_infected,0,np.inf)
    act_infected = data['ActiveInfected'].values
    MSLE_infected = mean_squared_log_error(pred_infected[-optim_days:,],act_infected[-optim_days:,],weights)
    pred_removed = sol1.y[3]*N
    pred_removed = np.clip(pred_removed,0,np.inf)
    act_removed = data['Removed'].values
    MSLE_removed = mean_squared_log_error(pred_removed[-optim_days:,],act_removed[-optim_days:,],weights)
    MSLE = np.mean([MSLE_infected,MSLE_removed])
    return MSLE;

#Plot SEIR Model Results
def plot_results(country,pred_df,intervention_param,best_estimate_param,Obs_start = '2020-03-01',Obs_end = '2020-08-01'):
    temp = pred_df[(pred_df.index >= Obs_start) & (pred_df.index <= Obs_end)]
    sns.set(style="whitegrid")
    fig, ax2 = plt.subplots(1, figsize=(12,6))
    ax2.set_title(country + ' SEIR Model Projection')
    temp.loc[temp.index,'ActiveInfected'].plot(label='Training Actual Infection', color='r', ax=ax2)
    temp.loc[temp.index,'Removed'].plot(label='Training Actual Removed', color='g', ax=ax2)
    temp.loc[temp.index,'without_itr_INF'].plot(label='Infected Prediction', color='b', ax=ax2)
    temp.loc[temp.index,'without_itr_RMV'].plot(label='Removed Prediction', color='y', ax=ax2)
    ax2.set_ylabel("Population")
    ax2.legend(loc='best')
    del(temp)
    return ;
	
#Predict Covid 19 Incidence
def predict_covid_instance(best_estimate_param,data,population,intervention_param,decay_L=70,decay_k=6,max_days = 460):    

    train = data[data.index<= max(data.index)]
    st_dt = pd.to_datetime(min(train.index),format='%Y-%m-%d')
    dates_all = np.array([ st_dt + timedelta(days=x) for x in range(0,max_days)])

    active_infected = train['ActiveInfected'].values
    N = population*best_estimate_param[3]
    
    n_infected = active_infected[0]
    initial_state = [(N - n_infected)/ N,0, n_infected / N, 0]
    
    
    if intervention_param[0] == None:
        without_intr_sol = solve_ivp(sir_model, [0, max_days], initial_state, args=best_estimate_param[:-2], 
                             t_eval=np.arange(max_days))
        without_intr_sol = without_intr_sol.y
        with_intr_sol = without_intr_sol
        
    else:        
        without_intr_sol = solve_ivp(sir_model, [0, max_days], initial_state, args=best_estimate_param[:-2], 
                                     t_eval=np.arange(max_days))
        without_intr_sol = without_intr_sol.y
        
        intervention_day = (pd.to_datetime(intervention_param[0],format='%Y-%m-%d')- st_dt).days
        
        def time_varying_reproduction(t):
            if t > intervention_day:
                return best_estimate_param[0]/(1 + (t/decay_L)**decay_k)
            else:
                return best_estimate_param[0]

        argms = [time_varying_reproduction,best_estimate_param[1],best_estimate_param[2]] 
        with_intr_sol = solve_ivp(sir_model, [0, max_days], initial_state, args=argms, t_eval=np.arange(max_days))
        with_intr_sol = with_intr_sol.y

    pred_df = pd.DataFrame({'without_intr_SUS': np.clip(without_intr_sol[0]*N,0,np.inf),
                          'without_itr_INF': np.clip(without_intr_sol[2]*N,0,np.inf),
                          'without_itr_RMV': np.clip(without_intr_sol[3]*N,0,np.inf),
                          'with_intr_RMV': np.clip(with_intr_sol[3]*N,0,np.inf),
                          'with_intr_INF': np.clip(with_intr_sol[2]*N,0,np.inf)},index=dates_all)
    pred_df=pred_df.join(data)
    pred_df['End_pct'] = pred_df['without_itr_RMV']/N
    if max(pred_df['End_pct'])<=0.95:
        End_dt = '0000-00-00'
    else:
        End_dt = min(pred_df[pred_df['End_pct']>=0.95].index)
    return End_dt, pred_df
	
def display_output(area_name,df,population,obs_start='2020-03-01',obs_end='2020-08-01',cutoff=1,intervention_param = [None],decay_L=70,decay_k=6):

    data = df.loc[area_name]
    data = data[data['ActiveInfected']>cutoff]
#     n_infected = data['ActiveInfected'].values[0]

    res_const = minimize(sir_param_estimate, [2.0,2,2,0.025], bounds=((1, 100),(1, 2000),(1, 2000),(0.0001,0.2)),
                         args=(data, population), method='L-BFGS-B')
    best_estimate_param = list(res_const.x) + [res_const.fun]  

    End_dt,pred_df = predict_covid_instance(best_estimate_param,data,population,intervention_param,decay_L=decay_L,decay_k=decay_k)
    forecast_df = pred_df[pred_df.index> max(data.index)]
    plot_results(area_name ,pred_df,intervention_param,best_estimate_param,obs_start,obs_end)
    
    if intervention_param[0] == None:        
        Max_infected_people = int(max(pred_df.without_itr_INF))
        st = pred_df[pred_df.without_itr_INF==max(pred_df.without_itr_INF)].index        
        if End_dt != '0000-00-00':
            print('95% Infection will be removed by :', End_dt.strftime("%b %d %Y"))
        print("Estimated Parameter: R0 %5.2f, | Training Mean Square Log Error: %5.3f " 
                 %(best_estimate_param[0],best_estimate_param[3]))
        print('Apex Day:', st[0].strftime("%b %d %Y"),'| Maximum Active Infected Population:',Max_infected_people)
        print('Forecasted Days: ',[x.strftime("%b %d") for x in forecast_df[0:8].index])
        print('Forecasted Active Infection: ',[int(x) for x in forecast_df[0:8].without_itr_INF.values])
    else:
        Max_infected_people = int(max(pred_df.without_itr_INF))
        st = pred_df[pred_df.without_itr_INF==max(pred_df.without_itr_INF)].index 
        
        Max_infected_people1 = int(max(pred_df.with_intr_INF))
        st1 = pred_df[pred_df.with_intr_INF==max(pred_df.with_intr_INF)].index 
        print('Apex Day Without Intervention:', st[0].strftime("%b %d %Y"),'| Maximum Active Infected Population:',
              Max_infected_people)
        print('Apex Day With Intervention:', st1[0].strftime("%b %d %Y"),'| Maximum Active Infected Population:',
              Max_infected_people1)
        print('Forecasted Value: ',[int(x) for x in forecast_df[0:8]])

    return best_estimate_param, pred_df;

