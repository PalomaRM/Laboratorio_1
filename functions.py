
from pathlib import Path
import pandas as pd
import glob
import pandas_datareader.data as web
import re
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import locale
locale.setlocale( locale.LC_ALL, '' )


# obtener tickers unicos para yfinance
def unique_tickers(df_final):
    tickers=df_final.value_counts().loc[lambda x: x>=25]
    tickers=tickers.drop(tickers.index[-1])
    tickers=list(tickers.index)
    for i in range(0,len(tickers)):
        tickers[i]=str(tickers[i]).replace('*','')
        tickers[i]=str(tickers[i]).replace('.1','-1')


    lista_de_listas = list(map(lambda x:re.findall(r'([A-Z]+).*', x) ,tickers))
    final = [ticker[0] + '.MX' for ticker in lista_de_listas]
    final[5]='LIVEPOLC-1.MX'
    final[30]='GCARSOA1.MX'
    final[10]='PE&OLES.MX'
    final
    return final

def post_ini(closes,pesos):
    p_ini=closes.iloc[0:32,0:2]
    #.reset_index(drop=True)
    p_ini['Peso']=pesos.iloc[:,1]
    p_ini=p_ini.rename(columns={'index':'Ticker','29/01/2021':'Precio'})
    p_ini['Postura']=p_ini['Peso']/100*k
    p_ini['Titulos']=round(p_ini['Postura']/p_ini['Precio'])
    p_ini['Postura']=(p_ini['Peso']/100*k)-(p_ini['Titulos']*p_ini['Precio']*com)    #considerando comisiones
    p_ini['Postura'].sum()
    return p_ini

def resultado_pasiva(closesfin,p_inicial,date):
    # Calculando columna capital
    capital=(closesfin.mul(postura_inicial.iloc[:,4].values, axis = 0)).sum()
    capital=capital.round(decimals=2)
    #DataFrame
    df_pasiva=pd.DataFrame(columns=['timestamp','capital','rend','rend_acum'])
    df_pasiva['timestamp']=date
    df_pasiva=df_pasiva.set_index('timestamp')
    df_pasiva['capital']=capital
    df_pasiva['rend']=df_pasiva['capital'].pct_change().dropna()
    df_pasiva['rend_acum']=df_pasiva['rend'].cumsum()
    df_pasiva
    return df_pasiva


def var(w,Sigma):
    return w.T.dot(Sigma).dot(w)
def menos_RS(w,re,rf,Sigma):
    E_port=re.T.dot(w)
    s_port=var(w,Sigma)**0.5
    RS=(E_port-rf)/s_port
    return -RS


def RS(data):
    summary=pd.DataFrame(columns=data.columns)
    summary.loc['Media']=data.mean()
    summary.loc['Volatilidad']=data.std()*(254**(1/2))
    corr=data.corr()
    S= np.diag(summary.loc['Volatilidad'].values)
    Sigma=S.dot(corr).dot(S)
    re=summary.loc['Media'].values
    rf=0.0429
    N=len(re)                                               
    w0=np.ones(N)/N                                        
    bnds=((0,1),)*N                                         
    cons=({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)
    emv=minimize(menos_RS, postura_inicial.iloc[:,2].values, args=(re,rf,Sigma),bounds=bnds, constraints=cons)
    w_emv1=emv.x
    E_emv=re.T.dot(w_emv1)
    s_emv=var(w_emv1,Sigma)**0.5
    RS_emv=(E_emv-rf)/s_emv
    return RS_emv
pasive_rs=RS(closes1)

def var(w,Sigma):
    return w.T.dot(Sigma).dot(w)
N=len(re)                                               # Número de activos
w0=np.ones(N)/N                                         # Dato inicial
bnds=((0,1),)*N                                         # Cotas de las variables
cons=({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)    # Restricciones

# Función objetivo
def menos_RS(w,re,rf,Sigma):
    E_port=re.T.dot(w)
    s_port=var(w,Sigma)**0.5
    RS=(E_port-rf)/s_port
    return -RS

def port_ef(pesos,data):
    k=1000000
    Peficiente=pd.DataFrame(columns=summary.columns)
    Peficiente.loc['Pesos%']=pesos*100
    Peficiente.loc['Postura']=Peficiente.iloc[0,:].values/100*k
    Peficiente.loc['Precio']=data.iloc[-1,:]
    Peficiente.loc['Titulos']=round(Peficiente.loc['Postura']/Peficiente.loc['Precio'])
    Peficiente=Peficiente.T.reset_index()
    return Peficiente
Peficiente=port_ef(w_emv,año1)


def df_operaciones(dates,ticker,data):
    df_op=pd.DataFrame(columns=['timestamp','titulos_totales','titulos_op','cash_acum','comision_x_op','comision_acum','rend','rend_acum'])
    df_op['timestamp']=dates
    c=pd.DataFrame(columns=[tickers],index=dates)
    com= 0.00125
    k=1000000
    df_op.iloc[0,1]=round((w_emv*k)/data.iloc[0,:]).sum()
    df_op.iloc[0,2]=df_op.iloc[0,1]
    df_op.iloc[0,3]=k
    c.iloc[0,:]=(round((w_emv*k)/data.iloc[0,:])).values
    for j in range(len(dates)-1):  
        change=1-(data.iloc[j+1,:]/data.iloc[j,:])
        for i in range(len(w_emv)-1):
            if change[i]>=0.05:
                c.iloc[j+1,i]=c.iloc[j,i]*(1-0.025)
            else:
                c.iloc[j+1,i]=c.iloc[j,i]*(1+0.025)
        df_op.iloc[j+1,1]=round(c.iloc[j+1,:].sum(),0)
        df_op.iloc[j+1,2]=df_op.iloc[j+1,1]-df_op.iloc[j,1]
        df_op.iloc[j+1,3]=df_op.iloc[j,3]+round((data.iloc[j+1,:]*(c.iloc[j+1,:]-c.iloc[j,:]).values).sum(),3)
        df_op.iloc[j+1,4]=round(df_op.iloc[j+1,3]*com,3)
        df_op['comision_x_op']=df_op['comision_x_op'].fillna(0)
        df_op['comision_acum']=df_op['comision_x_op'].cumsum() 
        df_op['rend']=df_op['cash_acum'].pct_change().fillna(0)
        df_op['rend_acum']=df_op['rend'].cumsum()  
    return df_op
df_op=df_operaciones(año2.index.values.tolist(),tickers,año2)
df_op

def rend_active(data):
    ret=data.iloc[250:].pct_change().dropna()
    plt.figure(figsize=(12.2,4.5)) 
    for i in ret.columns.values:
        plt.hist( ret[i],  label=i, bins = 50)
    plt.title('Histograma de los retornos')
    #plt.legend(ret.columns.values,loc='best')
    plt.show()
rend_plot=rend_active(closesa)

def RS(data):
    summary=pd.DataFrame(columns=data.columns)
    summary.loc['Media']=data.mean()
    summary.loc['Volatilidad']=data.std()*(254**(1/2))
    corr=data.corr()
    S= np.diag(summary.loc['Volatilidad'].values)
    Sigma=S.dot(corr).dot(S)
    re=summary.loc['Media'].values
    rf=0.085
    N=len(re)                                               
    w0=np.ones(N)/N                                        
    bnds=((0,1),)*N                                         
    cons=({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)
    emv=minimize(menos_RS, w_emv, args=(re,rf,Sigma),bounds=bnds, constraints=cons)
    w_emv1=emv.x
    E_emv=re.T.dot(w_emv1)
    s_emv=var(w_emv1,Sigma)**0.5
    RS_emv=(E_emv-rf)/s_emv
    return RS_emv
active_rs=RS(año2)
