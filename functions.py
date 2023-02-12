from pathlib import Path
import pandas as pd
import glob
import pandas_datareader.data as web
import re
import yfinance as yf
import main as mn

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

# Postura inicial
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

# Resultado pasiva
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