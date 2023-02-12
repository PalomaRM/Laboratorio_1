from pathlib import Path
import pandas as pd
import glob
import pandas_datareader.data as web
import re
import yfinance as yf
import functions as fn

data = pd.DataFrame()
path=(r'C:\Users\Paloma\Documents\ITESO\9\Trading\Lab1\Laboratorio_1\files')
filenames = glob.glob(path +"/*.csv")
df=[]

for file in range(len(filenames)):
    data_read=pd.read_csv(filenames[file], skiprows=2, usecols=['Ticker'],skip_blank_lines=True)
    df.append(data_read)
    df_final=pd.concat(df)
    
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

#Pesos
pesos=pd.read_csv(r'C:\Users\Paloma\Documents\ITESO\9\Trading\Lab1\files\NAFTRAC_20210129.csv'
                  ,skiprows=2, usecols=['Ticker','Peso (%)'],skip_blank_lines=True)
pesos=pesos.drop(index=pesos.index[-1:]).sort_values('Ticker').reset_index(drop=True)
pesos

# Precios de acciones
closes=pd.read_csv('closes_files.csv').set_index('Date')
dates=closes.reset_index(drop=False)
date=dates['Date']
date

closes=closes.T.reset_index(drop=False)
closes

# Solo los precios
closesfin=closes.drop('index',axis=1)
closesfin

#Postura inicial
postura_inicial=fn.post_ini(closes, pesos)
postura_inicial

# Calculando el efectivo
cash=k-(postura_inicial['Postura'].sum())
cash

# Resultado pasiva
res_pasiva=fn.resultado_pasiva(closesfin,postura_inicial,date)
res_pasiva

# Resultados
cap_final=res_pasiva['capital'].iloc[-1]
rend_m_p=res_pasiva['rend'].mean()
rend_c_p=res_pasiva['rend_acum'].mean()
rend_m_p, rend_c_p

print('Capital final: ',cap_final+cash)