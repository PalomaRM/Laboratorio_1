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
import functions as fn



data = pd.DataFrame()
path=(r'C:\Users\Paloma\Documents\ITESO\9\Trading\Lab1\files')
filenames = glob.glob(path +"/*.csv")
df=[]

for file in range(len(filenames)):
    data_read=pd.read_csv(filenames[file], skiprows=2, usecols=['Ticker'],skip_blank_lines=True)
    df.append(data_read)
    df_final=pd.concat(df)

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

unique=unique_tickers(df_final)
unique

#Pesos
pesos=pd.read_csv(r'C:\Users\Paloma\Documents\ITESO\9\Trading\Lab1\files\NAFTRAC_20210129.csv'
                  ,skiprows=2, usecols=['Ticker','Peso (%)'],skip_blank_lines=True)
pesos=pesos.drop(index=pesos.index[-1:]).sort_values('Ticker').reset_index(drop=True)
pesos

# #yf.pdr_override()    
# # Función para descargar precios de cierre ajustados:
# def get_adj_closes(tickers,
#                    start_date=None,
#                    end_date=None):
#     # Descargamos DataFrame con todos los datos
#     closes = web.get_data_yahoo(
#         tickers=tickers,
#         start=start_date,
#         end=end_date)
#     # Solo necesitamos los precios ajustados en el cierre
#     closes = closes['Adj Close']
#     # Se ordenan los índices de manera ascendente
#     closes.sort_index(inplace=True)
#     return closes

# # Información
# tickers = final
# start_date = "2021-01-29"
# end_date="2023-01-25"

# # Precios diarios ajustados en el cierre
# closes = get_adj_closes(
#     tickers=tickers,
#     start_date=start_date,
#     end_date=end_date)

# Precios de acciones
closes=pd.read_csv('closes_files.csv').set_index('Date')
dates=closes.reset_index(drop=False)
date=dates['Date']


closes1=closes
closes2=closes.T.reset_index(drop=False)


closesfin=closes2.drop('index',axis=1)


#Postura inicial
k=1000000
com= 0.00125
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

postura_inicial=post_ini(closes2, pesos)


# Calculando el efectivo
cash=k-(postura_inicial['Postura'].sum())

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

res_pasiva=resultado_pasiva(closesfin,postura_inicial,date)


cap_final=res_pasiva['capital'].iloc[-1]
rend_m_p=res_pasiva['rend'].mean()
rend_c_p=res_pasiva['rend_acum'].mean()
rend_m_p, rend_c_p

#print('Capital final: ',cap_final+cash)

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


### ACTIVA
# Precios de acciones
closesa=pd.read_csv('closes_files_activa.csv').reset_index(drop=True).set_index('Date')
closesa=closesa.drop(closesa.columns[23],axis=1)

#Seleccionar los datos del año 1
año1=closesa.iloc[:234]

#dataframe con medias y volatilidades 
summary=pd.DataFrame(columns=año1.columns)
summary.loc['Media']=año1.mean()
summary.loc['Volatilidad']=año1.std()*(254**(1/2))

#correlaciones
corr=año1.corr()

#rf
rf=0.0429

#matriz varianza-covarianza
S= np.diag(summary.loc['Volatilidad'].values)
Sigma=S.dot(corr).dot(S)
# rendimientos esperados activos individuales
re=summary.loc['Media'].values


# Función objetivo
def var(w,Sigma):
    return w.T.dot(Sigma).dot(w)
N=len(re)                                               # Número de activos
w0=np.ones(N)/N                                         # Dato inicial
bnds=((0,1),)*N                                         # Cotas de las variables
cons=({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)    # Restricciones

# Portafolio de mínima varianza
minvar=minimize(var, w0, args=(Sigma,),bounds=bnds, constraints=cons)
w_minvar=minvar.x
E_minvar=re.T.dot(w_minvar)
s_minvar=var(w_minvar,Sigma)**0.5
RS_minvar=(E_minvar-rf)/s_minvar

# Función objetivo
def menos_RS(w,re,rf,Sigma):
    E_port=re.T.dot(w)
    s_port=var(w,Sigma)**0.5
    RS=(E_port-rf)/s_port
    return -RS

# Portafolio EMV
emv=minimize(menos_RS, w0, args=(re,rf,Sigma),bounds=bnds, constraints=cons)

# Pesos, rendimiento y riesgo del portafolio EMV
w_emv=emv.x
E_emv=re.T.dot(w_emv)
s_emv=var(w_emv,Sigma)**0.5
RS_emv=(E_emv-rf)/s_emv


w_minvar = minvar.x
E_minvar =re.T.dot(w_minvar)
s_minvar = var(w_minvar, Sigma)**0.5
RS_minvar= (E_minvar - rf) / s_minvar


cov_emv_minvar=w_emv.T.dot(Sigma).dot(w_minvar)
corr_emv_minvar=cov_emv_minvar/(s_emv*s_minvar)
w_p=np.linspace(0,1)
frontera = pd.DataFrame(data={'Media' : w_p*E_emv + (1-w_p)*E_minvar,
                             'Vol': ((w_p*s_emv)*2 +((1-w_p)*s_minvar)**2 + 2 * w_p * (1-w_p)*cov_emv_minvar)*0.5})
frontera['RS']=(frontera['Media'] - rf)/frontera['Vol']

def frontera_ef(frontera):
    plt.figure(figsize=(6,4))
    plt.scatter(frontera['Vol'],frontera['Media'],c=frontera['RS'],cmap='RdYlBu')
    plt.grid()
    plt.xlabel('Volatilidad $\sigma$')
    plt.ylabel('Rendimiento Esperado $E[r]$')
    plt.colorbar()
    return plt.show()

def front():
    plt.figure(figsize=(10,6))
    # Frontera
    plt.scatter(frontera['Vol'], frontera['Media'], c=frontera['RS'], cmap = 'RdYlBu', label = 'Frontera de min var')
    # Activos ind
    for activo in list(summary.columns):
        plt.plot(summary.loc['Volatilidad', activo],
                 summary.loc['Media', activo],
                 'o',
                 ms=5,
                label = activo)
    # Port. óptimos
    plt.plot(s_minvar, E_minvar, '*g', ms=10, label='Portafolio de min var')
    plt.plot(s_emv, E_emv, '*r', ms=10, label='Portafolio eficiente en media var')
    plt.xlabel('Volatilidad $\sigma$')
    plt.ylabel('Rendimiento Esperado $E[r]$')
    plt.grid()
    #plt.legend(loc='best')
    plt.colorbar()
    return plt.show()

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

titulos_pef=Peficiente[Peficiente['Titulos']>0]

año2=closesa.iloc[274:]
año2=año2.reset_index()
año2=año2.set_index('Date')


tickers=['AC.MX', 'ALFAA.MX', 'ALSEA.MX', 'AMXL.MX', 'ASURB.MX', 'BBAJIOO.MX',
       'BIMBOA.MX', 'BOLSAA.MX', 'CEMEXCPO.MX', 'CUERVO.MX', 'ELEKTRA.MX',
       'FEMSAUBD.MX', 'GAPB.MX', 'GCARSOA1.MX', 'GCC.MX', 'GFINBURO.MX',
       'GFNORTEO.MX', 'GMEXICOB.MX', 'GRUMAB.MX', 'KIMBERA.MX', 'KOFUBL.MX',
       'LIVEPOLC-1.MX', 'MEGACPO.MX', 'OMAB.MX', 'ORBIA.MX', 'PE&OLES.MX',
       'PINFRA.MX', 'Q.MX', 'TLEVISACPO.MX', 'VESTA.MX', 'WALMEX.MX']


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

ventas=df_op[df_op['titulos_op']<0]

(df_op.iloc[:,6].mean()-0.0429)/df_op.iloc[:,6].std()

cash=round(df_op.iloc[-1,3]-df_op.iloc[-1,5],2)
locale.currency(cash, grouping=True )

def rend_active(data):
    ret=data.iloc[250:].pct_change().dropna()
    plt.figure(figsize=(12.2,4.5)) 
    for i in ret.columns.values:
        plt.hist( ret[i],  label=i, bins = 50)
    plt.title('Histograma de los retornos')
    #plt.legend(ret.columns.values,loc='best')
    rend_plot=rend_active(closesa)
    return plt.show()

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


df_medidas=pd.DataFrame(columns=['Medida','Descripcion','inv_Pasiva','inv_Activa'])
desc=['Rendimiento Promedio Mensual','Rendimiento Promedio Acumulado','Radio de Sharpe']
med=['rend_m','rend_c','sharpe']
df_medidas.iloc[:,0]=med
df_medidas.iloc[:,1]=desc
df_medidas.iloc[0,2]=res_pasiva.rend.mean()*100
df_medidas.iloc[1,2]=res_pasiva.rend_acum.mean()*100
df_medidas.iloc[2,2]=pasive_rs
df_medidas.iloc[0,3]=df_op.rend.mean()*100
df_medidas.iloc[1,3]=df_op.rend_acum.mean()*100
df_medidas.iloc[2,3]=active_rs
