# 1) Elegir y describir 6 empresas que cotizan en bolsa

# Constellation Brands, Inc. (STZ): Esta empresa estadounidense es productora y distribuidora de bebidas alcohólicas,
# incluyendo vinos, cervezas y licores. Conocida por marcas populares como Corona, Modelo y Svedka Vodka, 
# Constellation Brands es uno de los mayores productores de bebidas en Estados Unidos.

#Boston Beer Company (SAM): Conocida por su marca Samuel Adams, esta empresa es uno de los mayores fabricantes de cerveza artesanal 
#en los Estados Unidos. Además de cerveza, Boston Beer ha expandido su portafolio para incluir sidras, té helado y bebidas alcohólicas.

#Fomento Económico Mexicano S.A.B. de C.V. (FMX): También conocido como FEMSA, es un conglomerado mexicano que, 
#a través de su participación en Coca-Cola FEMSA, produce y distribuye bebidas en México y otros países de América Latina.
#FEMSA también es dueño de tiendas de conveniencia y tiene una amplia red de distribución.

#Sturm, Ruger & Co., Inc. (RGR): Fabricante estadounidense de armas de fuego para el mercado civil y de seguridad. 
#Sturm, Ruger produce una variedad de armas, incluyendo rifles, pistolas y revólveres, 
#y es uno de los fabricantes de armas más reconocidos en Estados Unidos.

#Oshkosh Corporation (OSK): Empresa estadounidense especializada en la fabricación de vehículos y equipos militares y de emergencia. 
#Produce vehículos blindados, camiones y equipos especializados para el ejército y servicios de emergencia, 
#además de vehículos comerciales y de construcción.

#BAE Systems plc (BAESY): Con sede en el Reino Unido, BAE Systems es una de las mayores empresas de defensa en el mundo, 
#dedicada a la producción de vehículos blindados, sistemas de artillería y tecnología militar avanzada, 
#así como de sistemas de seguridad y armas para las fuerzas armadas globales.
# 2.Grafique en un dashboard los precios y los retornos de las distintas acciones con un dropdown que permita seleccionar las acciones 
#a mostrar, otro que permita elegir entre mostrar precios de cierre y los retornos y un slicer para filtrar las fechas.
# Importación de librerías
import numpy as np
import plotly.express as px
import yfinance as yf
import datetime
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
# Listado de símbolos de acciones de las empresas seleccionadas
empresas_seleccionadas = ['STZ', 'SAM', 'FMX', 'RGR', 'OSK', 'BAESY']

# Definir el rango de fechas para el análisis
fecha_final = datetime.datetime.now()
fecha_inicial = fecha_final - datetime.timedelta(days=3*365)  # Últimos 3 años

# Descargar precios ajustados de cierre de las acciones
datos_historicos = yf.download(empresas_seleccionadas, start=fecha_inicial, end=fecha_final)['Adj Close']

# Calcular retornos diarios a partir de los precios históricos
retornos_diarios = datos_historicos.pct_change().dropna()

# Guardar los datos en archivos CSV (opcional)
datos_historicos.to_csv('precios_acciones.csv')
retornos_diarios.to_csv('retornos_acciones.csv')

# Imprimir primeras filas de los datos para verificar
print(datos_historicos.head())
print(retornos_diarios.head())
# Configuración de la aplicación Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Precios y Retornos de Acciones"),

# Dropdown para seleccionar las empresas
    html.Label("Seleccionar Empresa"),
    dcc.Dropdown(id='selector-empresa', 
                 options=[{'label': empresa, 'value': empresa} for empresa in empresas_seleccionadas], 
                 value=empresas_seleccionadas[0], 
                 multi=True),

# Dropdown para elegir entre precios o retornos
    html.Label("Seleccionar Tipo de Datos"),
    dcc.Dropdown(id='selector-datos', 
                 options=[
                     {'label': 'Precios de Cierre', 'value': 'precios'},
                     {'label': 'Retornos', 'value': 'retornos'}
                 ], 
                 value='precios'),

# Selección de rango de fechas
    dcc.DatePickerRange(
        id='rango-fechas',
        start_date=fecha_inicial,
        end_date=fecha_final,
        display_format='YYYY-MM-DD'
    ),

# Gráfico interactivo
    dcc.Graph(id='grafico-acciones')
])

# Callback para actualizar el gráfico en función de las opciones seleccionadas
@app.callback(
    Output('grafico-acciones', 'figure'),
    [Input('selector-empresa', 'value'),
     Input('selector-datos', 'value'),
     Input('rango-fechas', 'start_date'),
     Input('rango-fechas', 'end_date')]
)
def actualizar_grafico(empresas, tipo_dato, fecha_inicio, fecha_fin):
    # Filtrar los datos según el rango de fechas
    datos_filtrados = datos_historicos.loc[fecha_inicio:fecha_fin, empresas]
    if tipo_dato == 'retornos':
        datos_filtrados = retornos_diarios.loc[fecha_inicio:fecha_fin, empresas]
    
# Crear la figura del gráfico
    figura = px.line(datos_filtrados, x=datos_filtrados.index, y=datos_filtrados.columns)
    figura.update_layout(title="Visualización de Precios y Retornos", xaxis_title="Fecha", yaxis_title="Valor")

    return figura

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(port=1600, debug=True)
  #3.	Analice los retornos por acción de 3 años y calcule las siguientes métricas:
#a.	Retorno medio
#b.	Retorno anualizado
#c.	Volatilidad
#d.	Sharpe
#e.	Curtosis
#f.	Sesgo (skewness)
#Interprete sus resultados anteriores y argumente con qué acciones quisiera trabajar. 
pip install cvxpy
import pyfolio as pf
import warnings
warnings.filterwarnings('ignore')
# Performance de Constellation Brands (STZ)
performanceSTZ = pf.timeseries.perf_stats(retornos_diarios['STZ'])
print("Métricas de rendimiento para Constellation Brands (STZ):")
print(performanceSTZ)
# Performance de Boston Beer Company (SAM)
performanceSAM = pf.timeseries.perf_stats(retornos_diarios['SAM'])
print("Métricas de rendimiento para Boston Beer Company (SAM):")
print(performanceSAM)
# Performance de FEMSA (FMX)
performanceFMX = pf.timeseries.perf_stats(retornos_diarios['FMX'])
print("Métricas de rendimiento para FEMSA (FMX):")
print(performanceFMX)
# Performance de Sturm Ruger (RGR)
performanceRGR = pf.timeseries.perf_stats(retornos_diarios['RGR'])
print("Métricas de rendimiento para Sturm Ruger (RGR):")
print(performanceRGR)
# Performance de Oshkosh (OSK)
performanceOSK = pf.timeseries.perf_stats(retornos_diarios['OSK'])
print("Métricas de rendimiento para Oshkosh (OSK):")
print(performanceOSK)
# Performance de BAE Systems (BAESY)
performanceBAESY = pf.timeseries.perf_stats(retornos_diarios['BAESY'])
print("Métricas de rendimiento para BAE Systems (BAESY):")
print(performanceBAESY)
#4.	Arme un portafolio con las acciones recomendadas con los mismos pesos para todas y obtenga:
#a.	retorno del portafolio
#b.	retorno anualizado
#c.	retorno histórico (acumulado)
stocks = ["BAESY","FMX","STZ","OSK"]

end_date=datetime.datetime.now()

start_date=end_date - datetime.timedelta(days=3*365) 
#se realiza analizis sobre 3 años
historical_data=yf.download(stocks,start=start_date,end=end_date)["Adj Close"]

historical_data.to_csv("stock_prices.csv")

print(historical_data.head())
# se calculan retornos porcentuales para cada una de las acciones
#estos retornos representan el cambio porcental por DIA (Habil) de cada una

data = historical_data
returns = data.pct_change()
returns
#Se calcula el retorno promedio diario de cada accion

meanDailyReturns = returns.mean()
meanDailyReturns
#Calculo de retorno del portafolio utilizandoselo pesos equitativos para cada accion
#el peso es de 25% para cada una

pesos = np.array([0.25,0.25,0.25,0.25])

portReturns = np.sum(meanDailyReturns*pesos)
print(portReturns)

#EL portafolio estaria generando 0.05% de retorno diario en promedio

portReturnsAn = portReturns*365
print(portReturnsAn)
returns["Portfolio"] = returns.dot(pesos)
returns#retornos acumulados
#Se obtiene el retorno obtenido en los años presentados
total_return = (data["BAESY"][-1]-data["BAESY"][0])/data["BAESY"][0]
total_return 

#convertir a anualizado
annualized_return = ((1+total_return)**(12/36))-1
annualized_return

#El portafolio anual en promedio tendria un retorno de 34.36%

#Se comprueba que el "Anual Return" expresado en la funcion performance, presenta el mismo dato, por lo que se procede a utlizar ese
daily_cum_ret = (1+returns).cumprod()
print(daily_cum_ret.tail())
#Retorno historico del portafolio de los años evaluados

fig = px.line(daily_cum_ret, x=daily_cum_ret.index, y="Portfolio", line_shape="linear")
fig.show() 
#EJERICIO 5
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as pl
#Calculo de sigma y mu para realizat la optimizacion

mu = expected_returns.mean_historical_return(historical_data)
print(mu)

sigma = risk_models.sample_cov(historical_data)
print(sigma)

#Calculo de frontera de eficiencia
ef = EfficientFrontier(mu,sigma)
print(ef)
#Portafio que maximizadme Sharpe ratio

maxsharpe = ef.max_sharpe()
weights_maxsharpe = ef.clean_weights()
print(maxsharpe, weights_maxsharpe)
#Portafolio con menor volatilidad

ef = EfficientFrontier (mu,sigma)
minvol = ef.min_volatility()
weights_minvol = ef.clean_weights()
print(weights_minvol)
#EJERCICIO 7

#Pesos de cada portafolio (Maximizacion de Sharpe y minimizacion de Volatilidad)

pesosSharpe = np.array([0.93015,0.0698,0.0,0.0,])


pesosVol = np.array([0.32412,0.23115,0.05671,0.38802])
#Calculo del retorno del portafolio de maximización de sharpe
portReturnsSharpe = np.sum(meanDailyReturns*pesosSharpe)
portReturnsSharpe
#Calculo del retorno del portafolio de minimizacion de volatilidad
portReturnsVol = np.sum(meanDailyReturns*pesosVol)
portReturnsVol
meanDailyReturnSharpe = meanDailyReturns.drop(columns='Portfolio')
meanDailyReturnSharpe
returnsSharpe = returns.drop(columns=['Portfolio'])
returnsSharpe
returnsSharpe["Portfolio"] = returnsSharpe.dot(pesosSharpe)
returnsSharpe
performanceSharpe = pf.timeseries.perf_stats(returnsSharpe["Portfolio"])
performanceSharpe
fig = px.histogram(returnsSharpe, x="Portfolio")
fig.show()
meanDailyReturnVol = meanDailyReturns.drop(columns='Portfolio')
meanDailyReturnVol
returnsVol = returns.drop(columns=['Portfolio'])
returnsVol
returnsVol["Portfolio"] = returnsVol.dot(pesosVol)
returnsVol
performanceVol = pf.timeseries.perf_stats(returnsVol["Portfolio"])
performanceVol
ig = px.histogram(returnsVol, x="Portfolio")
fig.show()
