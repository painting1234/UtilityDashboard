# IMPORTS
# Import common packages
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import math
import re

# Import plotting tools
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly
import matplotlib.pyplot as plt
plt.get_backend()

# Import statistics packages
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm # tsaplots only works when this is imported separately from statsmodels
import statsmodels
from scipy import stats
from scipy.stats import norm

# Import machine learning packages
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# tqdm package to show progress in loops
from tqdm import tqdm

# Dashboard Imports
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


# GET CONSUMPTION DATA FOR ALL BUILDINGS
elec_df = pd.read_csv('NYCHA_Electric.csv')
gas_df = pd.read_csv('NYCHA_Gas.csv')

### Consumption Pre-processing
# Remove spaces from column names and replace with _
elec_df.columns = [c.replace(' ', '_') for c in elec_df.columns]
gas_df.columns = [c.replace(' ', '_') for c in gas_df.columns]

# Drop duplicates rows
gas_df.loc[:,gas_df.columns !='Bill_Analyzed']
elec_df = elec_df.loc[:,elec_df.columns !='Bill_Analyzed'] #Exclude "Bill Analyzed" column because it is sometimes the only difference between two rows of identical consumption
gas_df = gas_df.loc[:,gas_df.columns !='Bill_Analyzed']
elec_df.drop_duplicates(inplace=True) 
gas_df.drop_duplicates(inplace=True) 

# Get list of development names for dashboard inputs
developments = elec_df['Development_Name'].unique()
gas_developments = gas_df['Development_Name'].unique()

# Define numeric columns that can be aggregated by groupby
sum_cols_elec = ['#_days', 'Current_Charges', 'Consumption_(KWH)','KWH_Charges', 'Consumption_(KW)', 'KW_Charges', 'Other_charges']
sum_cols_gas = ['#_days', 'Current_Charges', 'Consumption_(Therms)']

# GET WEATHER DATA
# Read in two spreadsheets covering <=10 year periods and concatenate with datetime as index
weatherdata1 = pd.read_csv('temperature_1of2.csv')
weatherdata2 = pd.read_csv('temperature_2of2.csv')
weatherdata = pd.concat([weatherdata1, weatherdata2], axis=0)
weatherdata.index = pd.to_datetime(weatherdata['DATE'])
weatherdata.drop('DATE', axis=1, inplace=True)

# Perform linear regression to fill in NaN values, calculate HDD & CDD
def CleanWeatherData(weather, basetemp=65):
    
    # Drop NA values from X variable --> Use Groupby to eliminate duplicate timestamps
    drydata = weather.dropna(subset='HourlyDryBulbTemperature')
    dry_indexed = drydata.groupby(drydata.index).first()
    
    # Drop NA values from Y variable --> Use Groupby to eliminate duplicate timestamps
    wetdata = weather.dropna(subset='HourlyWetBulbTemperature')
    wet_indexed = wetdata.groupby(wetdata.index).first()
    
    # Merge cleaned WetBulb data into cleaned DryBulb data, preserving all DryBulb rows since it is the independent variable for predicting WetBulb
    weather = pd.merge(left=dry_indexed['HourlyDryBulbTemperature'], right=wet_indexed['HourlyWetBulbTemperature'], left_index=True, right_index=True, how='left')
    
    # Create training set without NA values
    train = weather.dropna()
    
    # Convert x_train as Series to 2D numpy array per LinearRegression() requirements
    x_train = train.iloc[:,0].to_numpy().reshape(-1,1)
    y_train = train.iloc[:,1]
    
    # Filter dataframe to contain rows that have missing y values
    test = weather[weather.iloc[:,1].isna()]
    
    # Create x_test as Series to 2D numpy array per LinearRegression() requirements
    x_test = test.iloc[:,0].to_numpy().reshape(-1,1)
    
    # Convert x as Series to 2D numpy array per LinearRegression() requirements
    x_train = train.iloc[:,0].to_numpy().reshape(-1,1)
    y_train = train.iloc[:,1]
    
    # fit model
    model = LinearRegression().fit(x_train, y_train)
    WBpredictions = model.predict(x_test)
    
    # Align predictions with appropriate index for filling NA values
    wetbulbs = pd.Series(WBpredictions, index=test.index)
    
    # Fill NA values
    weather.iloc[:,1].fillna(wetbulbs, inplace=True)

    # Calculate HDD and CDD
    weather['HDD Dry Bulb'] = (basetemp - weather['HourlyDryBulbTemperature']).apply(lambda x: x/24 if x>0 else 0) # 1hour is 1/24 of day
    weather['CDD Wet Bulb'] = (weather['HourlyWetBulbTemperature'] - basetemp).apply(lambda x: x/24 if x>0 else 0)
    weather['CDD Dry Bulb'] = (weather['HourlyDryBulbTemperature'] - basetemp).apply(lambda x: x/24 if x>0 else 0)

    return weather

dailyweather = CleanWeatherData(weatherdata)
weather_df = dailyweather.resample('MS').agg({'HourlyDryBulbTemperature':'mean', 'HourlyWetBulbTemperature': 'mean', 'HDD Dry Bulb': 'sum', 'CDD Wet Bulb': 'sum', 'CDD Dry Bulb': 'sum'})

# Align weather df index with consumption data index and use historical monthly averages as exogenous variable for forecasting
def GetExogVariables(y, horizon=24):

    # IN-SAMPLE EXOGENOUS
    # Match in-sample exogenous data range (weather) to timeseries index
    exog = weather_df[y.index[0]:y.index[-1]]

    # OUT-OF-SAMPLE EXOGENOUS FORECAST
    # Create empty dataframe to fill in monthly averages
    monthly_avgs = pd.DataFrame(np.zeros((12,exog.shape[1])), columns=exog.columns, index=np.arange(1,13))
    
    # Create lookup table for monthly averages
    for month in np.arange(1,13):
        monthly_avgs.loc[month] = np.array(exog[exog.index.month==month].mean())
    
        # Create exogenous forecast variable dataframe
        end_date = exog.index[-1]
        forecast_index = pd.date_range(end_date, periods=horizon+1, freq='MS')[1:]
        
        # Create empty dataframe to populate with monthly averages from lookup table
        exogforecast = pd.DataFrame(np.zeros((horizon, monthly_avgs.shape[1])), columns = monthly_avgs.columns, index=forecast_index)
    
    M = 1
    for date in forecast_index:
        for column in range(len(monthly_avgs.columns)):
            exogforecast.loc[date][column] = monthly_avgs.loc[M][column]
        if M <= 11:
            M +=1
        else:
            M = 1

    return exog, exogforecast

# APP
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
server = app.server
            
app.layout = html.Div([
    dbc.Row([
        dbc.Col([
    dbc.Label('Utility'),
    dcc.Dropdown(id='Utility',
                             placeholder='Choose a Utility',
                             value='Electric',
                             options=[utility for utility in ['Gas', 'Electric']])
        ], md=3, lg=3) # Close Utility Selection Column
    ]), # Close Utility Selection Row
    dbc.Row([
        dbc.Col([
    dbc.Label('Development'),
    dcc.Dropdown(id='Development',
                             placeholder='Choose a Development',
                             value='WASHINGTON',
                             options=[development for development in developments]),
    ], md=3, lg=3), # Close Development Selection Column
        dbc.Col([
            dbc.Label('Forecast Horizon (Months)'),
            dcc.Dropdown(id='Forecast Horizon',
                      placeholder='Number of Months to Forecast',
                      value=24,
                      options=np.arange(1,61)
                        )
        ], md=3, lg=3) # Close Forecast Horizon Column
    ]), # Close Development Selection / Forecast Row
    dbc.Row([
        dcc.Graph(id='Consumption')
                ]), # Close Graph Row
    dbc.Row([
        dbc.Label('Consumption Outliers'),
        html.Div(id='Outliers')
    ]) #Close Table Row
])

@app.callback(Output('Consumption', 'figure'),
              Output('Outliers', 'children'),
              Input('Utility', 'value'),
              Input('Development', 'value'),
              Input('Forecast Horizon', 'value'))
def make_plot(utility_selected, site_selected, horizon):

    # PROCURE DATA
    # Modify data set based on app inputs
    if utility_selected == 'Electric':
        development = elec_df[elec_df['Development_Name']==site_selected]
        y_val = 'Consumption_(KWH)'
        title = 'Monthly kWH'
        
        # Aggregate the consumption of different buildings in the site together
        dev_agg = development[['Revenue_Month'] + sum_cols_elec].groupby('Revenue_Month').sum()

    else:
        development = gas_df[gas_df['Development_Name']==site_selected]
        
        # Filter out Brokered Gas, leaving only Utility Gas to avoid double-counting consumption
        development = development[development['ES_Commodity']=='UTILITY GAS']
        y_val = 'Consumption_(Therms)'
        title = 'Monthly Therms'
        
        # Aggregate the consumption of different buildings in the site together
        dev_agg = development[['Revenue_Month'] + sum_cols_gas].groupby('Revenue_Month').sum()
        
    y = dev_agg[y_val]

    # Convert y index to datetime
    y.index=pd.to_datetime(y.index)

    # This aims to remove outliers from incomplete billing or bills incorrectly stacked on the same month that will interfere with timeseries forecasting
    def RemoveOutliers(rawdata, threshold=1.5):

        # Define cutoffs
        percentiles = [25, 75]
        quartiles = np.percentile(rawdata, percentiles)
        IQR = quartiles[1]-quartiles[0]
        upper_cutoff = quartiles[1] + 3*IQR
        lower_cutoff = quartiles[0] - 1.5*IQR

        # Retrieve outliers
        mask = (rawdata < lower_cutoff) | (rawdata > upper_cutoff) | (rawdata<=0)
        outliers = rawdata[mask]
        cleaned_data = rawdata.copy()
        cleaned_data[mask] = np.nan
        
        return cleaned_data
    
    cleaned_data = RemoveOutliers(y)
    
    # If datetime index is missing values, insert them and show them as N/A; MS indicates start-of-month index
    y = cleaned_data.asfreq('MS')

    # Define exogenous variables
    exog, exogforecast = GetExogVariables(y, horizon)
    
    # MODELING
    # Define and fit ARIMA model
    model = SARIMAX(y, order=(12,0,0), exog=exog)
    res = model.fit()

    # Get in-sample prediction, confidence intervals, and forecast
    # In-sample
    estimates = res.get_prediction(start=1, end=len(y)-1, information_set='smoothed').predicted_mean.apply(lambda x: 0 if x < 0 else x) #Set negative estimates = 0
    predictions = res.get_prediction(start=1, end=len(y)-1, information_set='predicted')
    conf_int = predictions.conf_int(alpha=0.05)
    lower_conf = conf_int.iloc[:,0].apply(lambda x: 0 if x < 0 else x) #remove negative values from confidence interval
    upper_conf = conf_int.iloc[:,1]
    #
    # Out-of-sample
    forecasts = res.get_forecast(steps=horizon , exog=exogforecast)
    future = forecasts.predicted_mean
    conf_int_forecast = forecasts.conf_int(alpha=0.05)
    lower_conf_forecast = conf_int_forecast.iloc[:,0].apply(lambda x: 0 if x < 0 else x) #remove negative values from confidence interval
    upper_conf_forecast = conf_int_forecast.iloc[:,1]

    # Concatenate in-sample and out-of-sample confidence intervals
    lower_conf_all = pd.concat([lower_conf, lower_conf_forecast])
    upper_conf_all = pd.concat([upper_conf, upper_conf_forecast])
    
    # Combine ARIMA predictions and actual consumption into single dataframe
    data = pd.concat([estimates, y, future], axis=1)
    data.columns=['Estimated', 'Measured', 'Forecast']
    
    # PLOTTING
    fig = px.line(data_frame=data,
                 x=data.index,
                 y=['Estimated', 'Measured', 'Forecast'],
                 title=title,
                 )

    #fig.update_layout(barmode='overlay', bargap=0)

    uppercolor = 'rgba(0,0,0,0.15)'
    lowercolor = 'rgba(0,0,0,0.15)'
    fillcolor = 'rgba(156, 158, 155, 0.2)'
    fig.add_trace(go.Scatter(x=upper_conf_all.index,
                                 y=upper_conf_all, 
                                 mode="lines",
                                 line=go.scatter.Line(color=uppercolor),
                                 name='95% CI Upper Bound'))
    fig.add_trace(go.Scatter(x=lower_conf_all.index,
                                 y=lower_conf_all, 
                                 mode="lines",
                                 line=go.scatter.Line(color=lowercolor),
                                 fill='tonexty',
                                 fillcolor=fillcolor,
                                 name='95% CI Lower Bound'
                            ))

    # TABLE  
    # Generate table of outliers
    y_matched = y[1:] #Start comparing y on 2nd data point because prediction is 1-step-ahead
    low_outliers = pd.DataFrame(y_matched[y_matched < lower_conf])
    low_outliers['Date'] = low_outliers.index
    low_outliers['Type of Outlier'] = 'Low'
    low_outliers = low_outliers.iloc[:,[1,0,2]] #rearrange columns to make date first
    
    high_outliers = pd.DataFrame(y_matched[y_matched > upper_conf])
    high_outliers['Date'] = high_outliers.index
    high_outliers['Type of Outlier'] = 'High'
    high_outliers = high_outliers.iloc[:,[1,0,2]] #rearrange columns to make date first
    outlier_df = pd.concat([low_outliers, high_outliers], axis=0)

    outlier_table = dash_table.DataTable(columns = [{'name': col, 'id': col} for col in outlier_df.columns],
                  data = outlier_df.to_dict('records'),
                  style_header={'whiteSpace': 'normal'},
                  fixed_rows={'headers': True},
                  virtualization=True,
                  style_table={'height': '400px'},
                  sort_action='native',
                  filter_action='native',
                  export_format='csv',
                  style_cell={'minWidth': '150px'}
                            )
    
    return fig, outlier_table

    print(table)

if __name__ == '__main__':
    app.run_server(debug=True)
