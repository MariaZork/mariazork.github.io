---
layout: post
title: Time Series Forecasting (Part 1)&#58; 
subtitle: LSTM-based model for metro traffic prediction (PyTorch)
author: Maria Zorkaltseva
categories: [Deep Learning]
tags: [machine learning, deep learning, pytorch, lstm, time series]
feature-img: "assets/img/sample_feature_img.png"
excerpt_begin_separator: <!--excerpt-->
excerpt_separator: <!--more-->
comments: true
---

<!--excerpt-->
This tutorial contains brief overview of statistical and machine learning methods for time series forecasting, experiments and comparative analysis of Long short-term memory (LSTM) based architectures for solving above mentioned problem. Single layer, two layer and bidirectional single layer LSTM cases are considered. Metro Interstate Traffic Volume Data Set from UCI Machine Learning Repository and PyTorch deep learning framework are used for analysis.
<!--more-->

<!-- TOC -->

- [Time series forecasting methods overview](#time-series-forecasting-methods-overview)
- [Dataset](#dataset)
- [Metro Traffic Prediction using LSTM-based recurrent neural network](#metro-traffic-prediction-using-lstm-based-recurrent-neural-network)
  - [Exploratory Data Analysis (EDA) and Scaling](#exploratory-data-analysis-eda-and-scaling)
  - [Preparing training dataset/Visualizations](#preparing-training-datasetvisualizations)
  - [LSTM Time Series Predictor Model](#lstm-time-series-predictor-model)
  - [Train and Evaluate Helping Functions](#train-and-evaluate-helping-functions)
  - [1-Layer LSTM model](#1-layer-lstm-model)
  - [2-Layer LSTM](#2-layer-lstm)
  - [Bidirectional 1-Layer LSTM](#bidirectional-1-layer-lstm)
  - [Experimental results on val data](#experimental-results-on-val-data)
- [Summary](#summary)

<!-- /TOC -->

### Time series forecasting methods overview

A time series is a set of observations, each one is being recorded at the specific time $$t$$. It can be weather observations, for example, a records of the temperature for a month, it can be observations of currency quotes during the day or any other process aggregated by time. **Time series forecasting** can be determened as the act of predicting the future by understanding the past. The model for forecasting can rely on one variable and this is a **univariate case** or when more than one variable taken into consideration it will be **multivariate case**.

**Stochastic Linear Models:**

- Autoregressive (AR);
- Moving Average (MA);
- Autoregressive Moving Average (ARMA);
- Autoregressive Integrated Moving Average (ARIMA);
- Seasonal ARIMA (SARIMA).

For above family of models, the **stationarity condition** must be satisfied. Loosely speaking, a stochastic process is stationary, if its statistical properties do not change with time.

**Stochastic Non-Linear Models:**

- nonlinear autoregressive exogenous models (NARX);
- autoregressive conditional heteroskedasticity (ARCH);
- generalized autoregressive conditional heteroskedasticity (GARCH).

**Machine Learning Models**

- Hidden Markov Model;
- Least-square SVM (LS-SVM);
- Dynamic Least-square SVM (DLS-SVM);
- Feed Forward Network (FNN);
- Time Lagged Neural Network (TLNN);
- Seasonal Artificial Neural Network (SANN);
- Recurrent Neural Networks (RNN).

### Dataset

We will use [Metro Interstate Traffic Volume Data Set](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume) from UC Irvine Machine Learning Repository, which contains large number of datasets for various tasks in machine learning. We will investigate how weather and holiday features influence the metro traffic in US.

Attribute Information:

<p style="font-size: 0.8em">
<b>holiday</b> Categorical US National holidays plus regional holiday, Minnesota State;<br />
<b>temp</b>    Numeric Average temp in kelvin;<br />
<b>rain_1h</b> Numeric Amount in mm of rain that occurred in the hour;<br />
<b>snow_1h</b> Numeric Amount in mm of snow that occurred in the hour;<br />
<b>clouds_all</b> Numeric Percentage of cloud cover;<br />
<b>weather_main</b> Categorical Short textual description of the current weather;<br />
<b>weather_description</b> Categorical Longer textual description of the current weather;<br />
<b>date_time</b> DateTime Hour of the data collected in local CST time;<br />
<b>traffic_volume</b> Numeric Hourly I-94 ATR 301 reported westbound traffic volume.<br />
</p>
Our target variable will be **traffic_volume**. Here we will consider multivariate multi-step prediction case with LSTM-based recurrent neural network architecture.

### Metro Traffic Prediction using LSTM-based recurrent neural network

I used Google Colab Notebooks to calculate experiments. Here, for convinience, I
mounted my Google drive where I stored the files.

<b style="font-size: 0.8em">In [1]:</b>

{% highlight python %}
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
{% endhighlight %}

<b style="font-size: 0.8em">Out [1]:</b>

    Mounted at /content/drive

<b style="font-size: 0.8em">In [2]:</b>

{% highlight python %}
import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')
{% endhighlight %}

<b style="font-size: 0.8em">In [3]:</b>

{% highlight python %}
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
{% endhighlight %}

<b style="font-size: 0.8em">In [4]:</b>

{% highlight python %}
data = pd.read_csv('/content/drive/My Drive/time series prediction/Metro_Interstate_Traffic_Volume.csv', parse_dates=True)
{% endhighlight %}

#### Exploratory Data Analysis (EDA) and Scaling

<b style="font-size: 0.8em">In [5]:</b>

{% highlight python %}
data.head()
{% endhighlight %}

<b style="font-size: 0.8em">Out [5]:</b>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>holiday</th>
      <th>temp</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_main</th>
      <th>weather_description</th>
      <th>date_time</th>
      <th>traffic_volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>288.28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40</td>
      <td>Clouds</td>
      <td>scattered clouds</td>
      <td>2012-10-02 09:00:00</td>
      <td>5545</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td>289.36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>Clouds</td>
      <td>broken clouds</td>
      <td>2012-10-02 10:00:00</td>
      <td>4516</td>
    </tr>
    <tr>
      <th>2</th>
      <td>None</td>
      <td>289.58</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>Clouds</td>
      <td>overcast clouds</td>
      <td>2012-10-02 11:00:00</td>
      <td>4767</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>290.13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>Clouds</td>
      <td>overcast clouds</td>
      <td>2012-10-02 12:00:00</td>
      <td>5026</td>
    </tr>
    <tr>
      <th>4</th>
      <td>None</td>
      <td>291.14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>Clouds</td>
      <td>broken clouds</td>
      <td>2012-10-02 13:00:00</td>
      <td>4918</td>
    </tr>
  </tbody>
</table>
</div>

**categorical** features: holiday, weather_main, weather_description.<br/>
**Continious** features: temp, rain_1h, show_1h, clouds_all.<br/>
**Target variable**: traffic_volume

<b style="font-size: 0.8em">In [6]:</b>

{% highlight python %}
data.info()
{% endhighlight %}

<b style="font-size: 0.8em">Out [6]:</b>

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 48204 entries, 0 to 48203
    Data columns (total 9 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   holiday              48204 non-null  object 
     1   temp                 48204 non-null  float64
     2   rain_1h              48204 non-null  float64
     3   snow_1h              48204 non-null  float64
     4   clouds_all           48204 non-null  int64  
     5   weather_main         48204 non-null  object 
     6   weather_description  48204 non-null  object 
     7   date_time            48204 non-null  object 
     8   traffic_volume       48204 non-null  int64  
    dtypes: float64(3), int64(2), object(4)
    memory usage: 3.3+ MB

Checking for Nan values

<b style="font-size: 0.8em">In [7]:</b>

{% highlight python %}
for column in data.columns:
    print(column, sum(data[column].isna()))
{% endhighlight %}

<b style="font-size: 0.8em">Out [7]:</b>

    holiday 0
    temp 0
    rain_1h 0
    snow_1h 0
    clouds_all 0
    weather_main 0
    weather_description 0
    date_time 0
    traffic_volume 0

Here I take into consideration two categorical variables, such as 'holiday' and
'weather_main' and then one-hot encode them.

<b style="font-size: 0.8em">In [8]:</b>

{% highlight python %}
data = pd.get_dummies(data, columns = ['holiday', 'weather_main'], drop_first=True)
{% endhighlight %}

<b style="font-size: 0.8em">In [9]:</b>

{% highlight python %}
data.head()
{% endhighlight %}

<b style="font-size: 0.8em">Out [9]:</b>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temp</th>
      <th>rain_1h</th>
      <th>snow_1h</th>
      <th>clouds_all</th>
      <th>weather_description</th>
      <th>date_time</th>
      <th>traffic_volume</th>
      <th>holiday_Columbus Day</th>
      <th>holiday_Independence Day</th>
      <th>holiday_Labor Day</th>
      <th>holiday_Martin Luther King Jr Day</th>
      <th>holiday_Memorial Day</th>
      <th>holiday_New Years Day</th>
      <th>holiday_None</th>
      <th>holiday_State Fair</th>
      <th>holiday_Thanksgiving Day</th>
      <th>holiday_Veterans Day</th>
      <th>holiday_Washingtons Birthday</th>
      <th>weather_main_Clouds</th>
      <th>weather_main_Drizzle</th>
      <th>weather_main_Fog</th>
      <th>weather_main_Haze</th>
      <th>weather_main_Mist</th>
      <th>weather_main_Rain</th>
      <th>weather_main_Smoke</th>
      <th>weather_main_Snow</th>
      <th>weather_main_Squall</th>
      <th>weather_main_Thunderstorm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>288.28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40</td>
      <td>scattered clouds</td>
      <td>2012-10-02 09:00:00</td>
      <td>5545</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>289.36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>broken clouds</td>
      <td>2012-10-02 10:00:00</td>
      <td>4516</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>289.58</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>overcast clouds</td>
      <td>2012-10-02 11:00:00</td>
      <td>4767</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>290.13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>90</td>
      <td>overcast clouds</td>
      <td>2012-10-02 12:00:00</td>
      <td>5026</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>291.14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75</td>
      <td>broken clouds</td>
      <td>2012-10-02 13:00:00</td>
      <td>4918</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<b style="font-size: 0.8em">In [10]:</b>

{% highlight python %}
data.shape
{% endhighlight %}

<b style="font-size: 0.8em">Out [10]:</b>

    (48204, 28)

Here we can see outliers in 'temp' variable, lets filter outliers with
Interquartile range (IQR) method

<b style="font-size: 0.8em">In [11]:</b>

{% highlight python %}
data['temp'].plot()
plt.show()
data['traffic_volume'].plot()
plt.show()
{% endhighlight %}

<b style="font-size: 0.8em">Out [11]:</b>

![png](/assets/img/nb_images/metro_traffic_forecasting_16_0.png)

![png](/assets/img/nb_images/metro_traffic_forecasting_16_1.png)

<b style="font-size: 0.8em">In [12]:</b>

{% highlight python %}
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
{% endhighlight %}

<b style="font-size: 0.8em">Out [12]:</b>

    temp                                   19.646
    rain_1h                                 0.000
    snow_1h                                 0.000
    clouds_all                             89.000
    traffic_volume                       3740.000
    holiday_Columbus Day                    0.000
    holiday_Independence Day                0.000
    holiday_Labor Day                       0.000
    holiday_Martin Luther King Jr Day       0.000
    holiday_Memorial Day                    0.000
    holiday_New Years Day                   0.000
    holiday_None                            0.000
    holiday_State Fair                      0.000
    holiday_Thanksgiving Day                0.000
    holiday_Veterans Day                    0.000
    holiday_Washingtons Birthday            0.000
    weather_main_Clouds                     1.000
    weather_main_Drizzle                    0.000
    weather_main_Fog                        0.000
    weather_main_Haze                       0.000
    weather_main_Mist                       0.000
    weather_main_Rain                       0.000
    weather_main_Smoke                      0.000
    weather_main_Snow                       0.000
    weather_main_Squall                     0.000
    weather_main_Thunderstorm               0.000
    dtype: float64

<b style="font-size: 0.8em">In [13]:</b>

{% highlight python %}
data = data[~((data['temp'] < (Q1['temp'] - 1.5 * IQR['temp'])) |(data['temp'] > (Q3['temp'] + 1.5 * IQR['temp'])))]
{% endhighlight %}

<b style="font-size: 0.8em">In [14]:</b>

{% highlight python %}
data['temp'].plot()
plt.show()
data['traffic_volume'].plot()
plt.show()
{% endhighlight %}

<b style="font-size: 0.8em">Out [14]:</b>

![png](/assets/img/nb_images/metro_traffic_forecasting_19_0.png)

![png](/assets/img/nb_images/metro_traffic_forecasting_19_1.png)

<b style="font-size: 0.8em">In [15]:</b>

{% highlight python %}
 data.drop(columns=['rain_1h', 'snow_1h', 'weather_description', 'date_time'], inplace=True)
{% endhighlight %}

**Normalizing the features**

<b style="font-size: 0.8em">In [16]:</b>

{% highlight python %}
features_to_norm = ['temp', 'clouds_all', 'traffic_volume']
{% endhighlight %}

<b style="font-size: 0.8em">In [17]:</b>

{% highlight python %}
TRAIN_SPLIT = 40000
STEP = 6

past_history = 720
future_target = 32
target_index = 2

tmp = data[features_to_norm].values
data_mean = tmp[:TRAIN_SPLIT].mean(axis=0)
data_std = tmp[:TRAIN_SPLIT].std(axis=0)
{% endhighlight %}

<b style="font-size: 0.8em">In [18]:</b>

{% highlight python %}
data[features_to_norm] = (data[features_to_norm]-data_mean)/data_std
{% endhighlight %}

<b style="font-size: 0.8em">In [19]:</b>

{% highlight python %}
data
{% endhighlight %}

<b style="font-size: 0.8em">Out [19]:</b>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temp</th>
      <th>clouds_all</th>
      <th>traffic_volume</th>
      <th>holiday_Columbus Day</th>
      <th>holiday_Independence Day</th>
      <th>holiday_Labor Day</th>
      <th>holiday_Martin Luther King Jr Day</th>
      <th>holiday_Memorial Day</th>
      <th>holiday_New Years Day</th>
      <th>holiday_None</th>
      <th>holiday_State Fair</th>
      <th>holiday_Thanksgiving Day</th>
      <th>holiday_Veterans Day</th>
      <th>holiday_Washingtons Birthday</th>
      <th>weather_main_Clouds</th>
      <th>weather_main_Drizzle</th>
      <th>weather_main_Fog</th>
      <th>weather_main_Haze</th>
      <th>weather_main_Mist</th>
      <th>weather_main_Rain</th>
      <th>weather_main_Smoke</th>
      <th>weather_main_Snow</th>
      <th>weather_main_Squall</th>
      <th>weather_main_Thunderstorm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.581321</td>
      <td>-0.261692</td>
      <td>1.145383</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.668789</td>
      <td>0.638749</td>
      <td>0.628729</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.686606</td>
      <td>1.024652</td>
      <td>0.754754</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.731150</td>
      <td>1.024652</td>
      <td>0.884797</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.812948</td>
      <td>0.638749</td>
      <td>0.830570</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>48199</th>
      <td>0.190147</td>
      <td>0.638749</td>
      <td>0.140191</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48200</th>
      <td>0.134265</td>
      <td>1.024652</td>
      <td>-0.242404</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48201</th>
      <td>0.131835</td>
      <td>1.024652</td>
      <td>-0.554707</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48202</th>
      <td>0.080003</td>
      <td>1.024652</td>
      <td>-0.910691</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>48203</th>
      <td>0.082432</td>
      <td>1.024652</td>
      <td>-1.159730</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>48194 rows × 24 columns</p>
</div>

#### Preparing training dataset/Visualizations

Here we will consider multiple future points prediction case given a past
history.

<b style="font-size: 0.8em">In [20]:</b>

{% highlight python %}
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)
{% endhighlight %}

<b style="font-size: 0.8em">In [21]:</b>

{% highlight python %}
trainX, trainY = multivariate_data(data.values, data.iloc[:, target_index].values, 0,
                                   TRAIN_SPLIT, past_history,
                                   future_target, STEP)

valX, valY = multivariate_data(data.values, data.iloc[:, target_index].values,
                               TRAIN_SPLIT, None, past_history,
                               future_target, STEP)
{% endhighlight %}

<b style="font-size: 0.8em">In [22]:</b>

{% highlight python %}
testX, testY = valX[:len(valX)//4], valY[:len(valY)//4]
valX, valY = valX[len(valX)//4:], valY[len(valY)//4:]
{% endhighlight %}

<b style="font-size: 0.8em">In [23]:</b>

{% highlight python %}
print('Train input features shape : {}'.format(trainX.shape))
print('\nTrain output shape : {}'.format(trainY.shape))
print('\nValidation input features shape : {}'.format(valX.shape))
print('\nValidation output shape : {}'.format(valY.shape))
print('\nTest input features shape : {}'.format(testX.shape))
print('\nTest output shape : {}'.format(testY.shape))
{% endhighlight %}

<b style="font-size: 0.8em">Out [23]:</b>

    Train input features shape : (39280, 120, 24)
    
    Train output shape : (39280, 32)
    
    Validation input features shape : (5582, 120, 24)
    
    Validation output shape : (5582, 32)
    
    Test input features shape : (1860, 120, 24)
    
    Test output shape : (1860, 32)

<b style="font-size: 0.8em">In [24]:</b>

{% highlight python %}
def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction, pic_name):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history), label='History', linewidth=2.5)
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b', label='True Future', linewidth=2.5)
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Predicted Future', linewidth=2.5)
    plt.legend(loc='upper left')
    plt.title(pic_name)
    plt.show()

multi_step_plot(trainX[0,:,target_index], trainY[0,:], np.array([0]), 'Train case')
multi_step_plot(valX[0,:,target_index], valY[0,:], np.array([0]), 'Val case')
multi_step_plot(testX[0,:,target_index], testY[0,:], np.array([0]), 'Test case')
{% endhighlight %}

<div style="max-height: 500px; overflow-x: hidden; overflow-y: scroll">
<b style="font-size: 0.8em">Out [24]:</b>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_31_0.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_31_1.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_31_2.png"/></p>
</div>

#### LSTM Time Series Predictor Model

For more efficient storing of long sequences, here I chose the LSTM
architecture. LSTM layer consists from cells which is shown below. Two vectors
are inputs to the LSTM-cell: the new input vector from the data $$x_t$$ and the
vector of the hidden state $$h_{t-1}$$, which is obtained from the hidden state of
this cell in the previous time step. The LSTM-cell consists of several number of
gates and in addition to the hidden state vector, there is a "memory vector" or
cell state $$C_t$$.<br/>
Cell state on time step *t* is a linear combination of cell state on *t-1* time
step $$C_{t-1}$$ with coefficients from **forget gate** and new *candidate cell
state* $$\tilde{C_t}$$ with coefficients from **input gate**. When values of **forget
gate** $$f_t$$ wiil be close to zero, cell state $$C_{t-1}$$ will be forgotten. Where
values of **input gate** vector will be large, the new input vector will be added
to that which already were in memory.

![lstm-cell](/assets/img/2020-06-20-metro-traffic-forecasting/LSTM-cell.png)

where,<br/>
$$i_t$$ - input gate; <br/>
$$f_t$$ - forget gate;<br/>
$$o_t$$ - output gate;<br/>
$$\tilde{C_t}$$ - new candidate cell state;<br/>
$$C_t$$ - cell state;<br/>
$$h_t$$ - block output.

Peepholes are often added to LSTM cells to increase model connectivity.

<b style="font-size: 0.8em">In [25]:</b>

{% highlight python %}
class LSTMTimeSeriesPredictor(nn.Module):

    def __init__(self,
                 num_classes,
                 n_features,
                 hidden_size,
                 num_layers,
                 bidirectional=False,
                 dp_rate=0.5):

        super(LSTMTimeSeriesPredictor, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dp_rate,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
       # dim(x) = (batch, seq_len, input_size)
       # dim(h_0) = (num_layers *num_directions, batch, hidden_size)
        if self.bidirectional:
            h_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        else:
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        o, (h_out,_) = self.lstm(x, (h_0, c_0))

        if self.bidirectional:
            h_out = h_out.view(self.num_layers, 2, x.size(0), self.hidden_size)
            # taking last hidden state
            # dim(h_out) = (num_directions, batch, hidden_size)
            h_out = h_out[-1]

        # in bidectional case we sum vectors over num_directions
        # in the case of multi-layer LSTM we sum over num_layers direction
        # dim(h_out) = (batch, hidden_size)
        h_out = h_out.sum(dim=0)

        # dim(out) = (batch, num_classes)
        out = self.fc(h_out)
        return out
{% endhighlight %}

<b style="font-size: 0.8em">In [26]:</b>

{% highlight python %}
trainX = torch.from_numpy(trainX).float()
trainY = torch.from_numpy(trainY).float()
valX = torch.from_numpy(valX).float()
valY = torch.from_numpy(valY).float()
testX = torch.from_numpy(testX).float()
testY = torch.from_numpy(testY).float()
{% endhighlight %}

<b style="font-size: 0.8em">In [27]:</b>

{% highlight python %}
train_data = torch.utils.data.TensorDataset(trainX, trainY)
val_data = torch.utils.data.TensorDataset(valX, valY)
test_data =  torch.utils.data.TensorDataset(testX, testY)
{% endhighlight %}

#### Train and Evaluate Helping Functions

<b style="font-size: 0.8em">In [28]:</b>

{% highlight python %}
def train_model(model, train_data, loss_criterion, optimizer, batch_size):

    running_loss = 0.
    model.train()

    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = loss_criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / (len(train_data) // batch_size)

    print("Epoch: %d, train loss: %1.5f" % (epoch+1, epoch_loss))

    return model, epoch_loss
{% endhighlight %}

<b style="font-size: 0.8em">In [29]:</b>

{% highlight python %}
def evaluate_model(model, val_data, loss_criterion, optimizer, batch_size):
    running_loss = 0.
    model.eval()

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0)

    for batch_x, batch_y in val_loader:

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(batch_x)
            loss = loss_criterion(outputs, batch_y)

        running_loss += loss.item()

    epoch_loss = running_loss / (len(val_data) // batch_size)

    print("Epoch: %d, val loss: %1.5f" % (epoch+1, epoch_loss))

    return model, epoch_loss, best_model_wts
{% endhighlight %}

#### 1-Layer LSTM model

<b style="font-size: 0.8em">In [30]:</b>

{% highlight python %}
learning_rate = 0.001
input_size = trainX.shape[2] # number of input features, multivariate case
hidden_size = 20
num_layers = 1
num_classes = trainY.shape[1] # future time window length
lstm_model = LSTMTimeSeriesPredictor(num_classes, input_size, hidden_size, num_layers, bidirectional=False)

config = {'batch_size': 128, 'num_epochs': 10, 'checkpoints_dir': '/content/drive/My Drive/time series prediction/', 'model_filename': '1-layer-lstm-best_model.pth.tar'}

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

train_history, val_history = [], []

best_model_wts = copy.deepcopy(lstm_model.state_dict())

best_loss = 10e10  # for validation phase

if not os.path.exists(config['checkpoints_dir']):
    os.mkdir(config['checkpoints_dir'])

for epoch in range(config['num_epochs']):
    print("="*20 + str(epoch+1) + "="*20)
    _, train_loss = train_model(lstm_model, train_data, criterion, optimizer, config['batch_size'])
    train_history.append(train_loss)

    _, val_loss, best_model_wts = evaluate_model(lstm_model, val_data, criterion, optimizer, config['batch_size'])
    val_history.append(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        print("Saving model for best loss")
        checkpoint = {
            'state_dict': best_model_wts
        }
        torch.save(checkpoint, config['checkpoints_dir'] + config['model_filename'])
        best_model_wts = copy.deepcopy(lstm_model.state_dict())
{% endhighlight %}

<b style="font-size: 0.8em">Out [30]:</b>

    ====================1====================
    Epoch: 1, train loss: 0.90311
    Epoch: 1, val loss: 0.78379
    Saving model for best loss
    ====================2====================
    Epoch: 2, train loss: 0.77797
    Epoch: 2, val loss: 0.73462
    Saving model for best loss
    ====================3====================
    Epoch: 3, train loss: 0.74900
    Epoch: 3, val loss: 0.70943
    Saving model for best loss
    ====================4====================
    Epoch: 4, train loss: 0.73103
    Epoch: 4, val loss: 0.69784
    Saving model for best loss
    ====================5====================
    Epoch: 5, train loss: 0.71739
    Epoch: 5, val loss: 0.68612
    Saving model for best loss
    ====================6====================
    Epoch: 6, train loss: 0.70665
    Epoch: 6, val loss: 0.67945
    Saving model for best loss
    ====================7====================
    Epoch: 7, train loss: 0.69868
    Epoch: 7, val loss: 0.67938
    Saving model for best loss
    ====================8====================
    Epoch: 8, train loss: 0.69284
    Epoch: 8, val loss: 0.67938
    ====================9====================
    Epoch: 9, train loss: 0.68811
    Epoch: 9, val loss: 0.66802
    Saving model for best loss
    ====================10====================
    Epoch: 10, train loss: 0.68397
    Epoch: 10, val loss: 0.67317

<b style="font-size: 0.8em">In [31]:</b>

{% highlight python %}
plt.plot(np.arange(config['num_epochs']), train_history, label='Train loss')
plt.plot(np.arange(config['num_epochs']), val_history, label='Val loss')
plt.xlabel("num of epochs")
plt.ylabel("train loss")
plt.show()
{% endhighlight %}

<b style="font-size: 0.8em">Out [31]:</b>

![png](/assets/img/nb_images/metro_traffic_forecasting_41_0.png)

**Test**

<b style="font-size: 0.8em">In [32]:</b>

{% highlight python %}
lstm_model = LSTMTimeSeriesPredictor(num_classes, input_size, hidden_size, num_layers)
lstm_model.load_state_dict(torch.load(os.path.join(config['checkpoints_dir'], config['model_filename']))['state_dict'])
lstm_model.eval()
for param in lstm_model.parameters():
    param.requires_grad = False
{% endhighlight %}

<b style="font-size: 0.8em">In [33]:</b>

{% highlight python %}
i = 0
while i < len(testX):
    test_x = testX[i].unsqueeze(0)
    test_y = testY[i].unsqueeze(0)
    multi_step_plot[test_x.numpy(](0, :, target_index), test_y.detach().numpy()[0, :], lstm_model(test_x).detach().numpy()[0, :], 'Test case')
    i += 100
{% endhighlight %}

<div style="max-height: 500px; overflow-x: hidden; overflow-y: scroll">
<b style="font-size: 0.8em">Out [33]:</b>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_0.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_1.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_2.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_3.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_4.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_5.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_6.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_7.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_8.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_9.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_10.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_11.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_12.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_13.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_14.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_15.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_16.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_17.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_44_18.png"/></p>
</div>

#### 2-Layer LSTM

<b style="font-size: 0.8em">In [34]:</b>

{% highlight python %}
learning_rate = 0.001
input_size = trainX.shape[2] # number of input features, multivariate case
hidden_size = 20
num_layers = 2
num_classes = trainY.shape[1] # future time window length
lstm_model = LSTMTimeSeriesPredictor(num_classes, input_size, hidden_size, num_layers, bidirectional=False)

config = {'batch_size': 128, 'num_epochs': 10, 'checkpoints_dir': '/content/drive/My Drive/time series prediction/', 'model_filename': '2-layer-lstm-best_model.pth.tar'}

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

train_history, val_history = [], []

best_model_wts = copy.deepcopy(lstm_model.state_dict())

best_loss = 10e10

if not os.path.exists(config['checkpoints_dir']):
    os.mkdir(config['checkpoints_dir'])

for epoch in range(config['num_epochs']):
    print("="*20 + str(epoch+1) + "="*20)
    _, train_loss = train_model(lstm_model, train_data, criterion, optimizer, config['batch_size'])
    train_history.append(train_loss)

    _, val_loss, best_model_wts = evaluate_model(lstm_model, val_data, criterion, optimizer, config['batch_size'])
    val_history.append(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        print("Saving model for best loss")
        checkpoint = {
            'state_dict': best_model_wts
        }
        torch.save(checkpoint, config['checkpoints_dir'] + config['model_filename'])
        best_model_wts = copy.deepcopy(lstm_model.state_dict())
{% endhighlight %}

<b style="font-size: 0.8em">Out [34]:</b>

    ====================1====================
    Epoch: 1, train loss: 0.89874
    Epoch: 1, val loss: 0.77490
    Saving model for best loss
    ====================2====================
    Epoch: 2, train loss: 0.77985
    Epoch: 2, val loss: 0.73741
    Saving model for best loss
    ====================3====================
    Epoch: 3, train loss: 0.75368
    Epoch: 3, val loss: 0.71219
    Saving model for best loss
    ====================4====================
    Epoch: 4, train loss: 0.73283
    Epoch: 4, val loss: 0.69299
    Saving model for best loss
    ====================5====================
    Epoch: 5, train loss: 0.71830
    Epoch: 5, val loss: 0.68409
    Saving model for best loss
    ====================6====================
    Epoch: 6, train loss: 0.70771
    Epoch: 6, val loss: 0.67732
    Saving model for best loss
    ====================7====================
    Epoch: 7, train loss: 0.69788
    Epoch: 7, val loss: 0.67322
    Saving model for best loss
    ====================8====================
    Epoch: 8, train loss: 0.69050
    Epoch: 8, val loss: 0.67194
    Saving model for best loss
    ====================9====================
    Epoch: 9, train loss: 0.68391
    Epoch: 9, val loss: 0.66734
    Saving model for best loss
    ====================10====================
    Epoch: 10, train loss: 0.67748
    Epoch: 10, val loss: 0.66578
    Saving model for best loss

<b style="font-size: 0.8em">In [35]:</b>

{% highlight python %}
plt.plot(np.arange(config['num_epochs']), train_history, label='Train loss')
plt.plot(np.arange(config['num_epochs']), val_history, label='Val loss')
plt.xlabel("num of epochs")
plt.ylabel("train loss")
plt.show()
{% endhighlight %}

<b style="font-size: 0.8em">Out [35]:</b>

![png](/assets/img/nb_images/metro_traffic_forecasting_47_0.png)

**Test**

<b style="font-size: 0.8em">In [36]:</b>

{% highlight python %}
lstm_model = LSTMTimeSeriesPredictor(num_classes, input_size, hidden_size, num_layers)
lstm_model.load_state_dict(torch.load(os.path.join(config['checkpoints_dir'], config['model_filename']))['state_dict'])
lstm_model.eval()
for param in lstm_model.parameters():
    param.requires_grad = False
{% endhighlight %}

<b style="font-size: 0.8em">In [37]:</b>

{% highlight python %}
i = 0
while i < len(testX):
    test_x = testX[i].unsqueeze(0)
    test_y = testY[i].unsqueeze(0)
    multi_step_plot[test_x.numpy(](0, :, target_index), test_y.detach().numpy()[0, :], lstm_model(test_x).detach().numpy()[0, :], 'Test case')
    i += 100
{% endhighlight %}

<b style="font-size: 0.8em">Out [37]:</b>

<div style="max-height: 500px; overflow-x: hidden; overflow-y: scroll">
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_0.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_1.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_2.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_3.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_4.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_5.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_6.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_7.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_8.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_9.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_10.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_11.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_12.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_13.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_14.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_15.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_16.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_17.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_50_18.png"/></p>
</div>

#### Bidirectional 1-Layer LSTM

<b style="font-size: 0.8em">In [38]:</b>

{% highlight python %}
learning_rate = 0.001
input_size = trainX.shape[2] # number of input features, multivariate case
hidden_size = 20
num_layers = 1
num_classes = trainY.shape[1] # future time window length
lstm_model = LSTMTimeSeriesPredictor(num_classes, input_size, hidden_size, num_layers, bidirectional=True)

config = {'batch_size': 128, 'num_epochs': 10, 'checkpoints_dir': '/content/drive/My Drive/time series prediction/', 'model_filename': 'bidir-1-layer-lstm-best_model.pth.tar'}

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

train_history, val_history = [], []

best_model_wts = copy.deepcopy(lstm_model.state_dict())

best_loss = 10e10  # for validation phase

if not os.path.exists(config['checkpoints_dir']):
    os.mkdir(config['checkpoints_dir'])

for epoch in range(config['num_epochs']):
    print("="*20 + str(epoch+1) + "="*20)
    _, train_loss = train_model(lstm_model, train_data, criterion, optimizer, config['batch_size'])
    train_history.append(train_loss)

    _, val_loss, best_model_wts = evaluate_model(lstm_model, val_data, criterion, optimizer, config['batch_size'])
    val_history.append(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        print("Saving model for best loss")
        checkpoint = {
            'state_dict': best_model_wts
        }
        torch.save(checkpoint, config['checkpoints_dir'] + config['model_filename'])
        best_model_wts = copy.deepcopy(lstm_model.state_dict())
{% endhighlight %}

<b style="font-size: 0.8em">Out [38]:</b>

    ====================1====================
    Epoch: 1, train loss: 0.92054
    Epoch: 1, val loss: 0.78858
    Saving model for best loss
    ====================2====================
    Epoch: 2, train loss: 0.77843
    Epoch: 2, val loss: 0.73070
    Saving model for best loss
    ====================3====================
    Epoch: 3, train loss: 0.75058
    Epoch: 3, val loss: 0.71063
    Saving model for best loss
    ====================4====================
    Epoch: 4, train loss: 0.73297
    Epoch: 4, val loss: 0.69804
    Saving model for best loss
    ====================5====================
    Epoch: 5, train loss: 0.71834
    Epoch: 5, val loss: 0.68801
    Saving model for best loss
    ====================6====================
    Epoch: 6, train loss: 0.70628
    Epoch: 6, val loss: 0.67993
    Saving model for best loss
    ====================7====================
    Epoch: 7, train loss: 0.69642
    Epoch: 7, val loss: 0.67463
    Saving model for best loss
    ====================8====================
    Epoch: 8, train loss: 0.68780
    Epoch: 8, val loss: 0.67683
    ====================9====================
    Epoch: 9, train loss: 0.68003
    Epoch: 9, val loss: 0.67213
    Saving model for best loss
    ====================10====================
    Epoch: 10, train loss: 0.67364
    Epoch: 10, val loss: 0.66894
    Saving model for best loss

<b style="font-size: 0.8em">In [39]:</b>

{% highlight python %}
plt.plot(np.arange(config['num_epochs']), train_history, label='Train loss')
plt.plot(np.arange(config['num_epochs']), val_history, label='Val loss')
plt.xlabel("num of epochs")
plt.ylabel("train loss")
plt.show()
{% endhighlight %}

<b style="font-size: 0.8em">Out [39]:</b>

![png](/assets/img/nb_images/metro_traffic_forecasting_53_0.png)

**Test**

<b style="font-size: 0.8em">In [40]:</b>

{% highlight python %}
from collections import OrderedDict

state_dict = torch.load(os.path.join(config['checkpoints_dir'], config['model_filename']))['state_dict']

new_state_dict = OrderedDict()
for k,v in state_dict.items():
    if k not in ["lstm.weight_ih_l0_reverse", "lstm.weight_hh_l0_reverse", "lstm.bias_ih_l0_reverse", "lstm.bias_hh_l0_reverse"]:
        new_state_dict[k] = v
{% endhighlight %}

<b style="font-size: 0.8em">In [41]:</b>

{% highlight python %}
lstm_model = LSTMTimeSeriesPredictor(num_classes, input_size, hidden_size, num_layers)
lstm_model.load_state_dict(new_state_dict)
lstm_model.eval()
for param in lstm_model.parameters():
    param.requires_grad = False
{% endhighlight %}

<b style="font-size: 0.8em">In [42]:</b>

{% highlight python %}
i = 0
while i < len(testX):
    test_x = testX[i].unsqueeze(0)
    test_y = testY[i].unsqueeze(0)
    multi_step_plot[test_x.numpy(](0, :, target_index), test_y.detach().numpy()[0, :], lstm_model(test_x).detach().numpy()[0, :], 'Test case')
    i += 100
{% endhighlight %}

<div style="max-height: 500px; overflow-x: hidden; overflow-y: scroll">
<b style="font-size: 0.8em">Out [42]:</b>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_0.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_1.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_2.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_3.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_4.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_5.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_6.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_7.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_8.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_9.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_10.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_11.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_12.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_13.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_14.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_15.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_16.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_17.png"/></p>
<p><img src="/assets/img/nb_images/metro_traffic_forecasting_57_18.png"/></p>
</div>

#### Experimental results on val data

| Model      | Best MSE loss          |
| ------------- |:-------------:|
| 1-layer LSTM    | 0.66802 |
| 2-layer LSTM      | 0.66578 |
|Bidirectional 1-layer LSTM  | 0.66894 |

### Summary

At the first glance, all three models based on LSTM showed approximately the same results, so you can choose a single-layer LSTM without loss of quality. In cases where computational efficiency is a crucial part you may chose GRU model, which has less parameters to optimize than LSTM model. The ways you can improve existing deep learning model:

- work on data cleanliness, look for essential features, which are strongly related to the target variable;
- manual and automatic feature engineering;
- optimization of hyperparameters (length of hidden state vector, batch size, number of epochs, learning rate, number of layers);
- try different optimization algorithm;
- try different loss function which will be differentiable and adequate to task you're solving;
- use ensembles of prediction vectors.
