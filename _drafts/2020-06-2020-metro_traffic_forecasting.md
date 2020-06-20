---
layout: post
title: Part 1 - Machine Learning Time Series Forecasting with Pytorch&#58; Using LSTM and GRU for metro traffic prediction
author: Maria Zorkaltseva
categories: [Deep Learning]
tags: [machine learning, deep learning, pytorch, lstm, gru, time series]
feature-img: "assets/img/sample_feature_img.png"
excerpt_begin_separator: <!--excerpt-->
excerpt_separator: <!--more-->
comments: true
---

<!--excerpt-->
description
<!--more-->

### Introduction
### Time series theory
### Metro Traffic Prediction using LSTM-based recurrent neural network

**In [1]:**

{% highlight python %}
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
{% endhighlight %}

**Out [1]:**

    Go to this URL in a browser: "Your generated link here"
    
    Enter your authorization code:
    ··········
    Mounted at /content/drive
    

**In [2]:**

{% highlight python %}
!ls
{% endhighlight %}

    drive  sample_data
    
 
https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume 

**In [0]:**

{% highlight python %}
import os
import copy 
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
{% endhighlight %}

**In [0]:**

{% highlight python %}
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
{% endhighlight %}

**In [0]:**

{% highlight python %}
data = pd.read_csv('/content/drive/My Drive/time series prediction/Metro_Interstate_Traffic_Volume.csv', parse_dates=True)
{% endhighlight %}

**In [6]:**

{% highlight python %}
data.head()
{% endhighlight %}




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
    .dataframe{
        font-size: 0.6em;
        table-layout: auto;
        width: 100;
        display: block;
        overflow-x: auto;
        white-space: nowrap;
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



**In [7]:**

{% highlight python %}
data.info()
{% endhighlight %}

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
    

**In [8]:**

{% highlight python %}
for column in data.columns:
    print(column, sum(data[column].isna()))
{% endhighlight %}

    holiday 0
    temp 0
    rain_1h 0
    snow_1h 0
    clouds_all 0
    weather_main 0
    weather_description 0
    date_time 0
    traffic_volume 0
    

**In [0]:**

{% highlight python %}
data = pd.get_dummies(data, columns = ['holiday', 'weather_main'], drop_first=True)
{% endhighlight %}

**In [10]:**

{% highlight python %}
data.head()
{% endhighlight %}




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



**In [11]:**

{% highlight python %}
data.shape
{% endhighlight %}




    (48204, 28)



**In [12]:**

{% highlight python %}
data['temp'].plot()
plt.show()
data['traffic_volume'].plot()
plt.show()
{% endhighlight %}

 
![png](/assets/img/metro_traffic_forecasting_12_0.png) 


 
![png](/assets/img/metro_traffic_forecasting_12_1.png) 


**In [13]:**

{% highlight python %}
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
{% endhighlight %}

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
    

**In [14]:**

{% highlight python %}
data = data[~((data['temp'] < (Q1['temp'] - 1.5 * IQR['temp'])) |(data['temp'] > (Q3['temp'] + 1.5 * IQR['temp'])))]
data.shape
{% endhighlight %}




    (48194, 28)



**In [15]:**

{% highlight python %}
data['temp'].plot()
plt.show()
data['traffic_volume'].plot()
plt.show()
{% endhighlight %}

 
![png](/assets/img/metro_traffic_forecasting_15_0.png) 


 
![png](/assets/img/metro_traffic_forecasting_15_1.png) 


**In [0]:**

{% highlight python %}
 data.drop(columns=['rain_1h',	'snow_1h',	'weather_description',	'date_time'], inplace=True)
{% endhighlight %}

**In [17]:**

{% highlight python %}
data.head()
{% endhighlight %}




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
      <td>288.28</td>
      <td>40</td>
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
      <td>75</td>
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
      <td>90</td>
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
      <td>90</td>
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
      <td>75</td>
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



**In [18]:**

{% highlight python %}
data.shape
{% endhighlight %}




    (48194, 24)



**In [0]:**

{% highlight python %}
features_to_norm = ['temp',	'clouds_all', 'traffic_volume']
{% endhighlight %}

**In [0]:**

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

**In [0]:**

{% highlight python %}
data[features_to_norm] = (data[features_to_norm]-data_mean)/data_std
{% endhighlight %}

**In [22]:**

{% highlight python %}
data
{% endhighlight %}




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



**In [0]:**

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

**In [0]:**

{% highlight python %}
trainX, trainY = multivariate_data(data.values, data.iloc[:, target_index].values, 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)

valX, valY = multivariate_data(data.values, data.iloc[:, target_index].values,
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
{% endhighlight %}

**In [0]:**

{% highlight python %}
testX, testY = valX[:len(valX)//4], valY[:len(valY)//4]
valX, valY = valX[len(valX)//4:], valY[len(valY)//4:]
{% endhighlight %}

**In [26]:**

{% highlight python %}
print ('Single window of past history : {}'.format(trainX[0].shape))
print ('\n Target temperature to predict : {}'.format(trainY[0].shape))
{% endhighlight %}

    Single window of past history : (120, 24)
    
     Target temperature to predict : (32,)
    

**In [27]:**

{% highlight python %}
print(trainX.shape)
print(trainY.shape)
print(valX.shape)
print(valY.shape)
print(testX.shape)
print(testY.shape)
{% endhighlight %}

    (39280, 120, 24)
    (39280, 32)
    (5582, 120, 24)
    (5582, 32)
    (1860, 120, 24)
    (1860, 32)
    

**In [28]:**

{% highlight python %}
def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction, pic_name):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b', label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.title(pic_name)
    plt.show()
    

multi_step_plot(trainX[0,:,target_index], trainY[0,:], np.array([0]), 'Train case')
multi_step_plot(valX[0,:,target_index], valY[0,:], np.array([0]), 'Val case')
multi_step_plot(testX[0,:,target_index], testY[0,:], np.array([0]), 'Test case')
{% endhighlight %}

 
![png](/assets/img/metro_traffic_forecasting_28_0.png) 


 
![png](/assets/img/metro_traffic_forecasting_28_1.png) 


 
![png](/assets/img/metro_traffic_forecasting_28_2.png) 


**In [0]:**

{% highlight python %}
class TimeSeriesPredictor(nn.Module):

    def __init__(self, num_classes, n_features, hidden_size, num_layers):
        super(TimeSeriesPredictor, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=0.5, 
                            batch_first=True, 
                            bidirectional=False)
        
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
       # dim(x) = (batch, seq_len, input_size)
       # dim(h_0) (num_layers * num_directions, batch, hidden_size)

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through LSTM
        o, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
{% endhighlight %}

**In [0]:**

{% highlight python %}
trainX = torch.from_numpy(trainX).float()
trainY = torch.from_numpy(trainY).float()
valX = torch.from_numpy(valX).float()
valY = torch.from_numpy(valY).float()
testX = torch.from_numpy(testX).float()
testY = torch.from_numpy(testY).float()
{% endhighlight %}

**In [0]:**

{% highlight python %}
train_data = torch.utils.data.TensorDataset(trainX, trainY)
val_data = torch.utils.data.TensorDataset(valX, valY)
test_data =  torch.utils.data.TensorDataset(testX, testY)
{% endhighlight %}

**In [32]:**

{% highlight python %}
len(train_data)
{% endhighlight %}




    39280



**In [33]:**

{% highlight python %}
print(trainX.shape)
print(trainY.shape)
{% endhighlight %}

    torch.Size([39280, 120, 24])
    torch.Size([39280, 32])
    

**In [0]:**

{% highlight python %}
def train_model(model, X, Y, loss_criterion, optimizer, batch_size):
      
    running_loss = 0.
    permutation = torch.randperm(X.size()[0])  # returns shuffled indeces
    model.train()

    for i in tqdm(range(0, X.size()[0], batch_size)):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X[indices], Y[indices]
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = loss_criterion(outputs, batch_y)
    
        loss.backward()
    
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / (X.shape[0] // batch_size)
    
    print("Epoch: %d, train loss: %1.5f" % (epoch+1, epoch_loss))

    return model, epoch_loss
{% endhighlight %}

**In [0]:**

{% highlight python %}
def train_model(model, train_data, loss_criterion, optimizer, batch_size):
      
    running_loss = 0.
    model.train()

    train_loader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=0)

    for batch_x, batch_y in tqdm(train_loader):
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

**In [0]:**

{% highlight python %}
def evaluate_model(model, X, Y, loss_criterion, optimizer, batch_size):     
    running_loss = 0.
    permutation = torch.randperm(X.size()[0])  # returns shuffled indeces
    model.eval()

    for i in tqdm(range(0, X.size()[0], batch_size)):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X[indices], Y[indices]
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(batch_x)
            loss = loss_criterion(outputs, batch_y)
                
        running_loss += loss.item()
    
    
    epoch_loss = running_loss / (X.shape[0] // batch_size)
    
    print("Epoch: %d, val loss: %1.5f" % (epoch+1, epoch_loss))

    return model, epoch_loss, best_model_wts
{% endhighlight %}

**In [0]:**

{% highlight python %}
def evaluate_model(model, val_data, loss_criterion, optimizer, batch_size):     
    running_loss = 0.
    model.eval()

    val_loader = torch.utils.data.DataLoader(val_data, 
                                             batch_size=batch_size, 
                                             shuffle=True, 
                                             num_workers=0)
    
    for batch_x, batch_y in tqdm(val_loader):

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(batch_x)
            loss = loss_criterion(outputs, batch_y)
                
        running_loss += loss.item()
    
    
    epoch_loss = running_loss / (len(val_data) // batch_size)
    
    print("Epoch: %d, val loss: %1.5f" % (epoch+1, epoch_loss))

    return model, epoch_loss, best_model_wts
{% endhighlight %}

**In [38]:**

{% highlight python %}
learning_rate = 0.001
input_size = trainX.shape[2] # number of input features, multivariate case
hidden_size = 24
num_layers = 1
num_classes = trainY.shape[1] # future time window length
lstm_model = TimeSeriesPredictor(num_classes, input_size, hidden_size, num_layers)

config = {'batch_size': 128, 'num_epochs': 10, 'checkpoints_dir': '/content/drive/My Drive/time series prediction/', 'model_filename': 'best_model.pth.tar'}

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate)

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

    ====================1====================
    

    /usr/local/lib/python3.6/dist-packages/torch/nn/modules/rnn.py:50: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
      "num_layers={}".format(dropout, num_layers))
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 1, train loss: 0.89345
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 1, val loss: 0.78886
    Saving model for best loss
    ====================2====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 2, train loss: 0.77582
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 2, val loss: 0.73505
    Saving model for best loss
    ====================3====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 3, train loss: 0.74655
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 3, val loss: 0.70192
    Saving model for best loss
    ====================4====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 4, train loss: 0.72560
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 4, val loss: 0.68362
    Saving model for best loss
    ====================5====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 5, train loss: 0.70978
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 5, val loss: 0.67458
    Saving model for best loss
    ====================6====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 6, train loss: 0.69738
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 6, val loss: 0.66958
    Saving model for best loss
    ====================7====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 7, train loss: 0.68958
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 7, val loss: 0.66839
    Saving model for best loss
    ====================8====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 8, train loss: 0.68312
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 8, val loss: 0.65649
    Saving model for best loss
    ====================9====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 9, train loss: 0.67716
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 9, val loss: 0.65513
    Saving model for best loss
    ====================10====================
    


    HBox(children=(FloatProgress(value=0.0, max=307.0), HTML(value='')))


    
    Epoch: 10, train loss: 0.67319
    


    HBox(children=(FloatProgress(value=0.0, max=44.0), HTML(value='')))


    
    Epoch: 10, val loss: 0.65272
    Saving model for best loss
    

**In [39]:**

{% highlight python %}
plt.plot(np.arange(config['num_epochs']), train_history, label='Train loss')
plt.plot(np.arange(config['num_epochs']), val_history, label='Val loss')
plt.xlabel("num of epochs")
plt.ylabel("train loss")
plt.show()
{% endhighlight %}

 
![png](/assets/img/metro_traffic_forecasting_39_0.png) 


**In [41]:**

{% highlight python %}
lstm_model = TimeSeriesPredictor(num_classes, input_size, hidden_size, num_layers)
lstm_model.load_state_dict(torch.load(os.path.join(config['checkpoints_dir'], config['model_filename']))['state_dict'])
lstm_model.eval()
for param in lstm_model.parameters():
    param.requires_grad = False
{% endhighlight %}

    /usr/local/lib/python3.6/dist-packages/torch/nn/modules/rnn.py:50: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
      "num_layers={}".format(dropout, num_layers))
    

**In [0]:**

{% highlight python %}
test_sample_x = testX[0].unsqueeze(0)
test_sample_y = testY[0].unsqueeze(0)
{% endhighlight %}

**In [43]:**

{% highlight python %}
multi_step_plot(
    test_sample_x.detach().numpy()[0, :, target_index], 
    test_sample_y.detach().numpy()[0, :], 
    lstm_model(test_sample_x).detach().numpy()[0, :], 
    'Test case')
{% endhighlight %}

 
![png](/assets/img/metro_traffic_forecasting_42_0.png) 


**In [44]:**

{% highlight python %}
i = 0
while i < len(testX):
    test_x = testX[i].unsqueeze(0)
    test_y = testY[i].unsqueeze(0)
    multi_step_plot(test_x.numpy()[0, :, target_index], test_y.detach().numpy()[0, :], lstm_model(test_x).detach().numpy()[0, :], 'Test case')
    i += 100
{% endhighlight %}

 
![png](/assets/img/metro_traffic_forecasting_43_0.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_1.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_2.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_3.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_4.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_5.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_6.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_7.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_8.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_9.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_10.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_11.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_12.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_13.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_14.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_15.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_16.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_17.png) 


 
![png](/assets/img/metro_traffic_forecasting_43_18.png) 

### Metro Traffic Prediction using GRU-based recurrent neural network
### Summary