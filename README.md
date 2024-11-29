# Feature Engineering (BCG X Data Science Project Part 2)
![Introductory Picture](Feature_Engineering.png)
## Introduction
This is Part 2 of a project from the [BCG X Data Science micro-internship](https://www.theforage.com/simulations/bcg/data-science-ccdz). The Boston Consulting Group (BCG) is an American global consulting firm that  partners with leaders in business and society to tackle their most important challenges. It is one of the world's 3 largest consulting firms along with McKinsey & Company and Bain & Company. BCG X is a new initiative from BCG that combines the firm's consulting expertise with tech building and design.

In this task, I take on the role of a junior data analyst employed at BCG X. BCG X's client, a major gas and electricity utility called PowerCo, is concerned about their customers leaving for better offers from other energy providers. **In this part of the project, I will conduct feature engineering by manipulating and transforming raw data to create new features to improve the performance of the machine learning model we will be using in part 3.**

## Problem Statement
PowerCo has expressed concern over their customers leaving them for better offers from competing energy companies. This concern is exacerbated by the fact that the energy market has had a lot of change in recent years and there are more options than ever for customers to choose from. During a meeting with the Associate Director of the Data Science team, **one potential reason for churn is price sensitivity.** I am tasked with investigating this hypothesis. **To improve the performance of the machine learning model we will be using in part 3, I will conduct feature engineering on the dataset and create new features using the current dataset.**

## Skills Demonstrated
* Python
* Feature Engineering
* Data Manipulation
* Data Visualization

## Data Sourcing
This data was provided to me by the BCG X Data Science microinternship hosted by Forage. A copy of the data is included in this repository under the file name: client_data (1).csv and price_data (1).csv.

## Data Attributes
The data provided by PowerCo is separated into 2 files: client_data(1).csv and price_data(1).csv. The client data contains information about power consumption, sales channels, forecasted power consumption, and whether the client has churned or not. Each row contains data for 1 client.

The price data contains information on the price of energy that each client pays during various peak times of the day. Most clients will have 12 rows of data, one row for each month in a year.

Attributes for client data:
* id - Client company identifier.
* channel_sales - Code of the sales channel.
* cons_12m - Electricity consumption of the past 12 months.
* cons_gas_12m - Gas consumption of the past 12 months.
* cons_last_month - Electricity consumption of the last month.
* date_activ - Date of activation of the contract.
* date_end - Registered date of the end of the contract.
* date_modif_prod - Date of the last modification of the product.
* date_renewal - Date of the next contract renewal.
* forecast_cons_12m - FForecasted electricity consumption for next 12 months.
* forecast_cons_year - Forecasted electricity consumption for the next calendar year.
* forecast_discount_energy - Forecasted value of current discount.
* forecast_meter_rent_12m - Forecasted bill of meter rental for the next 2 months.
* forecast_price_energy_off_peak - Forecasted energy price for 1st period (off peak).
* forecast_price_energy_peak - Forecasted energy price for 2nd period (peak).
* forecast_price_pow_off_peak - Forecasted power price for 1st period (off peak).
* has_gas - Indicated if client is also a gas client.
* imp_cons - Current paid consumption.
* margin_gross_pow_ele - Gross margin on power subscription.
* margin_net_pow_ele - Net margin on power subscription.
* nb_prod_act - Number of active products and services.
* net_margin - Total net margin.
* num_years_antig - Antiquity of the client (in number of years).
* origin_up - Code of the electricity campaign the customer first subscribed to.
* pow_max - Subscribed power.
* churn - Has the client churned over the next 3 months.

Attributes for price data:
* id - Client company identifier.
* price_date - Reference date.
* price_off_peak_var - Price of energy for the 1st period (off peak).
* price_peak_var - Price of energy for the 2nd period (peak).
* price_mid_peak_var - Price of energy for the 3rd period (mid peak).
* price_off_peak_fix - Price of power for the 1st period (off peak).
* price_peak_fix - Price of power for the 2nd period (peak).
* price_mid_peak_fix - Price of power for the 3rd period (mid peak).

## Feature Engineering and Data Visualizations
**Feature engineering is the process of selecting, manipulating and transforming raw data into features that can be used in supervised machine learning. A feature is any measurable input that can be used in a predictive model (each column that make up the data is a feature).**

**Supervised machine learning is the creation of data models by using labeled datasets (column names are the labels) to train a model to predict outcomes.**

A copy of this analysis is included in this repository under the file name: James Weber Feature Engineering.ipynb.

### 1. Importing Libraries and Data
We must first import libraries which contains the commands we need for feature engineering.
Then we import the data from the client_data(1).csv and price_data(1).csv files into dataframes.

```
# Importing libraries

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Use the read_csv() command to import .csv files.
# Create a client_df dataframe for the client data and a price_df dataframe for the price data.

client_df = pd.read_csv(r'C:/Users/jwebe/Desktop/client_data (1).csv')
price_df = pd.read_csv(r'C:/Users/jwebe/Desktop/price_data (1).csv')
```

### 2. Price Variance Between Peak Hours
When speaking wtth the Associate Director of the Data Science team, one hypothesis for PowerCo's customer churn is price sensitivity, the degree to which demand changes when the cost of a product or service changes. However, **there are no features in the raw data that reflect price change. We will use feature engineering techniques to create features that reflect price change.**

**One metric related to price change is the variance of price throughout the year between peak hours.** Variance is the spread between numbers in a dataset. **Variance will give us an inidcation of how much the price has changed over a year. We will also include the variance for the last 6 months in the year.**

The price_df dataframe (which contains the price data) separate the price of energy and the price of power and divide them between various peak hours (off peak, peak, and mid peak). The code below will add the 2 prices and create a new column for total price during off peak, peak, and mid peak hours.

```
# The price_df dataframe contains data for the price of energy and power for each customer.
# The price changes every month and during different peak periods (peak, off peak, and mid peak).
# Along with finding the variance in price for energy and power, we should also find the variance in price for the total.
# Create columns that contain total price by adding energy price with power price.

price_df['price_off_peak_total'] = price_df['price_off_peak_var'] + price_df['price_off_peak_fix']
price_df['price_peak_total'] = price_df['price_peak_var'] + price_df['price_peak_fix']
price_df['price_mid_peak_total'] = price_df['price_mid_peak_var'] + price_df['price_mid_peak_fix']
```
Three columns are added to the price_df dataframe: 
* price_off_peak_total - The price of energy and power during off peak hours.
* price_peak_total - The price of energy and power during peak hours.
* price_mid_peak_total - The price of energy and power during MID peak hours.

The code below will calculate the price variance throughout the year between peak hours and will add columns to the client_df (dataframe containing client data) for each variance calculated.

```
# Use the .groupby() command to group the data in the price_df dataframe by the id column.
# Use the .agg() command to use multiple aggregate functions (var) across multiple columns.
# Use the .reset_index() command to insert the outputs into a dataframe and reset the index column.

price_variance_1year = price_df.groupby('id').agg({'price_off_peak_var': 'var', 
                                                   'price_peak_var': 'var', 
                                                   'price_mid_peak_var': 'var', 
                                                   'price_off_peak_fix': 'var', 
                                                   'price_peak_fix': 'var', 
                                                   'price_mid_peak_fix': 'var', 
                                                   'price_off_peak_total': 'var', 
                                                   'price_peak_total': 'var', 
                                                   'price_mid_peak_total': 'var'}).reset_index()

# Use the .rename() command to rename columns.

price_variance_1year.rename(columns = {'price_off_peak_var':'variance_1y_off_peak_var', 
                                       'price_peak_var':'variance_1y_peak_var', 
                                       'price_mid_peak_var':'variance_1y_mid_peak_var', 
                                       'price_off_peak_fix':'variance_1y_off_peak_fix', 
                                       'price_peak_fix':'variance_1y_peak_fix', 
                                       'price_mid_peak_fix':'variance_1y_mid_peak_fix', 
                                       'price_off_peak_total':'variance_1y_off_peak_total', 
                                       'price_peak_total':'variance_1y_peak_total', 
                                       'price_mid_peak_total':'variance_1y_mid_peak_total'}, 
                                       inplace = True)

client_df = pd.merge(client_df, 
                     price_variance_1year, 
                     on = 'id')
```

Nine columns are added to the client_df dataframe:
* variance_1y_off_peak_var - The price variance of energy during off peak hours for 1 year.
* variance_1y_peak_var - The price variance of energy during peak hours for 1 year.
* variance_1y_mid_peak_var - The price variance of energy during mid peak hours for 1 year.
* variance_1y_off_peak_fix - The price variance of power during off peak hours for 1 year.
* variance_1y_peak_fix - The price variance of power during peak hours for 1 year.
* variance_1y_mid_peak_fix - The price variance of power during mid peak hours for 1 year.
* variance_1y_off_peak_total - The price variance of both energy and power during off peak hours for 1 year.
* variance_1y_peak_total - The price variance of both energy and power during peak hours for 1 year.
* variance_1y_mid_peak_total - The price variance of both energy and power during mid peak hours for 1 year.

The code below will calculate the price variance throughout the last six months of the year between peak hours and will add columns to the client_df.

```
# Use the .groupby() command to group the data in the price_df dataframe by the id column.
# Use the .tail(6) command after the .groupby() command to only include the last 6 months of the year per customer.
# Use the .agg() command to use multiple aggregate functions (var) across multiple columns.
# Use the .reset_index() command to insert the outputs into a dataframe and reset the index column.

price_variance_6months = price_df.groupby('id').tail(6).groupby('id').agg({'price_off_peak_var': 'var', 
                                                                           'price_peak_var': 'var', 
                                                                           'price_mid_peak_var': 'var', 
                                                                           'price_off_peak_fix': 'var', 
                                                                           'price_peak_fix': 'var', 
                                                                           'price_mid_peak_fix': 'var', 
                                                                           'price_off_peak_total': 'var', 
                                                                           'price_peak_total': 'var', 
                                                                           'price_mid_peak_total': 'var'}).reset_index()

# Use the .rename() command to rename columns.

price_variance_6months.rename(columns = {'price_off_peak_var':'variance_6m_off_peak_var', 
                                         'price_peak_var':'variance_6m_peak_var', 
                                         'price_mid_peak_var':'variance_6m_mid_peak_var', 
                                         'price_off_peak_fix':'variance_6m_off_peak_fix', 
                                         'price_peak_fix':'variance_6m_peak_fix', 
                                         'price_mid_peak_fix':'variance_6m_mid_peak_fix', 
                                         'price_off_peak_total':'variance_6m_off_peak_total', 
                                         'price_peak_total':'variance_6m_peak_total', 
                                         'price_mid_peak_total':'variance_6m_mid_peak_total'}, 
                                         inplace = True)

# Merge the client_df dataframe with the 6 month variance dataframes.

client_df = pd.merge(client_df, 
                     price_variance_6months, 
                     on = 'id')
```
Nine columns are added to the client_df dataframe:
* variance_6m_off_peak_var - The price variance of energy during off peak hours for the last 6 months of the year.
* variance_6m_peak_var - The price variance of energy during peak hours for the last 6 months of the year.
* variance_6m_mid_peak_var - The price variance of energy during mid peak hours for the last 6 months of the year.
* variance_6m_off_peak_fix - The price variance of power during off peak hours for the last 6 months of the year.
* variance_6m_peak_fix - The price variance of power during peak hours for the last 6 months of the year.
* variance_6m_mid_peak_fix - The price variance of power during mid peak hours for the last 6 months of the year.
* variance_6m_off_peak_total - The price variance of both energy and power during off peak hours for the last 6 months of the year.
* variance_6m_peak_total - The price variance of both energy and power during peak hours for the last 6 months of the year.
* variance_6m_mid_peak_total - The price variance of both energy and power during mid peak hours for the last 6 months of the year.

**In summary we:**
* Calculated the price variance of energy between different peak hours (off peak, peak, and mid peak) for 1 year.
* Calculated the price variance of power between different peak hours (off peak, peak, and mid peak) for 1 year.
* Calculated the price variance of energy and power between different peak hours (off peak, peak, and mid peak) for 1 year.
* Repeated the previous price variance calculations using the last 6 months of the year instead of the whole year.
* The results of the variance calculations were added to the client_dataframe. A total of 18 columns were added.

### 3. Difference in Price: Beginning of Year to End of Year
Another metric we can use to determine if price sensitivity may be a cause of churn is to determine the difference between the price of power at the end of the year and the price at the beginning of the year. **This will give us the price range for each of PowerCo's customers.**
