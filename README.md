# Feature Engineering (BCG X Data Science Project Part 2)
![Introductory Picture](Feature_Engineering.png)
## Introduction
This is Part 2 of a project from the [BCG X Data Science micro-internship](https://www.theforage.com/simulations/bcg/data-science-ccdz). The Boston Consulting Group (BCG) is an American global consulting firm that  partners with leaders in business and society to tackle their most important challenges. It is one of the world's 3 largest consulting firms along with McKinsey & Company and Bain & Company. BCG X is a new initiative from BCG that combines the firm's consulting expertise with tech building and design.

In this task, I take on the role of a junior data analyst employed at BCG X. BCG X's client, a major gas and electricity utility called PowerCo, is concerned about their customers leaving for better offers from other energy providers. **In this part of the project, I will conduct feature engineering by manipulating and transforming raw data to create new features to improve the performance of the machine learning model we will be using in part 3.**

## Problem Statement
PowerCo has expressed concern over their customers leaving them for better offers from competing energy companies. This concern is exacerbated by the fact that the energy market has had a lot of change in recent years and there are more options than ever for customers to choose from. During a meeting with the Associate Director of the Data Science team, **one potential reason for churn is price sensitivity.** I am tasked with investigating this hypothesis. **We will use a predictive machine learning model to determine what features are most influential to customer churn. To improve the performance of the machine learning model, which we will be using in part 3, I will conduct feature engineering on the dataset and create new features using the current dataset.**

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
**Feature engineering is the process of selecting, manipulating and transforming raw data into features that can be used in supervised machine learning. A feature is any measurable input that can be used in a predictive model. The features we have in our dataset are listed in the Data Attribures section above.**

**Supervised machine learning is the creation of data models by using labeled datasets to train a model to predict outcomes.**

A copy of this analysis is included in this repository under the file name: James Weber Feature Engineering.ipynb.

### 1. Importing Libraries and Data
We must first import libraries which contains the commands we need for feature engineering.
Then we import the data from the client_data(1).csv and price_data(1).csv files into client_df and price_df dataframes respectively.

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

### 2. Creating Price Sensitivty Features: Price Variance During Peak Hours
**When speaking wtth the Associate Director of the Data Science team, one hypothesis for PowerCo's customer churn is price sensitivity,** the degree to which demand changes when the cost of a product or service changes. However, **there are no features in the raw data that reflect price change. We will use feature engineering techniques to create features that reflect price change.**

**One metric related to price change is the variance of price throughout the year between peak hours.** Variance is the spread between numbers in a dataset. **Variance will give us an inidcation of how much the price has changed over a year. We will also include the variance for the last 6 months in the year.**

The price_df dataframe contains the price of energy and the price of power and during various peak hours (off peak, peak, and mid peak). To make the price_df dataframe more complete, we will sum together the price of energy and power to calculate a total price during peak hours. A new feature will be added to the price_df dataframe for each total price calculated.

```
# The price_df dataframe contains data for the price of energy and power for each customer.
# The price changes every month and during different peak periods (peak, off peak, and mid peak).
# Along with finding the variance in price for energy and power, we should also find the variance in price for the total.
# Create columns that contain total price by adding energy price with power price.

price_df['price_off_peak_total'] = price_df['price_off_peak_var'] + price_df['price_off_peak_fix']
price_df['price_peak_total'] = price_df['price_peak_var'] + price_df['price_peak_fix']
price_df['price_mid_peak_total'] = price_df['price_mid_peak_var'] + price_df['price_mid_peak_fix']
```
Three features are added to the price_df dataframe: 
* price_off_peak_total - The price of energy and power during off peak hours.
* price_peak_total - The price of energy and power during peak hours.
* price_mid_peak_total - The price of energy and power during mid peak hours.

The code below will calculate the energy, power, and total price variance throughout the year during peak hours. A new feature will be added to the client_df dataframe for each price variance calculated.

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

Nine features are added to the client_df dataframe:
* variance_1y_off_peak_var - The price variance of energy during off peak hours for 1 year.
* variance_1y_peak_var - The price variance of energy during peak hours for 1 year.
* variance_1y_mid_peak_var - The price variance of energy during mid peak hours for 1 year.
* variance_1y_off_peak_fix - The price variance of power during off peak hours for 1 year.
* variance_1y_peak_fix - The price variance of power during peak hours for 1 year.
* variance_1y_mid_peak_fix - The price variance of power during mid peak hours for 1 year.
* variance_1y_off_peak_total - The price variance of both energy and power during off peak hours for 1 year.
* variance_1y_peak_total - The price variance of both energy and power during peak hours for 1 year.
* variance_1y_mid_peak_total - The price variance of both energy and power during mid peak hours for 1 year.

We will also calculate the price variance for the last 6 months of the year and add the price variances to the3 client_df dataframe.
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
Nine features are added to the client_df dataframe:
* variance_6m_off_peak_var - The price variance of energy during off peak hours for the last 6 months of the year.
* variance_6m_peak_var - The price variance of energy during peak hours for the last 6 months of the year.
* variance_6m_mid_peak_var - The price variance of energy during mid peak hours for the last 6 months of the year.
* variance_6m_off_peak_fix - The price variance of power during off peak hours for the last 6 months of the year.
* variance_6m_peak_fix - The price variance of power during peak hours for the last 6 months of the year.
* variance_6m_mid_peak_fix - The price variance of power during mid peak hours for the last 6 months of the year.
* variance_6m_off_peak_total - The price variance of both energy and power during off peak hours for the last 6 months of the year.
* variance_6m_peak_total - The price variance of both energy and power during peak hours for the last 6 months of the year.
* variance_6m_mid_peak_total - The price variance of both energy and power during mid peak hours for the last 6 months of the year.

**In summary, we:**
* Calculated the price variance of energy, power, and total price between different peak hours (off peak, peak, and mid peak) for 1 year.
* Calculated the price variance of energy, power, and total price between different peak hours (off peak, peak, and mid peak) for the last 6 months of the year.
* Added the price variances to the client_df dataframe. The client_df dataframe gains 18 new features.

### 3. Creating Price Sensitivty Features: Price Difference Beginning of Year to End of Year
**Another metric we can use to determine if price sensitivity may be a cause of churn is to determine the difference between the price of energy and power at the end of the year and the price of energy and power at the beginning of the year.**

The code below will create a dataframe called price_differences. The price_differences dataframe contains data on the price of energy and power during off peak hours at the beginning of the year (2015-01-01) and at the end of the year (2015-12-01).
```
# Create a dataframe that contains the customer ID, price dates, and the off peak prices.
# Use the .reset_index() command to insert the outputs into a dataframe and reset the index column.

monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 
                                                                  'price_off_peak_fix': 'mean'}).reset_index()

# Create dataframes for the price of power at the beginning of the year and the end of the year.
# Use the .groupby() command to group the data by the id column.
# Use the .first() and .last() command to select the first and last datapoint in the group.
# Use the .reset_index() command to insert the outputs into a dataframe and reset the index column.

beginning_of_year_prices = monthly_price_by_id.groupby('id').first().reset_index()
end_of_year_prices = monthly_price_by_id.groupby('id').last().reset_index()

# Remove the price_date columns from the beginning_of_year_prices and end_of_year_prices dataframes.
# Use the .drop() command to delete a column.

beginning_of_year_prices = beginning_of_year_prices.drop(columns = 'price_date')
end_of_year_prices = end_of_year_prices.drop(columns = 'price_date')

# Rename the price_off_peak_var and price_off_peak_fix columns.
# Use the .rename() command to rename columns.

beginning_of_year_prices.rename(columns = {'price_off_peak_var':'beginning_year_price_energy', 
                                           'price_off_peak_fix':'beginning_year_price_power'}, 
                                inplace = True)
end_of_year_prices.rename(columns = {'price_off_peak_var':'ending_year_price_energy', 
                                     'price_off_peak_fix':'ending_year_price_power'}, 
                          inplace = True)

# Merge the end_of_year_prices and beginning_of_year_price into a single dataframe using the .merge() command.
# Merge the dataframes using the id column.

price_differences = pd.merge(end_of_year_prices, beginning_of_year_prices, on = 'id')
```
Once the price_differences dataframe is created, we will the difference between the price at the end of the year and the price at the beginning of the year. The price differences will be added to the client_df dataframe.
```
# Calcualte the differences between the beginnig of year prices and end of year prices.

price_differences['off_peak_price_difference_energy'] = price_differences['ending_year_price_energy'] - price_differences ['beginning_year_price_energy']
price_differences['off_peak_price_difference_power'] = price_differences['ending_year_price_power'] - price_differences ['beginning_year_price_power']

# Keep only the id, off_peak_price_difference_energy, and off_peak_price_difference_power columns.

price_differences = price_differences[['id', 
                                       'off_peak_price_difference_energy', 
                                       'off_peak_price_difference_power']]

# Merge the price_differences dataframe to the client_df dataframe.
# Merge the dataframes on the id column.

client_df = pd.merge(client_df, 
                     price_differences, 
                     on = 'id')
```
Two features are added to the client_df dataframe:
* off_peak_price_difference_energy - The price difference of energy during off peak hours from the beginning of the year to the end of the year.
* off_peak_price_difference_power - The price difference of power during off peak hours from the beginning of the year to the end of the year.

**In summary, we:**
* Created a dataframe called price_differences which contains the price of energy and power during off peak hours at the beginning and end of the year.
* Calcualted the price difference of energy and power from the beginning of the year to the end of the year.
* Added the price differences to the client_df dataframe. The client_df datafame gains 2 new features.
* The client_df dataframe gains a total of 20 new features.

### 4. Creating Price Sensitivty Features: Average Price Across Peak Hours
**Another metric we can use to determine if price sensitivity is a major cause of churn is the average price across peak hours. This will give us an indication of how much every customer has to pay when peak hours change.**

First, we create a dataframe called avg_prices. The avg_prices dataframe contains the average price of energy and power for each peak hour.
```
# Create a dataframe than contains the average peak hour prices grouped by companies.
# Use the .groupby() command to group the data by id.
# Use the .agg() command to use multiple aggregate functions (mean) across multiple columns.
# Use the .reset_index() command to insert the outputs into a dataframe and reset the index column.

avg_prices = price_df.groupby(['id']).agg({'price_off_peak_var': 'mean', 
                                           'price_peak_var': 'mean', 
                                           'price_mid_peak_var': 'mean', 
                                           'price_off_peak_fix': 'mean', 
                                           'price_peak_fix': 'mean', 
                                           'price_mid_peak_fix': 'mean'}).reset_index()
```
Once the average prices are calculated, we will calculate the difference in energy price and power price across peak hours (off peak hours to peak hours, peak hours to mid peak hours, and off peak hours to mid peak hours). The avearage price differences will be added to the client_df dataframe.
```
# Calculate the differences of average prices across peak periods.

avg_prices['energy_mean_diff_off_peak_peak'] = avg_prices['price_off_peak_var'] - avg_prices['price_peak_var']
avg_prices['energy_mean_diff_peak_mid_peak'] = avg_prices['price_peak_var'] - avg_prices['price_mid_peak_var']
avg_prices['energy_mean_diff_off_peak_mid_peak'] = avg_prices['price_off_peak_var'] - avg_prices['price_mid_peak_var']
avg_prices['power_mean_diff_off_peak_peak'] = avg_prices['price_off_peak_fix'] - avg_prices['price_peak_fix']
avg_prices['power_mean_diff_peak_mid_peak'] = avg_prices['price_peak_fix'] - avg_prices['price_mid_peak_fix']
avg_prices['power_mean_diff_off_peak_mid_peak'] = avg_prices['price_off_peak_fix'] - avg_prices['price_mid_peak_fix']

# Keep only the id, and mean differences columns.

avg_prices = avg_prices[['id', 
                         'energy_mean_diff_off_peak_peak',
                         'energy_mean_diff_peak_mid_peak', 
                         'energy_mean_diff_off_peak_mid_peak', 
                         'power_mean_diff_off_peak_peak', 
                         'power_mean_diff_peak_mid_peak', 
                         'power_mean_diff_off_peak_mid_peak']]

# Merge the avg_prices dataframe to the client_df dataframe.
# Merge the dataframes on the id column.

client_df = pd.merge(client_df, 
                     avg_prices, 
                     on = 'id')
```
Six features are added to the client dataframe:
* energy_mean_diff_off_peak_peak - The average difference in energy price between off peak and peak hours.
* energy_mean_diff_peak_mid_peak - The average difference in energy price between peak and mid peak hours.
* energy_mean_diff_off_peak_mid_peak - The average difference in energy price between off peak and mid peak hours.
* power_mean_diff_off_peak_peak - The average difference in power price between off peak and peak hours.
* power_mean_diff_peak_mid_peak - The average difference in power price between peak and mid peak hours.
* power_mean_diff_off_peak_mid_peak - The average difference in power price between off peak and mid peak hours.

**In summary, we:**
* Created a dataframe called avg_prices which contains the average price of energy and power during peak hours.
* Calculated the change in the average prices across peak hours (off peak hours to peak hours, peak hours to mid peak hours, and off peak hours to mid peak hours).
* Added the average price differences to the client_df dataframe. The client_df datafame gains 6 new features.
* The client_df dataframe gains a total of 26 new features.
  
### 5. Creating Price Sensitivty Features: Greatest Price Change Across Peak Hours
**The last metric we will use to determine if price sensitivity is a major cause of churn is the greatest price change across peak hours.** In the previous section we have calculated the avearage price differences between peak hours. In this section we will calculate the largest change between peak hours. **This metric will give us the price range for each of PowerCo's customers.**

First, we create a dataframe called monthly_prices. Then we calculate the change in energy price and power price across peak hours for every month.
```
# Create a dataframe than contains the id and price_date columns, and the peak hour columns for energy and power.

monthly_prices = price_df.drop(columns = ['price_off_peak_total', 
                                          'price_peak_total', 
                                          'price_mid_peak_total'])

# Calculate the difference between peak periods for every company and every month.

monthly_prices['monthly_diff_off_peak_peak_var'] = monthly_prices['price_off_peak_var'] - monthly_prices['price_peak_var']
monthly_prices['monthly_diff_peak_mid_peak_var'] = monthly_prices['price_peak_var'] - monthly_prices['price_mid_peak_var']
monthly_prices['monthly_diff_off_peak_mid_peak_var'] = monthly_prices['price_off_peak_var'] - monthly_prices['price_mid_peak_var']
monthly_prices['monthly_diff_off_peak_peak_fix'] = monthly_prices['price_off_peak_fix'] - monthly_prices['price_peak_fix']
monthly_prices['monthly_diff_peak_mid_peak_fix'] = monthly_prices['price_peak_fix'] - monthly_prices['price_mid_peak_fix']
monthly_prices['monthly_diff_off_peak_mid_peak_fix'] = monthly_prices['price_off_peak_fix'] - monthly_prices['price_mid_peak_fix']
```
We will then group the data by customer ID and find the greatest value of each price difference across peak hours. The greatest values in price difference will be added to the client_df dataframe.
```
# Use the .groupby() command to group the data by id.
# Use the .agg() command to use multiple aggregate functions (max) across multiple columns.
# Use the .reset_index() command to insert the outputs into a dataframe and reset the index column.

max_difference_across_peak_period = monthly_prices.groupby(['id']).agg({'monthly_diff_off_peak_peak_var': 'max', 
                                                                        'monthly_diff_peak_mid_peak_var': 'max', 
                                                                        'monthly_diff_off_peak_mid_peak_var': 'max', 
                                                                        'monthly_diff_off_peak_peak_fix': 'max', 
                                                                        'monthly_diff_peak_mid_peak_fix': 'max', 
                                                                        'monthly_diff_off_peak_mid_peak_fix': 'max'}).reset_index()

# Use the .rename() command to rename columns.
max_difference_across_peak_period.rename(columns = {'monthly_diff_off_peak_peak_var':'max_diff_off_peak_peak_var', 
                                                    'monthly_diff_peak_mid_peak_var':'max_diff_peak_mid_peak_var', 
                                                    'monthly_diff_off_peak_mid_peak_var':'max_diff_off_peak_mid_peak_var', 
                                                    'monthly_diff_off_peak_peak_fix':'max_diff_off_peak_peak_fix', 
                                                    'monthly_diff_peak_mid_peak_fix':'max_diff_peak_mid_peak_fix', 
                                                    'monthly_diff_off_peak_mid_peak_fix':'max_diff_off_peak_mid_peak_fix'}, 
                                                    inplace = True)

# Merge the avg_prices dataframe to the client_df dataframe.
# Merge the dataframes on the id column

client_df = pd.merge(client_df, 
                     max_difference_across_peak_period, 
                     on = 'id')
```
Six features are added to the client dataframe:
* max_diff_off_peak_peak_var - The greatest difference in energy price between off peak and peak hours.
* max_diff_peak_mid_peak_var - The greatest difference in energy price between peak and mid peak hours
*	max_diff_off_peak_mid_peak_var - The greatest difference in energy price between off peak and mid peak hours.
* max_diff_off_peak_peak_fix - The greatest difference in power price between off peak and peak hours.
* max_diff_peak_mid_peak_fix - The greatest difference in power price between peak and mid peak hours
*	max_diff_off_peak_mid_peak_fix - The greatest difference in power price between off peak and mid peak hours.

**In summary, we:**
* Created a dataframe called monthly_prices which contains the monthly price of energy and power during peak hours.
* Calculated the price change across peak hours for every month/
* Grouped the data by customer ID, and found the greatest price change across peak hours.
* Added greatest price differences to the client_df dataframe. The client_df datafame gains 6 new features.
* The client_df dataframe gains a total of 32 new features.

### 6. Transforming Dates: Number of Months
We will be using the client_df dataframe as input for a predictive machine learning model. However, **predictive machine learning models cannot use dates as input and the client_df dataframe contains 4 date features.** date_activ, date_end, date_modif_prod, and date_renewal. **To make our our data more useful for predictive machine learning, we can convert dates into number of months.** The Data Science team has agreed to use Jan. 2016 as a reference date.

The code below will create a function called convert_months. The inputs for the convert_months function are the reference date (Jan. 2016), the dataframe we will apply the function to (client_df), and the column we will apply the function to. The function will subtract the reference date from each date in a column and calculate the number of days the date is from the reference date. The function will then divide the number of days by 30 to convert the number of days into number of months.
```
# Use the def command to create a function where you  an input a column with date time and get the number of months from the reference date.
# Name the function convert_months with variables for reference dates, dataframe, and column.
# Create a variable called months which subtracts a date from the reference date, divides the number of days by 30 (1 month), and convert that number into an int data type.
# The function should return the months value.

def convert_months(reference_date, df, column):
    months = ((reference_date - df[column]) / np.timedelta64(30, 'D')).astype(int)
    return months
```
Once the function is created, we will use the function to convert the date columns into number of months based on the reference date.
```
# Use the datetime() command to create the reference date.

reference_date = datetime(2016, 1, 1)

# Use the convert_months function to calculate the number of months a date is from the reference date.
# Use the convert_months function on the months_activ, months_to_end, months_modif_prod, and months_renewal columns.

client_df['months_activ'] = convert_months(reference_date, client_df, 'date_activ')
client_df['months_to_end'] = -convert_months(reference_date, client_df, 'date_end')
client_df['months_modif_prod'] = convert_months(reference_date, client_df, 'date_modif_prod')
client_df['months_renewal'] = convert_months(reference_date, client_df, 'date_renewal')
```
Four features are added to the client_df dataframe:
* months_activ - Number of months the contract is activated.
* months_to_end - Number of months until the end of the contract.
* months_modif_prod - Number of months since the last modification of the contract.
* months_renewal - Number of months until the next renewal.

**In summary, we:**
* Created a function that will take a reference date and convert a date into number of months based on the reference date.
* Used the function to convert dates into number of months.
* The client_df dataframe gains 4 new features.
* The client_df dataframe gains a total of 36 new features.

## 7. Transforming Dates: Tenure
