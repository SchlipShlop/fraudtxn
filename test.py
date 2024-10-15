#!/usr/bin/env python3  
# -\*- coding: utf-8 -\*-  
"""  
Created on Tue Jul 23 18:39:53 2024  
@author: jenniferhamrick  
"""  
"""# \*\*Initialize analytics packages\*\*"""  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.compose import make_column_selector as selector  
from sklearn.preprocessing import OneHotEncoder  
from sklearn.pipeline import Pipeline  
from sklearn.compose import ColumnTransformer  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report  
"""# \*\*Load datasets\*\*"""  
# Specify the path to your CSV file  
file_path = 'fraudTest.csv'  
# Load the CSV file into a DataFrame  
test = pd.read_csv(file_path)  
# Create new column labelling all of these rows as "TEST"  
test['rowtype']="TEST"  
# Display the first few rows of the DataFrame  
print(test.head())  
# Specify the path to your CSV file  
file_path = 'fraudTrain.csv'  
# Load the CSV file into a DataFrame  
train = pd.read_csv(file_path)  
# Create new column labelling all of these rows as "TRAIN"  
train['rowtype']="TRAIN"  
# Display the first few rows of the DataFrame  
print(train.head())  
#Determine the dimensions of each:  
print(train.shape)  
print(test.shape)  
#Feature engineering  
#Union the TRAIN and TEST sets together for feature engineering  
df = pd.concat([train, test])  
"""# \*\*Build some simple cross-tabs\*\*"""  
#gender:  
cross_tab = pd.crosstab(df['gender'], df['is_fraud'], normalize='index')  
# Display the cross-tabulation table  
print(cross_tab)  
#merchant category:

cross_tab = pd.crosstab(df['category'], df['is_fraud'], normalize='index')  
# Display the cross-tabulation table  
print(cross_tab)  
#Merchant category:  
# Assuming you have already created the cross-tabulation table and stored it in the variable cross_tab  
# Sort the cross-tabulation table by the frequency of '1' in the 'is_fraud' column  in descending order  
cross_tab_sorted = cross_tab.sort_values(by=1, ascending=False)  
# Display the sorted cross-tabulation table  
print(cross_tab_sorted)  
import matplotlib.pyplot as plt  
# Plot the cross-tabulation table as a bar chart  
cross_tab_sorted.plot(kind='bar', stacked=True)  
# Set the title and labels  
plt.title('Fraud Distribution by Merchant Category')  
plt.xlabel('Merchant Category')  
plt.ylabel('Proportion Fraud vs. Not')  
# Show the plot  
plt.show()  
"""# \*\*Build a new variable -- online vs. in-person sales\*\*"""  
# Assuming you have a DataFrame named df with the 'category' column  
# Define a function to check if the last four characters of a string equal "_net"  (case insensitive)  
def check_net(category):  
    return 1 if category.lower().endswith('_net') else 0  
# Create the new 'NET' column using the apply() function  
df['ONLINE'] = df['category'].apply(lambda x: check_net(x))  
# Verify it worked as inspected by checking a cross-tabulation for "category" vs.  "NET" to make sure the classification worked  
cross_tab_category_net = pd.crosstab(df['category'], df['ONLINE'])  
print(cross_tab_category_net)  
#Now remove the "_net" and "_pos" suffixes since I've created the online variable.  
# Define a function to clean the category field  
def clean_category(category):  
    if category.lower().endswith('_net'):  
        return category[:-4] # Remove the last 4 characters  
    elif category.lower().endswith('_pos'):  
        return category[:-4] # Remove the last 4 characters  
    else:  
        return category # Return the original value if no suffix found  
# Create the new 'category_clean' column using the apply() function  
df['category_clean'] = df['category'].apply(lambda x: clean_category(x))

# Display the DataFrame to verify the new column  
print(df)  
#Now crosstab our "ONLINE" variable vs. the cleaned category  
cross_tab_category_net = pd.crosstab(df['category_clean'], df['ONLINE'])  
print(cross_tab_category_net)  
"""# \*\*Build a new variable -- time of day\*\*"""  
# Convert 'trans_date_trans_time' column to datetime type  
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])  
# Extract hour of the day and create a new column 'charge_hour'  
df['charge_hour'] = df['trans_date_trans_time'].dt.hour  
# Display the DataFrame to verify the new column  
print(df)  
#Group the hours into times of the day:  
# Define a function to categorize the hour of the day  
def categorize_hour(hour):  
    if 0 <= hour < 6:  
        return "Wee hours"  
    elif 6 <= hour < 12:  
        return "Morning"  
    elif 12 <= hour < 18:  
        return "Afternoon"  
    elif 18 <= hour < 21:  
        return "Evening"  
    else:  
        return "Night"  
# Apply the categorize_hour function to create a new column 'time_period'  
df['time_period'] = df['charge_hour'].apply(categorize_hour)  
# Display the DataFrame to verify the new column  
print(df)  
#Create indicator variables ("one hot encoding")  
# Apply one-hot encoding to time periods  
time_period_dummies = pd.get_dummies(df['time_period'], prefix='time_period')  
# Concatenate the one-hot encoded columns with the original DataFrame  
df = pd.concat([df, time_period_dummies], axis=1)  
# Display the DataFrame to verify the new columns  
print(df)  
print(df)  
"""# \*\*Now weekend vs. weekday\*\*"""  
# Convert 'trans_date_trans_time' column to datetime type  
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])  
# Extract the day of the week and create a new column 'day_of_week'  
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()

# Display the DataFrame to verify the new column  
print(df)  
# Create a new column indicating whether the day is a weekend or not  
df['is_weekend'] = df['trans_date_trans_time'].dt.dayofweek // 5 == 1  
# Convert the boolean values to integers (0 for False, 1 for True)  
df['is_weekend'] = df['is_weekend'].astype(int)  
# Display the DataFrame to verify the new column  
print(df)  
"""# \*\*Why do merchant names all start with 'fraud_'?\*\*  
Clean the data.  
"""  
# Create a new column for the first six characters of the "merchant" variable  
df['merchant_first_six'] = df['merchant'].str[:6]  
# Calculate the frequency table for the number of distinct values in the  "merchant_first_six" column  
frequency_table = df['merchant_first_six'].value_counts()  
# Display the frequency table  
print(frequency_table)  
# Create a new column "merchant_clean" by omitting the first six characters of the  "merchant" variable  
df['merchant_clean'] = df['merchant'].str[6:]  
# Display the DataFrame to verify the new column  
print(df)  
#Are these known vendors? NOPE! Gas stations are oddly misnamed.  
# Filter the DataFrame where the category is "gas_transport"  
gas_transport_df = df[df['category'] == 'gas_transport']  
# Calculate the frequency table for each unique merchant name  
frequency_table = gas_transport_df['merchant_clean'].value_counts()  
# Display the frequency table  
print(frequency_table)  
"""# \*\*Prepare dataset for analysis\*\*  
Keep only the variables of interest.  
"""  
#List the columns  
df.columns  
#rename the transaction ID field  
df.rename(columns={'Unnamed: 0': 'tran_id'}, inplace=True)  
df.columns

#Send only the columns I want to keep into a new dataframe  
df_data = df[['tran_id','cc_num','merchant_clean','category_clean','amt',  
'gender','city_pop','ONLINE',  
'time_period_Afternoon', 'time_period_Evening',  
'time_period_Morning', 'time_period_Night', 'time_period_Wee hours',  
'day_of_week', 'is_weekend','is_fraud','rowtype']]  
#Note: I'm ignoring potentially valuable location data (zip, lat, long, merch_lat, merch_log) --  
# Although certain zipcodes for the card owner / the merchant may be more susceptible to theft,  
# that would require creating a whole bunch of indicator variable columns and/or Geo data processing to  
# gather census/descriptive information about each zipcode.  
#Note: I'm also ignoring that some merchants may be more likely for fraudsters to try to spend money at.  
# I'm sticking just with merchant category rather than any more detailed encoding.  
#Note: Credit card numbers in this dataset are in plain text (that'd never happen in practice)  
#Also, credit card numbers in this data set do not adhere to block coding conventions  
#e.g., Visa cards don't begin with \___, and Amex Cards don't have 15 digits.  
print(df_data)  
df_data.columns  
"""\*\*Reshape columns: text -\> numeric\*\*  
"""  
# One-hot encode category_clean into indicator variables with a prefix  
category_clean_encoded = pd.get_dummies(df_data['category_clean'],  
prefix='merch_cat')  
# Concatenate the one-hot encoded columns with the original DataFrame  
df_data = pd.concat([df_data, category_clean_encoded], axis=1)  
# Display the DataFrame to verify the new columns  
print(df_data)  
#Display all variable names:  
#Need to hot encode gender  
gender = pd.get_dummies(df_data['gender'])  
# Concatenate the one-hot encoded columns with the original DataFrame  
df_data = pd.concat([df_data, gender], axis=1)  
df_data.columns  
"""\*\*Reshape columns: Fix skewness in numeric columns by logging them\*\*

\*\*Amount column\*\*  
"""  
#AMOUNT:  
# Calculate the skewness of the "amt" column  
skewness = df_data['amt'].skew()  
# Calculate the kurtosis of the "amt" column  
kurtosis = df_data['amt'].kurtosis()  
# Display the skewness and kurtosis  
print("Skewness:", skewness)  
print("Kurtosis:", kurtosis)  
# Generate summary statistics for the 'amt' column  
summary_stats = df_data['amt'].describe()  
# Print the summary statistics  
print(summary_stats)  
#My data has no negative amounts, so it is a candidate for a log transformation.  
#Create a logged variable  
df_data['amt_log'] = np.log(df_data['amt'])  
# Calculate the skewness of the "amt" column  
skewness = df_data['amt_log'].skew()  
# Calculate the kurtosis of the "amt" column  
kurtosis = df_data['amt_log'].kurtosis()  
# Display the skewness and kurtosis  
print("Skewness:", skewness)  
print("Kurtosis:", kurtosis)  
"""\*\*Owner's city's population\*\*"""  
#AMOUNT:  
# Calculate the skewness of the "city_pop" column  
skewness = df_data['city_pop'].skew()  
# Calculate the kurtosis of the "amt" column  
kurtosis = df_data['city_pop'].kurtosis()  
# Display the skewness and kurtosis  
print("Skewness:", skewness)  
print("Kurtosis:", kurtosis)  
# Generate summary statistics for the 'amt' column  
summary_stats = df_data['city_pop'].describe()  
# Print the summary statistics  
print(summary_stats)  
#My data has no negative amounts, so it is a candidate for a log transformation.  
#Create a logged variable  
df_data['city_pop_log'] = np.log(df_data['city_pop'])

# Calculate the skewness of the "amt" column  
skewness = df_data['city_pop_log'].skew()  
# Calculate the kurtosis of the "amt" column  
kurtosis = df_data['city_pop_log'].kurtosis()  
# Display the skewness and kurtosis  
print("Skewness:", skewness)  
print("Kurtosis:", kurtosis)  
"""# \*\*SPLIT: Separate 'train' dataset from 'test'\*\*"""  
df_data.columns  
train2 = df_data[df_data['rowtype'] == 'TRAIN'].copy()  
#How many rows in original dataset?  
print('Original train dataset had this number of observations:')  
print(len(train))  
#How many rows in transformed dataset?  
print('Transformed train dataset ("train2") has this number of observations:')  
print(len(train2))  
test2 = df_data[df_data['rowtype'] == 'TEST'].copy()  
#How many rows in original dataset?  
print('Original test dataset had this number of observations:')  
print(len(test))  
#How many rows in transformed dataset?  
print('Transformed test dataset ("test2") has this number of observations:')  
print(len(test2))  
"""# \*\*ANALYSIS: Logistic Regression\*\*"""  
# Display the data types of each column  
#print(train2.dtypes)  
# List of columns to exclude from the features (X)  
# Be sure to drop one of the indicator variables from each hot-encoded set!  
# The omitted variable will serve as the baseline for comparison...  
columns_to_exclude = ['tran_id', 'cc_num', 'merchant_clean', 'category_clean',  
'amt', 'gender', 'city_pop', 'day_of_week',  
'time_period_Morning',  
'merch_cat_grocery',  
'F', 'is_fraud', 'rowtype']  
"""BASELINE"""  
#The logistic regression will compare the time period indicators to:  
#time_period_Morning  
#The "M" (male indicator) will be compared to a baseline of:  
#F  
#The merchant category indicators will be compared to:  
#grocery

# separate features and target:  
X = train2.drop('is_fraud',axis=1)  
y = train2['is_fraud']  
# Create a filtered DataFrame without the excluded columns  
X = train2.drop(columns_to_exclude, axis=1)  
#drop any missing values  
X = X.dropna()  
y = y.dropna()  
#print(X.dtypes)  
"""Train the logistic model"""  
from sklearn.linear_model import LogisticRegression  
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)  
#, class_weight='balanced'  
prediction = clf.predict(X)  
import matplotlib.pyplot as plt  
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  
"""#Evaluate the trained logistic model using a standardized accuracy report:"""  
cm = confusion_matrix(y, prediction, labels=clf.classes_)  
print(classification_report(y, prediction))  
#Predictive accuracy:  
#Within training dataset:  
dp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes) 
dp.plot()  
print("")  
print(cm)  
"""Evaluate the logistic model against the holdout data set named "test" """  
#Identify true positive flag in test data  
y_true = test2['is_fraud']  
#Drop the columns not used to predict in the previous model.  
# Create a filtered DataFrame without the excluded columns  
X_test = test2.drop(columns_to_exclude, axis=1)  
# Make predictions on the test data  
y_pred = clf.predict(X_test)  
# Generate the classification report for the test data  
report = classification_report(y_true, y_pred)  
print(report)  
"""#Evaluate the logistic model's performance on the TEST data using the

standardized accuracy report:"""  
# Calculate the confusion matrix for the test data  
cm_test = confusion_matrix(y_true, y_pred)  
# Create the confusion matrix display object  
dp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test,  
display_labels=clf.classes_)  
# Print the confusion matrix as a text array  
print("")  
print(cm_test)  
# Plot the confusion matrix  
dp_test.plot()  
plt.show()  
#Precision = correctly predicted positives / total predicted positives  
# = ?? / ??  
# = ??%  
#"Precision, also known as positive predictive value, measures the accuracy of the  positive predictions.  
#It is the ratio of correctly predicted positive observations (true positives) to  the total predicted positives (true positives plus false positives).  
#High precision indicates that an algorithm returned substantially more relevant  results than irrelevant ones."  
#Recall = correctly predicted positives / true positives  
# = ?? / ??  
# = ??%  
#Recall, also known as sensitivity, measures the ability of the model to find all  the relevant cases within a dataset.  
#It is the ratio of correctly predicted positive observations (true positives) to  all the actual positives (true positives plus false negatives). High recall means that an algorithm returned most of the relevant results.  
# \*\*Random Forest Classifier\*\*  
from sklearn.ensemble import RandomForestClassifier  
clf2 = RandomForestClassifier(random_state=0, max_depth=10).fit(X, y)  
#, class_weight='balanced'  
prediction = clf2.predict(X)  
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  
cm = confusion_matrix(y, prediction, labels=clf.classes_)  
print(classification_report(y, prediction))  
dp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)  
dp.plot()  
# This ensures that the plot is displayed in your environment  
plt.show()

#Apply Trained Random Forest Model to Test Data  
# Make predictions on the test data  
y_pred = clf2.predict(X_test)  
# Generate the classification report for the test data  
report2 = classification_report(y_true, y_pred)  
print(report2)  
# Calculate the confusion matrix for the test data  
cm_test2 = confusion_matrix(y_true, y_pred)  
# Create the confusion matrix display object  
dp_test2 = ConfusionMatrixDisplay(confusion_matrix=cm_test2,  
display_labels=clf2.classes_)  
# Print the confusion matrix as a text array  
print("")  
print(cm_test2)  
# Plot the confusion matrix  
dp_test2.plot()  
plt.show()  
#INTERPRETABILITY: LOGISTIC:  
#Interpret coefficients  
import pandas as pd  
import matplotlib.pyplot as plt  
# Extracting coefficients  
feature_names = X.columns # Replace 'X.columns' with the actual column names of your dataset  
coefficients = clf.coef_[0] # Getting the first (and only) row of coefficients  
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})  
# Sorting the coefficients for better visualization  
coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)  
# Visualizing the coefficients  
plt.figure(figsize=(10, 8))  
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')  
plt.xlabel('Change in log-odds')  
plt.ylabel('Feature')  
plt.title('Impact of Features on Fraud Prediction')  
plt.gca().invert_yaxis() # Inverts the y-axis so the largest bar is on top  
plt.show()  
#State the coefficients in terms of probability relative to the baseline.  
import numpy as np  
# Extracting the intercept  
intercept = clf.intercept_[0]  
# Calculate the baseline probability using the logistic function  
baseline_probability = 1 / (1 + np.exp(-intercept))

# Add a new column to the DataFrame to store the probabilities  
coef_df['Probability'] = np.exp(intercept) / (np.exp(intercept) +  
np.exp(coef_df['Coefficient']))  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
# Calculate baseline probability using the logistic function  
intercept = clf.intercept_[0]  
baseline_probability = 1 / (1 + np.exp(-intercept))  
# Calculate the probability impact of each coefficient independently  
feature_effects = pd.DataFrame(X.columns, columns=['Feature'])  
feature_effects['Coefficient'] = clf.coef_[0]  
# We calculate the probability considering each coefficient separately  
# and all other coefficients as zero (i.e., their effect is not included).  
feature_effects['Probability'] = feature_effects['Coefficient'].apply(  
lambda coef: 1 / (1 + np.exp(-(intercept + coef)))  
)  
#Exclude continuous variable effects  
feature_effects = feature_effects[~feature_effects['Feature'].isin(['amt_log',  
'city_pop_log'])]  
# Divide features into positive and negative effects based on their coefficients  
positive_effects = feature_effects[feature_effects['Coefficient'] >  0].sort_values(by='Probability', ascending=False)  
negative_effects = feature_effects[feature_effects['Coefficient'] <  0].sort_values(by='Probability')  
# Set pandas display options to format float numbers with 10 decimal places  
pd.set_option('display.float_format', '{:.10f}'.format)  
print("Indicator vars with a decreasing effect vs. baseline")  
print(negative_effects)  
print(" ")  
print("Baseline probability of fraud")  
print(f"{baseline_probability:.10f}")  
#Print positive effects in reverse order to stick with the overall order of ascending probabilities  
print(" ")  
print("Indicator vars with an increasing effect vs. baseline")  
print(positive_effects.iloc[::-1])  
# Plotting  
plt.figure(figsize=(10, 8))  
# Plot bars for positive effects  
plt.barh(positive_effects['Feature'], positive_effects['Probability'],  
color='skyblue')

# Plot bars for negative effects  
plt.barh(negative_effects['Feature'], negative_effects['Probability'],  
color='salmon')  
# Draw a line for the baseline probability  
plt.axvline(x=baseline_probability, color='grey', linestyle='--', label='Baseline Probability')  
# Annotate the baseline probability line with the actual value  
plt.text(baseline_probability, 0, f'Baseline\\n{baseline_probability:.2f}',  
color='grey', ha='right')  
plt.xlabel('Absolute Probability of Fraud')  
plt.ylabel('Feature')  
plt.title('Impact of Binary Variable Features on Fraud Probability')  
plt.legend()  
plt.show()