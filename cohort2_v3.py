# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:10:55 2024

@author: HP
"""

from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import shapiro
from tabulate import tabulate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind


# for decrypting a file
import msoffcrypto
import io

file_name = "C://Users//HP//Documents//Masters//Research Work//Data//1258_postCABG_Outcome_FA_Dati.xlsx"
file = msoffcrypto.OfficeFile(open(file_name, "rb"))

file.load_key(password="Corino1258")  # Use password

decrypted = io.BytesIO()
file.decrypt(decrypted)

cohort2 = pd.read_excel(decrypted)

# Replace -999 with NaN
cohort2['BMI'].replace(-999, np.nan, inplace=True)
cohort2['LVEF'].replace(-999, np.nan, inplace=True)

cohort2 = cohort2.drop(cohort2[cohort2['PAST_AF'] == 'Unknown'].index)

##creating numerical and categorical features
numfeat = cohort2[['AGE', 'BMI', 'CREAT', 'LVEF', 'CPB_TIME', 'AGE_CAT']]

catfeat = cohort2[['RACE', 'AGE_CAT', 'GENDER', 'PAST_AF', 'PRE_AF', 'PREMAZE', 'HYPER', 'DIAB', 'PAST_MI', 'PULM', 'CREAT_CAT', 'LVEF35', 'PREACE', 'PREAA', 'PREASP','PREBB', 'PRECACB', 'PREDIUR', 'PRESTAT', 'ONCABG', 'OFFCABG', 'VALVCABG', 'OTHSURG', 'INTMAZE', 'INTRAINO', 'POSTSTAT', 'POSTBB', 'POSTAF', 'TREATAF', 'ON_AF', 'OFF_AF']]

##separate the data into male and female to make clear distinctions in target feature: Atrial Fibrillation

female = cohort2[cohort2['GENDER'] == 'Female']
male = cohort2[cohort2['GENDER'] == 'Male']

numfeat_female = female[['AGE', 'BMI', 'CREAT', 'LVEF', 'CPB_TIME']]

numfeat_male = male[['AGE', 'BMI', 'CREAT', 'LVEF', 'CPB_TIME']]

catfeat_female = female[['RACE', 'AGE_CAT', 'GENDER', 'PAST_AF', 'PRE_AF', 'PREMAZE', 'HYPER', 'DIAB', 'PAST_MI', 'PULM', 'CREAT_CAT', 'LVEF35', 'PREACE', 'PREAA', 'PREASP','PREBB', 'PRECACB', 'PREDIUR', 'PRESTAT', 'ONCABG', 'OFFCABG', 'VALVCABG', 'OTHSURG', 'INTMAZE', 'INTRAINO', 'POSTSTAT', 'POSTBB', 'POSTAF', 'TREATAF', 'ON_AF', 'OFF_AF']]

catfeat_male = male[['RACE', 'AGE_CAT', 'GENDER', 'PAST_AF', 'PRE_AF', 'PREMAZE', 'HYPER', 'DIAB', 'PAST_MI', 'PULM', 'CREAT_CAT', 'LVEF35', 'PREACE', 'PREAA', 'PREASP','PREBB', 'PRECACB', 'PREDIUR', 'PRESTAT', 'ONCABG', 'OFFCABG', 'VALVCABG', 'OTHSURG', 'INTMAZE', 'INTRAINO', 'POSTSTAT', 'POSTBB', 'POSTAF', 'TREATAF', 'ON_AF', 'OFF_AF']]

##calculating basic parameters in female##
basic_stats_female = numfeat_female.describe()
basic_stats_female.loc['variance'] = numfeat_female.var(skipna=True)
# checking for normality
for i, column in enumerate(numfeat_female.columns):
    stat, p_value = shapiro(numfeat_female[column].dropna())
    basic_stats_female.loc['p-value', column] = p_value

##calculating basic parameters in male##
basic_stats_male = numfeat_male.describe()
basic_stats_male.loc['variance'] = numfeat_male.var(skipna=True)
# checking for normality
for i, column in enumerate(numfeat_male.columns):
    stat, p_value = shapiro(numfeat_male[column].dropna())
    basic_stats_male.loc['p-value', column] = p_value

####calculating basic parameters in age category##
    
##<=59
numfeat_age1 = numfeat[(numfeat['AGE_CAT'] == 'AGE_50_59') | (numfeat['AGE_CAT'] == 'AGE_LTT_50')]
basic_stats_age1 = numfeat_age1.describe()
numfeat_age1.drop(columns = ['AGE_CAT'], inplace = True)
basic_stats_age1.loc['variance'] = numfeat_age1.var(skipna=True)
for i, column in enumerate(numfeat_age1.columns):
    stat, p_value = shapiro(numfeat_age1[column].dropna())
    basic_stats_age1.loc['p-value', column] = p_value
    
##60-69
numfeat_age2 = numfeat[numfeat['AGE_CAT'] == 'AGE_60_69']
basic_stats_age2 = numfeat_age2.describe()
numfeat_age2.drop(columns = ['AGE_CAT'], inplace = True)
basic_stats_age2.loc['variance'] = numfeat_age2.var(skipna=True)
for i, column in enumerate(numfeat_age2.columns):
    stat, p_value = shapiro(numfeat_age2[column].dropna())
    basic_stats_age2.loc['p-value', column] = p_value
    
##>70
numfeat_age3 = numfeat[(numfeat['AGE_CAT'] == 'AGE_70_79') | (numfeat['AGE_CAT'] == 'AGE_GT_79')]

basic_stats_age3 = numfeat_age3.describe()
numfeat_age3.drop(columns = ['AGE_CAT'], inplace = True)
basic_stats_age3.loc['variance'] = numfeat_age3.var(skipna=True)
for i, column in enumerate(numfeat_age3.columns):
    stat, p_value = shapiro(numfeat_age3[column].dropna())
    basic_stats_age3.loc['p-value', column] = p_value

#making dist plots for gender##
##female
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Female\n")
for i, column in enumerate(numfeat_female.columns):
    plt.subplot(2, 3, i+1)
    for disease in female['PAST_AF'].unique():
        sns.distplot(numfeat_female[column].loc[female['PAST_AF'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()

##male
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Male\n")
for i, column in enumerate(numfeat_male.columns):
    plt.subplot(2, 3, i+1)
    for disease in male['PAST_AF'].unique():
        sns.distplot(numfeat_male[column].loc[male['PAST_AF'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()
    
##age groups##

##<50
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in <50 years old\n")
for i, column in enumerate(numfeat_age1.columns):
    plt.subplot(2, 3, i+1)
    for disease in cohort2['PAST_AF'].unique():
        sns.distplot(numfeat_age1[column].loc[cohort2['PAST_AF'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()
    
##60-69
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in 60-69 years old\n")
for i, column in enumerate(numfeat_age2.columns):
    plt.subplot(2, 3, i+1)
    for disease in cohort2['PAST_AF'].unique():
        sns.distplot(numfeat_age2[column].loc[cohort2['PAST_AF'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()

    
##>70
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in >70 years old\n")
for i, column in enumerate(numfeat_age3.columns):
    plt.subplot(2, 3, i+1)
    for disease in cohort2['PAST_AF'].unique():
        sns.distplot(numfeat_age3[column].loc[cohort2['PAST_AF'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()

##making boxplots for gender##
#in female
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axs.flat):
    data = numfeat_female.iloc[:, i].dropna()
    sns.boxplot(x=female['PAST_AF'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_female.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)

#in male
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axs.flat):
    data = numfeat_male.iloc[:, i].dropna()
    sns.boxplot(x=male['PAST_AF'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_male.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)

##box plots in age group##

#<59
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axs.flat):
    data = numfeat_age1.iloc[:, i].dropna()
    sns.boxplot(x=cohort2['PAST_AF'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age1.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)
    
#60-69    
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axs.flat):
    data = numfeat_age2.iloc[:, i].dropna()
    sns.boxplot(x=cohort2['PAST_AF'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age2.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)
    
#>70
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axs.flat):
    data = numfeat_age3.iloc[:, i].dropna()
    sns.boxplot(x=cohort2['PAST_AF'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age3.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)

##numerical correlation in gender##
#in female
numfeat_female_corr = numfeat_female.dropna().corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_female_corr, annot = True)
sns.set(font_scale = 2)

##in male
numfeat_male_corr = numfeat_male.dropna().corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_male_corr, annot = True)
sns.set(font_scale = 2)

##numerical correlation in age group##
#<50
numfeat_age1_corr = numfeat_age1.dropna().corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_age1_corr, annot = True)
sns.set(font_scale = 2)

#60-69
numfeat_age2_corr = numfeat_age2.dropna().corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_age2_corr, annot = True)
sns.set(font_scale = 2)

#>70
numfeat_age3_corr = numfeat_age3.dropna().corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_age3_corr, annot = True)
sns.set(font_scale = 2)

##frequency plots in gender##

#in female
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_female.columns):
    plt.subplot(7,6, i+1)
    count_female_catfeat = sns.countplot(x=catfeat_female[column].dropna(), hue = female['PAST_AF'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_female_catfeat.containers:
        count_female_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##in male
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_male.columns):
    plt.subplot(7,6, i+1)
    count_male_catfeat = sns.countplot(x=catfeat_male[column].dropna(), hue = male['PAST_AF'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_male_catfeat.containers:
        count_male_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##frequency plots in age group##

#<60
catfeat_age1 = catfeat[(catfeat['AGE_CAT'] == 'AGE_50_59') | (catfeat['AGE_CAT'] == 'AGE_LTT_50')]
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age1.columns):
    plt.subplot(7,6, i+1)
    count_age1_catfeat = sns.countplot(x=catfeat_age1[column].dropna(), hue = cohort2['PAST_AF'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age1_catfeat.containers:
        count_age1_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##60-69
catfeat_age2 = catfeat[(catfeat['AGE_CAT'] == 'AGE_60_69')]
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age2.columns):
    plt.subplot(7,6, i+1)
    count_age2_catfeat = sns.countplot(x=catfeat_age2[column].dropna(), hue = cohort2['PAST_AF'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age2_catfeat.containers:
        count_age2_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##>70
catfeat_age3 = catfeat[(catfeat['AGE_CAT'] == 'AGE_70_79') | (catfeat['AGE_CAT'] == 'AGE_GT_79')]
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age3.columns):
    plt.subplot(7,6, i+1)
    count_age3_catfeat = sns.countplot(x=catfeat_age3[column].dropna(), hue = cohort2['PAST_AF'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age3_catfeat.containers:
        count_age3_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##statistical tests on categorical variable - chi2 test

##in female
# Create contingency tables for each categorical variable
contingency_tables = {}
for col in catfeat_female.columns:
    if col != 'PAST_AF':
        contingency_table = pd.crosstab(catfeat_female['PAST_AF'], catfeat_female[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_female = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_female = pd.concat([results_female, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_female = results_female.drop(results_female[results_female['P-value'] >= 0.05].index)


##in male
# Create contingency tables for each categorical variable
contingency_tables = {}
for col in catfeat_male.columns:
    if col != 'PAST_AF':
        contingency_table = pd.crosstab(catfeat_male['PAST_AF'], catfeat_male[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_male = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_male = pd.concat([results_male, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_male = results_male.drop(results_male[results_male['P-value'] >= 0.05].index)


##in age group
##<60
contingency_tables = {}
for col in catfeat_age1.columns:
    if col != 'PAST_AF':
        contingency_table = pd.crosstab(catfeat_age1['PAST_AF'], catfeat_age1[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age1 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age1 = pd.concat([results_age1, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age1 = results_age1.drop(results_age1[results_age1['P-value'] >= 0.05].index)

##<60-69
contingency_tables = {}
for col in catfeat_age2.columns:
    if col != 'PAST_AF':
        contingency_table = pd.crosstab(catfeat_age2['PAST_AF'], catfeat_age2[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age2 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age2 = pd.concat([results_age2, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age2 = results_age2.drop(results_age2[results_age2['P-value'] >= 0.05].index)


##>70
contingency_tables = {}
for col in catfeat_age3.columns:
    if col != 'PAST_AF':
        contingency_table = pd.crosstab(catfeat_age3['PAST_AF'], catfeat_age3[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age3 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age3 = pd.concat([results_age3, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age3 = results_age3.drop(results_age3[results_age3['P-value'] >= 0.05].index)


##plotting associations	

final_catfeat_female = catfeat_female[['LVEF35', 'PREAA','INTMAZE']]

plt.figure(figsize=(15, 5))
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)
for i, column in enumerate(final_catfeat_female.columns):
    plt.subplot(1,3, i+1)
    count_female_catfeat = sns.countplot(x=final_catfeat_female[column].dropna(), hue = catfeat_female['PAST_AF'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_female_catfeat.containers:
        count_female_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##	Variable CREAT_CAT PREAA PREASP PREDIUR VALVCABG INTMAZE INTRAINO POSTBB TREATAF 

final_catfeat_male = catfeat_male[['CREAT_CAT', 'PREAA', 'PREASP', 'PREDIUR', 'VALVCABG', 'INTMAZE', 'INTRAINO', 'POSTBB', 'TREATAF']]

plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)
for i, column in enumerate(final_catfeat_male.columns):
    plt.subplot(3,3, i+1)
    count_male_catfeat = sns.countplot(x=final_catfeat_male[column].dropna(), hue = catfeat_male['PAST_AF'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_male_catfeat.containers:
        count_male_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##	Variable PREAA PREASP PREDIUR INTMAZE TREATAF
final_catfeat_age2 = catfeat_age2[['CREAT_CAT', 'PREAA', 'PREASP', 'PREDIUR', 'VALVCABG', 'INTMAZE', 'INTRAINO', 'POSTBB', 'TREATAF']]

plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)
for i, column in enumerate(final_catfeat_age2.columns):
    plt.subplot(3,3, i+1)
    count_age2_catfeat = sns.countplot(x=final_catfeat_age2[column].dropna(), hue = catfeat_age2['PAST_AF'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age2_catfeat.containers:
        count_age2_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##num feats statistical tests in female
##for non normal data 
numfem = female[['AGE', 'CREAT', 'LVEF', 'CPB_TIME']]
                
results_numfem = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = female[cohort2['PAST_AF'] == 'Yes']
group2 = female[cohort2['PAST_AF'] == 'No']

# Iterate over each column in the numfem DataFrame
for column in numfem.columns:
    # Retrieve data for group 1 and group 2 based on the CHD recurrence post CABG variable
    group1_data = group1[column].dropna()
    group2_data = group2[column].dropna()
    
    # Perform the Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(group1_data, group2_data)
    
    result_row = {
    'Variable': column,  # Name of the variable being tested
    'U Statistic': u_statistic,
    'P-value': p_value
}

# Convert the dictionary to a DataFrame
    result_df =             pd.DataFrame([result_row])

# Concatenate the result_df with the results_numfem DataFrame
    results_numfem = pd.concat([results_numfem, result_df[result_df['P-value'] < 0.05]], ignore_index=True)

##for normal data
test_stat, p_value = ttest_ind(group1['BMI'], group2['BMI'].dropna()) ##result: no associations

##num feats statistical tests in male
##for non normal data 
nummale = male[['AGE', 'CREAT', 'LVEF', 'CPB_TIME', 'BMI']]
                
results_nummale = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = male[cohort2['PAST_AF'] == 'Yes']
group2 = male[cohort2['PAST_AF'] == 'No']

# Iterate over each column in the numfem DataFrame
for column in nummale.columns:
    # Retrieve data for group 1 and group 2 based on the CHD recurrence post CABG variable
    group1_data = group1[column].dropna()
    group2_data = group2[column].dropna()
    
    # Perform the Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(group1_data, group2_data)
    
    result_row = {
    'Variable': column,  # Name of the variable being tested
    'U Statistic': u_statistic,
    'P-value': p_value
}

# Convert the dictionary to a DataFrame
    result_df =             pd.DataFrame([result_row])

# Concatenate the result_df with the results_numfem DataFrame
    results_nummale = pd.concat([results_nummale, result_df[result_df['P-value'] < 0.05]], ignore_index=True)

##plotting associations in male 
data1 = male['AGE']
data2 = male['CREAT']
data3 = male['LVEF']

fig, axs = plt.subplots(1,3, figsize = (15,5))
sns.kdeplot(x = data1, hue = male['PAST_AF'], ax = axs[0])
sns.kdeplot(x = data2,hue = male['PAST_AF'],  ax = axs[1])
sns.kdeplot(x = data3, hue = male['PAST_AF'], ax = axs[2])

##num feats statistical tests in age1
##for non normal data 
numage1 = numfeat_age1[['AGE', 'CREAT', 'LVEF', 'CPB_TIME', 'BMI']]
                
results_age1 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numage1[cohort2['PAST_AF'] == 'Yes']
group2 = numage1[cohort2['PAST_AF'] == 'No']

# Iterate over each column in the numfem DataFrame
for column in numage1.columns:
    # Retrieve data for group 1 and group 2 based on the CHD recurrence post CABG variable
    group1_data = group1[column].dropna()
    group2_data = group2[column].dropna()
    
    # Perform the Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(group1_data, group2_data)
    
    result_row = {
    'Variable': column,  # Name of the variable being tested
    'U Statistic': u_statistic,
    'P-value': p_value
}

# Convert the dictionary to a DataFrame
    result_df =             pd.DataFrame([result_row])

# Concatenate the result_df with the results_numfem DataFrame
    results_age1 = pd.concat([results_age1, result_df[result_df['P-value'] < 0.05]], ignore_index=True)
    
##num feats statistical tests in age2
##for non normal data 
numage2 = numfeat_age2[['AGE', 'CREAT', 'LVEF', 'CPB_TIME', 'BMI']]
                
results_age2 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numage2[cohort2['PAST_AF'] == 'Yes']
group2 = numage2[cohort2['PAST_AF'] == 'No']

# Iterate over each column in the numfem DataFrame
for column in numage2.columns:
    # Retrieve data for group 1 and group 2 based on the CHD recurrence post CABG variable
    group1_data = group1[column].dropna()
    group2_data = group2[column].dropna()
    
    # Perform the Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(group1_data, group2_data)
    
    result_row = {
    'Variable': column,  # Name of the variable being tested
    'U Statistic': u_statistic,
    'P-value': p_value
}

# Convert the dictionary to a DataFrame
    result_df =             pd.DataFrame([result_row])

# Concatenate the result_df with the results_numfem DataFrame
    results_age2 = pd.concat([results_age2, result_df[result_df['P-value'] < 0.05]], ignore_index=True)
    
##plotting age2 associations
data1 = numfeat_age2['LVEF'].dropna()
data2 = numfeat_age2['BMI'].dropna()
 
fig, axs = plt.subplots(2,1, figsize = (10,15))
sns.kdeplot(x = data1, hue = catfeat_age2['PAST_AF'].dropna(), ax = axs[0])
sns.kdeplot(x = data2, hue = catfeat_age2['PAST_AF'].dropna(),  ax = axs[1])

##num feats statistical tests in age3
##for non normal data 
numage3 = numfeat_age3[['AGE', 'CREAT', 'LVEF', 'CPB_TIME', 'BMI']]
                
results_age3 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numage3[cohort2['PAST_AF'] == 'Yes']
group2 = numage3[cohort2['PAST_AF'] == 'No']

# Iterate over each column in the numfem DataFrame
for column in numage3.columns:
    # Retrieve data for group 1 and group 2 based on the CHD recurrence post CABG variable
    group1_data = group1[column].dropna()
    group2_data = group2[column].dropna()
    
    # Perform the Mann-Whitney U test
    u_statistic, p_value = mannwhitneyu(group1_data, group2_data)
    
    result_row = {
    'Variable': column,  # Name of the variable being tested
    'U Statistic': u_statistic,
    'P-value': p_value
}

# Convert the dictionary to a DataFrame
    result_df =             pd.DataFrame([result_row])

# Concatenate the result_df with the results_numfem DataFrame
    results_age3 = pd.concat([results_age3, result_df[result_df['P-value'] < 0.05]], ignore_index=True)

sns.kdeplot(x = numage3['CREAT'], hue = catfeat_age3['PAST_AF'])

#calculating missing values
total = cohort2.isnull().sum()
percent = (total / cohort2.isnull().count())*100
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_data['Percent'] = missing_data['Percent'].round()
missing_data1 = missing_data[missing_data["Percent"] > 0]


##plotting associations
fig, axs = plt.subplots(1, 2, figsize = (15,5))
plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.5)

missing_values = sns.barplot(missing_data1, x = missing_data1.index, y = 'Percent', ax = axs [0], palette = 'spring', saturation = 0.2)
axs[0].set_title('Missing values')
axs[0].tick_params(axis = 'x', rotation = 90)
sns.boxplot(x = male['PAST_AF'], y = male['AGE'], ax = axs[1], palette = 'spring', saturation = 0.2)
axs[1].set_title('In Male population')
for count_plot in [missing_values]:
    for container in count_plot.containers:
        count_plot.bar_label(container, padding=2)

