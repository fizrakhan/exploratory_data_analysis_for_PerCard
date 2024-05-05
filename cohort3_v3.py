# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:33:43 2024

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

file_name = "C://Users//HP//Documents//Masters//Research Work//Data//2445 postIMA. Outcome FA. Dati completamente anonimizzati_CohortB_ENG.xlsx"
file = msoffcrypto.OfficeFile(open(file_name, "rb"))

file.load_key(password="Corino2445")  # Use password

decrypted = io.BytesIO()
file.decrypt(decrypted)

cohort3 = pd.read_excel(decrypted)

##creating age group column
bins = [0, 49, 65, 79, float('inf')]  # Define the bin edges for age categories
labels = ['<50', '50-64', '65-79', '>80']  # Define labels for each age category

# Create the 'Age group' column
cohort3['Age group'] = pd.cut(cohort3['Age'], bins=bins, labels=labels, right=False)


##separating the whole dataset into numerical and categorical variables
numfeat = cohort3[['Age', 'Weight', 'Height', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']]

##exclusing the columns defined in numfeats 
numfeat_columns = numfeat.columns
included_columns = cohort3.columns.difference(numfeat_columns)
catfeat = cohort3[included_columns]
catfeat.shape[1]
catfeat.drop(columns = ['Patient ID', 'BMI', 'Other comorbidities', 'Location of AMI'], inplace=True)
catfeat.columns

##separate the data into male and female to make clear distinctions in target feature: Atrial Fibrillation

female = cohort3[cohort3['Gender'] == 'F']
male = cohort3[cohort3['Gender'] == 'M']

numfeat_female = female[['Age', 'Weight', 'Height', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']]

numfeat_male = male[['Age', 'Weight', 'Height', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']]

catfeat_female = catfeat[catfeat['Gender'] == 'F']

catfeat_male = catfeat[catfeat['Gender'] == 'M']

##separate the data into age groups to make clear distinctions in target feature: Atrial Fibrillation
numfeat_age1 = numfeat[(cohort3['Age group'] == '<50')]
numfeat_age2 = numfeat[(cohort3['Age group'] == '50-64')]
numfeat_age3 = numfeat[(cohort3['Age group'] == '65-79')]
numfeat_age4 = numfeat[(cohort3['Age group'] == '>80')]

catfeat_age1 = catfeat[(catfeat['Age group'] == '<50')]
catfeat_age2 = catfeat[(catfeat['Age group'] == '50-64')]
catfeat_age3 = catfeat[(catfeat['Age group'] == '65-79')]
catfeat_age4 = catfeat[(catfeat['Age group'] == '>80')]



####calculating basic parameters in female##
basic_stats_female = numfeat_female.describe()
basic_stats_female.loc['variance'] = numfeat_female.var(skipna=True)
# checking for normality
for i, column in enumerate(numfeat_female.columns):
    stat, p_value = shapiro(numfeat_female[column].dropna())
    basic_stats_female.loc['p-value', column] = p_value

plt.figure(figsize=(30, 10))
sns.heatmap(basic_stats_female, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
sns.set(font_scale=1.5)
plt.title("Summary Statistics Male")
plt.xlabel("Statistics")
plt.ylabel("Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

##calculating basic parameters in male##
basic_stats_male = numfeat_male.describe()
basic_stats_male.loc['variance'] = numfeat_male.var(skipna=True)
# checking for normality
for i, column in enumerate(numfeat_male.columns):
    stat, p_value = shapiro(numfeat_male[column].dropna())
    basic_stats_male.loc['p-value', column] = p_value
    

plt.figure(figsize=(30, 10))
sns.heatmap(basic_stats_male, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
sns.set(font_scale=1.5)
plt.title("Summary Statistics Male")
plt.xlabel("Statistics")
plt.ylabel("Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

####calculating basic parameters in age category##
##<50
basic_stats_age1 = numfeat_age1.describe()
basic_stats_age1.loc['variance'] = numfeat_age1.var(skipna=True)
for i, column in enumerate(numfeat_age1.columns):
    stat, p_value = shapiro(numfeat_age1[column].dropna())
    basic_stats_age1.loc['p-value', column] = p_value
    
plt.figure(figsize=(30, 10))
sns.heatmap(basic_stats_age1, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
sns.set(font_scale=1.5)
plt.title("Summary Statistics Age < 50")
plt.xlabel("Statistics")
plt.ylabel("Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

##50-64
basic_stats_age2 = numfeat_age2.describe()
basic_stats_age2.loc['variance'] = numfeat_age2.var(skipna=True)
for i, column in enumerate(numfeat_age2.columns):
    stat, p_value = shapiro(numfeat_age2[column].dropna())
    basic_stats_age2.loc['p-value', column] = p_value
    
plt.figure(figsize=(30, 10))
sns.heatmap(basic_stats_age2, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
sns.set(font_scale=1.5)
plt.title("Summary Statistics Age = 50 - 64")
plt.xlabel("Statistics")
plt.ylabel("Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
    
##65-79
basic_stats_age3 = numfeat_age3.describe()
basic_stats_age3.loc['variance'] = numfeat_age3.var(skipna=True)
for i, column in enumerate(numfeat_age3.columns):
    stat, p_value = shapiro(numfeat_age3[column].dropna())
    basic_stats_age3.loc['p-value', column] = p_value
    
plt.figure(figsize=(30, 10))
sns.heatmap(basic_stats_age2, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
sns.set(font_scale=1.5)
plt.title("Summary Statistics Age = 65 - 79")
plt.xlabel("Statistics")
plt.ylabel("Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

##>80

basic_stats_age4 = numfeat_age4.describe()
basic_stats_age4.loc['variance'] = numfeat_age4.var(skipna=True)
for i, column in enumerate(numfeat_age4.columns):
    stat, p_value = shapiro(numfeat_age4[column].dropna())
    basic_stats_age4.loc['p-value', column] = p_value
    
plt.figure(figsize=(30, 10))
sns.heatmap(basic_stats_age2, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
sns.set(font_scale=1.5)
plt.title("Summary Statistics Age > 80")
plt.xlabel("Statistics")
plt.ylabel("Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

#making dist plots for gender##
##female
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Female\n")
for i, column in enumerate(numfeat_female.columns):
    plt.subplot(2, 3, i+1)
    for disease in female['New onset atrial fibrilation (history of paroxistic AF allowed)'].unique():
        sns.distplot(numfeat_female[column].loc[female['New onset atrial fibrilation (history of paroxistic AF allowed)'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()

##male
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Male\n")
for i, column in enumerate(numfeat_male.columns):
    plt.subplot(2, 3, i+1)
    for disease in male['New onset atrial fibrilation (history of paroxistic AF allowed)'].unique():
        sns.distplot(numfeat_male[column].loc[male['New onset atrial fibrilation (history of paroxistic AF allowed)'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()
    
##box plots in age group##
#<50
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
plt.title('Age group: 40-49')
for i, ax in enumerate(axs.flat):
    data = numfeat_age1.iloc[:, i].dropna()
    sns.boxplot(x=cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age1.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)

#50 - 64
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
plt.title('Age group: 50-59')
for i, ax in enumerate(axs.flat):
    data = numfeat_age2.iloc[:, i].dropna()
    sns.boxplot(x=cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age2.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)
    
#65-79
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
plt.title('Age group: 60-69')
for i, ax in enumerate(axs.flat):
    data = numfeat_age3.iloc[:, i].dropna()
    sns.boxplot(x=cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age3.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)

#>=80
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
plt.title('Age group: 70-79')
for i, ax in enumerate(axs.flat):
    data = numfeat_age4.iloc[:, i].dropna()
    sns.boxplot(x=cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age4.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)
    
##numerical correlation in age group##
#<50
numfeat_age1_corr = numfeat_age1.dropna().corr().round(2)
plt.figure(figsize=(25,15))
sns.heatmap(numfeat_age1_corr, annot = True)
sns.set(font_scale = 2)

#50-64
numfeat_age2_corr = numfeat_age2.dropna().corr().round(2)
plt.figure(figsize=(25,15))
sns.heatmap(numfeat_age2_corr, annot = True)
sns.set(font_scale = 2)

#65-79
numfeat_age3_corr = numfeat_age3.dropna().corr().round(2)
plt.figure(figsize=(25,15))
sns.heatmap(numfeat_age3_corr, annot = True)
sns.set(font_scale = 2)

#>80
numfeat_age4_corr = numfeat_age4.dropna().corr().round(2)
plt.figure(figsize=(25,15))
sns.heatmap(numfeat_age4_corr, annot = True)
sns.set(font_scale = 2)
    
##frequency plots in gender##

#in female
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_female.columns):
    plt.subplot(7,6, i+1)
    count_female_catfeat = sns.countplot(x=catfeat_female[column].dropna(), hue = female['New onset atrial fibrilation (history of paroxistic AF allowed)'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_female_catfeat.containers:
        count_female_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##in male
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_male.columns):
    plt.subplot(7,6, i+1)
    count_male_catfeat = sns.countplot(x=catfeat_male[column].dropna(), hue = male['New onset atrial fibrilation (history of paroxistic AF allowed)'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_male_catfeat.containers:
        count_male_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##in age groups

##<50
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age1.columns):
    plt.subplot(7,6, i+1)
    count_age1_catfeat = sns.countplot(x=catfeat_age1[column].dropna(), hue = catfeat_age1['New onset atrial fibrilation (history of paroxistic AF allowed)'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age1_catfeat.containers:
        count_age1_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##50-64
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age2.columns):
    plt.subplot(7,6, i+1)
    count_age2_catfeat = sns.countplot(x=catfeat_age2[column].dropna(), hue = catfeat_age2['New onset atrial fibrilation (history of paroxistic AF allowed)'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age2_catfeat.containers:
        count_age2_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##65-79
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age3.columns):
    plt.subplot(7,6, i+1)
    count_age3_catfeat = sns.countplot(x=catfeat_age3[column].dropna(), hue = catfeat_age3['New onset atrial fibrilation (history of paroxistic AF allowed)'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age3_catfeat.containers:
        count_age3_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##>80
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age4.columns):
    plt.subplot(7,6, i+1)
    count_age4_catfeat = sns.countplot(x=catfeat_age4[column].dropna(), hue = catfeat_age4['New onset atrial fibrilation (history of paroxistic AF allowed)'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age4_catfeat.containers:
        count_age4_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##statistical tests on categorical variable - chi2 test

##in female
# Create contingency tables for each categorical variable
contingency_tables = {}
for col in catfeat_female.columns:
    if col != 'New onset atrial fibrilation (history of paroxistic AF allowed)':
        contingency_table = pd.crosstab(catfeat_female['New onset atrial fibrilation (history of paroxistic AF allowed)'], catfeat_female[col])
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
    if col != 'New onset atrial fibrilation (history of paroxistic AF allowed)':
        contingency_table = pd.crosstab(catfeat_male['New onset atrial fibrilation (history of paroxistic AF allowed)'], catfeat_male[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_male = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_male = pd.concat([results_male, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_male = results_male.drop(results_male[results_male['P-value'] >= 0.05].index)

##in age group

#<50
# Create contingency tables for each categorical variable
contingency_tables = {}
for col in catfeat_age1.columns:
    if col != 'New onset atrial fibrilation (history of paroxistic AF allowed)':
        contingency_table = pd.crosstab(catfeat_age1['New onset atrial fibrilation (history of paroxistic AF allowed)'], catfeat_age1[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age1 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age1 = pd.concat([results_age1, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age1 = results_age1.drop(results_age1[results_age1['P-value'] >= 0.05].index)

#<50-64
# Create contingency tables for each categorical variable
contingency_tables = {}
for col in catfeat_age2.columns:
    if col != 'New onset atrial fibrilation (history of paroxistic AF allowed)':
        contingency_table = pd.crosstab(catfeat_age2['New onset atrial fibrilation (history of paroxistic AF allowed)'], catfeat_age2[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age2 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age2 = pd.concat([results_age2, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age2 = results_age2.drop(results_age2[results_age2['P-value'] >= 0.05].index)

#<65-79
# Create contingency tables for each categorical variable
contingency_tables = {}
for col in catfeat_age3.columns:
    if col != 'New onset atrial fibrilation (history of paroxistic AF allowed)':
        contingency_table = pd.crosstab(catfeat_age3['New onset atrial fibrilation (history of paroxistic AF allowed)'], catfeat_age3[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age3 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age3 = pd.concat([results_age3, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age3 = results_age3.drop(results_age3[results_age3['P-value'] >= 0.05].index)

#>80
# Create contingency tables for each categorical variable
contingency_tables = {}
for col in catfeat_age4.columns:
    if col != 'New onset atrial fibrilation (history of paroxistic AF allowed)':
        contingency_table = pd.crosstab(catfeat_age4['New onset atrial fibrilation (history of paroxistic AF allowed)'], catfeat_age4[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age4 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age4 = pd.concat([results_age4, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age4 = results_age4.drop(results_age4[results_age4['P-value'] >= 0.05].index)

##plotting associated features 

#in females
final_catfeat_female = female[['AKI stage', 'Acute kidney injury', 'Acute pulmonary edema', 'Age group', 'Anticoagulants at admission', 'Antidiabetics at admission', 'CVVH/HD', 'FV/TV', 'Hyperglicemia (history of)', 'Hypertension (history of)', 'RMN performed', 'Tienopiridins at discharge from ACU', 'VAM', 'coronary angiography', 'shock(IABP/ECMO)', 'trasfusions']]

plt.figure(figsize=(40, 30))
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)
for i, column in enumerate(final_catfeat_female.columns):
    plt.subplot(5,4, i+1)
    count_female_catfeat = sns.countplot(x=final_catfeat_female[column].dropna(), hue = catfeat_female['New onset atrial fibrilation (history of paroxistic AF allowed)'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_female_catfeat.containers:
        count_female_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)


##in age group3 

final_catfeat_age3 = catfeat_age3[['AKI stage', 'Acute kidney injury', 'Acute pulmonary edema', 'Anticoagulants at admission', 'CVVH/HD','FV/TV','Therapy of ACS', 'VAM', 'coronary angiography', 'shock(IABP/ECMO)', 'trasfusions']]

plt.figure(figsize=(40, 30))
plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)
for i, column in enumerate(final_catfeat_age3.columns):
    plt.subplot(4,3, i+1)
    count_age3_catfeat = sns.countplot(x=final_catfeat_age3[column].dropna(), hue = catfeat_age3['New onset atrial fibrilation (history of paroxistic AF allowed)'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age3_catfeat.containers:
        count_age3_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##num feats statistical tests in female
##for non normal data 
numfem = female[['Age', 'Weight', 'Height', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']]
                
results_numfem = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = female[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 0]
group2 = female[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 1]

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
    
##num feats statistical tests in male
##for non normal data 
nummale = male[['Age', 'Weight', 'Height', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']]
                
results_nummale = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = male[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 0]
group2 = male[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 1]

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
    
##num feats statistical tests in age1
##for non normal data 
numage1 = numfeat_age1[['Age', 'Weight', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']] ##height var was removed bec it was normal
                
results_numage1 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numfeat_age1[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 0]
group2 = numfeat_age1[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 1]

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
    results_numage1 = pd.concat([results_numage1, result_df[result_df['P-value'] < 0.05]], ignore_index=True)
    
##num feats statistical tests in age2
##for non normal data 
numage2 = numfeat_age2[['Age', 'Weight', 'Height', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']]
                
results_numage2 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numfeat_age2[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 0]
group2 = numfeat_age2[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 1]

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
    results_numage2 = pd.concat([results_numage2, result_df[result_df['P-value'] < 0.05]], ignore_index=True)
    

##num feats statistical tests in age3
##for non normal data 
numage3 = numfeat_age3[['Age', 'Weight', 'Height', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']]
                
results_numage3 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numfeat_age3[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 0]
group2 = numfeat_age3[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 1]

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
    results_numage3 = pd.concat([results_numage3, result_df[result_df['P-value'] < 0.05]], ignore_index=True)
    
##num feats statistical tests in age4
##for non normal data 
numage4 = numfeat_age4[['Age', 'Weight', 'Height', 'Time-to-Pres(h)', 'Days in CCU', 'Total Chol', 'LDL-C', 'HDL-C', 'Triglycerides', 'Ejection fraction (%)','Troponin at admission', 'Troponin peak', 'Glycemia ad admission', 'Glycemia at fasting', 'Gliceted hb at admission', 'Creatinine at admission', 'Max creatinin in ACU', 'eGFR', 'delta crea', 'Haemoblogin', 'HsCRP']]
                
results_numage4 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numfeat_age4[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 0]
group2 = numfeat_age4[cohort3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == 1]

# Iterate over each column in the numfem DataFrame
for column in numage4.columns:
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
    result_df = pd.DataFrame([result_row])

# Concatenate the result_df with the results_numfem DataFrame
    results_numage4 = pd.concat([results_numage4, result_df[result_df['P-value'] < 0.05]], ignore_index=True)



##plotting associations in female
data = female[results_numfem['Variable']].dropna()
data.drop(labels=['Days in CCU'], axis=1, inplace=True)

plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5) 
plt.title("Distribution plots in Female\n")
for i, column in enumerate(data.columns):
    plt.subplot(3, 5, i+1)
    for disease in female['New onset atrial fibrilation (history of paroxistic AF allowed)'].unique():
        sns.kdeplot(data[column].loc[female['New onset atrial fibrilation (history of paroxistic AF allowed)'] == disease].dropna(), label=disease)
    plt.legend()
    
##plotting associations in male
data = male[results_nummale['Variable']].dropna()
data.drop(labels=['Days in CCU'], axis=1, inplace=True)

plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5) 
plt.title("Distribution plots in Male\n")
for i, column in enumerate(data.columns):
    plt.subplot(3, 5, i+1)
    for disease in male['New onset atrial fibrilation (history of paroxistic AF allowed)'].unique():
        sns.kdeplot(data[column].loc[male['New onset atrial fibrilation (history of paroxistic AF allowed)'] == disease].dropna(), label=disease)
    plt.legend()

##plotting associations in age1
data = numfeat_age1[results_numage1['Variable']].dropna()
data.drop(labels=['Days in CCU'], axis=1, inplace=True)

plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5) 
plt.title("Distribution plots in Age < 50 years\n")
for i, column in enumerate(data.columns):
    plt.subplot(3, 5, i+1)
    for disease in catfeat_age1['New onset atrial fibrilation (history of paroxistic AF allowed)'].unique():
        sns.kdeplot(data[column].loc[catfeat_age1['New onset atrial fibrilation (history of paroxistic AF allowed)'] == disease].dropna(), label=disease)
    plt.legend()
    
##plotting associations in age2
data = numfeat_age2[results_numage2['Variable']].dropna()
data.drop(labels=['Days in CCU'], axis=1, inplace=True)

plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5) 
plt.title("Distribution plots in Age < 50 years\n")
for i, column in enumerate(data.columns):
    plt.subplot(3, 5, i+1)
    for disease in catfeat_age2['New onset atrial fibrilation (history of paroxistic AF allowed)'].unique():
        sns.kdeplot(data[column].loc[catfeat_age2['New onset atrial fibrilation (history of paroxistic AF allowed)'] == disease].dropna(), label=disease)
    plt.legend()
    
##plotting associations in age3
data = numfeat_age3[results_numage3['Variable']].dropna()
data.drop(labels=['Days in CCU'], axis=1, inplace=True)

plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5) 
plt.title("Distribution plots in Age = 65-79 years\n")
for i, column in enumerate(data.columns):
    plt.subplot(2, 3, i+1)
    for disease in catfeat_age3['New onset atrial fibrilation (history of paroxistic AF allowed)'].unique():
        sns.kdeplot(data[column].loc[catfeat_age3['New onset atrial fibrilation (history of paroxistic AF allowed)'] == disease].dropna(), label=disease)
    plt.legend()
    
##plotting associations in age4
data = numfeat_age4[results_numage4['Variable']].dropna()
data.drop(labels=['Days in CCU'], axis=1, inplace=True)

plt.figure(figsize=(15, 10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5) 
plt.title("Distribution plots in Age > 80 years\n")
for i, column in enumerate(data.columns):
    plt.subplot(3, 3, i+1)
    for disease in catfeat_age4['New onset atrial fibrilation (history of paroxistic AF allowed)'].unique():
        sns.kdeplot(data[column].loc[catfeat_age4['New onset atrial fibrilation (history of paroxistic AF allowed)'] == disease].dropna(), label=disease)
    plt.legend()
    
    
#calculating missing values
total = cohort3.isnull().sum()
percent = (total / cohort3.isnull().count())*100
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_data['Percent'] = missing_data['Percent'].round()
missing_data1 = missing_data[missing_data["Percent"] > 0]


##plotting all associations together
fig, axs = plt.subplots(2, 3, figsize = (45,25))
plt.subplots_adjust(hspace = 0.75)
plt.subplots_adjust(wspace = 0.5)

missing_values = sns.barplot(missing_data1, x = missing_data1.index, y = 'Percent', ax = axs[0,0], palette = 'spring', saturation = 0.2)
axs[0,0].set_title('Missing values')
axs[0,0].tick_params(axis = 'x', rotation = 90)
fem_age = sns.boxplot(x = female['New onset atrial fibrilation (history of paroxistic AF allowed)'], y = female['Age'], ax = axs[0,1], palette = 'spring', saturation = 0.2)
axs[0,1].set_title('In Female population')
male_age = sns.boxplot(x = male['New onset atrial fibrilation (history of paroxistic AF allowed)'], y = male['Age'], ax = axs[0,2], palette = 'spring', saturation = 0.2)
axs[0,2].set_title('In Male population')
male_hyperten = sns.countplot(x = male['Hypertension (history of)'], hue = male['New onset atrial fibrilation (history of paroxistic AF allowed)'], ax = axs[1,0], palette = 'spring', saturation = 0.2)
axs[1,0].set_title('In Male population')
age1_hyperten = sns.countplot(x = catfeat_age1['Hypertension (history of)'], hue = catfeat_age1['New onset atrial fibrilation (history of paroxistic AF allowed)'], ax = axs[1,1], palette = 'spring', saturation = 0.2)
axs[1,1].set_title('In Age group 1')
age4_hyperten = sns.countplot(x = catfeat_age4['Hypertension (history of)'], hue = catfeat_age4['New onset atrial fibrilation (history of paroxistic AF allowed)'], ax = axs[1,2], palette = 'spring', saturation = 0.2)
axs[1,2].set_title('In Age group 4')
for count_plot in [missing_values, male_hyperten, age1_hyperten, age4_hyperten]:
    for container in count_plot.containers:
        count_plot.bar_label(container, padding=2)
