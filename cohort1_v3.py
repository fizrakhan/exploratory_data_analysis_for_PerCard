# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:28:58 2024

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

file_name = "C://Users//HP//Documents//Masters//Research Work//Data//1248_postCABG_Outcome_CHD_recurrences.xlsx"
file = msoffcrypto.OfficeFile(open(file_name, "rb"))

file.load_key(password="Corino1248")  # Use password

decrypted = io.BytesIO()
file.decrypt(decrypted)

cohort1 = pd.read_excel(decrypted)

##pre-processing of the data
bins = [39.5, 49.5, 59.5, 69.5, 79.5]
labels = ['40-49', '50-59', '60-69', '70-79']

# Create the 'Age group' column
cohort1['Age group'] = pd.cut(cohort1['Age at CABG (years)'], bins=bins, labels=labels)

cohort1['Age group'].value_counts()

numfeat = cohort1[['Age at CABG (years)', 'Days from CABG to recurrence', 'Euroscore II', 'BMI', 'Creatinine (mg/dl)', 'Ejection fraction at admission %', 'Age group']]

catfeat = cohort1[['CHD recurrence post CABG', 'Hard (1) or soft (2) recurrence', 'Diabetes','Hypercholesterolemia', 'BMI category', 'Hypertension ( (>140/90 or on therapy)',
'Cigarette smoke (0=no, 1=ex, 2= current)', 'NYHA Class',
 'Precedent acute myocardial infarction', 'Extra-coronary atherosclerosis','Anti-platelets at admission', 'Beta-blockers at admission','RAAS inhibitors at admission', 'Statins at admission', 'Number of coronary vessels with significant disease',
 'Main left CA > 50% stenosis','History of previous coronary interventions', 'CABG Redo','Isolated CABG or combined intervention', 'Number of substituted valves', 'CABG + carotid endarterectomy',
 'Perioperatory myocardial infarction', 'n Saphenous vein bypass grafts',
 'n Arterial bypass grafts', 'Totale bypass',
 'Aortic valve substitution with biologic valve',
 'Aortic valve substitution with mechanic valve',
 'Aortic valve repairment',
 'Mitral valve substitution with mechanic valve',
 'Mitral valve substitution with biologic valve',
 'Mitral valve repairment', 'With (CABG) or without extracorporeal circulatory assistance (OPCABG)', 'Age group']]

##separate the data into male and female to make clear distinctions in target feature: CHD occurence

female = cohort1[cohort1['Sex'] == 'Female']
male = cohort1[cohort1['Sex'] == 'male']

numfeat_female = female[['Age at CABG (years)', 'Days from CABG to recurrence', 'Euroscore II', 'BMI', 'Creatinine (mg/dl)', 'Ejection fraction at admission %']]

numfeat_male = male[['Age at CABG (years)', 'Days from CABG to recurrence', 'Euroscore II', 'BMI', 'Creatinine (mg/dl)', 'Ejection fraction at admission %']]

catfeat_female = female[['CHD recurrence post CABG', 'Hard (1) or soft (2) recurrence', 'Diabetes','Hypercholesterolemia', 'BMI category', 'Hypertension ( (>140/90 or on therapy)',
'Cigarette smoke (0=no, 1=ex, 2= current)', 'NYHA Class',
 'Precedent acute myocardial infarction', 'Extra-coronary atherosclerosis','Anti-platelets at admission', 'Beta-blockers at admission','RAAS inhibitors at admission', 'Statins at admission', 'Number of coronary vessels with significant disease',
 'Main left CA > 50% stenosis','History of previous coronary interventions', 'CABG Redo','Isolated CABG or combined intervention', 'Number of substituted valves', 'CABG + carotid endarterectomy',
 'Perioperatory myocardial infarction', 'n Saphenous vein bypass grafts',
 'n Arterial bypass grafts', 'Totale bypass',
 'Aortic valve substitution with biologic valve',
 'Aortic valve substitution with mechanic valve',
 'Aortic valve repairment',
 'Mitral valve substitution with mechanic valve',
 'Mitral valve substitution with biologic valve',
 'Mitral valve repairment', 'With (CABG) or without extracorporeal circulatory assistance (OPCABG)']]

catfeat_male = male[['CHD recurrence post CABG', 'Hard (1) or soft (2) recurrence', 'Diabetes','Hypercholesterolemia', 'BMI category', 'Hypertension ( (>140/90 or on therapy)',
'Cigarette smoke (0=no, 1=ex, 2= current)', 'NYHA Class',
 'Precedent acute myocardial infarction', 'Extra-coronary atherosclerosis','Anti-platelets at admission', 'Beta-blockers at admission','RAAS inhibitors at admission', 'Statins at admission', 'Number of coronary vessels with significant disease',
 'Main left CA > 50% stenosis','History of previous coronary interventions', 'CABG Redo','Isolated CABG or combined intervention', 'Number of substituted valves', 'CABG + carotid endarterectomy',
 'Perioperatory myocardial infarction', 'n Saphenous vein bypass grafts',
 'n Arterial bypass grafts', 'Totale bypass',
 'Aortic valve substitution with biologic valve',
 'Aortic valve substitution with mechanic valve',
 'Aortic valve repairment',
 'Mitral valve substitution with mechanic valve',
 'Mitral valve substitution with biologic valve',
 'Mitral valve repairment', 'With (CABG) or without extracorporeal circulatory assistance (OPCABG)']]

numfeat_age1 = numfeat[numfeat['Age group'] == '40-49']
numfeat_age2 = numfeat[numfeat['Age group'] == '50-59']
numfeat_age3 = numfeat[numfeat['Age group'] == '60-69']
numfeat_age4 = numfeat[numfeat['Age group'] == '70-79']


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
    
##calculating basic parameters for age categories##
##40-49
basic_stats_age1 = numfeat_age1.describe()
numfeat_age1.drop(columns = ['Age group'], inplace = True)
basic_stats_age1.loc['variance'] = numfeat_age1.var(skipna=True)
for i, column in enumerate(numfeat_age1.columns):
    stat, p_value = shapiro(numfeat_age1[column].dropna())
    basic_stats_age1.loc['p-value', column] = p_value
    
##50-59
basic_stats_age2 = numfeat_age2.describe()
numfeat_age2.drop(columns = ['Age group'], inplace = True)
basic_stats_age2.loc['variance'] = numfeat_age2.var(skipna=True)
for i, column in enumerate(numfeat_age2.columns):
    stat, p_value = shapiro(numfeat_age2[column].dropna())
    basic_stats_age2.loc['p-value', column] = p_value
    
##60-69
basic_stats_age3 = numfeat_age3.describe()
numfeat_age3.drop(columns = ['Age group'], inplace = True)
basic_stats_age3.loc['variance'] = numfeat_age3.var(skipna=True)
for i, column in enumerate(numfeat_age3.columns):
    stat, p_value = shapiro(numfeat_age3[column].dropna())
    basic_stats_age3.loc['p-value', column] = p_value
    
##70-79
basic_stats_age4 = numfeat_age4.describe()
numfeat_age4.drop(columns = ['Age group'], inplace = True)
basic_stats_age4.loc['variance'] = numfeat_age4.var(skipna=True)
for i, column in enumerate(numfeat_age4.columns):
    stat, p_value = shapiro(numfeat_age4[column].dropna())
    basic_stats_age4.loc['p-value', column] = p_value
    
#making dist plots for gender##
##female
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Female\n")
for i, column in enumerate(numfeat_female.columns):
    plt.subplot(2, 3, i+1)
    for disease in female['CHD recurrence post CABG'].unique():
        sns.distplot(numfeat_female[column].loc[female['CHD recurrence post CABG'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()

##male
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Male\n")
for i, column in enumerate(numfeat_male.columns):
    plt.subplot(2, 3, i+1)
    for disease in male['CHD recurrence post CABG'].unique():
        sns.distplot(numfeat_male[column].loc[male['CHD recurrence post CABG'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()

##age groups##
##40-49

plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Age: 40-49\n")
for i, column in enumerate(numfeat_age1.columns):
    plt.subplot(2, 3, i+1)
    for disease in cohort1['CHD recurrence post CABG'].unique():
        sns.distplot(numfeat_age1[column].loc[cohort1['CHD recurrence post CABG'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()
    
##50-59

plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Age: 50-59\n")
for i, column in enumerate(numfeat_age2.columns):
    plt.subplot(2, 3, i+1)
    for disease in cohort1['CHD recurrence post CABG'].unique():
        sns.distplot(numfeat_age2[column].loc[cohort1['CHD recurrence post CABG'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()
    
##60-69
plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Age: 60-69\n")
for i, column in enumerate(numfeat_age3.columns):
    plt.subplot(2, 3, i+1)
    for disease in cohort1['CHD recurrence post CABG'].unique():
        sns.distplot(numfeat_age3[column].loc[cohort1['CHD recurrence post CABG'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()
    
##70-79

plt.figure(figsize=(20, 10))
plt.subplots_adjust(wspace=0.5)  
plt.title("Distribution plots in Age: 70-79\n")
for i, column in enumerate(numfeat_age3.columns):
    plt.subplot(2, 3, i+1)
    for disease in cohort1['CHD recurrence post CABG'].unique():
        sns.distplot(numfeat_age3[column].loc[cohort1['CHD recurrence post CABG'] == disease].dropna(), label=disease, kde_kws={'bw_adjust':1})
    plt.legend()


##making boxplots for gender##
#in female
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)  
plt.title('Female population')
for i, ax in enumerate(axs.flat):
    data = numfeat_female.iloc[:, i].dropna()
    sns.boxplot(x=female['CHD recurrence post CABG'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_female.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)

#in male
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5) 
plt.title('Male population')
for i, ax in enumerate(axs.flat):
    data = numfeat_male.iloc[:, i].dropna()
    sns.boxplot(x=male['CHD recurrence post CABG'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_male.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)
    
##box plots in age group##
#40-49
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
plt.title('Age group: 40-49')
for i, ax in enumerate(axs.flat):
    data = numfeat_age1.iloc[:, i].dropna()
    sns.boxplot(x=cohort1['CHD recurrence post CABG'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age1.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)

#50-59
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
plt.title('Age group: 50-59')
for i, ax in enumerate(axs.flat):
    data = numfeat_age2.iloc[:, i].dropna()
    sns.boxplot(x=cohort1['CHD recurrence post CABG'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age2.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)
    
#60-69
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
plt.title('Age group: 60-69')
for i, ax in enumerate(axs.flat):
    data = numfeat_age3.iloc[:, i].dropna()
    sns.boxplot(x=cohort1['CHD recurrence post CABG'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age3.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)

#70-79
fig, axs = plt.subplots(2, 3, figsize = (20,10))
plt.subplots_adjust(wspace=0.5) 
plt.subplots_adjust(hspace=0.5)
plt.title('Age group: 70-79')
for i, ax in enumerate(axs.flat):
    data = numfeat_age4.iloc[:, i].dropna()
    sns.boxplot(x=cohort1['CHD recurrence post CABG'], y=data, ax=ax, palette='spring', saturation=0.2)
    ax.set_title(numfeat_age4.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis = 'y', labelsize=14)
    sns.set(font_scale = 1.2)
    
##numerical correlation in gender##
#in female
numfeat_female_corr = numfeat_female.corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_female_corr, annot = True)
sns.set(font_scale = 2)

##in male
numfeat_male_corr = numfeat_male.corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_male_corr, annot = True)
sns.set(font_scale = 2)

##numerical correlation in age group##

#40-49
numfeat_age1_corr = numfeat_age1.corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_age1_corr, annot = True)
sns.set(font_scale = 2)

#50-59
numfeat_age2_corr = numfeat_age2.corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_age2_corr, annot = True)
sns.set(font_scale = 2)

#60-69
numfeat_age3_corr = numfeat_age3.corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_age3_corr, annot = True)
sns.set(font_scale = 2)

#70-79
numfeat_age4_corr = numfeat_age4.corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(numfeat_age4_corr, annot = True)
sns.set(font_scale = 2)

##frequency plots in gender##

#in female
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_female.columns):
    plt.subplot(7,6, i+1)
    count_female_catfeat = sns.countplot(x=catfeat_female[column].dropna(), hue = female['CHD recurrence post CABG'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_female_catfeat.containers:
        count_female_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
##in male
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_male.columns):
    plt.subplot(7,6, i+1)
    count_male_catfeat = sns.countplot(x=catfeat_male[column].dropna(), hue = male['CHD recurrence post CABG'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_male_catfeat.containers:
        count_male_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##frequency plots in age group##

#40-49
catfeat_age1 = catfeat[catfeat['Age group'] == '40-49']
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age1.columns):
    plt.subplot(7,6, i+1)
    count_age1_catfeat = sns.countplot(x=catfeat_age1[column].dropna(), hue = cohort1['CHD recurrence post CABG'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age1_catfeat.containers:
        count_age1_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
#50-59
catfeat_age2 = catfeat[catfeat['Age group'] == '50-59']
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age2.columns):
    plt.subplot(7,6, i+1)
    count_age2_catfeat = sns.countplot(x=catfeat_age2[column].dropna(), hue = cohort1['CHD recurrence post CABG'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age2_catfeat.containers:
        count_age2_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)
    
#60-69
catfeat_age3 = catfeat[catfeat['Age group'] == '60-69']
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age3.columns):
    plt.subplot(7,6, i+1)
    count_age3_catfeat = sns.countplot(x=catfeat_age3[column].dropna(), hue = cohort1['CHD recurrence post CABG'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age3_catfeat.containers:
        count_age3_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

#70-79
catfeat_age4 = catfeat[catfeat['Age group'] == '70-79']
plt.figure(figsize=(45, 50))
for i, column in enumerate(catfeat_age4.columns):
    plt.subplot(7,6, i+1)
    count_age4_catfeat = sns.countplot(x=catfeat_age4[column].dropna(), hue = cohort1['CHD recurrence post CABG'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age4_catfeat.containers:
        count_age4_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##statistical tests on categorical variable - chi2 test

##in female
# Create contingency tables for each categorical variable
contingency_tables = {}
for col in catfeat_female.columns:
    if col != 'CHD recurrence post CABG':
        contingency_table = pd.crosstab(catfeat_female['CHD recurrence post CABG'], catfeat_female[col])
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
    if col != 'CHD recurrence post CABG':
        contingency_table = pd.crosstab(catfeat_male['CHD recurrence post CABG'], catfeat_male[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_male = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_male = pd.concat([results_male, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_male = results_male.drop(results_male[results_male['P-value'] >= 0.05].index)

##in age group

##40-49
contingency_tables = {}
for col in catfeat_age1.columns:
    if col != 'CHD recurrence post CABG':
        contingency_table = pd.crosstab(catfeat_age1['CHD recurrence post CABG'], catfeat_age1[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age1 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age1 = pd.concat([results_age1, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age1 = results_age1.drop(results_age1[results_age1['P-value'] >= 0.05].index)

##50-59
contingency_tables = {}
for col in catfeat_age2.columns:
    if col != 'CHD recurrence post CABG':
        contingency_table = pd.crosstab(catfeat_age2['CHD recurrence post CABG'], catfeat_age2[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age2 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age2 = pd.concat([results_age2, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age2 = results_age2.drop(results_age2[results_age2['P-value'] >= 0.05].index)

##60-69
contingency_tables = {}
for col in catfeat_age3.columns:
    if col != 'CHD recurrence post CABG':
        contingency_table = pd.crosstab(catfeat_age3['CHD recurrence post CABG'], catfeat_age3[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age3 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age3 = pd.concat([results_age3, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age3 = results_age3.drop(results_age3[results_age3['P-value'] >= 0.05].index)

##70-79
contingency_tables = {}
for col in catfeat_age4.columns:
    if col != 'CHD recurrence post CABG':
        contingency_table = pd.crosstab(catfeat_age4['CHD recurrence post CABG'], catfeat_age4[col])
        contingency_tables[col] = contingency_table

# Perform chi-squared test for each variable and store results in a DataFrame
results_age4 = pd.DataFrame(columns=['Variable', 'Chi2', 'P-value'])
for col, contingency_table in contingency_tables.items():
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results_age4 = pd.concat([results_age4, pd.DataFrame({'Variable': [col], 'Chi2': [chi2], 'P-value': [p]})], ignore_index=True)

results_age4 = results_age4.drop(results_age4[results_age4['P-value'] >= 0.05].index)


##plotting final results only
##no significant difference in female in any variables

final_male_cat = male[['History of previous coronary interventions', 'Number of substituted valves', 'Totale bypass', 'CHD recurrence post CABG']]

plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.5)  
for i, column in enumerate(final_male_cat.columns):
    plt.subplot(2,2, i+1)
    count_male_catfeat = sns.countplot(x=final_male_cat[column].dropna(), hue = catfeat_male['CHD recurrence post CABG'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_male_catfeat.containers:
        count_male_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)


##no significant difference other than 60-69 years old
final_age2_cat = catfeat_age2[['NYHA Class', 'Main left CA > 50% stenosis', 'History of previous coronary interventions']]

#60-69
plt.figure(figsize=(20, 5))
plt.subplots_adjust(hspace=0.5)  
for i, column in enumerate(final_age2_cat.columns):
    plt.subplot(1,3, i+1)
    count_age2_catfeat = sns.countplot(x=final_age2_cat[column].dropna(), hue = catfeat_age2['CHD recurrence post CABG'], palette = 'spring', saturation = 0.2)
    plt.xticks(rotation=45)
    for container in count_age2_catfeat.containers:
        count_age2_catfeat.bar_label(container, padding=2)
    sns.set(font_scale=1.5)

##num statistical test in female

##for non-normal data
numfem = female[['Age at CABG (years)', 'Euroscore II', 'BMI', 'Creatinine (mg/dl)', 'Ejection fraction at admission %']]

results_numfem = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = female[female['CHD recurrence post CABG'] == 1]

group2 = female[female['CHD recurrence post CABG'] == 0]

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


####num statistical test in male
nummale = male[['Age at CABG (years)', 'Euroscore II', 'BMI', 'Creatinine (mg/dl)', 'Ejection fraction at admission %']]

##for non-normal data 
results_nummale = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = male[male['CHD recurrence post CABG'] == 1]

group2 = male[male['CHD recurrence post CABG'] == 0]

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




sns.kdeplot(x = male['Euroscore II'], hue = male['CHD recurrence post CABG'])
plt.xlim(0, 15)
plt.title('Distribution in Male population')

##in age group

#in age group1

numage1 = numfeat_age1[['Age at CABG (years)', 'Euroscore II', 'Creatinine (mg/dl)', 'Ejection fraction at admission %']]

results_numage1 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numfeat_age1[cohort1['CHD recurrence post CABG'] == 1]

group2 = numfeat_age1[cohort1['CHD recurrence post CABG'] == 0]

##for non normal data
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

    results_numage1 = pd.concat([results_numage1, result_df[result_df['P-value'] < 0.05]], ignore_index=True)
    
## for normal data
t_statistic, p_value = ttest_ind(group1['BMI'], group2['BMI'].dropna()) ## result: no association was found


 
#in age group2
##for non normal data
numage2 = numfeat_age2[['Age at CABG (years)', 'Euroscore II', 'BMI', 'Creatinine (mg/dl)', 'Ejection fraction at admission %']]

results_numage2 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numfeat_age2[cohort1['CHD recurrence post CABG'] == 1]

group2 = numfeat_age2[cohort1['CHD recurrence post CABG'] == 0]

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


#in age group3
##for non normal data 
numage3 = numfeat_age3[['Age at CABG (years)', 'Euroscore II', 'BMI', 'Creatinine (mg/dl)', 'Ejection fraction at admission %']]

results_numage3 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numfeat_age3[cohort1['CHD recurrence post CABG'] == 1]

group2 = numfeat_age3[cohort1['CHD recurrence post CABG'] == 0]

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
    
#in age group4
##for non normal data 
numage4 = numfeat_age4[['Age at CABG (years)', 'Euroscore II', 'BMI', 'Creatinine (mg/dl)', 'Ejection fraction at admission %']]

results_numage4 = pd.DataFrame(columns=['Variable', 'U Statistic', 'P-value'])

group1 = numfeat_age4[cohort1['CHD recurrence post CABG'] == 1]

group2 = numfeat_age4[cohort1['CHD recurrence post CABG'] == 0]

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
    result_df =             pd.DataFrame([result_row])

# Concatenate the result_df with the results_numfem DataFrame
    results_numage4 = pd.concat([results_numage4, result_df[result_df['P-value'] < 0.05]], ignore_index=True)
    
##plotting age4 associations
sns.kdeplot(x = numage4['Age at CABG (years)'], hue = catfeat_age4['CHD recurrence post CABG'])
plt.title('Distribution in 70-79 years old')



#calculating missing values
total = cohort1.isnull().sum()
percent = (total / cohort1.isnull().count())*100
missing_data = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
missing_data['Percent'] = missing_data['Percent'].round()
missing_data1 = missing_data[missing_data["Percent"] > 0]



##plotting association for male + age group 2 + age group 3

from matplotlib import rcParams

# Set global font size
rcParams.update({'font.size': 30})  # Adjust the font size as needed

fig, axs = plt.subplots(1, 5, figsize = (60,10))
plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.5)

missing_values = sns.barplot(missing_data1, x = missing_data1.index, y = 'Percent', ax = axs [0], palette = 'spring', saturation = 0.2)
axs[0].set_title('Missing values\n')
axs[0].tick_params(axis = 'x', rotation = 90)
count_male_totale_bypass = sns.countplot(x = male['Totale bypass'], hue = male['CHD recurrence post CABG'], ax = axs [1], palette = 'spring', saturation = 0.2)
axs[1].set_title('In Male population\n')
count_age2_nyha_class = sns.countplot(x = catfeat_age2['NYHA Class'], hue = catfeat_age2['CHD recurrence post CABG'], ax = axs [2], palette = 'spring', saturation = 0.2)
axs[2].set_title('In Age group 2\n')
count_age3_totale_bypass = sns.countplot(x = catfeat_age3['Totale bypass'], hue = catfeat_age3['CHD recurrence post CABG'], ax = axs [3], palette = 'spring', saturation = 0.2)
axs[3].set_title('In Age group 3\n')
for count_plot in [missing_values, count_male_totale_bypass, count_age2_nyha_class, count_age3_totale_bypass]:
    for container in count_plot.containers:
        count_plot.bar_label(container, padding=2)
box_male_age = sns.boxplot(x = male['CHD recurrence post CABG'], y = male['Age at CABG (years)'], ax = axs[4], palette = 'spring', saturation = 0.2)
axs[4].set_title('In Male population\n')
