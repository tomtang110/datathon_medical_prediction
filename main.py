import pandas as pd
from scipy.stats import norm,skew
import numpy as np
import json
import seaborn as sns
import re
from copy import deepcopy
from scipy.special import boxcox1p
import math
age_dict_10 = {'sixty':60, 'eighty':80 ,'fifty':50, 'forty':40, 'twenty':20 ,'ninety':90,'thirty':30,'seventy':70}
age_dict_1 = {'one':1,'two':2,'three':3,'four':4,'five':5,'fivee':5,'six':6,'seven':7,'eight':8,'nine':9}
age_dict_ = {'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'eleven':11,'twelve':12}
sex_list = ['m','M','MALE','f','F','FEMALE','other']
def job_living(job_status_living):
    def string_deal(string_a):
        a_st = string_a.lower()
        if a_st == 'null' or a_st == 'n.a' or a_st == '' or a_st == 'nan':
            return ''
        if a_st in {'government', 'govt.'}:
            return 'government'
        if a_st in {'business owner', 'business_owner', 'biz'}:
            return 'business owner'
        if a_st in {'private', 'private sector', 'private_sector', 'privattte'}:
            return 'private sector'
        if a_st in {'parental_leave', 'parental leave'}:
            return 'parental leave'
        if a_st in {'city', 'c'}:
            return 'city'
        if a_st in {'remote', 'remotee'}:
            return 'remote'
        if a_st in {'unemployed'}:
            return 'unemployed'
        return ''
    all_types = set()
    for i in job_status_living.values:
        if isinstance(i, str):
            split_i = i.split('?')
            for k in split_i:
                all_types.add(k)
        else:
            all_types.add('nan')
    job_status = {'government', 'business owner', 'parental leave', 'private sector', 'unemployed'}
    living = {'remote', 'city'}
    JS = []
    LV = []
    for i in job_status_living.values:
        if isinstance(i, str):
            split_i = i.split('?')[:2]
            single_JS = ''
            single_LV = ''
            for k in split_i:
                k_deal = string_deal(k)
                if k_deal in job_status:
                    single_JS += k_deal
                elif k_deal in living:
                    single_LV += k_deal
            JS.append(single_JS)
            LV.append(single_LV)
        else:
            JS.append('')
            LV.append('')
    JS_LV = pd.DataFrame({'job_status': JS, 'living': LV})
    JS_LV['job_status'] = JS_LV['job_status'].replace('', np.nan)
    JS_LV['living'] = JS_LV['living'].replace('', np.nan)
    return JS_LV


def process_age(x):
    pat = re.compile(r'[ -]')
    if type(x) == str:
        if not x:
            return np.nan
        x = x.strip()
        row = x.split(',')
        if not row[1] and not row[0]:
            return np.nan
        if not row[0] or not row[1]:
            if not row[1]:
                if row[0][0].isnumeric():
                    if '.' not in row[0]:
                        return int(row[0])
                    else:
                        return float(row[0])
                elif row[0] in sex_list:
                    return np.nan
                else:
                    for dic in [age_dict_, age_dict_1, age_dict_10]:
                        if row[0] in dic.keys():
                            return dic[row[0]]
                    return np.nan
            if not row[0]:
                if row[1][0].isnumeric():
                    if '.' not in row[1]:
                        return int(row[1])
                    else:
                        return float(row[1])
                elif row[1] in sex_list:
                    return np.nan
                else:
                    for dic in [age_dict_, age_dict_1, age_dict_10]:
                        if row[1] in dic.keys():
                            return dic[row[1]]
                    return np.nan
        sex, age = row[0], row[1].strip()
        if row[0][0].isnumeric() or (age in sex_list):
            sex, age = age, sex
        if age[0].isnumeric():
            if '.' not in age:
                return int(age)
            else:
                return float(age)
        age = re.split(pat, age)
        if len(age) == 1:
            for dic in [age_dict_, age_dict_1, age_dict_10]:
                if age[0] in dic.keys():
                    return dic[age[0]]
        else:
            return age_dict_10[age[0]] + age_dict_1[age[1]]
    else:
        return np.nan

def process_sex(x):
    if type(x) == str:
        if not x:
            return np.nan
        x = x.strip()
        row = x.split(',')
        if not row[1] and not row[0]:
            return np.nan
        if not row[0] or not row[1]:
            if not row[1]:
                if row[0] not in sex_list:
                    return np.nan
            else:
                if row[1] not in sex_list:
                    return np.nan
        sex, age = row[0], row[1].strip()
        if sex[0].isnumeric() or (age in sex_list):
            sex, age = age, sex
        if sex in ['m', 'M', 'MALE']:
            return 1
        elif sex in ['F', 'FEMALE', 'f']:
            return 0
        else:
            return 2
    else:
        return np.nan

def process_BP(x):
    if not x:
        return np.nan
    if type(x)==str:
        if x=='0':
            return 0
        elif x=='1':
            return 1
        else:
            return np.nan
    if math.isnan(x):
        return np.nan
    elif type(x) == int or type(x)==float:
        if x==np.nan:
            return x
        if int(x)==0:
            return 0
        elif int(x)==1:
            return 1
        return np.nan

def process_heart_condition(x):
    if not x : return np.nan
    if type(x)==str:
        if x=='1':
            return 1
        elif x=='0':
            return 0
        else:
            return np.nan
    else:
        return x

def process_married(x):
    if type(x)==str:
        s = str(x)
        if s.startswith('0'):
            return 0
        elif s.startswith('1'):
            return 1
        else:
            return np.nan
    elif type(x)==int:
        if x==1:
            return 1
        elif x == 0:
            return 0
        elif x//10 == 1:
            return 1
        else:
            return np.nan

def help_replace(x):
    if type(x)==str:
        if x=='0':
            return 0
        elif x=='1':
            return 1
        else:
            return 2
    return x

def process_job(x):
    if type(x)==str:
        dic = {'private sector':0,'business owner':1,'parental leave':2,'government':3,'unemployed':4}
        return dic[x]
    else:
        return x

def process_target(x):
    if type(x) == float or type(x) == int:
        if all(s in '10.'for s in str(x)):
            if x in ['1.0','1','0.0','0']:
                return int(x)
            else:
                return np.nan
        else:
            return np.nan
    elif type(x) == str:
        if all(s in '01.'for s in x):
            if x in ['1.0','1','0.0','0']:
                return int(x)
            else:
                return np.nan
        else:
            return np.nan
    else:
        return np.nan

def process_living(x):
    dic = {'city':0,'remote':1}
    if type(x)==str:
        return dic[x]
    return x

def process_AB(x):
    if type(x) == float:
        return x
    elif type(x) == str:
        if all(s in '0123456789.'for s in x):
            return float(x)
        else:
            return np.nan
    else:
        return np.nan

def process_smoker(x):
    dic = {'non-smoker':0,'quit':1,'quit?':1,'active_smoker':2}
    if type(x)==str:
        if x in dic.keys():
            return dic[x]
        else:
            return np.nan
    else:
        return x

data_train = pd.read_csv('train.csv',encoding='ISO-8859-1')
target_train = data_train['stroke_in_2018']
data_train_drop = data_train.drop(['stroke_in_2018'],axis=1)
data_test = pd.read_csv('test.csv',encoding='ISO-8859-1')
# concat train and test to deal with totally
data_total = pd.concat((data_train_drop,data_test)).reset_index(drop=True)

#deal with job_status and living area

job_status_living = data_total['job_status and living_area']
JS_LV = job_living(job_status_living)

data_total.insert(5,'job_status',JS_LV['job_status'])
data_total.insert(6,'living',JS_LV['living'])

del data_total['job_status and living_area']


data_total.insert(1,'sex',data_total['sex and age'].apply(process_sex))
data_total.insert(2,'age',data_total['sex and age'].apply(process_age))

del data_total['sex and age']

print(data_total.head())
#BP
data_total['high_BP'] = data_total['high_BP'].apply(process_BP)
#heart
data_total['heart_condition_detected_2017'] = data_total['high_BP'].apply(process_heart_condition)
#married
data_total['married'] = data_total['married'].apply(process_married)
#treatment
for i in ['TreatmentA','TreatmentB','TreatmentC','TreatmentD']:
    data_total[i] = data_total[i].fillna(2)
data_total['TreatmentD'] = data_total['TreatmentD'].apply(help_replace)
# job_status
data_total['job_status'] = data_total['job_status'].apply(process_job)

#living
data_total['living'] = data_total['living'].apply(process_living)

#smoker
data_total['smoker_status'] = data_total['smoker_status'].apply(process_smoker)

# average_blood_sugar
data_total['average_blood_sugar'] = data_total['average_blood_sugar'].apply(process_AB)
# BMI
data_total['BMI'] = data_total['BMI'].apply(process_AB)


data_new_target = process_target(data_train['stroke_in_2018'])
