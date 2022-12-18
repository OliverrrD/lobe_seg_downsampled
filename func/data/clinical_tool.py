import os
import pandas as pd
import nibabel as nib
import numpy as np
from skimage import transform, util
import re
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
import shutil

def get_static_info(csv_path, item, phase):
    """
    This function is designed to add the table to SPIE paper. For the discrete variable.
    exp: get_static_info('/share5/wangj46/fromlocal/demographic/final_4fold.csv', 'Histologic Type', 'test')
    :param csv_path:
    :param item:
    :param phase:
    :return:
    """
    df = pd.read_csv(csv_path)
    res = {}
    count = 0
    ph_dict = {'train': [1,2,3], 'val':[4], 'test': [0]}
    for i in range(len(df[item])):
        if df['trainvalnew'][i] not in ph_dict[phase]:
            continue
        count += 1
        if df[item][i] in res.keys():
            res[df[item][i]] += 1
        else:
            res[df[item][i]] = 1
    print ('the num is ', count)
    return res

def get_static_ave_std(csv_path, item, phase):
    '''
    This function is designed to add the table to SPIE paper. For the continuous variable.
    :param csv_path:
    :param item:
    :param phase:
    :return:
    '''
    df = pd.read_csv(csv_path)
    res = []
    ph_dict = {'train': [1,2,3], 'val':[4], 'test': [0]}
    for i in range(len(df[item])):
        if df['trainvalnew'][i] not in ph_dict[phase]:
            continue
        if df[item][i] != df[item][i]:
            print ("it is none")
            continue
        res.append(df[item][i])
    res = np.array(res)
    return np.mean(res), np.std(res)

def get_clinical_step1(ori_path, new_path):
    df = pd.read_csv(ori_path)
    meanaq = df["Age Quit"].mean()
    meanas = df["Age Started Cig"].mean()
    meancpd = df['CIGSPERDAY'].mean()
    for index, item in df.iterrows():
        if (item['Weight'] == item['Weight'] and item['Height'] == item['Height']):  # not nan
            df.at[index, 'BMI'] = item['Weight'] / item['Height'] / item['Height'] * 703
        if (item["Age Quit"] != item["Age Quit"]):  # nan
            if (item['CIGARETTE_Q1'] == "Never smoked"):
                df.at[index, 'Age Quit'] = 0
            elif (item['CIGARETTE_Q1'] == "Current smoker"):
                df.at[index, 'Age Quit'] = 100
            else:
                df.at[index, 'Age Quit'] = meanaq
        if (item["Age Started Cig"] != item["Age Started Cig"]):  #  nan
            if (item['CIGARETTE_Q1'] == "Never smoked"):
                df.at[index, 'Age Started Cig'] = 100
            else:
                df.at[index, 'Age Started Cig'] = meanas
        if (item['CIGSPERDAY'] != item['CIGSPERDAY']):  #  nan
            if (item['CIGARETTE_Q1'] == "Never smoked"):
                df.at[index, 'CIGSPERDAY'] = 0
            else:
                df.at[index, 'CIGSPERDAY'] = meancpd
    data = pd.DataFrame()
    data['Subject'] = df['Lookup MCL']
    data['Gender'] = df['Gender'].apply(getgender)
    data['BMI'] = df['BMI'].fillna(df['BMI'].mean())

    data['History'] = df['CIGARETTE_Q1'].apply(getcigaretteq1)
    data['Age Started Cig'] = df['Age Started Cig']
    data['Age Quit'] = df['Age Quit']
    data['Cigs Per Day'] = df["CIGSPERDAY"].fillna(0)
    data['Pack Years'] = df['PACKYEARS'].fillna(0)  # =SMOKINGPACKS*SMOKINGDURATION
    data.to_csv(new_path)

def get_clinical_step2(ori_path,label_path, new_path):
    df = pd.read_csv(ori_path)
    lb = pd.read_csv(label_path)
    nll_index = []
    df_sub = df['Subject'].tolist()
    lb_sub = lb['subject'].tolist()
    for i in range(len(df_sub)):
        if df_sub[i] not in lb_sub:
            nll_index.append(i)
    new_df = df.drop(nll_index)
    label_list = []
    trainval_list = []
    new_df_sub = new_df['Subject'].tolist()
    for i in range(len(new_df_sub)):
        index_in_labeldf = lb_sub.index(new_df_sub[i])
        label_list.append(lb['label'][index_in_labeldf])
        trainval_list.append(lb['trainvalnew'][index_in_labeldf])
    new_df['label'] = label_list
    new_df['trainvalnew'] = trainval_list
    new_df.to_csv(new_path)

def get_clinical(ori_path, label_path, new_path):
    df = pd.read_csv(ori_path)
    lb = pd.read_csv(label_path)
    nll_index = []
    df_sub = df['Lookup MCL'].tolist()
    lb_sub = lb['subject'].tolist()
    for i in range(len(df_sub)):
        if df_sub[i] not in lb_sub:
            nll_index.append(i)
    new_df = df.drop(nll_index)
    meanaq = new_df["Age Quit"].mean()
    meanas = new_df["Age Started Cig"].mean()
    meancpd = new_df['CIGSPERDAY'].mean()
    for index, item in new_df.iterrows():
        if (item['Weight'] == item['Weight'] and item['Height'] == item['Height']):  # not nan
            new_df.at[index, 'BMI'] = item['Weight'] / item['Height'] / item['Height'] * 703
        if (item["Age Quit"] != item["Age Quit"]):  # nan
            if (item['CIGARETTE_Q1'] == "Never smoked"):
                new_df.at[index, 'Age Quit'] = 0
            elif (item['CIGARETTE_Q1'] == "Current smoker"):
                new_df.at[index, 'Age Quit'] = 100
            else:
                new_df.at[index, 'Age Quit'] = meanaq
        if (item["Age Started Cig"] != item["Age Started Cig"]):  #  nan
            if (item['CIGARETTE_Q1'] == "Never smoked"):
                new_df.at[index, 'Age Started Cig'] = 100
            else:
                new_df.at[index, 'Age Started Cig'] = meanas
        if (item['CIGSPERDAY'] != item['CIGSPERDAY']):  #  nan
            if (item['CIGARETTE_Q1'] == "Never smoked"):
                new_df.at[index, 'CIGSPERDAY'] = 0
            else:
                new_df.at[index, 'CIGSPERDAY'] = meancpd
    data = pd.DataFrame()
    data['subject'] = new_df['Lookup MCL']
    data['Gender'] = new_df['Gender'].apply(getgender)
    data['BMI'] = new_df['BMI'].fillna(new_df['BMI'].mean())

    data['History'] =new_df['CIGARETTE_Q1'].apply(getcigaretteq1)
    data['Age Started Cig'] =new_df['Age Started Cig']
    data['Age Quit'] = new_df['Age Quit']
    data['Cigs Per Day'] = new_df["CIGSPERDAY"].fillna(0)
    data['Pack Years'] = new_df['PACKYEARS'].fillna(0)  # =SMOKINGPACKS*SMOKINGDURATION
    label_list = []
    trainval_list = []
    data_sub = data['subject'].tolist()
    for i in range(len(data_sub)):
        index_in_labeldf = lb_sub.index(data_sub[i])
        label_list.append(lb['label'][index_in_labeldf])
        trainval_list.append(lb['trainvalnew'][index_in_labeldf])
    data['label'] = label_list
    data['trainvalnew'] = trainval_list
    data.to_csv(new_path)

    # label_list = []
    # trainval_list = []




def getgender(x):
        if x == 'Female':
            return 1
        else:
            return 0

def getcigaretteq1( x):
        if (x == "Ex-smoker"):
            return 1
        elif (x == "Current smoker"):
            return 2
        else:
            return 0

def getshssmokers( x):
        if (x == 'No' or x == np.nan):
            return 0
        if (x == 'Yes'):
            return 1

def getdegreerelatives(x):
        if (x == 'No' or x == np.nan):
            return 0
        if (x == 'Yes'):
            return 1

def get_notNull_dict(csv_path, item_list):
    df = pd.read_csv(csv_path)
    notNull = {}
    for item in item_list:
        tmp_list = df[item]
        count = 0
        for cell in tmp_list:
            if cell == cell:
                count += 1
        notNull[item] = count
    return notNull

def get_item_dict(csv_path, item_list):
    df = pd.read_csv(csv_path)
    item_dict = {}
    for item in item_list:
        tmp_list = df[item]
        tmp_dict = {}
        for tmp in tmp_list:
            if tmp != tmp:
                tmp = 'null'
            if tmp not in tmp_dict.keys():
                tmp_dict[tmp] = 0
            tmp_dict[tmp] += 1
        item_dict[item] = tmp_dict
    return item_dict


def get_newTable(old_table_path, new_table_path, item_list, dsct_item_list, id_list):
    '''
    exp: get_newTable('/share5/gaor2/data/MCL_match_070618.csv', '/share5/gaor2/data/MCL_new.csv', use_item, dsct_item, df['Lookup MCL'])
    :param old_table_path:
    :param new_table_path:
    :param item_list:
    :param dsct_item_list:
    :param id_list:
    :return:
    '''
    df = pd.read_csv(old_table_path)
    #item_dict = get_item_dict(old_table_path, item_list)
    data = pd.DataFrame()
    data['MCL id'] = id_list
    info = {}
    for item in item_list:
        if item not in dsct_item_list:
            data[item] = df[item].fillna(df[item].mean())
            continue
        tmp_info = {'null': 0}
        tmp_count = 0
        tmp_list = []
        for i in range(len(df[item])):
            if df[item][i] != df[item][i]:
                tmp_list.append(tmp_info['null'])
                continue
            if df[item][i] not in tmp_info.keys():
                tmp_count += 1
                tmp_info[df[item][i]] = tmp_count
            tmp_list.append(tmp_info[df[item][i]])
        info[item] = tmp_info
        data[item] = tmp_list
    data.to_csv(new_table_path)
    return info

def add_column_csv(old_table_path, new_table_path,new_item_name, new_item_list):
    df = pd.read_csv(old_table_path)

    df[new_item_name] = new_item_list
    df.to_csv(new_table_path)

def gather_folder_item(folder_path, save_txt_path):
    it_list = os.listdir(folder_path)
    f = open(save_txt_path, 'w')
    for i in range(len(it_list)):
        f.write(it_list[i] + '\n')
    f.close()


def label2y(csv_path):
    '''
    exp: label2y('/share5/gaor2/data/MCL/MCL277/clinical3.csv')
    '''
    def get_label(x):
        if x in ['Normal', 'Negative for Dysplasia and Metaplasia', 
                                  'Negative for Malignant Cells','Granuloma ', 'Granuloma','Squamous Metaplasia']:
            return 0
        else:
            return 1
    data = pd.read_csv(csv_path)
    data['y'] = data["label"].apply(get_label)
    data.to_csv(csv_path)


def check_csv(merge_csv):  # just  a example
    df3 = pd.read_csv('/share5/gaor2/DeepLungNet/data_clinical.csv')
    th = 0.001
    for i, item in df3.iterrows():
        if abs(item['Gender_x'] != item['Gender_y'] ) > th:
            print (item['subject'], ' Gender problem')
        if abs(item['trainvalnew_x'] - item['trainvalnew_y']) > th:
            print (item['subject'], ' valtest problem')
        if abs(item['History_x'] - item['History_y'] ) > th:
            print (item['subject'], ' History problem')
        if abs(item['Age Started Cig_x'] - item['Age Started Cig_y']) > th:
            print (item['subject'], ' Age Started Cig problem')
        if abs(item['Age Quit_x'] - item['Age Quit_y']) > th:
            print (item['subject'], ' Age Quit problem')
        if abs(item['Cigs Per Day_x'] - item['Cigs Per Day_y']) > th:
            print (item['subject'], ' Cigs Per Day problem')
        if abs(item['Pack Years_x'] - item['Pack Years_y']) > th:
            print (item['subject'], ' Pack Years problem')
            
def PLCOm2012(age, race, education, body_mass_index, copd, phist, fhist,
              smoking_status, smoking_intensity, duration, quit_time):  # this for spore, please also refer to Norm_cancer_risk
    def get_num(x):
        if x == 'yes' or x == 'current':
            return 1
        else:
            return 0
    
    def get_race(x):
        d = {'White': 0, 'Black': 0.3944778, 
             'Hispanic': -0.7434744, 'Asian': -0.466585, 
             'American Indian or Alaskan Native': 0,
             'Native Hawaiian or Pacific Islander': 1.027152}
        return d[x]
    age_item = 0.0778868 * (age - 62)
    edu_item = -0.0812744 * (education - 4)
    bmi_item = -0.0274194 * (body_mass_index - 27)
    copd_item = 0.3553063 * get_num(copd)
    phist_item = 0.4589971 * get_num(phist)
    fhist_item = 0.587185 * get_num(fhist)
    sstatus_item = 0.2597431 * (smoking_status - 1)  #  change at 20200423
    sint_item = - 1.822606 * (10 / smoking_intensity - 0.4021541613)
    duration_item = 0.0317321 * (duration - 27)
    qt_item = -0.0308572 * (quit_time - 10)
    res = age_item + get_race(race) + edu_item + bmi_item \
          +copd_item + phist_item + fhist_item + sstatus_item \
          + sint_item + duration_item + qt_item - 4.532506
    res = np.exp(res) / (1 + np.exp(res))
    return res

def SPORE_PLCO_table(csv_path, new_csv):
    def get_edu(x):
        if x in ['9-11th grade', '9-11th gra']:
            return 1
        if x in ['High school graduate', 'High schoo']:
            return 2
        if x in ['Post high school trai','Post high'] :
            return 3
        if x in ['Associate degree some', 'Associate']:
            return 4
        if x in ['Bachelors degree', 'Bachelors']:
            return 5
        if x in ['Graduate or professio', 'Graduate o']:
            return 6
    def get_phist(x):
        #assert len(x) > 0
        if x!=x:
            return 0 
        if x == 'The patient has no personal history of other cancer.':
            return 0
        else:
            return 1
    def get_fhist(x):
        if x != x:
            return 0 
        #assert len(x) > 0
        if x == 'The patient has no family history of other cancer.':
            return 0
        else:
            return 1
        
    def get_smostatus(x):   # I not sure, need check, 03/25
        if x == 'Current Smoker':
            return 2
        elif x == 'Former Smoker':
            return 1
        else:
            return 0
        
    def get_copd(x):
        if x == 'Yes':
            return 1
        if x == 'No':
            return 0
        
    def get_race(x):
        if x == 'Caucasian/ White':
            return 1
        if x == 'African American/ Black':
            return 2
        if x == 'Asian':
            return 3
        if x == 'Native Hawaiian/ Pacific Islan':
            return 5
        if x == 'Hispanic/ Latnio':
            return 2.5
        if x == 'American Indian, Alaska Native':
            return 4
        
    def get_sm_dur(x):
        if x > 50:
            duration = 50
            smoking_intensity = 20 * (x / 50.0)
        else:
            duration = x
            smoking_intensity = 20
        smoking_intensity = min(80, smoking_intensity)
        duration = min(50, duration)    
        return smoking_intensity, duration
        
    df = pd.read_excel(csv_path, 'history12-31-17')
    Age, Education, BMI, COPD, Phist, Fhist, Smo_status, Smo_intensity, Duration, Quit_time = [], [], [],[],[], [], [], [],[],[]
    Pid, Race = [],[]
    for i, item in df.iterrows():
        #print (i)
        try:
            age = item['Age']
            education = get_edu(item['education'])
            body_mass_index = ( item['weightpounds'] * 0.45359237)/  (item['heightinches'] * 2.54 / 100) / (item['heightinches'] * 2.54 / 100)
            copd = get_copd(item['copd'])
            #print (item['personal_cancer_history'])
            phist= get_phist(item['personal_cancer_history'])
            fhist = get_fhist(item['family_cancer_history'])
            smoking_status = get_smostatus(item['smokingstatus'])
            smoking_intensity, duration = get_sm_dur(item['packyearsreported'])
            quit_time = item['Age'] - item['yearssincequit']
            race = get_race(item['race'])
            pid = item['sub_name']
        except:
            print ('problem 1')
            continue
        if item['Age'] != item['Age'] or item['education'] != item['education'] or item['weightpounds'] != item['weightpounds'] or item['heightinches'] != item['heightinches'] or item['copd'] != item['copd'] or item['yearssincequit'] != item['yearssincequit'] or item['packyearsreported'] != item['packyearsreported'] or item['race'] != item['race']:
            print ('problem 2')
            continue
            
        Pid.append(pid)
        Age.append(age)
        Education.append(education)
        BMI.append(body_mass_index)
        COPD.append(copd)
        Phist.append(phist)
        Fhist.append(fhist)
        Smo_status.append(smoking_status)
        Smo_intensity.append(smoking_intensity)
        Duration.append(duration)
        Quit_time.append(quit_time)
        Race.append(race)
    #print (Pid)
    data = pd.DataFrame()
    data['pid'] = Pid
    data['age'] = Age
    data['education'] = Education
    data['bmi'] = BMI
    data['copd'] = COPD
    data['phist'] = Phist
    data['fhist'] = Fhist
    data['smo_status'] = Smo_status
    data['smo_intensity'] = Smo_intensity
    data['duration'] = Duration
    data['quit_time'] = Quit_time
    data['race'] = Race
    data.to_csv(new_csv, index = False)
    
def handleNull(x):
    if x != x:
        x = ''
    return x
    
def NLST_PLCO_table(csv_path, new_csv):
    df = pd.read_csv(csv_path)
    Age, Education, BMI, COPD, Phist, Fhist, Smo_status, Smo_intensity, Duration, Quit_time = [], [], [],[],[], [], [], [],[],[]
    Pid, Race = [],[]
    for i, item in df.iterrows():
        #try:
        age = handleNull(item['age'])
        education = handleNull(max(item['educat'], 2) - 1)  # becasue the educat definiation in NLST is a little different from PLCOM
        body_mass_index = handleNull(( item['weight'] * 0.45359237)/  (item['height'] * 2.54 / 100) / (item['height'] * 2.54 / 100))
        copd = handleNull (item['diagcopd'])

        try:
            phist =  int(item['cancblad']) | int(item['cancbrea']) | int(item['canccerv']) | int(item['canccolo']) | int(item['cancesop']) | int(item['canckidn']) | int(item['canclary']) | int(item['canclung']) | int(item['cancnasa']) | int(item['cancoral']) | int(item['cancpanc']) | int(item['cancphar']) | int(item['cancstom']) | int(item['cancthyr']) | int(item['canctran'])
        except:
            phist = 0
        try:
            fhist = int(item['fambrother']) | int(item['famchild']) | int(item['famfather']) | int(item['fammother']) | int(item['famsister'])
        except:
            fhist = 0
        smoking_status = handleNull(item['cigsmok'])
        smoking_intensity = handleNull(item['smokeday'])
        duration = handleNull (item['smokeyr'])
        quit_time = handleNull (item['age'] - item['age_quit'])
        race = handleNull (item['race'])
        pid = handleNull (item['pid'])
#         except:
#             #print ('problems')
#             continue
#         if item['age'] != item['age'] or item['educat'] != item['educat'] or item['weight'] != item['weight'] or item['height'] != item['height'] or item['cigsmok'] != item['cigsmok'] or item['smokeday'] != item['smokeday'] or item['smokeyr'] != item['smokeyr'] or item['age_quit'] != item['age_quit'] or item['race'] != item['race']:
#             continue
        Pid.append(pid)
        Age.append(age)
        Education.append(education)
        BMI.append(body_mass_index)
        COPD.append(copd)
        Phist.append(phist)
        Fhist.append(fhist)
        Smo_status.append(smoking_status)
        Smo_intensity.append(smoking_intensity)
        Duration.append(duration)
        Quit_time.append(quit_time)
        Race.append(race)
    data = pd.DataFrame()
    data['pid'] = Pid
    data['age'] = Age
    data['education'] = Education
    data['bmi'] = BMI
    data['copd'] = COPD
    data['phist'] = Phist
    data['fhist'] = Fhist
    data['smo_status'] = Smo_status
    data['smo_intensity'] = Smo_intensity
    data['duration'] = Duration
    data['quit_time'] = Quit_time
    data['race'] = Race
    data.to_csv(new_csv)
        
def Norm_cancer_risk(forplco_csv, risk_csv):  
    # for PLCOm risk calculation
    df = pd.read_csv(forplco_csv)
    def get_race(x):
        d = {1: 0, 2: 0.3944778, 
             2.5: -0.7434744, 3: -0.466585, 
             4: 0,
             5: 1.027152}
        return d[x]
    risk_list = []
    for i, item in df.iterrows():
        print (i)
        if item['race'] > 5:
            risk_list.append('')
            continue
        age_item = 0.0778868 * (item['age'] - 62)
        edu_item = -0.0812744 * (item['education'] - 4)
        bmi_item = -0.0274194 * (item['bmi'] - 27)
        copd_item = 0.3553063 * item['copd']
        phist_item = 0.4589971 * item['phist']
        fhist_item = 0.587185 * item['fhist']
        sstatus_item = 0.2597431 * item['smo_status']
        sint_item = - 1.822606 * (10 / item['smo_intensity'] - 0.4021541613)
        duration_item = 0.0317321 * (item['duration'] - 27)
        qt_item = -0.0308572 * (item['quit_time'] - 10)
        print (age_item, get_race(int(item['race'])), edu_item, bmi_item)
        res = age_item + get_race(int(item['race'])) + edu_item + bmi_item \
              +copd_item + phist_item + fhist_item + sstatus_item \
              + sint_item + duration_item + qt_item - 4.532506
        res = np.exp(res) / (1 + np.exp(res))
        risk_list.append(res)
    df['plco_risk'] = risk_list
    df.to_csv(risk_csv, index = False)
    
def Mayo(age, ciga, cancer, diameter, spicul, upper):
    ori_val = -6.827 + 0.0391 * age + 0.7917 * ciga +  1.3388 * cancer + 0.1274 * diameter + 1.0407 * spicul + 0.7838 * upper
    prob = np.exp(ori_val) / (1 + np.exp(ori_val))
    return round(ori_val, 3), round(prob, 3)

def BrockRisk(age, sex, fhist, copd, nodule_size, nodule_type, uplobe, nodule_cnt, spic):
    '''
    https://www.uptodate.com/contents/calculator-solitary-pulmonary-nodule-malignancy-risk-in-adults-brock-university-cancer-prediction-equation
    age: true age 
    sex: Female 1, male 0
    fhist: True 1, False 0
    copd: True 1, False 0
    Nodule size: mm
    nodule_type: nonsolid or ground-glass: "nonsoild", Partially solid: 'partial', solid.
    uplobe: True 1, False 0
    spiculation: True 1, False 0
    '''
    type_dict = {'Non-Solid': -0.127, 'Part-Solid': 0.377, 'Solid': 0}
    logodds = 0.0287 * (age - 62) + 0.6011 * sex + 0.2961 * fhist + 0.2953 * copd  - (5.3854 * ( 1. / np.sqrt(nodule_size/10)- 1.58113883)) + type_dict[nodule_type] + 0.6581*uplobe -(0.0824 * (nodule_cnt - 4)) + 0.7729 *spic - 6.7892 
    prob = np.exp(logodds) / (1 + np.exp(logodds))
    return round(logodds, 4), round(prob, 4)

def PLCOM_factors(csv_root, new_root):
    df = pd.read_csv(csv_root)
    def get_race(x):
        d = {1: 0, 2: 0.3944778, 
             2.5: -0.7434744, 3: -0.466585, 
             4: 0,
             5: 1.027152}
        return d[x]
    risk_list = []
    age, education, bmi, copd, phist, fhist = [], [], [], [], [], []
    smo_status,smo_intensity,duration, quit_time, race = [], [], [], [], []

    for i, item in df.iterrows():
        #print (i)
        if item['race'] > 5:
            item['race'] = 1
        age.append(item['age'] - 62)
        education.append(item['education'] - 4)
        bmi.append(item['bmi'] - 27)
        copd.append(item['copd'])
        phist.append(item['phist'])
        fhist.append(item['fhist'])
        smo_status.append(item['smo_status'])
        smo_intensity.append(10. / item['smo_intensity'] - 0.4021541613)
        duration.append (item['duration'] - 27)
        quit_time.append(item['quit_time'] - 10)
        race.append(get_race(int(item['race'])))
    df['age'] = age
    df['education'] = education
    df['bmi'] = bmi
    df['copd'] = copd
    df['phist'] = phist
    df['fhist'] = fhist
    df['smo_status'] = smo_status
    df['smo_intensity'] = smo_intensity
    df['duration'] = duration
    df['quit_time'] = quit_time
    df['race'] = race
    df.to_csv(new_root, index = False)
    
def cnt_empty(file_path):
    data = pd.read_excel(file_path)
    count = 0
    lst = ['personal_cancer_history', 'family_cancer_history', 'Age', 'race',
            'education', 'weightpounds', 'heightinches','copd', 'smokingstatus', 'packyearsreported','yearssincequit']
    for i, item in data.iterrows():
        for it in lst:
            if item[it] != item[it]:
                count += 1
                break
    print ('the total number is ', len(data), ", and the empty number is ", count)