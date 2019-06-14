import re
from datetime import datetime, date
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup 
import urllib.request 
import ssl

#functions for dat scraping
def find_links(soup, list_links):
    cars=soup.findAll("div", {'class':'talalati_lista'})
    for c in cars:
        list_links.append(c.h2.a["href"]) 
    return list_links


def open_link(url_page):
    user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36"
    headers={'User-Agent':user_agent,} 
    request=urllib.request.Request(url_page,None,headers) 
    ssl._create_default_https_context = ssl._create_unverified_context
    response = urllib.request.urlopen(request)
    data = response.read()
    soup=BeautifulSoup(data, 'html.parser')
    return soup

#functions for transforming the variables
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_nan(x): 
    try: 
        return math.isnan(x) 
    except: 
        return False

def get_float(x):
    if is_nan(x):
        return None
    else:
        if is_float(x):
            return x
        else :
            return float(''.join(ch for ch in x if ch.isdigit()))

def get_int(x):
    if is_nan(x):
        return None
    elif x=='Ár nélkül':
        return None 
    else:
        if is_int(x):
            return x
        else :
            return int(''.join(ch for ch in x if ch.isdigit()))
def get_AC(x):
    if is_nan(x):
        return 0
    else:
        return 1

def get_run_time_new(x):
    if is_nan(x):
        return None
    elif x=='Újjárm?':
        return 1
    else:
        return 0

def get_propulsion(x):
    if is_nan(x):
        return None
    elif x=='Els? kerék':
        return 'First_wheel'
    elif x=='Hátsó kerék':
        return 'Back_wheel'
    else:
        return 'Both_wheel'
        
def get_km(x):
    if is_nan(x):
        return None
    elif x=='Nincs megadva':
        return None
    else:
        return get_int(x)
    
def get_date(x):
    if is_nan(x):
        return None
    else:
        numbers=re.findall(r'\d+', x)
        if len(numbers)>=2:
            if int(numbers[1])>12:
                return date(int(numbers[0]),1,1)
            else: 
                return date(int(numbers[0]), int(numbers[1]), 1)
        elif len(numbers)==1:
            return date(int(numbers[0]),1,1)
        else:
            return date(1900,1,1)
        
def get_gear(x):
    if is_nan(x):
        return None
    elif x.lower().find('manuális')!=-1:
        return 0
    else:
        return 1
            
def get_color(x):
    if is_nan(x):
        return None
    elif x=='Fehér':
        return 0
    else:
        return 1  

    
def get_condition(x):
    if is_nan(x):
        return None
    elif x.lower().find('sérült')!=-1:
        return 'damaged'
    elif x.lower().find('hibás')!=-1:
        return 'faulty'
    else:
        return 'good'

def date_diff(d1, d2):
    delta=d1-d2
    return delta.days

def get_diesel(x):
    if is_nan(x):
        return None
    elif x.lower().find('dízel')!=-1:
        return 1
    else:
        return 0    
    
def get_benzin(x):
    if is_nan(x):
        return None
    elif x.lower().find('benzin')!=-1:
        return 1
    else:
        return 0  

def get_electro(x):
    if is_nan(x):
        return None
    elif (x.lower().find('elektomos')!=-1) or (x.lower().find('hibrid')!=-1):
        return 1
    else:
        return 0

def get_back(x):
    if x=='Back_wheel' or x=='Both_wheel':
        return 1
    else:
        return 0 
    
def get_first(x):
    if x=='First_wheel' or x=='Both_wheel':
        return 1
    else:
        return 0 

def get_condition(x):
    if x=='good':
        return 1
    else:
        return 0

def nr_person_group(x):
    if x<5:
        return 1
    elif x==5:
        return 2
    else:
        return 3

def rating_group(x):
    if x<3:
        return 1
    elif x==3:
        return 2
    else:
        return 3
        
#functions for plotting thevariables

def check_features(df, feature):
    print('The number of missing values in feature %s: %d' % (feature, df[feature].isnull().sum()))
    fig=plt.figure(figsize=(10,5))
    if len(df[feature].unique())<20:
        df[feature].value_counts().plot(kind='bar')
    else:
        plt.hist(df[feature].dropna(),25)
    plt.xlabel(feature)
    plt.ylabel('# of cars')
    plt.title('Distribution of '+str(feature))
    name=feature+'_dist1.png'
    plt.savefig('distributions/'+str(name))

def get_missing(col, dframe):
    for i, row in dframe[dframe[col].isnull()].iterrows():        
        try:
            cartype=row['type']
            cmode=dframe[dframe['type']==cartype][col].value_counts().argmax()
            ad=row['ad_nr']
            dframe.loc[(dframe['ad_nr']==ad),col]=cmode
        except:
            pass

def plot_bar(df, feature, figsize=(12,4), width=0.4, n_cols=None):
    
    temp=df.groupby(feature)['PriceEUR'].agg(['mean','count']).reset_index()
    if n_cols!=None:
        temp=temp.head(n_cols)
    fig, ax1 = plt.subplots(figsize=figsize)
    labels=temp[feature].values
    ind=np.arange(len(labels))
    #ax1.bar(temp[feature],temp['count'], width, color='b',align='center')
    ax1.bar(ind,temp['count'], width, color='b',align='center')

    ax1.set_label('Count cars')
    ax1.set_ylabel('Count cars', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.scatter(ind,temp['mean'],s=70 , color='r' )
    ax2.set_ylabel('Price in EUR', color='r')
    ax2.tick_params('y', colors='r')

    ax2.set_title(str(feature) + ' - count and average prices')
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels)

def treat_outliers(df, feature, threshold, final, filt):
        
    print('The number of outliers in the dataset regarding the {} is : {}'.format(feature, len(df[df[feature]>threshold])))
    
    col_final='{}_final'.format(feature)
    col_filt='{}_filt'.format(feature)
    df[col_final]=df[feature].apply(lambda x: threshold if x>threshold else x)
    df[col_filt]=df[feature].apply(lambda x: None if x>threshold else x)

    final.append(col_final)
    filt.append(col_filt)

def plot_box_reg(df, feature):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 4))
    ax = sns.boxplot(x=df[feature],whis=1.5, ax=axes[0])
    ax.set_title('Boxplot - ' + str(feature))
    ax = sns.regplot(x=df[feature],y=df['PriceEUR'], ax=axes[1])
    ax.set_title('Regressionplot - ' + str(feature))

#treating the cartypes in order to match the data

def get_cartype(x):
    if x=='VOLVO 200-AS SOROZAT' or x=='VOLVO 700-AS SOROZAT' or x=='VOLVO 900-AS SOROZAT' or x=='VOLVO 850':
        return 'VOLVO 70'
    elif x=='VOLVO 400-AS SOROZAT':
        return 'VOLVO 440'
    elif x=='VOLVO V50' or x=='VOLVO C30':
        return 'VOLVO 40'
    elif x[:5]=='VOLVO':
        return 'VOLVO '+x[-2:]
    elif x[-11:]=='-ES SOROZAT' or x[-11:]=='-AS SOROZAT'  or x[-11:]=='-OS SOROZAT' or x[-11:]=='-ÖS SOROZAT':
        return x[:-11]+'ER'
    elif x=='BMW M SOROZAT':
        return 'BMW Z1'
    elif x[-8:]==' SOROZAT':
        return x[:-8]+'3'
    elif x[:13]=='MERCEDES-BENZ':
        if x[14:17] in ('GLS','CLK', 'CLS', 'GLA', 'GLK', 'GLS', 'GLE', 'SLK'):
            return 'MERCEDES'+x[13:17]
        elif x[14:16] in ( 'SL'):
            return 'MERCEDES'+x[13:16]
        elif x=='MERCEDES-BENZ 500':
            return 'MERCEDES SL'
        elif x[14:16] in ('GL', 'CL'):
            return 'MERCEDES'+x[13:16]+'-KLASSE'
        elif x=='MERCEDES-BENZ W-OSZTÁLY' or x=='MERCEDES-BENZ 200' or x=='MERCEDES-BENZ 230' or x=='MERCEDES-BENZ 300' or x=='MERCEDES-BENZ 280' or x=='MERCEDES-BENZ 260' or x=='MERCEDES-BENZ 400':
            return 'MERCEDES E-KLASSE'
        elif x[-8:]=='-OSZTÁLY': 
            return 'MERCEDES'+x[13:-8]+'-KLASSE'
        else:
            return 'MERCEDES'+x[13:]
    elif x[:10]=='VOLKSWAGEN' and (x[11:]=='BOGÁR (KÄFER)' or x[11:]=='NEW BEETLE'):
        return 'VW BEETLE'
    elif x=='VOLKSWAGEN BORA' or x=='VOLKSWAGEN VENTO':
        return 'VW JETTA'
    elif x=='VOLKSWAGEN CC' or x=='VOLKSWAGEN ARTEON':
        return 'VW PASSAT'
    elif x=='VOLKSWAGEN CARAVELLE':
        return 'VW TRANSPORTER'
    elif x[:10]=='VOLKSWAGEN':
        return 'VW'+x[10:]
    elif x=='CHRYSLER 300 C':
        return 'CHRYSLER 300C'
    elif x=='CHRYSLER 300 M':
        return 'CHRYSLER 300M'
    elif x=='ALFA ROMEO 168' or x=='ALFA ROMEO 164':
        return 'ALFA ROMEO ALFA 166'
    elif x=='ALFA ROMEO 155' or x=='ALFA ROMEO 75':
        return 'ALFA ROMEO ALFA 156' 
    elif x=='ALFA ROMEO 33':
        return 'ALFA ROMEO ALFA 145'
    elif x[:10]=='ALFA ROMEO' and x!='ALFA ROMEO GIULIA':
        return 'ALFA ROMEO ALFA'+x[10:]
    elif x[:11]=='CITROEN C3 ' or x[:11]=='CITROEN C4 ':
        return x[:10]
    elif x=='DAIHATSU GRAN MOVE':
        return 'DAIHATSU GRANMOVE'
    elif x=='FORD TRANSIT':
        return 'FORD TRANSIT COURIER'
    elif x=='CHEVROLET TACUMA' or x=='DAEWOO TACUMA':
        return 'CHEVROLET ORLANDO'
    elif x=='FORD KA+':
        return 'FORD KA'
    elif x=='FORD COURIER':
        return 'FORD TRANSIT COURIER'
    elif x=='FIAT 500L': 
        return 'FIAT MULTIPLA'
    elif x=='FIAT 500X':
        return 'FIAT SEDICI'
    elif x=='FIAT PALIO':
        return 'FIAT PALIO WEEKEND'
    elif x=='HYUNDAI IONIQ':
        return 'HYUNDAI I 30'
    elif x=='HYUNDAI H-1':
        return 'HYUNDAI H-1 STAREX'
    elif x[:9]=='HYUNDAI I' and x[:10]!='HYUNDAI IX':
        return 'HYUNDAI I'+' '+x[9:]
    elif x=='ROVER 800':
        return 'MG ROVER 75'
    elif x=='ROVER 200' or x=='ROVER STREETWISE':
        return 'MG ROVER 25'
    elif x=='ROVER 400':
        return 'MG ROVER 45'
    elif x=='MG ZR':
        return 'MG MG ZR'
    elif x[:8]=='NISSAN 3':
        return 'NISSAN 350Z'
    elif x=='NISSAN TERRANO':
        return 'NISSAN TERRANO II'
    elif x=='OPEL MOKKA X':
        return 'OPEL MOKKA'
    elif x[:7]=='PORSCHE':
        return 'PORSCHE 911'
    elif x=='PEUGEOT 301' or x=='PEUGEOT 305' or x=='PEUGEOT 309':
        return 'PEUGEOT 306'
    elif x=='PEUGEOT 605':
        return 'PEUGEOT 607'
    elif x=='RENAULT R 19':
        return 'RENAULT R19'
    elif x[:5]=='ROVER':
        return 'MG ROVER'+x[5:]
    elif x=='SMART EGYÉB':
        return 'SMART FORTWO'
    elif x[:9]=='SSANGYONG':
        return 'SSANGYONG TIVOLI'
    elif x=='SUZUKI WAGON R+':
        return 'SUZUKI WAGON R'
    elif x=='SUZUKI SX4 S-CROSS':
        return 'SUZUKI SX4'
    elif x=='TOYOTA GT86':
        return 'TOYOTA GT 86'
    elif x=='TOYOTA LAND CRUISER':
        return 'TOYOTA LANDCRUISER'
    elif x=='TOYOTA MR 2':
        return 'TOYOTA MR-2'
    elif x=='TOYOTA PRIUS+':
        return 'TOYOTA PRIUS PLUS'
    elif x=='DAEWOO ESPERO' or x=='DAEWOO LEGANZA' or x=='DAEWOO EVANDA':
        return 'CHEVROLET EPICA'
    elif x=='DAEWOO NEXIA' or x=='DAEWOO RACER':
        return 'CHEVROLET NUBIRA'
    elif x=='DAEWOO TICO':
        return 'CHEVROLET MATIZ'
    elif x[:6]=='DAEWOO':
        return 'CHEVROLET'+x[6:]
    elif x=="KIA CEE'D":
        return 'KIA CEED'
    elif x=='RENAULT THALIA':
        return 'RENAULT CLIO'
    elif x=='AUDI ALLROAD' or x=='AUDI 80':
        return 'AUDI A4'    
    elif x=='LADA NOVA' or x=='LADA VESTA' or x=='LADA ZHIGULI' or x=='LADA 1300' or x=='LADA 1500' or x=='LADA 110':
        return 'LADA 1118'    
    elif x=='OPEL CROSSLAND X' or x=='OPEL GRANDLAND X':
        return 'OPEL ANTARA'    
    elif x=='MITSUBISHI SPACE':
        return 'MITSUBISHI SPACE STAR'
    elif x=='SUBARU OUTBACK':
        return 'SUBARU LEGACY'
    elif x=='TRABANT 601' or x=='WARTBURG 1.3':
        return 'C'    
    elif x=='CITROEN C-ELYSEE':
        return 'CITROEN ZX'
    elif x=='MARUTI 800':
        return 'SUZUKI ALTO'
    elif x[:5]=='MINI ':
        return 'MINI MINI'
    elif x=='SKODA FAVORIT':
        return 'SKODA FELICIA'
    elif x=='POLSKI FIAT 126':
        return 'FIAT CINQUECENTO'
    elif x=='FIAT ALBEA' or x=='LADA GRANTA' or x=='LADA KALINA':
        return 'FIAT PUNTO'
    elif x=='CITROEN DS3':
        return 'CITROEN C2'
    elif x=='CITROEN DS4':
        return 'CITROEN C4'
    elif x=='AUDI 100':
        return 'AUDI A6'
    elif x=='AUDI TTS':
        return 'AUDI TT'
    elif x=='FORD CONNECT':
        return 'FORD TRANSIT CONNECT'
    elif x=='FORD ORION':
        return 'FORD ESCORT'
    elif x=='HONDA CITY':
        return 'HONDA CIVIC'
    elif x=='SKODA KODIAQ':
        return 'SKODA YETI'
    elif x=='INFINITI M':
        return 'NISSAN MAXIMA'
    elif x=='SAAB 900':
        return 'SAAB 9-3'
    elif x=='DODGE CARAVAN':
        return 'DODGE JOURNEY'
    elif x=='HONDA CR-X':
        return 'HONDA CR-Z'
    elif x=='LANCIA THEMA' or x=='LANCIA THESIS':
        return 'LANCIA KAPPA'
    elif x=='HONDA LEGEND' or x=='CADILLAC CTS' or x=='RENAULT LATITUDE':
        return 'E'
    elif x=='FIAT TEMPRA' or x=='DACIA 1310':
        return 'C'  
    elif x=='INFINITI EX' or x=='HYUNDAI GALLOPER':
        return 'SUV'
    elif x=='FIAT QUBO' or x=='PEUGEOT TRAVELLER' or x=='NISSAN EVALIA':
        return 'M'
    elif x=='CADILLAC BLS' or x=='FORD SIERRA':
        return 'D'
    elif x=='ABARTH 500':
        return 'A'
    elif x=='HYUNDAI PONY':
        return 'B'
    elif x=='FIAT SPIDER' or x=='HONDA INTEGRA':
        return 'S'
    elif x=='FIAT TALENTO':
        return 'FIAT DUCATO'
    else:
        return x

def get_class_aggr(x):
    if x in ('V', 'M'):
        return 'V'
    elif x in ('SUV', 'G', 'U'):
        return 'SUV'
    else:
        return x

def df_groupby_to_plot(df,col_groupby,col_values, list_values):
    return df.groupby(col_groupby)[col_values].agg(list_values).reset_index()

def plot_producers(df, x_axis, y1_axis,y1_label, y2_axis,y2_label, lim, title):
    temp=df.sort_values(y2_axis, ascending=False)
    x=np.arange(1,len(temp[y2_axis])+1)
    y=temp[y2_axis]
    x_list=temp[x_axis].values

    fig, ax=plt.subplots(figsize=(15,6))
    l1=ax.bar(x,temp[y1_axis],alpha=0.2, color='r',width=0.5,align='center' )
    ax.set_ylabel(y1_label, color='r')
    ax.tick_params('y', colors='r')

    #ax.set_ylim(0,10000)
    #ax.set_xlim(0,25)
    ax.set_ylim(lim[2],lim[3])
    ax.set_xlim(lim[0],lim[1])
    ax.set_title(title)
    ax2 = ax.twinx()
    l2=ax2.scatter(x,y,alpha=0.8 , label=y2_label )
    l3=ax2.plot(x,y,'b-',alpha=0.8  )

    ax2.set_ylabel(y2_label, color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_xlim(lim[0],lim[1])
    plt.legend( (l1,l2), (y1_label,y2_label), loc = 'upper right')

    ax.set_xticks(x + 0.3 / 2)
    ax.set_xticklabels(x_list, rotation=30, fontsize=8)
    ax.grid(False)
    ax2.grid(False)

def plot_stacked(df, x_axis, y1_axis,y1_label, y2_axis,y2_label, lim, title, legend):
    temp=df.sort_values(y2_axis, ascending=False)
    x=np.arange(1,len(temp[y2_axis])+1)
    y=temp[y2_axis]
    x_list=temp[x_axis].values
    colors=['r', 'b', 'g', 'cyan', 'yellow', 'brown', 'magenta', 'black', 'indigo']
    fig, ax=plt.subplots(figsize=(15,6))
    sum_bottom=0
    legend_list=[]
    
    for i,col in enumerate(y1_axis):
        name='l'+str(i)
        name=ax.bar(x,temp[col],alpha=0.2, color=colors[i],width=0.5,align='center', bottom=sum_bottom)
        sum_bottom+=temp[col]
        legend_list.append(name)
    ax.set_ylabel(y1_label, color='r')
    ax.tick_params('y', colors='r')

    ax.set_ylim(lim[2],lim[3])
    ax.set_xlim(lim[0],lim[1])
    ax.set_title(title)
    ax2 = ax.twinx()
    l10=ax2.scatter(x,y,alpha=0.8 , label=y2_label )
    l11=ax2.plot(x,y,'b-',alpha=0.8  )

    ax2.set_ylabel(y2_label, color='b')
    ax2.tick_params('y', colors='b')
    ax2.set_xlim(lim[0],lim[1])
    
    legend_list.append(l10)
    legend.append(y2_label)
    legend_list = tuple(legend_list)
    legend_tuple= tuple(legend)
                  
                  
                  
    plt.legend( legend_list, legend_tuple, loc = 'upper right')

    ax.set_xticks(x)
    ax.set_xticklabels(x_list, rotation=45, fontsize=7)
    ax.grid(False)
    ax2.grid(False)