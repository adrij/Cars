#it takes a very long time to run
#data exported from 01/2018
from bs4 import BeautifulSoup 
import pandas as pd
import numpy as np 
from datetime import datetime
import cars_functions

list_links=[]
url = "https://www.hasznaltauto.hu/talalatilista/auto/YHUF9UJK0EZD3UAK03ODCKAUG5QKKEE00W9SRDAOHFUWSQHQE7GUMEESAG2JJA2FOWC5HUSF7D0SESGAQ4WDSS/page"
for i in np.arange(1,10000,1): 
    url_page=url+str(i)
    soup=cars_functions.open_link(url_page)
    cars_functions.find_links(soup, list_links) 

### 2. Open the carlinks and save the features of each cars
#some cars were already deleted from the web
all_cars=[]
for link in list_links:
    try:
        data_car=cars_functions.open_link(link)
        car_features={}
        car_features["ad_nr"]=link[-8:]
        #get rating
        try:
            rating=data_car.findAll("div", {"class": "otpontozas"})[0]
            car_features["rating"]=rating.div.text
        except:
            pass
    
        #get the title of the ad
        try:
            name=data_car.findAll("div", {"class": "feherbox"})[0]
            car_features["title"]=name.h1.span.text
        except:
            pass
    
        #get producer
        try:
            prod=data_car.findAll("div", {"class": "navbar"})[0]
            producer=prod.findAll("span")
            car_features["producer"]=producer[4].text
            car_features["type"]=producer[6].text
        except:
            pass
    

        tbl=data_car.findAll("table", {"class": "hirdetesadatok"})
        rows=tbl[0].findAll("tr")

        for row in rows:
            try:
                cols=row.findAll("td")
                key=cols[0].text.strip()
                value=cols[1].text.strip()
                car_features[key]=value
            except:
                pass
    
        extras=[]
        try:
            tbl2=data_car.findAll("table", {"class": "felszereltseg"})[0]
            items=tbl2.findAll("li")
            for item in items:
                extras.append(item.text)
            car_features['extras']=extras
            #check the radar
            if ('APS (parkolóradar)' in extras) or ('tolatóradar)' in extras) or ('tolatókamera' in extras):
                car_features['radar']=1
            else: 
                car_features['radar']=0
            #check tempomat
            if ('tempomat' in extras) or ('távolságtartó tempomat') in extras:
                car_features['tempomat']=1
            else:
                car_features['tempomat']=0 
            #check boardcomputer
            if 'tempomat' in extras:
                car_features['boardcomputer']=1
            else:
                car_features['boardcomputer']=0
            #servocontrol
            if ('sebességfüggő szervókormány' in extras) or ('szervokormány' in extras):
                car_features['servocontrol']=1
            else:
                car_features['servocontrol']=0    

        except:
            pass
        all_cars.append(car_features)
    except:
            pass

df_cars_temp=pd.DataFrame.from_dict(all_cars)
#print(len(df_cars_temp))

#export 
df_cars_temp.to_csv('data/df_cars_temp.csv')

