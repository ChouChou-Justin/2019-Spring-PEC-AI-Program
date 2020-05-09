# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:26:19 2019

@author: ShinTing
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

EncodeDataset = pd.read_csv('suicide_rates_overview_1985_to_2016_EncodeDataset.csv')
Continentdata =  [[EncodeDataset['continent_encode_'][i],EncodeDataset['continent'][i],EncodeDataset['age_encode_'][i],
                   EncodeDataset['age'][i],EncodeDataset['suicides/100k pop'][i]]for i in range(len(EncodeDataset))]

Africa = [data for data in Continentdata if data[0]==0]
America = [data for data in Continentdata if data[0]==1]
Asia = [data for data in Continentdata if data[0]==2]
Europe = [data for data in Continentdata if data[0]==3]
Oceania = [data for data in Continentdata if data[0]==4]
def AgeDistube(lst):
    age_suicide = []
    for i in range(6):
        age_suicide.append(float(sum([data[4] for data in lst if data[2]==i])/len([data[4] for data in lst if data[2]==i])))
    return(age_suicide)
fig1, ax = plt.subplots(figsize=(10,10))



# original :['15-24 years','25-34 years','35-54 years','5-14 years','55-74 years','75+ years']
# encode [0,1,2,3,4,5]
xlabels=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']

plt.xlabel('Age')
plt.ylabel('Suicides Rate')
ax.set_xticklabels(xlabels, rotation=10)
plt.plot(xlabels,[AgeDistube(Africa)[3],AgeDistube(Africa)[0],AgeDistube(Africa)[1],AgeDistube(Africa)[2],AgeDistube(Africa)[4],AgeDistube(Africa)[5]],
                  c="B",label="Africa")
plt.plot(xlabels,[AgeDistube(America)[3],AgeDistube(America)[0],AgeDistube(America)[1],AgeDistube(America)[2],AgeDistube(America)[4],AgeDistube(America)[5]],
                  c="R",label="America")
plt.plot(xlabels,[AgeDistube(Asia)[3],AgeDistube(Asia)[0],AgeDistube(Asia)[1],AgeDistube(Asia)[2],AgeDistube(Asia)[4],AgeDistube(Asia)[5]],
                  c="G",label="Asia")
plt.plot(xlabels,[AgeDistube(Europe)[3],AgeDistube(Europe)[0],AgeDistube(Europe)[1],AgeDistube(Europe)[2],AgeDistube(Europe)[4],AgeDistube(Europe)[5]],
                  c="Orange",label="Europe")
plt.plot(xlabels,[AgeDistube(Oceania)[3],AgeDistube(Oceania)[0],AgeDistube(Oceania)[1],AgeDistube(Oceania)[2],AgeDistube(Oceania)[4],AgeDistube(Oceania)[5]],
                  c="PINK",label="Oceania")

plt.legend(loc=(1, 0))  
plt.tight_layout()
plt.savefig("Suicide_all_continent.png")
plt.show()

x=np.arange(6) 
total_width, n = 0.8, 5
width = total_width / n 
x = x - (total_width - width) / 2 
fig1, ax = plt.subplots(figsize=(10,10))
plt.bar(x,[AgeDistube(Africa)[3],AgeDistube(Africa)[0],AgeDistube(Africa)[1],AgeDistube(Africa)[2],AgeDistube(Africa)[4],AgeDistube(Africa)[5]],
                  label="Africa",width=width)
plt.bar(x+ width,[AgeDistube(America)[3],AgeDistube(America)[0],AgeDistube(America)[1],AgeDistube(America)[2],AgeDistube(America)[4],AgeDistube(America)[5]],
                  label="America",width=width)
plt.bar(x+ 2*width,[AgeDistube(Asia)[3],AgeDistube(Asia)[0],AgeDistube(Asia)[1],AgeDistube(Asia)[2],AgeDistube(Asia)[4],AgeDistube(Asia)[5]],
                  label="Asia",width=width)
plt.bar(x+ 3*width,[AgeDistube(Europe)[3],AgeDistube(Europe)[0],AgeDistube(Europe)[1],AgeDistube(Europe)[2],AgeDistube(Europe)[4],AgeDistube(Europe)[5]],
                  label="Europe",width=width)
plt.bar(x+ 4*width,[AgeDistube(Oceania)[3],AgeDistube(Oceania)[0],AgeDistube(Oceania)[1],AgeDistube(Oceania)[2],AgeDistube(Oceania)[4],AgeDistube(Oceania)[5]],
                  label="Oceania",width=width)

plt.xticks([0,1,2,3,4,5],[r'$5-14 years$', r'$15-24 years$', r'$25-34 years$',r'$35-54 years$', r'$55-74 years$', r'$75+ years$'])
plt.legend(loc=(1, 0))  
plt.tight_layout()
plt.savefig("Barchart_Suicide_all_continent.png") 
plt.show()
#Age_suicide-------------------------
#AGEdata = [[item,sum(EncodeDataset.loc[EncodeDataset['age_encode_']==item,'suicides/100k pop']),len(EncodeDataset.loc[EncodeDataset['age_encode_']==item,'suicides/100k pop'])] for item in range(6)]
#AGE = pd.DataFrame({"age_encode":[AGEdata[i][0] for i in range(len(AGEdata))],
#                    "age":['15-24 years','25-34 years','35-54 years','5-14 years','55-74 years','75+ years'],
#                    "suicidesrate":[int(AGEdata[i][1]/AGEdata[i][2]) for i in range(len(AGEdata))]})
#
##print(AGE)
#fig1, ax = plt.subplots()
#xlabels=AGE['age']
#plt.xlabel('Age')
#plt.ylabel('Suicides Rate')
##ax.set_xticklabels(xlabels, rotation=10)
#plt.plot([AGE['age'][3],AGE['age'][0],AGE['age'][1],AGE['age'][2],AGE['age'][4],AGE['age'][5]],[AGE['suicidesrate'][3],AGE['suicidesrate'][0],AGE['suicidesrate'][1],AGE['suicidesrate'][2],AGE['suicidesrate'][4],AGE['suicidesrate'][5]])
#plt.tight_layout()
#plt.savefig("Age_suicide.png")
#plt.show()
##Continent_suicide----------------------------
#Continentdata = [[item,sum(EncodeDataset.loc[EncodeDataset['continent_encode_']==item,'suicides/100k pop']),
#                  len(EncodeDataset.loc[EncodeDataset['continent_encode_']==item,'suicides/100k pop'])] for item in range(5)]
#Continent = pd.DataFrame({"continent_encode_":[Continentdata[i][0] for i in range(len(Continentdata))],
#                    "continent":['Africa','America','Asia','Europe','Oceania'],
#                    "suicidesrate":[int(Continentdata[i][1]/Continentdata[i][2]) for i in range(len(Continentdata))]})
#fig1, ax = plt.subplots()
#xlabels=Continent['continent']
#plt.xlabel('Continent')
#plt.ylabel('Suicides Rate')
##ax.set_xticklabels(xlabels, rotation=10)
#plt.plot(Continent['continent'],Continent['suicidesrate'])
#plt.tight_layout()
#plt.savefig("Continent_suicide.png")
#plt.show()
##Year_suicide----------------------------
#Yeardata = [[EncodeDataset['year'][i],EncodeDataset['suicides/100k pop'][i]]for i in range(len(EncodeDataset))]
#Year = [i+1985 for i in range(32)]
#Year_suicide=[]
#for i in Year:
#    Year_suicide.append(sum([data[1] for data in Yeardata if data[0]==i])/len([data[1] for data in Yeardata if data[0]==i]))
##print(Year_suicide) 
#fig1, ax = plt.subplots()
#xlabels=Year
#plt.xlabel('Year')
#plt.ylabel('Suicides Rate')
##ax.set_xticklabels(xlabels, rotation=10)
#plt.plot(Year,Year_suicide)
#plt.tight_layout()
#plt.savefig("Year_suicide.png")
#plt.show()
#    
#    
#    
#    
#    