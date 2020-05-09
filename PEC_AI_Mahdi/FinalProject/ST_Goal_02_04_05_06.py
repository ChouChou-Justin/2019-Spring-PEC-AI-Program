# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:16:28 2019

@author: ShinTing
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
le = preprocessing.LabelEncoder()    

EncodeDataset = pd.read_csv('suicide_rates_overview_1985_to_2016_EncodeDataset.csv')
#Goal2-------------------------
AGEdata = [[item,sum(EncodeDataset.loc[EncodeDataset['age_encode_']==item,'suicides/100k pop']),len(EncodeDataset.loc[EncodeDataset['age_encode_']==item,'suicides/100k pop'])] for item in range(6)]
AGE = pd.DataFrame({"age_encode":[AGEdata[i][0] for i in range(len(AGEdata))],
                    "age":['15-24 years','25-34 years','35-54 years','5-14 years','55-74 years','75+ years'],
                    "suicidesrate":[int(AGEdata[i][1]/AGEdata[i][2]) for i in range(len(AGEdata))]})

#print(AGE)
fig1, ax = plt.subplots()
xlabels=AGE['age']
plt.xlabel('Age')
plt.ylabel('Suicides Rate')
ax.set_xticklabels(xlabels, rotation=10)
plt.plot([AGE['age'][3],AGE['age'][0],AGE['age'][1],AGE['age'][2],AGE['age'][4],AGE['age'][5]],[AGE['suicidesrate'][3],AGE['suicidesrate'][0],AGE['suicidesrate'][1],AGE['suicidesrate'][2],AGE['suicidesrate'][4],AGE['suicidesrate'][5]])
plt.tight_layout()
plt.savefig("Goal2.png")
plt.show()
#Goal4-------------------------------
Generationdata = [[item,sum(EncodeDataset.loc[EncodeDataset['generation_encode_']==item,'suicides/100k pop']),len(EncodeDataset.loc[EncodeDataset['generation_encode_']==item,'suicides/100k pop'])] for item in range(6)]
print(Generationdata)
Generation = pd.DataFrame({"generation_encode_":[Generationdata[i][0] for i in range(len(Generationdata))],
                    "generation":['Boomers','G.I. Generation','Generation X','Generation Z','Millenials','Silent'],
                    "suicidesrate":[int(Generationdata[i][1]/Generationdata[i][2]) for i in range(len(Generationdata))]})
fig1, ax = plt.subplots()
xlabels=Generation['generation']
plt.xlabel('Generation')
plt.ylabel('Suicides Rate')
ax.set_xticklabels(xlabels, rotation=10)
plt.plot(Generation['generation'],Generation['suicidesrate'])
plt.tight_layout()
plt.savefig("Goal4.png")
plt.show()

#Goal5----------------------------
Continentdata = [[item,sum(EncodeDataset.loc[EncodeDataset['continent_encode_']==item,'suicides/100k pop']),
                  len(EncodeDataset.loc[EncodeDataset['continent_encode_']==item,'suicides/100k pop'])] for item in range(5)]
print(Continentdata)
Continent = pd.DataFrame({"continent_encode_":[Continentdata[i][0] for i in range(len(Continentdata))],
                    "continent":['Africa','America','Asia','Europe','Oceania'],
                    "suicidesrate":[int(Continentdata[i][1]/Continentdata[i][2]) for i in range(len(Continentdata))]})
fig1, ax = plt.subplots()
xlabels=Continent['continent']
plt.xlabel('Continent')
plt.ylabel('Suicides Rate')
ax.set_xticklabels(xlabels, rotation=10)
plt.plot(Continent['continent'],Continent['suicidesrate'])
plt.tight_layout()
plt.savefig("Goal5.png")
plt.show()
#Goal6----------------------------
GDPSuicideGeneration=[[EncodeDataset['suicides/100k pop'][i],EncodeDataset['gdp_per_capita ($)'][i],EncodeDataset['generation_encode_'][i]] for i in range(len(EncodeDataset))]
#GDPSuicideGeneration=[[EncodeDataset['suicides/100k pop'][i],EncodeDataset['suicides/100k pop ($)'][i]] for i in range(len(EncodeDataset))]
#print(GDPSuicideGeneration)
#
#fig1, ax = plt.subplots()

Boomers=[]
GIGeneration=[]
GenerationX=[]
GenerationZ=[]
Millenials=[]
Silent=[]
for i in range(len(GDPSuicideGeneration)):
    if GDPSuicideGeneration[i][2]==0:
        Boomers.append([ GDPSuicideGeneration[i][0],GDPSuicideGeneration[i][1] ])
    elif GDPSuicideGeneration[i][2]==1:
        GIGeneration.append([ GDPSuicideGeneration[i][0],GDPSuicideGeneration[i][1] ])
    elif GDPSuicideGeneration[i][2]==2:
        GenerationX.append([ GDPSuicideGeneration[i][0],GDPSuicideGeneration[i][1] ])
    elif GDPSuicideGeneration[i][2]==3:
        GenerationZ.append([ GDPSuicideGeneration[i][0],GDPSuicideGeneration[i][1] ])
    elif GDPSuicideGeneration[i][2]==4:
        Millenials.append([ GDPSuicideGeneration[i][0],GDPSuicideGeneration[i][1] ])
    elif GDPSuicideGeneration[i][2]==5:
        Silent.append([ GDPSuicideGeneration[i][0],GDPSuicideGeneration[i][1] ])        
print(len(Boomers),len(GIGeneration),len(GenerationX),len(Silent))
plt.scatter([item[0] for item in Boomers] ,[item[1] for item in Boomers],c='Red',label='Boomers')
my_x_ticks = np.arange(0,250,50)
my_y_ticks = np.arange(0,140000,20000)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('suicides/100k pop')
plt.ylabel('GDP_per_capita ($)')
plt.legend(loc=(1, 0))     
plt.savefig("Goal6_Boomers.png")
plt.show()
plt.scatter([item[0] for item in GIGeneration] ,[item[1] for item in GIGeneration],c='Orange',label='G.I.Generation')
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('suicides/100k pop')
plt.ylabel('GDP_per_capita ($)')
plt.legend(loc=(1, 0))     
plt.savefig("Goal6_GIGeneration.png")
plt.show()
plt.scatter([item[0] for item in GenerationX] ,[item[1] for item in GenerationX],c='Blue',label='GenerationX')
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('suicides/100k pop')
plt.ylabel('GDP_per_capita ($)')
plt.legend(loc=(1, 0))     
plt.savefig("Goal6_GenerationX.png")
plt.show()
plt.scatter([item[0] for item in GenerationZ] ,[item[1] for item in GenerationZ],c='Pink',label='GenerationZ')
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('suicides/100k pop')
plt.ylabel('GDP_per_capita ($)')
plt.legend(loc=(1, 0))     
plt.savefig("Goal6_GenerationZ.png")
plt.show()
plt.scatter([item[0] for item in Millenials] ,[item[1] for item in Millenials],c='Green',label='Millenials')
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('suicides/100k pop')
plt.ylabel('GDP_per_capita ($)')
plt.legend(loc=(1, 0))     
plt.savefig("Goal6_Millenials.png")
plt.show()
plt.scatter([item[0] for item in Silent] ,[item[1] for item in Silent],c='yellowgreen',label='Silent')
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.tight_layout()
plt.xlabel('suicides/100k pop')
plt.ylabel('GDP_per_capita ($)')
plt.legend(loc=(1, 0))     
plt.savefig("Goal6_Silent.png")
plt.show()


