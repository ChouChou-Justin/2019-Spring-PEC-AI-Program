# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:32:18 2019

@author: ShinTing Wu, HsinTing Chou, ChiaWei Hsu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from collections import Counter
from sklearn.cluster import KMeans
from pylab import pcolor,colorbar,bone
from mpl_toolkits.mplot3d import Axes3D

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

xlabels=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']
x=np.arange(6) 
total_width, n = 0.8, 5
width = total_width / n 
x = x - (total_width - width) / 2 
fig1, ax = plt.subplots(figsize=(8,8))
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
#------------------------------------------------------------------------------
EncodeData = [[EncodeDataset['country'][i],EncodeDataset['year'][i],EncodeDataset['age_encode_'][i],
                   EncodeDataset['age'][i],EncodeDataset['suicides/100k pop'][i]]for i in range(len(EncodeDataset))]
for i in EncodeData:
    if i[2] == 3: i[2] = 0
    elif i[2] == 0: i[2] = 0
    elif i[2] == 1: i[2] = 1
    elif i[2] == 2: i[2] = 1
    elif i[2] == 4: i[2] = 2
    elif i[2] == 5: i[2] = 2
Country=['Albania','Antigua and Barbuda','Argentina','Armenia','Aruba','Australia','Austria','Azerbaijan',
         'Bahamas','Bahrain','Barbados','Belarus','Belgium','Belize','Bosnia and Herzegovina','Brazil','Bulgaria','Cabo Verde',
         'Canada','Chile','Colombia','Costa Rica','Croatia','Cuba','Cyprus','Czech Republic','Denmark','Dominica','Ecuador',
         'El Salvador','Estonia','Fiji','Finland','France','Georgia','Germany','Greece','Grenada','Guatemala','Guyana',
         'Hungary','Iceland','Ireland','Israel','Italy','Jamaica','Japan','Kazakhstan','Kiribati','Kuwait','Kyrgyzstan',
         'Latvia','Lithuania','Luxembourg','Macau','Maldives','Malta','Mauritius','Mexico','Mongolia','Montenegro',
         'Netherlands','New Zealand','Nicaragua','Norway,Oman','Panama','Paraguay','Philippines','Poland','Portugal',
         'Puerto Rico','Qatar','Republic of Korea','Romania','Russian Federation','Saint Kitts and Nevis','Saint Lucia',
         'Saint Vincent and Grenadines','San Marino','Serbia','Seychelles','Singapore','Slovakia','Slovenia','South Africa',
         'Spain','Sri Lanka','Suriname','Sweden','Switzerland','Thailand','Trinidad and Tobago','Turkey','Turkmenistan',
         'Ukraine','United Arab Emirates','United Kingdom','United States','Uruguay','Uzbekistan']   

BeforeMillenial=[data for data in EncodeData if data[1] >=1985 and data[1] < 2000]
AfterMillenial=[data for data in EncodeData if data[1] >=2000 ]

CountryBeforeMillenial = []   
for eachcountry in Country:
    Suicide_AGE1 = []
    Suicide_AGE2 = []
    Suicide_AGE3 = []
    for index,item in enumerate(BeforeMillenial): 
        if(item[2]==0 and item[0] == eachcountry):
            Suicide_AGE1.append(item[4])
        if(item[2]==1 and item[0] == eachcountry):
            Suicide_AGE2.append(item[4])
        if(item[2]==2 and item[0] == eachcountry):
            Suicide_AGE3.append(item[4])    
    if(Suicide_AGE1 != [] and Suicide_AGE2 != [] and Suicide_AGE3 != []): 
        CountryBeforeMillenial.append(eachcountry)
    Suicide_AGE1 = []
    Suicide_AGE2 = []
    Suicide_AGE3 = []

CountryAfterMillenial = []
for eachcountry in CountryBeforeMillenial:
    Suicide_AGE1 = []
    Suicide_AGE2 = []
    Suicide_AGE3 = []
    for index,item in enumerate(AfterMillenial): 
        if(item[2]==0 and item[0] == eachcountry):
            Suicide_AGE1.append(item[4])
        if(item[2]==1 and item[0] == eachcountry):
            Suicide_AGE2.append(item[4])
        if(item[2]==2 and item[0] == eachcountry):
            Suicide_AGE3.append(item[4])    
    if(Suicide_AGE1 != [] and Suicide_AGE2 != [] and Suicide_AGE3 != []):
        CountryAfterMillenial.append(eachcountry)
    Suicide_AGE1 = []
    Suicide_AGE2 = []
    Suicide_AGE3 = []    

BeforeMillenial_AGE1=[]
BeforeMillenial_AGE2=[]
BeforeMillenial_AGE3=[]
AfterMillenial_AGE1=[]
AfterMillenial_AGE2=[]
AfterMillenial_AGE3=[]    

s1 = set(CountryBeforeMillenial)
s2 = set(CountryAfterMillenial)
AllCountry=list(s1.intersection(s2))

for eachcountry in AllCountry:
    Suicide_AGE1 = []
    Suicide_AGE2 = []
    Suicide_AGE3 = []
    for index,item in enumerate(BeforeMillenial): 
        if(item[2]==0 and item[0] == eachcountry):
            Suicide_AGE1.append(item[4])
        if(item[2]==1 and item[0] == eachcountry):
            Suicide_AGE2.append(item[4])
        if(item[2]==2 and item[0] == eachcountry):
            Suicide_AGE3.append(item[4])    
    if(Suicide_AGE1 != [] and Suicide_AGE2 != [] and Suicide_AGE3 != []):
        BeforeMillenial_AGE1.append([eachcountry,float(sum(Suicide_AGE1)/len(Suicide_AGE1))])     
        BeforeMillenial_AGE2.append([eachcountry,float(sum(Suicide_AGE2)/len(Suicide_AGE2))])     
        BeforeMillenial_AGE3.append([eachcountry,float(sum(Suicide_AGE3)/len(Suicide_AGE3))]) 
    Suicide_AGE1 = []
    Suicide_AGE2 = []
    Suicide_AGE3 = []    

for eachcountry in AllCountry:
    Suicide_AGE1 = []
    Suicide_AGE2 = []
    Suicide_AGE3 = []
    for index,item in enumerate(AfterMillenial): 
        if(item[2]==0 and item[0] == eachcountry):
            Suicide_AGE1.append(item[4])
        if(item[2]==1 and item[0] == eachcountry):
            Suicide_AGE2.append(item[4])
        if(item[2]==2 and item[0] == eachcountry):
            Suicide_AGE3.append(item[4])    
    if(Suicide_AGE1 != [] and Suicide_AGE2 != [] and Suicide_AGE3 != []):
        AfterMillenial_AGE1.append([eachcountry,float(sum(Suicide_AGE1)/len(Suicide_AGE1))])     
        AfterMillenial_AGE2.append([eachcountry,float(sum(Suicide_AGE2)/len(Suicide_AGE2))])     
        AfterMillenial_AGE3.append([eachcountry,float(sum(Suicide_AGE3)/len(Suicide_AGE3))]) 
    Suicide_AGE1 = []
    Suicide_AGE2 = []
    Suicide_AGE3 = []    

AllData = []
for i in range(len(AllCountry)):
    AllData.append([BeforeMillenial_AGE1[i][1],BeforeMillenial_AGE2[i][1],BeforeMillenial_AGE3[i][1],
                    AfterMillenial_AGE1[i][1],AfterMillenial_AGE2[i][1],AfterMillenial_AGE3[i][1]])

data=np.array(AllData)

fig, ax = plt.subplots(figsize=(15,10))
xlabels=['Before Millenial 5-24 years','Before Millenial 25-55 years','Before Millenial 55+ years',
         'After Millenial 5-24 years','After Millenial 25-55 years','After Millenial  55+ years']
plt.xlabel('Age Group',fontsize=15)
#ax.set_xticklabels(xlabels, rotation=10)
plt.ylabel('Suicides Rate',fontsize=15)

for country_index,country in enumerate(AllCountry) :
    plt.plot(xlabels,AllData[country_index],label = country)

plt.legend(loc=(1, 0))  
#plt.tight_layout()
plt.savefig("AllCountrySuicide_colorful.png")
plt.show()

fig, ax = plt.subplots(figsize=(15,10))
xlabels=['Before Millenial 5-24 years','Before Millenial 25-55 years','Before Millenial 75+ years',
         'After Millenial 5-24 years','After Millenial 25-55 years','After Millenial  75+ years']
plt.xlabel('Age',fontsize=15)
#ax.set_xticklabels(xlabels, rotation=10)
plt.ylabel('Suicides Rate',fontsize=15)

for country_index,country in enumerate(AllCountry) :
    plt.plot(xlabels,AllData[country_index],c="b")
plt.tight_layout()
plt.savefig("AllCountrySuicide.png")
plt.show()


### Initialization and training ###
som = MiniSom(10,10,6,sigma=1.0,learning_rate=0.5)
som.random_weights_init(AllData)
som.train_random(AllData,10000) # training with 100 iterations

fig, ax = plt.subplots(figsize=(10,10))
bone()
pcolor(som.distance_map().T) # distance map as background
colorbar()
weightlist =[]
countrylist1=[]
countrylist2=[]
countryNumberList1=[]
countryNumberList2=[]
for cnt,xx in  enumerate(data):
    w = som.winner(xx) # getting the winner
    weightlist.append(w)

highcounts = Counter(weightlist)
top_three = highcounts.most_common(10)
#print(top_three)
#print("activition\n",som.activation_response(data))
for cnt,xx in enumerate(data):
    w = som.winner(xx) # getting the winner
#    print(w,xx)
    if (w[0]==top_three[1][0][0] and w[1]==top_three[1][0][1]):
        ax.text(w[0]+.25,w[1]+.5,str(AllCountry[cnt]),color='R',fontsize=20)
        countrylist1.append(AllCountry[cnt])
        countryNumberList1.append(cnt)
    elif (w[0]==top_three[2][0][0] and w[1]==top_three[2][0][1]):
        ax.text(w[0]+.25,w[1]+.5,str(AllCountry[cnt]),color='cyan',fontsize=20)    
        countrylist2.append(AllCountry[cnt])
        countryNumberList2.append(cnt)
    else:    ax.text(w[0]+.25,w[1]+.5,str(AllCountry[cnt]),color='darkgreen')

#print("Red word :\n",countrylist1)
#print("Blue word :\n",countrylist2)

ax.axis([0,som._weights.shape[0],0,som._weights.shape[1]])
plt.tight_layout()
plt.savefig("ST_miniSOM.png")
plt.show()

Bcolor = ['blue','darkblue','aqua','lightskyblue','mediumblue','mediumslateblue','SlateBlue','LightBlue','cyan']
Rcolor = ['Firebrick','red','Tomato','Pink','Orange','Magenta','Brown','Maroon','Tan']
fig1, ax = plt.subplots(figsize=(15,10))
xlabels=['Before Millenial 5-24 years','Before Millenial 25-55 years','Before Millenial 75+ years',
         'After Millenial 5-24 years','After Millenial 25-55 years','After Millenial  75+ years']
plt.xlabel('Age Group',fontsize=15)
plt.ylabel('Suicides Rate',fontsize=15)

for index,AllC in enumerate(AllCountry):
    for i,c1 in enumerate(countrylist1):
        if(str(AllC) == str(c1)):
#            print(AllC,c1)
#            print(AllC,AllData[index])
            plt.plot(xlabels,AllData[index],c=Rcolor[i],label = AllC)
    for j,c2 in enumerate(countrylist2):
         if(str(AllC) == str(c2)):
#            print(AllC,c2)
#            print(AllC,AllData[index])
            plt.plot(xlabels,AllData[index],c=Bcolor[j],label = AllC)
plt.legend(loc=(1, 0))  
plt.tight_layout()
plt.savefig("cluster.png")
plt.show()
#------------------------------------------------------------------------------
#k-means
clustersNumber = 5
AllData_transposeGet = list(map(list, zip(*AllData)))
kmeans = KMeans(n_clusters=clustersNumber, init = 'random', n_init=1)
kmeans = kmeans.fit(AllData)
labels = kmeans.predict(AllData)
centroids = kmeans.cluster_centers_
print('centroids are:\n', centroids)
colorList=[]
for i in range(len(labels)):
    if labels[i]==0:
        colorList.append('red')
    elif labels[i]==1:
        colorList.append('orange')        
    elif labels[i]==2:
        colorList.append('green')        
    elif labels[i]==3:
        colorList.append('cyan')        
    elif labels[i]==4:
        colorList.append('blue')       
    elif labels[i]==5:
        colorList.append('black')       
    elif labels[i]==6:
        colorList.append('purple')       
    elif labels[i]==7:
        colorList.append('Brown')        
    elif labels[i]==8:
        colorList.append('Maroon')        
    elif labels[i]==9:
        colorList.append('Pink')        
checkColor1=[]
checkColor2=[]       
for i in range(len(colorList)):
    if i in countryNumberList1:
        checkColor1.append(colorList[i])
    elif i in countryNumberList2:
        checkColor2.append(colorList[i])
#------------------------------------------------------------------------------
#before Millenial k-means
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(AllData_transposeGet[0], AllData_transposeGet[1], AllData_transposeGet[2], c=colorList)
ax.set_xlabel('5-24 years old before Millenial')
ax.set_ylabel('25-54 years old before Millenial')
ax.set_zlabel('55+ years old before Millenial')
ax.set_xlim3d(0,25)
ax.set_ylim3d(0,60)
ax.set_zlim3d(0,80)
plt.savefig("k-means result before Millenial")
plt.show()
print(countrylist1)
print(countrylist2)
x11 = [AllData_transposeGet[0][i] for i in countryNumberList1]
y11 = [AllData_transposeGet[1][i] for i in countryNumberList1]
z11 = [AllData_transposeGet[2][i] for i in countryNumberList1]
x12 = [AllData_transposeGet[0][i] for i in countryNumberList2]
y12 = [AllData_transposeGet[1][i] for i in countryNumberList2]
z12 = [AllData_transposeGet[2][i] for i in countryNumberList2]
#before Millenial k-means proof SOM
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x11, y11, z11, c=checkColor1, label=countrylist1)
ax.legend(loc='upper right')
ax.set_xlabel('5-24 years old before Millenial')
ax.set_ylabel('25-54 years old before Millenial')
ax.set_zlabel('55+ years old before Millenial')
ax.set_xlim3d(0,25)
ax.set_ylim3d(0,60)
ax.set_zlim3d(0,80)
ax.scatter(x12, y12, z12, c=checkColor2, label=countrylist2)
ax.legend(loc='upper right')
ax.set_xlabel('5-24 years old before Millenial')
ax.set_ylabel('25-54 years old before Millenial')
ax.set_zlabel('55+ years old before Millenial')
ax.set_xlim3d(0,25)
ax.set_ylim3d(0,60)
ax.set_zlim3d(0,80)
plt.savefig("proof result before Millenial")
plt.show()
#------------------------------------------------------------------------------
#after Millenial k-means
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(AllData_transposeGet[3], AllData_transposeGet[4], AllData_transposeGet[5], c=colorList)
ax.set_xlabel('5-24 years old after Millenial')
ax.set_ylabel('25-54 years old after Millenial')
ax.set_zlabel('55+ years old after Millenial')
ax.set_xlim3d(0,25)
ax.set_ylim3d(0,60)
ax.set_zlim3d(0,80)
plt.savefig("k-means result after Millenial")
plt.show()
print(countrylist1)
print(countrylist2)
x21 = [AllData_transposeGet[3][i] for i in countryNumberList1]
y21 = [AllData_transposeGet[4][i] for i in countryNumberList1]
z21 = [AllData_transposeGet[5][i] for i in countryNumberList1]
x22 = [AllData_transposeGet[3][i] for i in countryNumberList2]
y22 = [AllData_transposeGet[4][i] for i in countryNumberList2]
z22 = [AllData_transposeGet[5][i] for i in countryNumberList2]
#after Millenial k-means proof SOM
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x21, y21, z21, c=checkColor1, label=countrylist1)
ax.legend(loc='upper right')
ax.set_xlabel('5-24 years old after Millenial')
ax.set_ylabel('25-54 years old after Millenial')
ax.set_zlabel('55+ years old after Millenial')
ax.set_xlim3d(0,25)
ax.set_ylim3d(0,60)
ax.set_zlim3d(0,80)
ax.scatter(x22, y22, z22, c=checkColor2, label=countrylist2)
ax.legend(loc='upper right')
ax.set_xlabel('5-24 years old after Millenial')
ax.set_ylabel('25-54 years old after Millenial')
ax.set_zlabel('55+ years old after Millenial')
ax.set_xlim3d(0,25)
ax.set_ylim3d(0,60)
ax.set_zlim3d(0,80)
plt.savefig("proof result after Millenial")
plt.show()