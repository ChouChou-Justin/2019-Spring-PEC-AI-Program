# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:00:21 2019

@author: Jarwy
"""


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydotplus
import random
import math
import time


startTime = time.time()
file = pd.read_csv("suicide_rates_overview_1985_to_2016_EncodeDataset.csv")
Africa = file[file.continent_encode_==0]
America = file[file.continent_encode_==1]
Asia = file[file.continent_encode_==2]
Europe = file[file.continent_encode_==3]
Oceania = file[file.continent_encode_==4]


def sketch_proportion(data,year):
    data_year=data[data.year==year]
    data_m = data_year[data_year.sex=="male"]
    data_f = data_year[data_year.sex=="female"]
    X=data_m.population.sum()
#    print(X)
    Y=data_f.population.sum()
#    print(Y)
    proportion=Y/X
#    print(proportion)
    plt.scatter(year,proportion,c="r")
    plt.xlabel("Year")
    plt.ylabel("Propotion(female/male)")  
    
    notch_x=[i for i in range(1985,2017,5)]
    plt.xticks(notch_x)
    

#=============================================================================#
###AFRICA###
sketch_proportion(Africa,1985)
sketch_proportion(Africa,1986)
sketch_proportion(Africa,1987)    
sketch_proportion(Africa,1988)    
sketch_proportion(Africa,1989)    
sketch_proportion(Africa,1990)    
sketch_proportion(Africa,1991)    
sketch_proportion(Africa,1992) 
sketch_proportion(Africa,1993) 
sketch_proportion(Africa,1994) 
sketch_proportion(Africa,1995) 
sketch_proportion(Africa,1996) 
sketch_proportion(Africa,1997) 
sketch_proportion(Africa,1998)  
sketch_proportion(Africa,1999) 
sketch_proportion(Africa,2000) 
sketch_proportion(Africa,2001) 
sketch_proportion(Africa,2002) 
sketch_proportion(Africa,2003)
sketch_proportion(Africa,2004) 
sketch_proportion(Africa,2005) 
sketch_proportion(Africa,2006) 
sketch_proportion(Africa,2007) 
sketch_proportion(Africa,2008)     
sketch_proportion(Africa,2009)     
sketch_proportion(Africa,2010)  
sketch_proportion(Africa,2011)
sketch_proportion(Africa,2012)
sketch_proportion(Africa,2013)  
sketch_proportion(Africa,2014)
sketch_proportion(Africa,2015)
sketch_proportion(Africa,2016)
plt.title("Female to male ratio in in Africa from 1985 to 2016")

notch_y=[i for i in np.arange(0.96,1.08,0.02)]
plt.yticks(notch_y)
plt.show()
#=============================================================================#
#=============================================================================#
###AMERICA###
sketch_proportion(America,1985)
sketch_proportion(America,1986)
sketch_proportion(America,1987)    
sketch_proportion(America,1988)    
sketch_proportion(America,1989)    
sketch_proportion(America,1990)    
sketch_proportion(America,1991)
sketch_proportion(America,1992) 
sketch_proportion(America,1993) 
sketch_proportion(America,1994) 
sketch_proportion(America,1995) 
sketch_proportion(America,1996) 
sketch_proportion(America,1997) 
sketch_proportion(America,1998)  
sketch_proportion(America,1999) 
sketch_proportion(America,2000) 
sketch_proportion(America,2001) 
sketch_proportion(America,2002) 
sketch_proportion(America,2003)
sketch_proportion(America,2004) 
sketch_proportion(America,2005) 
sketch_proportion(America,2006) 
sketch_proportion(America,2007) 
sketch_proportion(America,2008)     
sketch_proportion(America,2009)     
sketch_proportion(America,2010)  
sketch_proportion(America,2011)
sketch_proportion(America,2012)
sketch_proportion(America,2013)  
sketch_proportion(America,2014)
sketch_proportion(America,2015)
sketch_proportion(America,2016)
plt.title("Female to male ratio in America from 1985 to 2016")

notch_y=[i for i in np.arange(1.02,1.06,0.02)]
plt.yticks(notch_y)
plt.show()
#=============================================================================#
#=============================================================================#
###ASIA###
sketch_proportion(Asia,1985)
sketch_proportion(Asia,1986)
sketch_proportion(Asia,1987)    
sketch_proportion(Asia,1988)    
sketch_proportion(Asia,1989)    
sketch_proportion(Asia,1990)    
sketch_proportion(Asia,1991)
sketch_proportion(Asia,1992) 
sketch_proportion(Asia,1993) 
sketch_proportion(Asia,1994) 
sketch_proportion(Asia,1995) 
sketch_proportion(Asia,1996) 
sketch_proportion(Asia,1997) 
sketch_proportion(Asia,1998)  
sketch_proportion(Asia,1999) 
sketch_proportion(Asia,2000) 
sketch_proportion(Asia,2001) 
sketch_proportion(Asia,2002) 
sketch_proportion(Asia,2003)
sketch_proportion(Asia,2004) 
sketch_proportion(Asia,2005) 
sketch_proportion(Asia,2006) 
sketch_proportion(Asia,2007) 
sketch_proportion(Asia,2008)     
sketch_proportion(Asia,2009)     
sketch_proportion(Asia,2010)  
sketch_proportion(Asia,2011)
sketch_proportion(Asia,2012)
sketch_proportion(Asia,2013)  
sketch_proportion(Asia,2014)
sketch_proportion(Asia,2015)
sketch_proportion(Asia,2016)
plt.title("Female to male ratio in Asia from 1985 to 2016")

notch_y=[i for i in np.arange(0.96,1.08,0.02)]
plt.yticks(notch_y)
plt.show()
#=============================================================================#
#=============================================================================#
###EUROPE###
sketch_proportion(Europe,1985)
sketch_proportion(Europe,1986)
sketch_proportion(Europe,1987)    
sketch_proportion(Europe,1988)    
sketch_proportion(Europe,1989)    
sketch_proportion(Europe,1990)    
sketch_proportion(Europe,1991)
sketch_proportion(Europe,1992) 
sketch_proportion(Europe,1993) 
sketch_proportion(Europe,1994) 
sketch_proportion(Europe,1995) 
sketch_proportion(Europe,1996) 
sketch_proportion(Europe,1997) 
sketch_proportion(Europe,1998)  
sketch_proportion(Europe,1999) 
sketch_proportion(Europe,2000) 
sketch_proportion(Europe,2001) 
sketch_proportion(Europe,2002) 
sketch_proportion(Europe,2003)
sketch_proportion(Europe,2004) 
sketch_proportion(Europe,2005) 
sketch_proportion(Europe,2006) 
sketch_proportion(Europe,2007) 
sketch_proportion(Europe,2008)     
sketch_proportion(Europe,2009)     
sketch_proportion(Europe,2010)  
sketch_proportion(Europe,2011)
sketch_proportion(Europe,2012)
sketch_proportion(Europe,2013)  
sketch_proportion(Europe,2014)
sketch_proportion(Europe,2015)
sketch_proportion(Europe,2016)
plt.title("Female to male ratio in Europe from 1985 to 2016")

notch_y=[i for i in np.arange(1.04,1.10,0.02)]
plt.yticks(notch_y)
plt.show()
#=============================================================================#
#=============================================================================#
###OCEANIA###
sketch_proportion(Oceania,1985)
sketch_proportion(Oceania,1986)
sketch_proportion(Oceania,1987)    
sketch_proportion(Oceania,1988)    
sketch_proportion(Oceania,1989)    
sketch_proportion(Oceania,1990)    
sketch_proportion(Oceania,1991)
sketch_proportion(Oceania,1992) 
sketch_proportion(Oceania,1993) 
sketch_proportion(Oceania,1994) 
sketch_proportion(Oceania,1995) 
sketch_proportion(Oceania,1996) 
sketch_proportion(Oceania,1997) 
sketch_proportion(Oceania,1998)  
sketch_proportion(Oceania,1999) 
sketch_proportion(Oceania,2000) 
sketch_proportion(Oceania,2001) 
sketch_proportion(Oceania,2002) 
sketch_proportion(Oceania,2003)
sketch_proportion(Oceania,2004) 
sketch_proportion(Oceania,2005) 
sketch_proportion(Oceania,2006) 
sketch_proportion(Oceania,2007) 
sketch_proportion(Oceania,2008)     
sketch_proportion(Oceania,2009)     
sketch_proportion(Oceania,2010)  
sketch_proportion(Oceania,2011)
sketch_proportion(Oceania,2012)
sketch_proportion(Oceania,2013)  
sketch_proportion(Oceania,2014)
sketch_proportion(Oceania,2015)
sketch_proportion(Oceania,2016) 
plt.title("Female to male ratio in Oceania from 1985 to 2016")

notch_y=[i for i in np.arange(1.00,1.04,0.02)]
plt.yticks(notch_y)
plt.show()
#=============================================================================#

runningTime = time.time() - startTime
print ("time: {}".format(runningTime))
    

