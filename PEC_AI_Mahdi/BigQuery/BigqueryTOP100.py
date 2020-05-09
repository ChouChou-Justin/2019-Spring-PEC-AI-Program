# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:04:58 2019

@author: ShinTing
"""

import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1
import pandas as pd  
import numpy as np  

D = np.array([['Name', 'Gender', 'No']])
client = bigquery.Client(project="graphic-avenue-240206")
#2017
query_job2017M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2017`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)
results = query_job2017M.result()  # Waits for job to complete.
for row in results:
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0)
        #print(Year,row.name,row.gender,row.count)
     
query_job2017F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2017`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2017F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0)
#2016        
query_job2016M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2016`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2016M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0)  
query_job2016F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2016`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2016F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0)     
#2015        
query_job2015M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2015`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2015M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
        
query_job2015F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2015`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2015F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
#2014
query_job2014M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2014`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2014M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
        
query_job2014F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2014`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2014F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0)       
#2013
query_job2013M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2013`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2013M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
        
query_job2013F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2013`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2013F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
#2012 
query_job2012M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2012`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2012M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
        
query_job2012F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2012`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2012F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
#2011        
query_job2011M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2011`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2011M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
        
query_job2011F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2011`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2011F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
#2010
query_job2010M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2010`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2010M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
        
query_job2010F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2010`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2010F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
#2009
query_job2009M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2009`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2009M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
        
query_job2009F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2009`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2009F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
#2008       
query_job2008M = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2008`
    WHERE gender = 'M'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2008M.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 
        
query_job2008F = client.query("""     
    (SELECT name,gender,count
    FROM `graphic-avenue-240206.babynames.BN_2008`
    WHERE gender = 'F'
    ORDER BY count DESC
    LIMIT 100)
    """)  
for row in query_job2008F.result():
        D = np.append(D,[[row.name,row.gender,row.count]], axis=0) 

year2017=['2017' for i in range(0,200)]
year2016=['2016' for i in range(0,200)]
year2015=['2015' for i in range(0,200)]
year2014=['2014' for i in range(0,200)]
year2013=['2013' for i in range(0,200)]
year2012=['2012' for i in range(0,200)]
year2011=['2011' for i in range(0,200)]
year2010=['2010' for i in range(0,200)]
year2009=['2009' for i in range(0,200)]
year2008=['2008' for i in range(0,200)]

year=np.concatenate((year2017,year2016,year2015,year2014,year2013,
                     year2012,year2011,year2010,year2009,year2008),axis=0)

df=pd.DataFrame(D[1:],columns=D[0])  
df['Year']=year
df = df.reindex(columns=['Year','Name', 'Gender', 'No'])

#print(df)
df.to_csv('BigqueryTOP100NameFrom2017.csv',index=False) 











