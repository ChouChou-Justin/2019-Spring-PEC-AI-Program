# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:26:09 2019
@author: ChouHsingTing
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
startTime = time.time()
def suicideRate_and_population_versus_gdpPerCapita(file):
    for year in range(1985,2017):
        df = file[file.year==year]
        suicideRate = [float(i) for i in df['suicides/100k pop']] 
        country_year = [i for i in df['country-year']]  
        gdpPerCapita = [i for i in df['gdp_per_capita ($)']]
        population = [i for i in df['population']]
        totalPopulation = 0
        suicideMean = 0
        counter = 0
        i = 0
        populationRecord = []
        suicideMeanRecord = []
        gdpRecord = []
        for i in range(len(country_year)-1):
            if country_year[i]==country_year[i+1]:  
                totalPopulation += population[i]
                suicideMean +=  suicideRate[i]
                counter += 1
                if i==len(country_year)-2:
                    gdpRecord.append(gdpPerCapita[i])
                    suicideMean += suicideRate[i+1]
                    totalPopulation += population[i+1]
                    counter += 1
                    suicideMean /= counter
                    suicideMeanRecord.append(suicideMean)
                    populationRecord.append(totalPopulation)
                    counter = 0
                    suicideMean = 0
                    totalPopulation = 0
            else:
                gdpRecord.append(gdpPerCapita[i])
                totalPopulation += population[i]
                suicideMean += suicideRate[i]
                counter += 1
                suicideMean /= counter
                suicideMeanRecord.append(suicideMean)
                populationRecord.append(totalPopulation)
                counter = 0
                suicideMean = 0
                totalPopulation = 0 
        plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        plt.scatter(gdpRecord, suicideMeanRecord, c='gold', marker = 'o')    
        plt.xlabel('GDP Per Capita ($)')
        plt.ylabel('Average Suicide Rate (%)')
        plt.title('Suicide rate Versus GDP per capita in {}'.format(year))
        plt.subplot(1,2,2)
        plt.scatter(gdpRecord, populationRecord, c='red', marker = 'x')    
        plt.xlabel('GDP Per Capita ($)')
        plt.ylabel('Total Population')
        plt.title('Total Population Versus GDP per capita in {}'.format(year))
#        plt.savefig("{}.jpg".format(year))
        plt.show()        
def populationRatio_versus_gdpPerCapita_and_year(file):
    country = [i for i in file['country']] 
    countryRecord = []
    for i in range(len(country)-1):
        if country[i]!=country[i+1]:  
            countryRecord.append(country[i])
        if i==len(country)-2: 
            countryRecord.append(country[i])
    for countryEncode in range(101):
        df = file[file.country_encode_==countryEncode]              
        year = [i for i in df['year']]
        gdpPerCapita = [i for i in df['gdp_per_capita ($)']]
        malePopulation = 0
        femalePopulation = 0
        ratio = []
        maleRecord = []
        femaleRecord = []
        gdpRecord = []
        newYear = []
        for i in range(len(year)-1):
            if year[i]!=year[i+1]:
                newYear.append(year[i])
            elif i==len(year)-2:
                newYear.append(year[i])
            if gdpPerCapita[i]!=gdpPerCapita[i+1]:
                gdpRecord.append(gdpPerCapita[i])
            elif i==len(year)-2:
                gdpRecord.append(gdpPerCapita[i])
        for i in newYear:
            df2 = df[df.year==i]   
            sex = [i for i in df2['sex']]
            population = [i for i in df2['population']]
            for i in range(len(sex)):
                if sex[i]=='male':  
                    malePopulation += population[i]
                else:
                    femalePopulation +=  population[i]
            maleRecord.append(malePopulation)
            femaleRecord.append(femalePopulation)
            ratio.append(malePopulation/femalePopulation)
            malePopulation = 0
            femalePopulation = 0
        plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        plt.scatter(newYear, ratio, c = 'gold', marker = '*')
        plt.xlabel('Year')
        plt.ylabel('Population Ratio (Male/Female)')
        plt.title('Male Versus Female in {}'.format(countryRecord[countryEncode]))
        plt.subplot(1,2,2)
        plt.scatter(gdpRecord, ratio, c = 'purple', marker = 'v')
        plt.xlabel('GDP Per Capita ($)')
        plt.ylabel('Ratio (Male/Female)')
        plt.title('Population Ratio Versus GDP per capita in {}'.format(countryRecord[countryEncode]))
#        plt.savefig("{}.jpg".format(countryRecord[countryEncode]))
        plt.show()
def main():
    file = pd.read_csv('suicide_rates_overview_1985_to_2016_EncodeDataset.csv')
    suicideRate_and_population_versus_gdpPerCapita(file)
    populationRatio_versus_gdpPerCapita_and_year(file)
main()
runningTime = time.time() - startTime
print ("time: {}".format(runningTime))