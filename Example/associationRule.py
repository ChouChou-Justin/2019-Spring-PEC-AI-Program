import tensorflow as tf 
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import networkx as nx
from apyori import apriori 
#pip install apriori
from wordcloud import WordCloud 
#pip install wordcloud

def testTensorflow():
    hello = tf.constant('hello tensorflow!') 
    sess = tf.Session()
    print("hello") 
    print(sess.run(hello))
    
#conda install -c conda-forge wordcloud
#pip install wordcloud
def wordCloud():
    plt.figure(figsize=(9,6)) 
    data=np.array([
      ['Milk','Bread','Apple'],
      ['Milk','Bread'],
      ['Milk','Bread','Apple', 'Banana'],
      ['Milk', 'Banana','Rice','Chicken'],
      ['Apple','Rice','Chicken'],
      ['Milk','Bread', 'Banana'],
      ['Rice','Chicken'],
      ['Bread','Apple', 'Chicken'],
      ['Bread','Chicken'],
      ['Apple', 'Banana']])
    #convert the array to text 
    text_data=[]
    for i in data:
        for j in i:
            text_data.append(j)    
    products=' '.join(map(str, text_data))
    print(products) 
    wordcloud = WordCloud(relative_scaling = 1.0,stopwords = {}).generate(products)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def draw(df):
    plt.style.use('ggplot')
    plt.figure(figsize=(9,6))
    print(df.iloc[6:19][['items','support']]) # Only get items with two pair sets. They start from index 6 to 19
    ar=(df.iloc[6:19]['items'])
    G = nx.Graph()
    G.add_edges_from(ar)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, font_size=16, with_labels=False, edge_color='green',node_size=800,node_color=['red','green','blue','cyan','orange','magenta'])
    for p in pos:  
        pos[p][1] += 0.07
    nx.draw_networkx_labels(G, pos)
    plt.show()

def simple_bar_chart(support,products):
    labels=np.array(products)
    colors = ['#008000','#808000','#FFFF00','#000000','#FF0000','#00FF00','#0000FF','#008080','#aa22ff','#aa22ff','#dd0022','#ff00cc','#eeaa22','#22bbaa','#C0C0C0']
    y_pos = np.arange(len(labels))
    x_pos = np.array(support)
    plt.barh(y_pos, x_pos, color=colors, align='center' ,edgecolor='green')
    plt.yticks(y_pos, labels)
    plt.ylabel('Products',fontsize=18)
    plt.xlabel('Support',fontsize=18)
    plt.title('Consumer Buying Behaviour\n',fontsize=20)
    plt.show()
    
def testApriori_s():        
    data=np.array([
               ['Milk','Bread','Apple'],
               ['Milk','Bread'],
               ['Milk','Bread','Apple', 'Banana'],
               ['Milk', 'Banana','Rice','Chicken'],
               ['Apple','Rice','Chicken'],
               ['Milk','Bread', 'Banana'],
               ['Rice','Chicken'],
               ['Bread','Apple', 'Chicken'],
               ['Bread','Chicken'],
               ['Apple', 'Banana']])
    for i in data:
        print(i)
    print("\n\n")
    result=list(apriori(data))
    df=pd.DataFrame(result)
    df.to_csv("appriori_results.csv") #Save to csv formart for detailed view
    print(df.head()) # Print the first 5 items
    #print(df) 
    draw(df)
    support=df.iloc[0:19]['support']*100
    products=df.iloc[0:19]['items']
    simple_bar_chart(support,products)

def testApriori():    
    records = []      
    store_data = pd.read_csv('D:\\SpyderProject\\Example\\store_data.csv', header=None)  
    #print(store_data)
    print(store_data.head())
    #perprocessing
    #convert our pandas dataframe into a list of lists
    for i in range(0, 7501):  
        #records.append([str(store_data.values[i,j]) for j in range(0, 20)])  
        records.append([str(store_data.values[i,j]) for j in range(0, 20) if str(store_data.values[i,j]) != 'nan'])
        # remove NaN value
    #print(records)  
    association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)  
    #min_length: at least 2 product in the rules
    association_results = list(association_rules)  
    print(len(association_results))  
    #print(association_results)  
    print(association_results[0])      

    for item in association_results:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0] 
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])
        #second index of the inner list
        print("Support: " + str(item[1]))
        #third index of the list located at 0th
        #of the third index of the inner list
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")    
    
def main():
    
    #testApriori_s()
    #testApriori()
    wordCloud()

main()    