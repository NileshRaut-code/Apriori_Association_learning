import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#header us this to header =none to make the titile of data to no from string
dataset=pd.read_csv("c:/Users/A Kumar/Desktop/ml2/items_purchased_data.csv",header=None)
#list 
transaction=[]
for i in range(0,7501):#str us to save string value 
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])

#Traing the Aporiro Data set to get recommendation
from apyori import apriori
rules =apriori(transaction ,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#visulising the result

results=list(rules)#this containes list of rules

print(rules.head())




