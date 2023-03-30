import time
from tkinter.tix import Tree
import pandas as pd
import json
from math import sqrt
import scipy.stats as stats
import sklearn.metrics as metrics
import numpy as np


def preprocess(DatasetLocation):


	df = pd.DataFrame(pd.read_csv(DatasetLocation,low_memory=False))  #this will change the csv dataset to pandas dataframe
	df = df.drop(["Timestamp"], axis=1)
	df = df.drop(["Dst Port"], axis=1) # drop these 2 columns 

	df1 = pd.get_dummies(df["Protocol"]) #one-hot encoding
	df1 = pd.concat((df,df1), axis=1)	

	df1 = df1.drop(["Protocol"], axis=1)  #drop the protocol feature after one hot encoding it 

	features = df1.iloc[:,0:80] #slicing the dataset into features and a label 
	
	features = features.drop(["Label"], axis=1) #dropping the label from the features slice
	features = features.replace([np.inf, -np.inf], 0.0) #changing the infs to nans
	features = features.fillna(0.0)
	#print(features)
	#input("stoppppp")
	 #changing the nans to 0s 
	
	for col in list(features.columns):
		features = features.rename(columns={col: str(col)})
	
	label = df1.loc[:,["Label"]] 
	label = pd.get_dummies(label)
	label = label.drop(["Label_Benign"], axis=1)

		
	return(features,label)	


def check_feature_type(feature):
	
	with open('dictionary.txt') as f:
		data = f.read()
	js = json.loads(data)
	return(js.get(feature))

def detect_correlation_type(x,y):
	if x=="numerical continuous" and y=="numerical continuous":
		return("Pearson")
	
	elif x=="numerical continuous" and y=="dichotomous categorical":
		return("Point Biserial")
	
	elif x=="numerical continuous" and y=="numerical discrete":
		return("Pearson")
	
	elif x=="numerical discrete" and y=="numerical discrete":
		return("Pearson")
	
	elif x=="numerical discrete" and y=="numerical continuous":
		return("Pearson")
	
	elif x=="numerical discrete" and y=="dichotomous categorical":
		return("Point Biserial")
	
	elif x=="dichotomous categorical" and y=="numerical discrete":
		return("Point Biserial")
	
	elif x=="dichotomous categorical" and y=="numerical continuous":
		return("Point Biserial")
	
	elif x=="dichotomous categorical" and y=="dichotomous categorical":
		return("MCC")



def correlate(c,x,y,df):
	
	
	if c=="Point Biserial":
		#df1 = pd.DataFrame(df[x])
		a = df[x]
		a = pd.to_numeric(a,  errors='coerce').fillna(0).astype("Float64")
		
		#df2 = pd.DataFrame(df[y])
		b = df[y]
		b = pd.to_numeric(b,  errors='coerce').fillna(0).astype("Float64")

		a = a.to_numpy()
		b = b.to_numpy()

		a[a==np.inf] = 0
		b[b==np.inf] = 0
		
		return(stats.pointbiserialr(a,b))
		
	
	if c=="Pearson":
		#df1 = pd.DataFrame(df[x])
		a = df[x]
		a = pd.to_numeric(a,  errors='coerce').fillna(0).astype("Float64") 
		
		#df2 = pd.DataFrame(df[y])
		b = df[y]
		b = pd.to_numeric(b,  errors='coerce').fillna(0).astype("Float64")

		a = a.to_numpy()
		b = b.to_numpy()

		a[a==np.inf] = 0
		b[b==np.inf] = 0	

		return(stats.pearsonr(a, b))
	if c=="MCC":
		list = []
		
		a = df[x]
		a = pd.to_numeric(a, errors='coerce').fillna(0).astype("float32")

		b = df[y]
		b = pd.to_numeric(b, errors='coerce').fillna(0).astype("float32")

		a = a.to_numpy()
		b = b.to_numpy()

		a[a==np.inf] = 0
		b[b==np.inf] = 0

		c = metrics.matthews_corrcoef(a, b)
		list.append(c)
		return(list)



def merit_compute(subset, label):
	k = len(subset)
	check_list = []
	overall1 = 0.0
	overall2 = 0.0
	for i in range(k): #getting overall feature-feature correlation 
		for j in range(k):
			if subset[i]==subset[j] or subset[i]+":"+subset[j] in check_list or subset[j]+":"+subset[i] in check_list:
				continue
			x = check_feature_type(subset[i])
			y = check_feature_type(subset[j])
			co_type = detect_correlation_type(x,y)
			correlation = correlate(co_type,subset[i],subset[j],features)
			check_list.append(subset[i]+":"+subset[j])
			check_list.append(subset[j]+":"+subset[i])
			overall1 = overall1 + correlation[0]

	for i in range(k): #getting overall feature-label correlation
		feature_type = check_feature_type(subset[i])
		y = "dichotomous categorical"
		correlation_type = detect_correlation_type(feature_type,y)
		correlation = correlate(correlation_type,subset[i],label,df)
		overall2 = overall2 + correlation[0]

	overall1 = overall1/k
	overall2 = overall2/k
	
	merit = (k * overall2)/sqrt(k + k * (k-1) * overall1)
	return(merit) 


class PriorityQueue:
    def  __init__(self):
        self.queue = []

    def isEmpty(self):
        return len(self.queue) == 0
    
    def push(self, item, priority):
        """
        item already in priority queue with smaller priority:
        -> update its priority
        item already in priority queue with higher priority:
        -> do nothing
        if item not in priority queue:
        -> push it
        """
        for index, (i, p) in enumerate(self.queue):
            if (set(i) == set(item)):
                if (p >= priority):
                    break
                del self.queue[index]
                self.queue.append( (item, priority) )
                break
        else:
            self.queue.append( (item, priority) )
        
    def pop(self):
        # return item with highest priority and remove it from queue
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if (self.queue[max_idx][1] < p):
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return (item, priority)



if __name__ == "__main__":
	start_time = time.time()
	
	df1 = input("please input the location of the csv file:\n")
	features, label = preprocess(df1)
	l = len(list(label.columns))
	list1 = list(label.columns)
	print(list1)
	inp = input("The label has the following labels, which one do you want to use from the list above:(choose by index number)\n")
	inp = int(inp)
	for i in range(l):
		if i==int(inp):
			continue
		print("dropping", list1[i])
		label = label.drop([list1[i]], axis=1)
	best_feature = ""
	best_value = -1
	df = pd.concat([features,label])
	new_label = list1[inp]
	
	features_l = list(features.columns)
	for feature in features_l:
		type = check_feature_type(feature)
		correlation_type = detect_correlation_type(type, "dichotomous categorical")
		correlation = correlate(correlation_type, feature, new_label, df)
		print(correlation) #choose only the r value without the p value 
		correlation = list(correlation)
		if correlation[0] == "nan":
			df = df.drop([feature], axis=1)
			features = features.drop([feature], axis=1)
			continue
		abs_coeff = abs(correlation[0])
		if abs_coeff > best_value:
			best_value = abs_coeff
			best_feature = feature
	print("The label data type is ",label.dtypes)
	print("The feature of the highest coorelation with the label is:\n", best_feature)
	queue = PriorityQueue()
	queue.push([best_feature], best_value)
	visited = []
	n_backtrack = 0
	max_backtrack = 5

	n = 1
	while not queue.isEmpty():

		subset, priority = queue.pop()
		if (priority < best_value):
			print("The priority of the current merit is: ",priority,"\nAnd the best value is: ", best_value)
			n_backtrack +=1
			print(n_backtrack)
		else:
			best_value = priority
			best_subset = subset
		if n==1:
			best_value = -1 #set the best value to -1 after the first iteration
			n = n-1 
		if (n_backtrack == max_backtrack):
			print("reached max backtrack")
			break
		for feature in features_l:
			temp_subset = subset + [feature]
			
			for node in visited:
				if (set(node) == set(temp_subset)):
					print("node = tempsubset")
					break
			else:
				visited.append(temp_subset)
				merit = merit_compute(temp_subset, new_label)
				queue.push(temp_subset, merit)
		print(subset)
	time_taken = time.time() - start_time
	print("Time took to finish the process is: ",time_taken/60, " minutes" )
	print("finish"
