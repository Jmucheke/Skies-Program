import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

#housing_df = pd.read_csv('housing_dataset.csv')
housing_df = pd.read_csv('housing_dataset.csv', usecols=['longitude','latitude','median_house_value'])

'''
# Visualizing the data
print(f"=============== 5 rows sample of the dataset:===========\n {housing_df.head()}")
print("=============== Dispaying the information about the dataframe: =============\n")
print(f" {housing_df.info()}")


# Histogram showing distribution of Households
sns.histplot(housing_df['households'], kde=True)

plt.title('Histogram showing the frequency of households')
plt.show()

# Plotting population v ocean proximity

X = np.array(housing_df['population'])
Y = np.array(housing_df['housing_median_age'])
y = np.array(housing_df['ocean_proximity'])

plt.plot(X,y, '.g')
plt.title('Total Population vs Proximity to the Ocean')
plt.xlabel('Total Population')
plt.ylabel('Proximity to the Ocean')
plt.show()

'''

sns.scatterplot(housing_df,x='longitude',y='latitude',hue='median_house_value')
#plt.show()

x_train,x_test,y_train,y_test = train_test_split(housing_df[['latitude','longitude']], housing_df[['median_house_value']],test_size=0.33, random_state=0)

# Normalize
from sklearn import preprocessing
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)

print(f"============== The first 5 rows of x_train:==========\n {x_train.head()}")

# Model fitting, training and evaluation

from sklearn.cluster import KMeans
km_model = KMeans(n_clusters=3, random_state=0, n_init='auto')
km_model.fit(x_train_norm)

# Visualizing the results

sns.scatterplot(data=x_train, x='longitude', y='latitude', hue=km_model.labels_)
#plt.show()

# Look at the ditribution of median house prices in the 3 groups. A box plit can be useful
sns.boxplot(x=km_model.labels_,y=y_train['median_house_value'])
#plt.show()

''' people in the 1st and 3rd cluster have similar distributions
 of median house values and are higher than those of second cluster
'''

# Evaluation
'''
Evaluation performance of the clustering algorithm using a silhouette score which is part of sklearn.metrics.
A lower score represents a better fit.
'''

from sklearn.metrics import silhouette_score
perf = silhouette_score(x_train_norm, km_model.labels_, metric='euclidean')
print(perf)

'''
The silhouette score is a metric used to evaluate the performance of clustering algorithms like KMeans. It provides a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette score ranges from -1 to 1, where:

A score close to +1 indicates that the sample is far away from the neighboring clusters, meaning it is well-clustered.
A score of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters.
A score close to -1 indicates that those samples might have been assigned to the wrong clusters.

'''


#  How many clusters to use?
# need to test a range of them

K = range(2,8)
fits=[]
score=[]
for k in K:
    # train the model for current value of k on training data
    new_Km_model = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(x_train_norm)

    # Append the model fits
    fits.append(new_Km_model)

    # append the silhouette_score
    score.append(silhouette_score(x_train_norm, new_Km_model.labels_, metric='euclidean'))    

print(score)
print(fits)

# Visualizing a few, start with k = 2

sns.scatterplot(data=x_train, x='longitude',y='latitude',hue=fits[0].labels_)
plt.show()

# 2 halves not good looking

# What about k = 4
sns.scatterplot(data=x_train, x='longitude',y='latitude',hue=fits[2].labels_)
plt.show()

# Is it better? worse?
sns.scatterplot(data=x_train, x='longitude',y='latitude',hue=fits[5].labels_)
plt.show()

# 7 is too many


# Use the elbow plot to compare
sns.lineplot(x=K, y=score)
plt.show()

# Choose the point where the performance
#start to flatten or get worse. Here k=5
sns.scatterplot(data=x_train, x='longitude', y='latitude', hue=fits[3].labels_)
plt.show()
sns.boxplot(x=fits[3].labels_, y=y_train['median_house_value'])
plt.show()
