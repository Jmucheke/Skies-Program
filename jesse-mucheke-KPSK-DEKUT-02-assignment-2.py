import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

customers_df = pd.read_csv("Mall_Customers.csv")

print(f" Print the size of the dataset:\n {customers_df.size}")

print("Print the information about the columns/features in the dataframe: \n ")
print(f"{customers_df.info()}")

print(f" Show if there are null values: \n {customers_df.isnull().sum()}")


# Boxplot for Age by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Age', data=customers_df)
plt.title('Boxplot of Age by Gender')

# Boxplot for Annual Income by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Annual Income (k$)', data=customers_df)
plt.title('Boxplot of Annual Income by Gender')

# Boxplot for Spending Score by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=customers_df)
plt.title('Boxplot of Spending Score by Gender')

# Scatter plot for Annual Income vs. Spending Score by Gender
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', data=customers_df, s=100, ax=ax)
ax.set_title('Scatter Plot of Annual Income vs. Spending Score by Gender')
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')

plt.show()




# Splitting the data into train and testing sets
x_train,x_test,y_train,y_test = train_test_split(customers_df[['Annual Income (k$)','Age']], customers_df[['Spending Score (1-100)']],test_size=0.33, random_state=0)

# Normalizing the tests
from sklearn import preprocessing
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)

print(f"============== The first 5 rows of x_train:==========\n {x_train.head()}")

# Model fitting, training and evaluation

from sklearn.cluster import KMeans
km_model = KMeans(n_clusters=3, random_state=0, n_init='auto')
km_model.fit(x_train_norm)

# Visualizing the results

sns.scatterplot(data=x_train, x='Annual Income (k$)', y='Age', hue=km_model.labels_)
plt.title('Scatter Plot of Annual Income vs. Spending Score by Gender in 3 clusters')
plt.show()

# Look at the distribution of Spending score in the 3 groups.
sns.boxplot(x=km_model.labels_,y=y_train['Spending Score (1-100)'])
plt.title("Boxplot of Model labels by Spending score")
plt.show()



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
sns.scatterplot(data=x_train, x='Annual Income (k$)',y='Age',hue=fits[0].labels_)
plt.title("Scatter plot showing distribution of annual income vs. Age where k = 2")
plt.show()

# 2 halves not good looking

# What about k = 4
sns.scatterplot(data=x_train, x='Annual Income (k$)',y='Age',hue=fits[2].labels_)
plt.title("Scatter plot showing distribution of annual income vs. Age where k = 4")
plt.show()

# Where k = 6
sns.scatterplot(data=x_train, x='Annual Income (k$)',y='Age',hue=fits[5].labels_)
plt.title("Scatter plot showing distribution of annual income vs. Age where k = 7")
plt.show()



# Use the elbow plot to compare
sns.lineplot(x=K, y=score)
plt.title("Elbow plot for suggesting the optimal number of clusters")
plt.show()

# Choose the point where the performance
sns.scatterplot(data=x_train, x='Annual Income (k$)', y='Age', hue=fits[3].labels_)
plt.title("Scatter plot of Annual income vs Age with optimal number of clusters")
plt.show()
