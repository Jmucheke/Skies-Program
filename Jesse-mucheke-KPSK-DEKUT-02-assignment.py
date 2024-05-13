# 1(a) Loading data locally from the computer
# Importing the necessary libraries
import pandas as pd

# Loading the dataset
study_performance_df = pd.read_csv("study_performance.csv")

# Printing some information about the data to confirm it has been loaded successfully
print(f"========= Printing the last 11 rows of the dataset: ==================\n{study_performance_df.tail(11)}")
print(f"============ Printing the information about the columns/features and the data types associated with the columns: ==========")
print(study_performance_df.info())


# 1(b) Loading data from a given website
# Importing the necessary libraries 
import requests

# Getting the data using the requests library
response = requests.get("https://api.github.com/users/naveenkrnl")
data = response.json()

# Converting the json file into a dataframe for easier manipulation and visualization
data_df = pd.DataFrame([data])
print(f"============== Displaying the first 5 columns of the dataset: ==============\n {data_df.head()}")
print(f"============== Displaying information about the columns/features: =========\n")
print(data_df.info())

# Since the data is not displayed in a meaningful way, Below is a formated way
print("GitHub User Information:")
print(f"Username: {data_df['login'].values[0]}")
print(f"Name: {data_df['name'].values[0]}")
print(f"Location: {data_df['location'].values[0]}")
print(f"Number of public repos: {data_df['public_repos'].values[0]}")
print(f"Number of followers: {data_df['followers'].values[0]}")
print(f"Number of following: {data_df['following'].values[0]}")



# 1(c) Loading data from a stored file on a given website
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['SepalLengthcm','SepalWidthcm','PetalLengthcm','PetalWidthcm','Species']
iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=column_names)

# 2 a) first 8 records
print(f"============== Displaying the first 8 rows of the iris dataframe: =============\n{iris_df.head(8)}")

# 2 b) number of records and their features
print(f"============== Visualizing the number of records * the iris dataframe features: =================\n {iris_df.shape}")
# 2 c) data type of each feature 
print(f"============== Visualizing the data type of each column/feature: ===============\n {iris_df.dtypes}")
# 2 d) existence of missing values
print(f"============== Visualizing the sum of missing values that exist on each column/feature: ===============\n {iris_df.isnull().sum()}")

# 2 e) relationship between some of the features
# 2 e) i)
# Importing the necessary library 
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the Species Column using a bar graph to show the distribution between the classes
plt.hist(iris_df['Species'], bins=9)
plt.title('Bar graph showing the distribution of the various classes of species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Visualizing the Sepal width and Petal width using a Histogram
plt.scatter(iris_df['SepalWidthcm'], iris_df['PetalWidthcm'])
plt.xlabel("Sepal Width(cm)")
plt.ylabel("Petal Width (cm)")
plt.title('Scatter plot showing the distribution of values of Sepal Width Vs Petal Width')
plt.show()

# Visualizing the distribution of Species classes with respect to sepal length an petal length using a scatter plot
sns.scatterplot(x='SepalLengthcm', y='PetalLengthcm',hue='Species', data=iris_df, ) 
plt.title('Scatterplot showing the distribution of the classes of species with respect to sepal length and petal length')
plt.legend(bbox_to_anchor=(1, 1), loc=2)  
plt.show()


# Histogram showing distribution of the petal width
sns.histplot(iris_df['PetalWidthcm'], kde=True)
plt.title('Histogram showing count of the petal width')
plt.show()

# Catplot of Sepal Width
sns.catplot(x='SepalWidthcm', data=iris_df, kind='count')
plt.title('Catplot showing the distribution of values of the Sepal length')
plt.show()

# Distplot showing the distribution and frequencies of the different features in our dataframe
plot = sns.FacetGrid(iris_df, hue="Species") 
plot.map(sns.distplot, "SepalLengthcm").add_legend() 
  
plot = sns.FacetGrid(iris_df, hue="Species") 
plot.map(sns.distplot, "SepalWidthcm").add_legend() 
  
plot = sns.FacetGrid(iris_df, hue="Species") 
plot.map(sns.distplot, "PetalLengthcm").add_legend() 
  
plot = sns.FacetGrid(iris_df, hue="Species") 
plot.map(sns.distplot, "PetalWidthcm").add_legend() 
  
plt.show()


# Report

'''
From the Visualization of the Datafram Iris we can gather the following insights:
The Species Iris-Ssetosa, Iris-versicolor and iris-virginica have the same number of sample,
that is, each of the species has 50 samples in the dataframe.
From the scatter plot showing the distribution of sepal length and petal length on the various species,
we can gather the following information:
i) Iris-virginica has the largest petal length and sepal length
ii) Iris-setosa has the smallest petal length and sepal length
iii) Iris-versicolor has falls between the iris-virginica and iris-setosa in terms of the sepal and petal length.

From the Histogram showing the frequency of petal width we can conclude that:
The highest count of petal width are those that lies between 0.0 to 0.5 cm.

The highest frequency of Sepal width is between 2.8 and 3.2

From the overall visualization, iris-setosa has the smallest values, iris-versicolor is in the middle
and iris-virginica takes the largest values.

'''

