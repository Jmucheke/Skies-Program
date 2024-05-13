import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = sns.load_dataset('titanic')

print(titanic_df.head())
print(titanic_df.tail())
print(titanic_df.shape)
print(titanic_df.info)
print(titanic_df.dtypes)
print(titanic_df.isna().sum())



sns.set(style='ticks')
plt.style.use("fivethirtyeight")

#Plotting a bar graph
sns.catplot(x='sex', data=titanic_df, kind='count')
plt.show

#Plotting a Facet Grid
sns.catplot(x='pclass', data=titanic_df, hue='sex', kind='count')
plt.show

#Splitting males,females and children according to age

titanic_df['who'] = titanic_df.sex
titanic_df.loc[titanic_df['age'] < 16, 'Person'] = 'Child'

#Checking the distribution of Male, female and children
print(titanic_df.who.unique())
print(titanic_df.who.value_counts())
print(titanic_df.age.mean())

#Facet grid of male female and children
sns.catplot(x='pclass', data=titanic_df, hue='who', kind='count')

#Visualizing age distribution using a histogram

titanic_df.age.hist(bins=80)

#Visualizing the data using FacetGrid to plot multiple kedplots on one plot
fig = sns.FacetGrid(titanic_df, hue="sex", aspect=4)
fig.map(sns.kdeplot, 'age', fill=True)

oldest = titanic_df['age'].max()

fig.set(xlim=(0, oldest))
fig.add_legend()

# Visualizing data for 'person' column using Facet Grid
fig = sns.FacetGrid(titanic_df, hue="Person", aspect=4)
fig.map(sns.kdeplot, 'age', fill=True)
oldest = titanic_df['age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()

# Visualizing data for Pclass column using Facet Grid
fig = sns.FacetGrid(titanic_df, hue="pclass",aspect=4)
fig.map(sns.kdeplot, 'age', fill=True)
oldest = titanic_df['age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()


'''
Determining what deck the passangers were and how that relates to their class
'''

print("Determining what deck the passangers were and how that relates to their class\n")

print(titanic_df.head(10))

deck = titanic_df['deck'].dropna()
print(deck)

# Histogram of classes represented on the deck
sns.catplot(x='deck', data=titanic_df, kind='count')

print("Where did the passengers come from? \n")
titanic_df.head()

# Factor plot to check results
sns.catplot(x='embarked', data=titanic_df, hue='pclass', kind='count', order=['C', 'Q', 'S'])

print("Who was alone and who was with family? \n")
