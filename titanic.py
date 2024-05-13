import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = sns.load_dataset('titanic')

#Plotting a bar graph
sns.catplot(x='sex', data=titanic_df, kind='count')
print(plt.show)
