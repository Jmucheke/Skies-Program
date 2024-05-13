import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

diamond_df = sns.load_dataset('diamonds')
print(diamond_df.head())

print("\n=======================Welcome to the Diamonds dataset Visualization=======================\n")


plt.figure(figsize=(6,10))
sns.histplot(diamond_df['cut'], kde=True)



sns.catplot(x='carat', data=diamond_df, kind='count')
plt.show()

