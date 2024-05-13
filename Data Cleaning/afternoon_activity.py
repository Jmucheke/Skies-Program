import plotly.express as px
import seaborn as sns

'''
x=[1,2,3,4]
y=[1,2,3,4]
fig = px.line(x,y)
fig.show()
'''

diamond_df = sns.load_dataset('diamonds')
'''
fig = px.line(diamond_df.sample(50), x='price', y='carat')
#fig.show()

fig = px.histogram(diamond_df, x='cut')
#fig.show()

fig=px.violin(diamond_df, x='cut', y='price')
fig.show()

fig=px.violin(diamond_df, x='cut', y='carat')
fig.show()
'''
fig =px.scatter(diamond_df, x='cut', y='carat')
fig.show()
