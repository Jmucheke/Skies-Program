# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Get data
x = np.arange(10).reshape((-1,1))
y = np.array([0,0,0,0,1,1,1,1,1,1])

# Viewing the data
print(f"========== The following are values of x: ============\n {x}")

print(f"=========== The following are the values of y: ===========\n {y}")

# Creating a model amd training it
ex4_2 = LogisticRegression(solver='liblinear', random_state=0)

# Fit the data and train it

ex4_2.fit(x,y)

# Evaluate the model
print(f"=========The prediction probability is:======== \n {ex4_2.predict_proba(x)}")

# Generating the Actual predictions
print(f"========== These are the values of the predictions: ==========\n {ex4_2.predict(x)}")

# Accuracy of the model

print(f"=========== The accuracy of the model is: ==========\n {ex4_2.score(x,y)}")


# Confusion matrix: Provides the actual and predicted results

print(f"=============These are values for the confusion matrix:=========\n {confusion_matrix(y,ex4_2.predict(x))}")

# Visualize the data
cm = confusion_matrix(y, ex4_2.predict(x))




fig,ax = plt.subplots(figsize=(8,8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1),ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0,1),ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5,-0.5)

for i in range(2):
    
    for j in range(2):        
        ax.text(j,i,cm[i,j],ha='center',va='center',color='red')

plt.show()

print(f"The following is the report of the perfomance of the model:\n {classification_report(y,ex4_2.predict(x))}")

# Alternative


'''
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Predicted 0s', 'Predicted 1s'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Actual 0s', 'Actual 1s'])
ax.set_ylim(1.5, -0.5)

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.show()
'''
