import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

# Read dataset
df = pd.read_csv('housing.csv')
print(df.info())
# print(df.head())
# print(df.describe())

# Dirty value detection
print(df.isnull().sum())
df.dropna(axis=0, inplace=True)
print(df.shape)

# Draw heat map
heatmap_data = df
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 8})
plt.show()