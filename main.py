import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
warnings.filterwarnings('ignore')

def EncodingScaling(df, col):

    # Encoding the dataset
    df_ordinal = df.copy()
    df_label = df.copy()

    X = pd.DataFrame(df[col])
    # Convert categorical features to numeric values using ordinalEncoder
    ordinalEncoder = OrdinalEncoder()
    ordinalEncoder.fit(X)
    df_ordinal[col] = pd.DataFrame(ordinalEncoder.transform(X))
    # TODO: encoding 하면 null값이 생김......
    df_ordinal.dropna(axis=0, inplace=True)


    # Convert categorical features to numeric values using labelEncoder
    labelEncoder = LabelEncoder()
    labelEncoder.fit(X)
    df_label[col] = pd.DataFrame(labelEncoder.transform(X))
    df_label.dropna(axis=0, inplace=True)


    # Scaling the dataset
    df_ordinal_standard = df_ordinal.copy()
    df_ordinal_robust = df_ordinal.copy()
    df_ordinal_minmax = df_ordinal.copy()
    df_ordinal_maxabs = df_ordinal.copy()
    df_label_standard = df_label.copy()
    df_label_robust = df_label.copy()
    df_label_minmax = df_label.copy()
    df_label_maxabs = df_label.copy()

    # Scaling the dataset using StandardScaler
    scaler = StandardScaler()
    df_ordinal_standard = scaler.fit_transform(df_ordinal)
    df_ordinal_standard = pd.DataFrame(df_ordinal_standard, columns=df_ordinal.columns)
    df_label_standard = scaler.fit_transform(df_label)
    df_label_standard = pd.DataFrame(df_label_standard, columns=df_label.columns)

    # Scaling the dataset using RobustScaler
    scaler = RobustScaler()
    df_ordinal_robust = scaler.fit_transform(df_ordinal)
    df_ordinal_robust = pd.DataFrame(df_ordinal_robust, columns=df_ordinal.columns)
    df_label_robust = scaler.fit_transform(df_label)
    df_label_robust = pd.DataFrame(df_label_robust, columns=df_label.columns)

    # Scaling the dataset using MinMaxScaler
    scaler = MinMaxScaler()
    df_ordinal_minmax = scaler.fit_transform(df_ordinal)
    df_ordinal_minmax = pd.DataFrame(df_ordinal_minmax, columns=df_ordinal.columns)
    df_label_minmax = scaler.fit_transform(df_label)
    df_label_minmax = pd.DataFrame(df_label_minmax, columns=df_label.columns)

    # Scaling the dataset using MaxAbsScaler
    scaler = MaxAbsScaler()
    df_ordinal_maxabs = scaler.fit_transform(df_ordinal)
    df_ordinal_maxabs = pd.DataFrame(df_ordinal_maxabs, columns=df_ordinal.columns)
    df_label_maxabs = scaler.fit_transform(df_label)
    df_label_maxabs = pd.DataFrame(df_label_maxabs, columns=df_label.columns)


    # Show the results using OrdinalEncoder and MinMaxScaler
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 15))
    ax1.set_title('Before Scaling the OrdinalEncoded dataset')
    ax2.set_title('After MinMax Scaling the OrdinalEncoded dataset')

    for i in df_ordinal_minmax.columns:
        sns.kdeplot(df_ordinal[i], ax=ax1)
        sns.kdeplot(df_ordinal_minmax[i], ax=ax2)

    # plt.show()

    # Return the one of the encoded and scaled datasets
    return df_ordinal_minmax


def autoML(df):
    doDBSCAN(df)


def doDBSCAN(df):
    print(df)





# Read dataset
df = pd.read_csv('housing.csv')
# print(df.info())
# print(df.head())
# print(df.describe())

# Dirty value detection
# print(df.isnull().sum())
df.dropna(axis=0, inplace=True)
# print(df.shape)

# Encoding and Scaling the dataset
df_encoded_scaled = EncodingScaling(df, 'ocean_proximity')

# Draw heat map
heatmap_data = df
colormap = plt.cm.PuBu
plt.figure(figsize=(15, 15))
plt.title("Correlation of Features", y=1.05, size=15)
sns.heatmap(heatmap_data.corr(), linewidths=0.1, square=False, cmap=colormap, linecolor="white",
            annot=True, annot_kws={"size": 8})
# plt.show()

# Feature selection
df_median_house_value = df_encoded_scaled['median_house_value']
df_median_house_value.columns = ['median_house_value']

# Combinations of features
# TODO: 일단 제 마음대로 했는데 바꾸셔도 돼요
col1 = ['total_rooms', 'total_bedrooms', 'population', 'households']
col2 = ['median_income', 'median_house_value']
col3 = ['longitude', 'latitude', 'ocean_proximity']
col4 = ['median_income', 'households', 'total_rooms']
col5 = ['latitude', 'total_rooms', 'households', 'median_income']

# Make dataframes with various combination
df1 = df_encoded_scaled[col1]
df1.columns = col1
df2 = df_encoded_scaled[col2]
df2.columns = col2
df3 = df_encoded_scaled[col3]
df3.columns = col3
df4 = df_encoded_scaled[col4]
df4.columns = col4
df5 = df_encoded_scaled[col5]
df5.columns = col5


# TODO: 여기 feature selection도 autoML 안으로 넣을지,,,
autoML(df1)
# autoML(df2)
# autoML(df3)
# autoML(df4)
# autoML(df5)





