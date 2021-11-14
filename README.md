# Manual

# def auto_ml(dataset, model)

Select features randomly.  
Run algorithms with the selected combination.

##	Parameter  
dataset: DataFrame to be used  
model: A model to be used

##	How to operate
1.	Randomly select features from the list of numeric features.
2.	Encoding and Scaling using scale_encode_combination().
3.	Run the selected algorithm; one of [ test_kmeans(), test_gaussian(), test_clarans(), test_dbscan() and test_mean_shift() ].

##	Examples
    df = pd.read_csv(‘housing.csv’)
    df.fillna(df.mean(), inplace=True)
    medianHouseValue = df['median_house_value']
    df.drop(['median_house_value'], axis=1, inplace=True)
    auto_ml(df, ‘kmeans’)

##	Return
	All the results of the selected model.


# def scale_encode_combination(dataset, numerical_feature_list, categorical_feature_list)
Scaling and Encoding with 15 combinations.

## Parameters
dataset: DataFrame to be scaled and encoded  
numerical_feature_list: Features to scale  
categorical_feature_list: Features to encode

## How to operate
1.	for in scalers [StandardScaler(), MinMaxScaler(), RobustScaler(),  MaxAbsScaler(), Normalizer()]
2.	for in encoders [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
3.	Save each dataset in dictionary

## Examples
    for combination in feature_combination_list:
        data_combination = scale_encode_combination(dataset, combination, ['ocean_proximity'])
        for data_name, data in data_combination.items():
            data = data[combination]
            test_kmeans(data)
            test_gaussian(data)
            test_clarans(data)
            test_dbscan(data)
            test_mean_shift(data)

## Return
	Dictionary included all the dataframe combinations of Scalers and Encoders.
