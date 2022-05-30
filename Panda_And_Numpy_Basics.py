import numpy as np
import pandas as pd


"""PANDAS"""
#%%Excersice 1

a = pd.DataFrame({'Apples': [30] , 'Bananas': [21]})
print(a)

#%%
b = pd.DataFrame({'Apples': [35,41] , 'Bananas': [21,34]}, index = ['2017 Sales','2018 Sales'])

print(b)

#%%

series1 = pd.Series(['4 cups','1 cup','2 large','1 can'], index =['Flour','Milk','Eggs','Spam'], name = 'Dinner' )
print(series1)

#%%

data = pd.read_csv(r'D:\upes\3rd year\Pattern and Anomaly Detection\Lab\winemag-data_first150k.csv')
print(data)

#%%

animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals.head()


#%%Excersice 2


reviews = pd.read_csv(r'D:\upes\3rd year\Pattern and Anomaly Detection\Lab\winemag-data_first150k.csv')
desc = reviews.loc[:,['description']]
print(desc)
#%%

first_description = reviews.loc[0,['description']]
print(first_description)
#%%

first_row = reviews.iloc[1,:]
print(first_row)
#%%

first_descriptions=reviews['description'][10]
print(first_descriptions)
#%%

sample_reviews = reviews.iloc[[1, 2, 3,5,8], :]
print(sample_reviews)

#%%

df = reviews.iloc[[0,1,10,100],[1,6,7,8]]
print(df)

#%%

italian_wines = reviews.country == 'Italy'
print(italian_wines)

#%%

top_oceania_wines = reviews.loc[(reviews.country == 'Australia') & (reviews.points >= 95) |(reviews.country == 'New Zealand') & (reviews.points >= 95) ]

print(top_oceania_wines)

#%%Excersice 3

#%%
data = pd.read_csv(r'D:\upes\3rd year\Pattern and Anomaly Detection\Lab\winemag-data_first150k.csv')
median_points = data.points.mean()

print(" Ques 1\n",median_points)

#%%
countries = data.country.unique()
print(" Ques 2\n",countries)

#%%

reviews_per_country = data.country.value_counts()
print("\n Ques 3:counts of country = \n", reviews_per_country)

#%%
mean_price = data.price.mean()
centered_price = data.price.map(lambda p: p - mean_price)
print("\n Ques 4\n",centered_price)

#%% Excersice 4

reviews = pd.read_csv(r'D:\upes\3rd year\Pattern and Anomaly Detection\Lab\winemag-data_first150k.csv')
#%% 
reviews_written = reviews.groupby('designation').designation.count()
print("\n ques 1 : \n",reviews_written)

#%%

best_rating_per_price = reviews.groupby('price').points.max().sort_index()
print("\n ques 2 : \n",best_rating_per_price)

#%%

price_extremes = reviews.groupby('variety').price.agg([min,max])
print("\n ques 3 : \n",price_extremes)

#%%

sorted_varieties = reviews.groupby('variety').price.agg([min,max]).sort_values(by=['min', 'max'], ascending=False)
print("\n ques 4 : \n",sorted_varieties)


#%%

country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
print(country_variety_counts)

#%% Excersice 5

reviews = pd.read_csv(r'D:\upes\3rd year\Pattern and Anomaly Detection\Lab\winemag-data_first150k.csv')
datatype = reviews.points.dtype
print("\n ques 1:\n",datatype)

#%%

point_strings = reviews.points.astype('str')
print("\n ques 2:\n",point_strings)

#%%

n_missing_prices = reviews[pd.isnull(reviews.price)]
print("\n ques 3:\n",n_missing_prices)

#%%

reviews_per_region = reviews.region_1.fillna("Unknown")
reviews_per_region = reviews_per_region.value_counts().sort_values(ascending=False)

print("\n ques 4:\n",reviews_per_region)

#%%Excersice 6

reviews = pd.read_csv(r'D:\upes\3rd year\Pattern and Anomaly Detection\Lab\winemag-data_first150k.csv')

renamed = reviews.rename(columns={'region 1':'region','region 2':'locale'})
print("\n ques1 :  \n",renamed)

#%%

reindexed = reviews.rename(index={0:'wine'})
print("\n ques2 :  \n",reindexed)

#%% NUMPY Excersice 

#1D, 2D and 3D Array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
arrs=np.array(['apple', 'banana', 'cherry'])
print(arr,arr2,arr3)

#%%

print(arr[2],arr2[0][1])

#%%

print(arr[1:5])
print(arr[-3:-1])
print(arr2[0:2, 2])

#%%

print(arr.dtype)
print(arrs.dtype)

#%%

newarr = arr.astype('i')
print(newarr.dtype)

#%%

x = arr.copy()
print(x)
print(x.base)

#%%

print(arr.shape)

#%%

newarr = arr.reshape(2, 5)
print(newarr)

#%%

arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)
print(newarr)

#%%

arr = np.array([1, 2, 3])
for x in arr:
  print(x)

print("")
  
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
  for y in x:
    for z in y:
      print(z)

#%%

arr = np.array([1, 2, 3])
for idx, x in np.ndenumerate(arr):
  print(idx, x)
  
#%%

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)

#%%

arr = np.stack((arr1, arr2), axis=1)
print(arr)

#%%

arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(newarr)

#%%

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3)
print(newarr)

#%%

arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)

#%%

arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7)
print(x)

#%%

arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr))
arr = np.array([[3, 2, 4], [5, 0, 1]])
print(np.sort(arr))

#%%

arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]
print(newarr)

#%%

arr = np.array([41, 42, 43, 44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

#%%