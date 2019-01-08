import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('leukemia.dat')
train, test = train_test_split(data, test_size=0.2)
train.count()
test.count()
train.to_csv('leukemia_train.dat', index = False)
test.to_csv('leukemia_test.dat', index = False)
