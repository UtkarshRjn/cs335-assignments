import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#DO NOT MODIFY THIS FILE#


def get_dataset(): 
	data = pd.read_csv('data/dataset.csv', index_col=0)
	x = data.iloc[:,:-1].to_numpy()
	y = data.iloc[:,-1].to_numpy()
	return train_test_split(x, y, test_size=0.2, random_state=42)
	
def find_loss(predicted, actual):
	mse = mean_squared_error(predicted, actual)
	rmse = math.sqrt(mse)
	return rmse
