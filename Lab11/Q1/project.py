from sklearn.decomposition import PCA
from utils import *

points_4D = load_points_from_json('points_4D.json')
pca = PCA(n_components=2)
points_2D = pca.fit_transform(points_4D)
store_points_to_json(points_2D, 'points_2D.json')