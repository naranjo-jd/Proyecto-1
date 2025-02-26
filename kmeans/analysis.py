import numpy as np
import pandas as pd
import clustering as clust
import metric

data_2d = pd.read_csv("data_2d.csv")
data_3d = pd.read_csv("data_3d.csv")

np.random.seed(41)