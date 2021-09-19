import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
import pandas as pd
from sktime.datasets import load_arrow_head  # univariate dataset
#from sktime.datasets.base import load_japanese_vowels  # multivariate dataset

from sktime.transformations.panel.rocket import Rocket


X_train, y_train = load_arrow_head(split="train", return_X_y=True)
print((X_train.iat[0,0]))
print(type(X_train.shape[1]))
df = pd.DataFrame(X_train)



X_test, y_test = load_arrow_head(split="test", return_X_y=True)
rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
rocket.fit(X_train)
X_train_transform = rocket.transform(X_train)


classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(X_train_transform, y_train)


X_test_transform = rocket.transform(X_test)

print(classifier.score(X_test_transform, y_test))


X_test, y_test = load_arrow_head(split="test", return_X_y=True)
X_test_transform = rocket.transform(X_test)