import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from features import get_pack_check_features


if __name__ == '__main__':
    x_data = []
    y_data = []

    unpack_path = ''
    packed_path = ''

    for file in os.listdir(unpack_path):
        x_data.append(get_pack_check_features(os.path.join(unpack_path, file)))
        y_data.append(0)

    for file in os.listdir(packed_path):
        x_data.append(get_pack_check_features(os.path.join(packed_path, file)))
        y_data.append(1)

    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    print(rf.score(x_test, y_test))

    joblib.dump(rf, 'model/RF.model')
