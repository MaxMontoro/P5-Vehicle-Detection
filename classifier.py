from utils import *
from feature_extraction import car_features, notcar_features
from hog import CONFIG

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
# scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)


def get_classifier_from_pickle():
    with open(f'classifiers/classifier{str(CONFIG)}.p', 'rb') as classifier_file:
        svc = pickle.load(classifier_file)
    return svc

if not __name__ == "__main__":
    svc = get_classifier_from_pickle()

if __name__ == "__main__":
    from sklearn.svm import LinearSVC
    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    with open(f'classifiers/classifier{str(CONFIG)}.p', 'wb') as classifier_file:
        pickle.dump(svc, classifier_file)
    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
    print('My SVC predicts: ', svc.predict(X_test[0:10]))
    print('For labels:      ', y_test[0:10])
