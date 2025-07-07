import random

def train_model():
    images, labels, label_map = load_tomato_dataset()
    features = [extract_features(img) for img in images]
    X = np.array(features)
    y = np.array(labels)

    # Gunakan random_state acak setiap kali
    random_state = random.randint(1, 10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, label_map, accuracy
