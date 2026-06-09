import joblib


def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model


def save_model(model, path):

    joblib.dump(model, path)
    