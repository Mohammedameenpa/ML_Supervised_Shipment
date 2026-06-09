from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def tune_models(X_train, y_train):

    models = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [3, 5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        ),

        "Logistic Regression": (
            LogisticRegression(random_state=42, max_iter=1000),
            {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        )
    }

    best_models = {}

    for name, (model, params) in models.items():

        print(f"\nTuning {name}...")

        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring="f1",
            cv=5,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        print("Best Parameters:", grid.best_params_)
        print("Best F1 Score:", grid.best_score_)

        best_models[name] = grid.best_estimator_

    return best_models
    