"""
train_models.py
Trains and evaluates Decision Tree, Naive Bayes, ANN, Random Forest,
and a Voting Ensemble classifier.
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_ann(X_train, y_train):
    """Train ANN with hyperparameter tuning via GridSearchCV."""
    param_grid = {
        'hidden_layer_sizes': [(32, 16), (64, 32, 16), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    grid = GridSearchCV(
        MLPClassifier(max_iter=3000, random_state=42, early_stopping=True),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best ANN params: {grid.best_params_}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning via GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best Random Forest params: {grid.best_params_}")
    return grid.best_estimator_


def train_voting_ensemble(X_train, y_train, rf, ann, nb):
    """Combine top models into a soft-voting ensemble."""
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('ann', ann), ('nb', nb)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate_model(name, model, X_test, y_test):
    """Print accuracy and classification report for a model."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n{'=' * 40}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return acc


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all models and return their accuracies."""
    results = {}

    dt = train_decision_tree(X_train, y_train)
    results['Decision Tree'] = evaluate_model('Decision Tree', dt, X_test, y_test)

    nb = train_naive_bayes(X_train, y_train)
    results['Naive Bayes'] = evaluate_model('Naive Bayes', nb, X_test, y_test)

    ann = train_ann(X_train, y_train)
    results['ANN'] = evaluate_model('ANN', ann, X_test, y_test)

    rf = train_random_forest(X_train, y_train)
    results['Random Forest'] = evaluate_model('Random Forest', rf, X_test, y_test)

    ensemble = train_voting_ensemble(X_train, y_train, rf, ann, nb)
    results['Voting Ensemble'] = evaluate_model('Voting Ensemble', ensemble, X_test, y_test)

    return results