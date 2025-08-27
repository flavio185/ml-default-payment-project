from loguru import logger

def logistic_regression_model():
    logger.info("Loading Logistic Regression model...")
    from sklearn.linear_model import LogisticRegression
    model_params = {
        "C": 1.0,
        "max_iter": 100,
        "solver": "liblinear",
    }
    model = LogisticRegression(**model_params)
    return model

    
def svm_model():
    logger.info("Loading SVM model...")
    from sklearn.svm import SVC
    model_params = {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
    }
    model = SVC(**model_params)
    return model


def random_forest_model():
    logger.info("Loading Random Forest model...")
    from sklearn.ensemble import RandomForestClassifier
    model_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    }
    model = RandomForestClassifier(**model_params)
    return model