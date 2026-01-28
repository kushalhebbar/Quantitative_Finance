"""Train and compare baseline ML models on a labeled equity dataset."""

# Fitting various models on the data and a comparative analysis of performance

import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Default predictors (adjust as needed)
# train_names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
DEFAULT_FEATURES = [
    'MACD 12 26 9',
    'MACD 12 26 9 Signal',
    'MACD 12 26 9 Hist',
    '5d Avg Vol vs 30d Avg Vol',
    '14d RSI',
    '%K Full',
    '%D Full',
]
DEFAULT_LABEL = '15d Label'


def load_dataset(path, features, label):
    """Load data and validate required columns."""
    df = pd.read_csv(path)
    missing = [col for col in features + [label] if col not in df.columns]
    if missing:
        raise ValueError("Missing columns in dataset: {}".format(", ".join(missing)))
    return df


def time_series_split(df, features, label, date_col, test_size):
    """Split data chronologically to avoid lookahead bias."""
    if date_col not in df.columns:
        raise ValueError("Missing date column for time split: {}".format(date_col))

    df_sorted = df.sort_values(date_col)
    split_idx = int((1 - test_size) * len(df_sorted))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]

    return X_train, X_test, y_train, y_test

def logisticRegression(X_train, X_test, y_train, y_test):
    """Train/evaluate a Logistic Regression classifier."""
    # Initialize LogisticRegression Model
    lr = LogisticRegression()
    # Fit data and train
    lr.fit(X_train, y_train)
    # Predict test set
    lr_y_predict = lr.predict(X_test)

    # Analyze the results
    print("Accuracy of Logistic Regression Classifier:", lr.score(X_test, y_test))
    print(classification_report(y_test, lr_y_predict))


def sgdClassifier(X_train, X_test, y_train, y_test):
    """Train/evaluate a linear classifier optimized with SGD."""
    # Initialize SGDClassifier Model
    sgdc = SGDClassifier()
    sgdc.fit(X_train, y_train)
    sgdc_y_predict = sgdc.predict(X_test)

    # Analyze the results
    print("Accuracy of SGD Classifier:", sgdc.score(X_test, y_test))
    print(classification_report(y_test, sgdc_y_predict))


def linearSVC(X_train, X_test, y_train, y_test):
    """Train/evaluate a linear SVM classifier."""
    lsvc = LinearSVC()
    lsvc.fit(X_train, y_train)
    y_predict = lsvc.predict(X_test)

    print("Accuracy of Linear SVC:", lsvc.score(X_test, y_test))
    print(classification_report(y_test, y_predict))


def NBClassifier(X_train, X_test, y_train, y_test):
    """Train/evaluate a Naive Bayes classifier."""
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_predict = gnb.predict(X_test)

    print("Accuracy of NB Classifier:", gnb.score(X_test, y_test))
    print(classification_report(y_test, y_predict))


def kNNClassifier(X_train, X_test, y_train, y_test):
    """Train/evaluate a k-Nearest Neighbors classifier."""
    knc = KNeighborsClassifier()
    knc.fit(X_train, y_train)
    y_predict = knc.predict(X_test)

    print("Accuracy of kNN Classifier:", knc.score(X_test, y_test))
    print(classification_report(y_test, y_predict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train baseline models on a labeled equity dataset.")
    parser.add_argument("--data", default="AAPL.csv", help="Path to the input CSV dataset")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Name of the label column")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test split fraction")
    parser.add_argument("--time-split", action="store_true", help="Use time-based split instead of random split")
    parser.add_argument("--date-col", default="Date", help="Date column name for time-based split")
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.info("Loading dataset: %s", args.data)

    # Load the data from csv file
    data = load_dataset(args.data, DEFAULT_FEATURES, args.label)

    # Split the dataset into train set and test set
    if args.time_split:
        logging.info("Using time-based split on column: %s", args.date_col)
        X_train, X_test, y_train, y_test = time_series_split(
            data,
            DEFAULT_FEATURES,
            args.label,
            args.date_col,
            args.test_size,
        )
    else:
        logging.info("Using random train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            data[DEFAULT_FEATURES],
            data[args.label],
            test_size=args.test_size,
            random_state=33,
        )

    # Standardize features using z-score normalization
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # Run all baseline models in sequence
    logisticRegression(X_train, X_test, y_train, y_test)
    sgdClassifier(X_train, X_test, y_train, y_test)
    linearSVC(X_train, X_test, y_train, y_test)
    NBClassifier(X_train, X_test, y_train, y_test)
    kNNClassifier(X_train, X_test, y_train, y_test)

    # clf = svm.SVC(kernel='linear', C=1)
    # scores = cross_validation.cross_val_score(clf, X_train, y_train, cv = 10)
    # print(type(scores))
    # print(scores)
