# | filename: script.py
# | code-line-numbers: true

import ast
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

def _read_data_from_input_csv_files(base_directory):
    """Read the data from the input CSV files.

    This function reads the input data from a CSV file 
    and does some simple cleaning.
    """
    input_directory = Path(base_directory) / "input"
    files = list(input_directory.glob("*.csv"))

    if len(files) == 0:
        message = f"The are no CSV files in {input_directory.as_posix()}/"
        raise ValueError(message)

    raw_data = [pd.read_csv(file) for file in files]

    raw_train_test_data = pd.concat(raw_data)
    raw_train_test_data = raw_train_test_data.drop(columns='Unnamed: 0')

    return raw_train_test_data

def _get_early_discount_target(df):
    """Filter for games that actually went on sale.

    This function uses a SaleType columns to only consider games that actually went on sale
    Then drops that column and creates a new binary one for games that went on sale on release or not
    """
    discounted_games = df[df['SaleType'] == 'went on sale']
    discounted_games = discounted_games.drop(columns='SaleType')
    discounted_games['DiscountedEarly'] = discounted_games['TimeDelta'].apply(lambda x: 'discounted within 2 days' if x < 3 else 'discounted after 3 days')
    return discounted_games

def _encoding_multilabel_column(df, feature, frequency_threshold):
    #replace labels with spacing with a dash so that they remain one word
    #then split into list based on commas
    df[feature] = df[feature].apply(lambda x: x.replace(' ', '-').split(','))

    all_labels = [label for sublist in df[feature] for label in sublist]
    labels_counter = Counter(all_labels)

    frequent_cats = {label for label, count in labels_counter.items() if count >= frequency_threshold}
    df[f"Filtered_{feature}"] = df[feature].apply(lambda x: [label for label in x if label in frequent_cats])

    mlb = MultiLabelBinarizer()

    # Fit and transform the data
    one_hot_encoded = mlb.fit_transform(df[f"Filtered_{feature}"])

    # Create a DataFrame with the one-hot encoded data
    one_hot_df_labels = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)

    return one_hot_df_labels

def _split_data_discount_on_release(df):
    """Split the data into train and test for classification.

    This function splits the data into training and testing sets
    for classification - predicting if a game went on sale within 2 days or not
    """
    # pass in preprocessed_tabular_df as input
    y = df['DiscountedEarly']
    X = df.drop(columns=['DiscountedEarly', 'TimeDelta'])

    # duplicate columns to manually drop
    X = X.drop(columns=['Co-op', 'PvP'])

    label_encoding = {'discounted within 2 days': 0, 'discounted after 3 days': 1}
    y = y.apply(lambda x: label_encoding[x])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoding

def _split_data_discount_after_release(df):
    """Split the data into train and test for regression.
    
    This function splits the data into training and testing sets 
    for predicting how many months until a game goes one sale,
    given that it didn't within the first 2 days.
    """
    regression_df = df[df['DiscountedEarly'] == 'discounted after 3 days']
    # target for classification task that we don't need anymore after filtering
    regression_df = regression_df.drop(columns='DiscountedEarly')

    #removing games that went on sale after 6 months for optimal model performance
    regression_df['TimeDelta'] = regression_df[['TimeDelta']] // 30 
    regression_df = regression_df[regression_df['TimeDelta'] < 6]

    # log transform since data has exponential decay
    y = np.log1p(regression_df['TimeDelta'])
    X = regression_df.drop(columns=['TimeDelta'])

    # duplicate columns to manually drop
    X = X.drop(columns=['Co-op', 'PvP'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def _save_splits(
    base_directory,
    X_train,  # noqa: N803
    y_train,
    X_test,  # noqa: N803
    y_test,
    train_path_name,
    test_path_name
):
    """Save data splits to disk.

    This function concatenates the transformed features
    and the target variable, and saves each one of the split
    sets to disk.

    train_path_name (str): should be either "train_clf" or train_reg"
    test_path_name (str): should be either "test_clf" or test_reg"
    """
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train_path = Path(base_directory) / train_path_name
    test_path = Path(base_directory) / test_path_name

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(train_path / f"{train_path_name}.csv", header=True, index=False)
    test.to_csv(test_path / f"{test_path_name}.csv", header=True, index=False)


def preprocess(base_directory):
    df = _read_data_from_input_csv_files(base_directory)

    discounted_games = _get_early_discount_target(df)

    tabular_df = discounted_games[
        [
        'Achievements', 'Supported languages',
        'Mac', 'Linux', 'Categories', 'Tags', 
        'ReleaseDate', 'TimeDelta', 'DiscountedEarly'
        ]
    ]

    # converting binary columns to ints
    tabular_df['Achievements'] = tabular_df['Achievements'].astype(int)
    tabular_df['Mac'] = tabular_df['Mac'].astype(int)
    tabular_df['Linux'] = tabular_df['Linux'].astype(int)

    # converting supported languages to just the number of them
    tabular_df['Supported languages'] = tabular_df['Supported languages'].apply(ast.literal_eval)
    tabular_df['Supported languages'] = tabular_df['Supported languages'].apply(lambda lst: len(lst))

    # converting release date to the month only
    tabular_df['ReleaseDate'] = pd.to_datetime(tabular_df['ReleaseDate'])
    tabular_df['month'] = tabular_df['ReleaseDate'].dt.month

    tabular_df = tabular_df.dropna(subset=['Categories', 'Tags'])
    tabular_df = tabular_df.reset_index(drop=True)

    one_hot_df_cats = _encoding_multilabel_column(tabular_df, 'Categories', 50)
    one_hot_df_tags = _encoding_multilabel_column(tabular_df, 'Tags', 100)

    preprocessed_tabular_df = tabular_df.drop(columns=['Categories', 'Tags', 'Filtered_Categories', 'Filtered_Tags', 'ReleaseDate'])
    preprocessed_tabular_df = pd.concat([preprocessed_tabular_df, one_hot_df_cats, one_hot_df_tags], axis=1)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf, label_encoding = _split_data_discount_on_release(preprocessed_tabular_df)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = _split_data_discount_after_release(preprocessed_tabular_df)

    _save_splits(base_directory, X_train_clf, y_train_clf, X_test_clf, y_test_clf, "train_clf", "test_clf")
    _save_splits(base_directory, X_train_reg, y_train_reg, X_test_reg, y_test_reg, "train_reg", "test_reg")


if __name__ == "__main__":
    preprocess(base_directory="/opt/ml/processing")
