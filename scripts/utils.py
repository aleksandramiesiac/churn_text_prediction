from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.notebook import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def extract_labels(
        data,
        label_name
):
    labels = []
    for sample in data:
        value = sample[label_name]
        labels.append(value)
    labels = np.array(labels).astype('int')

    return labels.reshape(labels.shape[0], -1)


def convert_label(
        df
):
    df.churn = df.churn.replace("no", 0)
    df.churn = df.churn.replace("yes", 1)
    return df


def split_data(
        df,
        label_name,
        test_size
):
    """Splits data and creates json format.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[label_name], axis=1),
        df[label_name],
        test_size=test_size,
        random_state=123,
        stratify=df[label_name])
    X_train[label_name] = y_train
    X_test[label_name] = y_test

    return X_train, X_test


def save_feature_names(
        numerical_feature_names,
        categorical_feature_names,
        textual_feature_names,
        filepath
):
    feature_names = {
        'numerical': numerical_feature_names,
        'categorical': categorical_feature_names,
        'textual': textual_feature_names
    }
    with open(filepath, 'w') as f:
        json.dump(feature_names, f)


def load_feature_names(filepath):
    with open(filepath, 'r') as f:
        feature_names = json.load(f)
    numerical_feature_names = feature_names['numerical']
    categorical_feature_names = feature_names['categorical']
    textual_feature_names = feature_names['textual']
    return numerical_feature_names, categorical_feature_names, textual_feature_names


def get_feature_names(
        df
):
    num_columns = df.select_dtypes(include=np.number).columns.tolist()
    numerical_feature_names = [i for i in num_columns if i not in ['churn']]

    cat_columns = df.select_dtypes(include='object').columns.tolist()
    categorical_feature_names = [i for i in cat_columns if i not in ['chat_log']]

    textual_feature_names = ['chat_log']
    label_name = 'churn'

    return numerical_feature_names, categorical_feature_names, textual_feature_names, label_name


class PolishBertEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='dkleczek/bert-base-polish-cased-v1'):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, token_embeddings, attention_mask):
        return torch.sum(token_embeddings, 1) / attention_mask.size()[1]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sentence_emb = []
        for sentence in tqdm(X):
            encoded_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
            # Compute token embeddings
            with torch.no_grad():
                model_output = \
                self.model(attention_mask=encoded_input['attention_mask'], input_ids=encoded_input['input_ids'])[
                    0]  # First element of model_output contains all token embeddings

            sentence_embeddings = self.mean_pooling(model_output, encoded_input[
                'attention_mask'])  # Perform pooling. In this case, mean pooling.
            sentence_emb.append(sentence_embeddings)
        return sentence_emb


def main_transformation(df, model_dir, model_name, use_existing=False, test_size=0.33):
    """
    Args:
        df: Pandas dataframe with raw data
        use_existing: Set to True if you want to use locally stored, 
        already prepared train/test data. Set to False if you want 
        to rerun the data preparation pipeline.
    Returns:
        Train and test data as well as train labels and test labels.
  """
    train_file = Path(model_dir, 'train.csv')
    labels_file = Path(model_dir, 'labels.csv')
    test_file = Path(model_dir, 'test.csv')
    labels_test_file = Path(model_dir, 'labels_test.csv')
    feature_names_file = Path(model_dir, "feature_names.json")
    oh_feature_names_file = Path(model_dir, "one_hot_feature_names.json")
    all_file_paths = [train_file, labels_file, test_file, labels_test_file,
                      feature_names_file, oh_feature_names_file]

    if use_existing == True and sum([file.exists() for file in all_file_paths]) == 6:
        features = np.array(pd.read_csv(train_file))
        labels = np.array(pd.read_csv(labels_file))
        features_test = np.array(pd.read_csv(test_file))
        labels_test = np.array(pd.read_csv(labels_test_file))
        print("Using already prepared data.")

    else:
        print("Running data preparation pipeline...")
        # convert label to binary numeric
        df = convert_label(df)

        # get features categorial names
        numerical_feature_names, categorical_feature_names, textual_feature_names, label_name = get_feature_names(df)

        # split data
        X_train, X_test = split_data(df, label_name, test_size)

        # extract features & label
        print('extracting features')
        numerical_features = X_train[numerical_feature_names].values
        categorical_features = X_train[categorical_feature_names].values
        textual_features = X_train[textual_feature_names].values
        textual_features = [str(chat[0]) for chat in textual_features.tolist()]

        labels = X_train[label_name].values

        # define preprocessors
        print('defining preprocessors')
        numerical_transformer = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        textual_transformer = PolishBertEncoder(model_name)

        # fit preprocessors
        print('fitting numerical_transformer')
        numerical_transformer.fit(numerical_features)
        print('saving numerical_transformer')
        joblib.dump(numerical_transformer, Path(model_dir, "numerical_transformer.joblib"))
        print('fitting categorical_transformer')
        categorical_transformer.fit(categorical_features)
        print('saving categorical_transformer')
        joblib.dump(categorical_transformer, Path(model_dir, "categorical_transformer.joblib"))

        # transform features
        print('transforming numerical_features')
        numerical_features = numerical_transformer.transform(numerical_features)
        print('transforming categorical_features')
        categorical_features = categorical_transformer.transform(categorical_features)
        print('transforming textual_features')
        textual_features = textual_transformer.transform(textual_features)

        numerical_features_test = X_test[numerical_feature_names].values
        categorical_features_test = X_test[categorical_feature_names].values
        textual_features_test = X_test[textual_feature_names].values
        textual_features_test = [str(chat[0]) for chat in textual_features_test.tolist()]

        labels_test = X_test[label_name].values

        # transform features (for test data)
        print('transforming numerical_features_test')
        numerical_features_test = numerical_transformer.transform(numerical_features_test)
        print('transforming categorical_features_test')
        categorical_features_test = categorical_transformer.transform(categorical_features_test)
        print('transforming textual_features_test')
        textual_features_test = textual_transformer.transform(textual_features_test)

        # concat features
        print('concatenating features')
        categorical_features = categorical_features.toarray()
        textual_features = np.array([t[0].squeeze(0).numpy() for t in textual_features])
        features = np.concatenate([
            numerical_features,
            categorical_features,
            textual_features
        ], axis=1)

        # concat features (test data)
        print('concatenating features of test data')
        categorical_features_test = categorical_features_test.toarray()
        textual_features_test = np.array([t[0].squeeze(0).numpy() for t in textual_features_test])
        features_test = np.concatenate([
            numerical_features_test,
            categorical_features_test,
            textual_features_test
        ], axis=1)

        # save to disk
        pd.DataFrame(features).to_csv(Path(model_dir, "train.csv"), index=False)
        pd.DataFrame(labels).to_csv(Path(model_dir, "labels.csv"), index=False)
        pd.DataFrame(features_test).to_csv(Path(model_dir, "test.csv"), index=False)
        pd.DataFrame(labels_test).to_csv(Path(model_dir, "labels_test.csv"), index=False)

        save_feature_names(
            numerical_feature_names,
            categorical_feature_names,
            textual_feature_names,
            Path(model_dir, "feature_names.json")
        )
        # one-hot encoded feature names (for feat_imp)
        save_feature_names(
            numerical_feature_names,
            categorical_transformer.get_feature_names_out().tolist(),
            textual_feature_names,
            Path(model_dir, "one_hot_feature_names.json")
        )

    return features, features_test, labels, labels_test
