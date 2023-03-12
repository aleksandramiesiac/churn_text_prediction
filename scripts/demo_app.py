# import packages

import os, yaml, joblib, warnings
from train import Net
from pathlib import Path

import torch
from torch import Tensor

import numpy as np
import streamlit as st
from utils import load_feature_names, PolishBertEncoder

warnings.filterwarnings("ignore")
with open("../model_cased/all_translate/params.yaml", "r") as params_file:
    params_cased = yaml.safe_load(params_file)
with open("../model_sdadas/all_translate/params.yaml", "r") as params_file:
    params_sdadas = yaml.safe_load(params_file)
with open("../model_uncased/all_translate/params.yaml", "r") as params_file:
    params_uncased = yaml.safe_load(params_file)

model_dir = params_cased['model_dir']


# function to clean the text
@st.cache
def transform_text(text, model_name):
    # Transform text to embeddings

    # load the transformator
    tt = PolishBertEncoder(model_name)
    text_transform = tt.transform(text)

    # Return an embedding
    return text_transform[0].squeeze(0).numpy()


@st.cache
def transform_category(category_features, params):
    # Transform category features to one hot encoded features
    # load the onehotencoder
    ct = joblib.load(Path(params['model_dir'], "categorical_transformer.joblib"))
    category_features_transform = ct.transform(category_features)

    # Return an categorized features
    return category_features_transform.toarray()[0]


@st.cache
def predict(
        features,
        params
):
    # get size of num/cat & text data to specify neural net
    numerical_feature_names, categorical_feature_names, _ = load_feature_names(
        Path(params['model_dir'], "one_hot_feature_names.json"))
    number_cat_num_features = len(numerical_feature_names) + len(categorical_feature_names)
    x1_size = len(features[:number_cat_num_features])
    x2_size = len(features[number_cat_num_features:])

    model = Net(x1_size, x2_size)
    with open(os.path.join(params['model_dir'], 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.eval()
    with torch.no_grad():
        output = model(Tensor(features[:number_cat_num_features]).unsqueeze(0),
                       Tensor(features[number_cat_num_features:]).unsqueeze(0)).reshape(-1)
        result = torch.sigmoid(output)
    probability = "{:.2f}".format(float(result))

    return result, probability


def make_prediction(numeric_data, category_data, text_data, params):
    # clear the data
    new_text = transform_text(text_data, params['model_name'])
    new_cat = transform_category(category_data, params)

    features = np.concatenate([
        numeric_data,
        new_cat,
        new_text
    ])
    # load the model and make prediction
    return predict(features, params)


# Set the app title
st.title("Churn Analyisis App")
st.write(
    "A simple machine laerning app to predict the churn"
)

# Declare a form to receive a movie's review
review = st.text_area(label="Enter the text of your chat log",
                      value="""Klient: Chciałbym zaktualizować usługę głosową. Będę musiał podpisać nową umowę.
                      Agent Telcom: To świetny pomysł Leroy. Mogę zacząć od poszukiwania obecnych stawek dla telefonu VoIP.
                      Klient: Możemy uaktualnić z obecnego głosu do VoIP.
                      Agent Telcom: Byłoby świetnie. Mogę zacząć od przeczytania nowych dostępnych stawek z Telcom, jeśli chcesz.
                      Klient: Chciałbym zaktualizować z bieżącego tekstu do VoIP.
                      Agent Telcom: mogę zacząć od sprawdzenia bieżącego""")
state = st.selectbox(label="Choose your state",
                     options=['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                              'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                              'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                              'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                              'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])
account_length = st.number_input(label="Number of months the customer has been with the current telco provider",
                                 value=115)
area_code = 'area_code' + str(st.selectbox(label="Choose your area code", options=[408, 415, 510]))
international_plan = st.selectbox(label="Choose if you have an International plan", options=["yes", "no"])
voice_mail_plan = st.selectbox(label="Choose if you have an Voice mail plan", options=["yes", "no"])
number_vmail_messages = st.number_input(label="Enter how many vmail massages did you have", value=0)
total_day_minutes = st.number_input(label="Total minutes of day calls", value=345.3)
total_day_calls = st.number_input(label="Total number of day calls", value=81)
total_day_charge = st.number_input(label="Total charge of day calls", value=58.7)
total_eve_minutes = st.number_input(label="Total minutes of evening calls", value=203.4)
total_eve_calls = st.number_input(label="Total number of evening calls", value=106)
total_eve_charge = st.number_input(label="Total charge of evening calls", value=17.29)
total_night_minutes = st.number_input(label="Total minutes of night calls", value=217.5)
total_night_calls = st.number_input(label="Total number of night calls", value=107)
total_night_charge = st.number_input(label="Total charge of night calls", value=9.79)
total_intl_minutes = st.number_input(label="Total minutes of international calls", value=11.8)
total_intl_calls = st.number_input(label="Total number of international calls", value=8)
total_intl_charge = st.number_input(label="Total charge of international calls", value=3.19)
number_customer_service_calls = st.number_input(label="Number of calls to customer service", value=1)

if st.button(label="Make Prediction"):
    # make prediction from the input text build on a cased model
    result_cased, probability_cased = make_prediction(
        [
            account_length, number_vmail_messages, total_day_minutes, total_day_calls, total_day_charge,
            total_eve_minutes,
            total_eve_calls, total_eve_charge, total_night_minutes, total_night_calls, total_night_charge,
            total_intl_minutes, total_intl_calls, total_intl_charge, number_customer_service_calls
        ],
        [[
            state, area_code, international_plan, voice_mail_plan
        ]],
        [
            review
        ],
        params=params_cased
    )
    # Display results of the NLP task build on cased model
    st.header("Results of the NLP task build on cased model")
    st.write("Probability of churn is: ", probability_cased)

    # make prediction from the input text build on a sdadas model
    result_sdadas, probability_sdadas = make_prediction(
        [
            account_length, number_vmail_messages, total_day_minutes, total_day_calls, total_day_charge,
            total_eve_minutes,
            total_eve_calls, total_eve_charge, total_night_minutes, total_night_calls, total_night_charge,
            total_intl_minutes, total_intl_calls, total_intl_charge, number_customer_service_calls
        ],
        [[
            state, area_code, international_plan, voice_mail_plan
        ]],
        [
            review
        ],
        params=params_sdadas
    )
    # Display results of the NLP task build on sdadas model
    st.header("Results of the NLP task build on sdadas model")
    st.write("Probability of churn is: ", probability_sdadas)

    # make prediction from the input text build on a uncased model
    result_uncased, probability_uncased = make_prediction(
        [
            account_length, number_vmail_messages, total_day_minutes, total_day_calls, total_day_charge,
            total_eve_minutes,
            total_eve_calls, total_eve_charge, total_night_minutes, total_night_calls, total_night_charge,
            total_intl_minutes, total_intl_calls, total_intl_charge, number_customer_service_calls
        ],
        [[
            state, area_code, international_plan, voice_mail_plan
        ]],
        [
            review
        ],
        params=params_uncased
    )
    # Display results of the NLP task build on uncased model
    st.header("Results of the NLP task build on uncased model")
    st.write("Probability of churn is: ", probability_uncased)


