import os
import ast
import numpy as np
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
from dotenv import load_dotenv
from steam_web_api import Steam
import tarfile
from io import BytesIO
import boto3
import joblib

load_dotenv('.env')
api_key = os.getenv('API_KEY')
bucket_name = os.environ["BUCKET"]
aws_access_key = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

# configure s3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
)

# get the mlb objects for pre processing
object_key = 'price_history/preprocessing/mlb/mlb.tar.gz'
tar_obj = s3.get_object(Bucket=bucket_name, Key=object_key)
tar_data = tar_obj['Body'].read()

def load_sklearn_object(tar, filename):
    for member in tar.getmembers():
        if member.name == filename:
            f = tar.extractfile(member)
            return joblib.load(f)
    return None

# Extract the contents of the tar file
tar = tarfile.open(fileobj=BytesIO(tar_data), mode='r:gz')
mlb_cat = load_sklearn_object(tar, 'mlb_cat.joblib')
mlb_tag = load_sklearn_object(tar, 'mlb_tag.joblib')

# and now let's load the models
def load_xgboost_model(model_path):
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model
local_model_path = Path("local_model_dir")
xgboost_clf = load_xgboost_model(local_model_path / "discount_on_release-xgboost")
xgboost_reg = load_xgboost_model(local_model_path / "time_until_discount-xgboost")

# enable steam api access
steam_api_key = os.getenv('STEAM_API_KEY')
steam_client = Steam(steam_api_key)

def query_games(name, steam_client=steam_client):
    """
    This function takes in a query and searches the steam database/API
    for relevant games to that query. 
    """
    search_games_lst = steam_client.apps.search_games(name)
    games_lst = search_games_lst['apps']

    games_dct = {}

    for game in games_lst:
        # don't want game collections or empty id's
        if len(game['id']) == 1:
            games_dct[game['name']] = game['id'][0]

    # decode unicode escape sequences
    decoded_game_dict = {name.encode('latin1').decode('unicode_escape'): app_id for name, app_id in games_dct.items()}

    return decoded_game_dict

# helper function to make requests and prevent repetitive code
def request_game_data(endpoint, params):
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f'Error: {response.status_code}'

def get_game_id(appid, api_key=api_key, endpoint = 'https://api.isthereanydeal.com/games/lookup/v1'):
    params = {
        'key': api_key,
        'appid': appid
    }

    game_lookup = request_game_data(endpoint, params)
    
    return game_lookup['game']['id']

#with the kaggle data, we already have release dates, so this is unnecessary for now
def get_all_game_info(game_id, api_key=api_key, endpoint = 'https://api.isthereanydeal.com/games/info/v2'):
    params = {
        'key': api_key,
        'id': game_id
    }

    game_info = request_game_data(endpoint, params)
    
    return game_info


def process_raw_steam_data(steam_data):
    """ Gathering raw data from the steam API and putting it in proper form
    Categories were originally in the form of one string, seperated by a comma
    Mac and Linux data come in the form of lists, but we just want to know if they are supported
    i.e. are they empty or not.
    """
    categories = steam_data['categories']
    game_categories = ''
    
    for cat in categories:
        game_categories += cat['description'] + ','
    
    game_categories = game_categories[:-1] # remove last comma

    game_supported_languages = steam_data['supported_languages']

    # to match the format of the original games_df,
    # remove white space and then split on the commas
    game_supported_languages = game_supported_languages.replace(' ', '').split(',')

    # mac and linux are True/False columns, so just need to check if empty
    if steam_data['mac_requirements']:
        mac = True
    else:
        mac = False

    if steam_data['linux_requirements']:
        linux = True
    else:
        linux = False

    return game_categories, game_supported_languages, mac, linux


def get_game_data(steam_app_id, any_deals_game_info, steam_game_details):
    """ This function just gets data from API data and makes sure it is in a proper form
    """
    release_date = any_deals_game_info['releaseDate']
    achievements = any_deals_game_info['achievements']
    tags = any_deals_game_info['tags']
    tags = [tag.replace(' ',  '-') for tag in any_deals_game_info['tags']]
    ','.join(tags)

    steam_data = steam_game_details[f"{steam_app_id}"]['data']
    categories, supported_languages, mac, linux = process_raw_steam_data(steam_data)

    # need to turn lists into strings to convert to dataframe later
    supported_languages = str(supported_languages)

    return achievements, supported_languages, mac, linux, categories, tags, release_date

def preprocess_game(game_df):
    """This function preprocesses the columns
    Taking Binary columns and making sure they are int values
    Converting lists in the Supported languages column from string type to list, 
    and then taking the length
    Finally, getting just the month of release and dropping the original release date
    """
    game_df['Achievements'] = game_df['Achievements'].astype(int)
    game_df['Mac'] = game_df['Mac'].astype(int)
    game_df['Linux'] = game_df['Linux'].astype(int)

    # converting supported languages to just the number of them
    game_df['Supported languages'] = game_df['Supported languages'].apply(ast.literal_eval)
    game_df['Supported languages'] = game_df['Supported languages'].apply(lambda lst: len(lst))

    # converting release date to the month only
    game_df['ReleaseDate'] = pd.to_datetime(game_df['ReleaseDate'])
    game_df['month'] = game_df['ReleaseDate'].dt.month

    # don't need ReleaseDate anymore
    game_df = game_df.drop(columns=['ReleaseDate'])

    return game_df


def main():
    """
    This is the main streamlit app function.

    Let's user search a key word and presents a selection of relevant games
    Then the user picks a game and we predict how long until the game goes on sale.
    The xgboost classifier will predict if the game goes on sale within 2 days
    If it doesn't, then we predict how many months until it does, rounding to the nearest int.
    """
    st.markdown(
        """
        <h1 style='text-align: center; color: #4FC3F7;'>Steam Game Sale Prediction</h1>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <p>Not sure if you want to buy a new release at full price? No problem!</p>
            <p>Here you can predict how long you'll have to wait for a game to go on sale.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    predictions_col, box_art_col = st.columns(2)


    with predictions_col:
        # Input text box for user to enter a game name
        game_name = st.text_input("Search for a relevant game:")

        if game_name:
            # Query the games using the input text
            query_results_dct = query_games(game_name)


            # Display the list of game names (keys of the output dictionary)
            if query_results_dct:
                # empty spacing for aesthetics
                st.write("")
                selected_game = st.selectbox("We found the following:", list(query_results_dct.keys()))
                
                if st.button("Estimate Sale Date"):
                    steam_app_id = query_results_dct[selected_game]

                    game_id = get_game_id(steam_app_id, api_key)
                    any_deals_game_info = get_all_game_info(game_id)

                    boxart = any_deals_game_info['assets']['boxart']

                    steam_game_details = steam_client.apps.get_app_details(
                        app_id=steam_app_id, 
                        country='US', 
                        filters='basic,categories'
                    )

                    achievements, supported_languages, mac, linux, categories, tags, release_date = get_game_data(steam_app_id, any_deals_game_info, steam_game_details)
                    # use a dict with proper column order as preprocessing steps, and then convert to dataframe.
                    selected_game_dct = {
                        'Achievements': achievements, 'Supported languages': supported_languages, 
                        'Mac': mac, 'Linux': linux, 'Categories': categories, 
                        'Tags': tags, 'ReleaseDate': release_date
                    }
                    selected_game_df = pd.DataFrame([selected_game_dct])
                    selected_game_df = preprocess_game(selected_game_df)

                    # we need to one hot encode the multilabel columens and remove originals after
                    one_hot_tags = mlb_tag.transform(selected_game_df['Tags'])
                    one_hot_tags_df = pd.DataFrame(one_hot_tags, columns=mlb_tag.classes_)

                    one_hot_cats = mlb_cat.transform(selected_game_df['Categories'])
                    one_hot_cats_df = pd.DataFrame(one_hot_cats, columns=mlb_cat.classes_)

                    selected_game_df = selected_game_df.drop(columns=['Categories', 'Tags'])
                    selected_game_df = pd.concat([selected_game_df, one_hot_cats_df, one_hot_tags_df], axis=1)

                    # these were duplicate columns discovered when one-hot-encoding, we removed them for simplicity
                    selected_game_df = selected_game_df.drop(columns=['Co-op', 'PvP'])

                    # we need to convert the data to be inputted into xgboost models 
                    input_data = xgb.DMatrix(selected_game_df)
                    discount_on_release = xgboost_clf.predict(input_data)

                    # if discount_on_release is less than 0.5 
                    # then the game will be on sale within 2 days
                    # else we calculate how long until it does go on sale (months)
                    if discount_on_release < 0.5:
                        sale_prediction = 'less than a week'
                    else:
                        time_until_discount = xgboost_reg.predict(input_data)
                        months_until_sale = np.expm1(time_until_discount[0])
                        months_until_sale = round(months_until_sale)

                        # plural gramer
                        if months_until_sale == 1:
                            sale_prediction = f"{months_until_sale} month"
                        else:
                            sale_prediction = f"{months_until_sale} months"

                    st.write("")
                    st.write("")
                    st.markdown("<p style='font-size: 24px; text-align: center;'>Will go on sale in:</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 32px; font-weight: bold; text-align: center;'>{sale_prediction}</p>", unsafe_allow_html=True)

            else:
                st.write("No results found for the entered game name.")

            with box_art_col:
                try:
                    if 'boxart' in locals():
                        st.image(boxart, caption='Game Box Art')
                except Exception:
                    pass

if __name__ == "__main__":
    main()