import os
import pandas as pd
import streamlit
import xgboost as xgb
from dotenv import load_dotenv
from steam_web_api import Steam

load_dotenv('../.env')
steam_api_key = os.getenv('STEAM_API_KEY')
steam = Steam(steam_api_key)

def query_games(name, steam_client=steam):
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

