# main_trainer.py

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import sqlite3
from pathlib import Path
from modelagem.feature_eng.match_analysis import get_storage_ranks, create_main_cols
from modelagem.utils.logs import setup_logger

# Definindo diretórios base
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'database'
FT_DIR = BASE_DIR / 'modelagem' / 'feature_eng' / 'data'
LOG_DIR = DATA_DIR / 'logs'

logger = setup_logger(LOG_DIR, "main_train.log")

# Função para criar conexão com banco de dados SQLite
def create_connection(db_path):
    try:
        connection = sqlite3.connect(db_path)
        logger.debug("Conexão SQLite estabelecida.")
        return connection
    except sqlite3.Error as e:
        logger.error(f"Erro ao conectar ao SQLite: {e}")
        return None

# Função principal
def main():
    try:
        conn = create_connection("database/soccer_data.db")
        country = "Brazil"
        query = "SELECT * FROM soccer_data WHERE country = ?"
        df = pd.read_sql_query(query, conn, params=(country,))
        logger.info("Dados carregados do banco de dados com sucesso.")
    except Exception as exception:
        logger.error(f"ERROR: {exception}")
        return
    finally:
        if conn:
            conn.close()

    # Preprocessamento inicial
    df['match_name'] = df['home_team'] + ' - ' + df['away_team']
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    df = df[["season", "datetime", "home_team", "away_team", 
             "home_score", "away_score", "result", "match_name"]]
    
    to_int = ['season', 'home_score', 'away_score']
    df[to_int] = df[to_int].astype(int)

    # Encoding de times
    label_encoder = LabelEncoder()
    df['home_team_encoder'] = label_encoder.fit_transform(df['home_team'])
    df['away_team_encoder'] = label_encoder.fit_transform(df['away_team'])

    # Calculando vencedores e pontos
    dict_result = {'DRAW': 0, 'AWAY_WIN': 1, 'HOME_WIN': 2}
    df['winner'] = np.where(df['home_score'] > df['away_score'], dict_result["HOME_WIN"], 
                            np.where(df['home_score'] < df['away_score'], dict_result['AWAY_WIN'], dict_result['DRAW']))
    df['h_match_points'] = np.where(df['winner'] == dict_result['HOME_WIN'], 3, 
                                    np.where(df['winner'] == dict_result['DRAW'], 1, 0))
    df['a_match_points'] = np.where(df['winner'] == dict_result['AWAY_WIN'], 3, 
                                    np.where(df['winner'] == dict_result['DRAW'], 1, 0))

    # Reordenando colunas
    cols_order = ['season', 'datetime', 'match_name', 'home_team', 'away_team',
                  'home_team_encoder', 'away_team_encoder', 'winner', 'home_score', 
                  'away_score', 'h_match_points', 'a_match_points']
    df = df[cols_order]

    # Feature Engineering
    cols = ['_rank', '_ls_rank', '_days_ls_match', '_points', '_l_points', 
            '_l_wavg_points', '_goals', '_l_goals', '_l_wavg_goals', '_goals_sf', 
            '_l_goals_sf', '_l_wavg_goals_sf', '_wins', '_draws', '_losses', 
            '_win_streak', '_loss_streak', '_draw_streak']
    ht_cols = ['ht' + col for col in cols]
    at_cols = ['at' + col for col in cols]

    logger.debug("Iniciando o feature engineering do time da casa.")
    df[ht_cols] = pd.DataFrame(
        df.apply(lambda x: create_main_cols(x, x.home_team, df), axis=1).to_list(), index=df.index)
    
    logger.debug("Iniciando o feature engineering do time visitante.")
    df[at_cols] = pd.DataFrame(
        df.apply(lambda x: create_main_cols(x, x.away_team, df), axis=1).to_list(), index=df.index)

    # Removendo colunas desnecessárias e salvando resultados
    cols_to_drop = ['match_name', 'datetime', 'home_team', 'away_team', 
                    'home_score', 'away_score', 'h_match_points', 'a_match_points']
    df.drop(columns=cols_to_drop, inplace=True)
    df.fillna(-33, inplace=True)
    df_dum = pd.get_dummies(df)

    output_path = FT_DIR / 'ft_df.csv'
    os.makedirs(FT_DIR, exist_ok=True)
    df_dum.to_csv(output_path, index=False)
    logger.info(f"Feature DataFrame salvo em {output_path}")

    
if __name__ == "__main__":
    main()
