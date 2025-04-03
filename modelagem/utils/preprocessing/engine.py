import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from modelagem.feature_eng.match_analysis import get_storage_ranks, create_main_cols
from modelagem.utils.logs import logger


# Definindo diretórios base
# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = BASE_DIR / 'database'
# FT_DIR = BASE_DIR / 'modelagem' / 'feature_eng' / 'data'
# LOG_DIR = DATA_DIR / 'logs'

BASE_DIR = os.path.dirname(Path(__file__).resolve().parent)
DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data', 'ft_df.csv')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'database', 'models')
LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'logs')

# Diretórios
FT_DIR = Path("features")
LOG_DIR = Path("logs")

# Função para encoding dos times
def encode_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Faz uma cópia explícita para evitar warnings
    
    label_encoder = LabelEncoder()
    all_teams = pd.concat([df['home_team'], df['away_team']])
    label_encoder.fit(all_teams)

    df['home_team_encoder'] = label_encoder.transform(df['home_team'])
    df['away_team_encoder'] = label_encoder.transform(df['away_team'])

    return df

# Função para calcular vencedores e pontos
def calculate_match_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Evita o warning

    dict_result = {'DRAW': 0, 'AWAY_WIN': 1, 'HOME_WIN': 2}
    
    # Garantir que df seja uma cópia independente
    df = df.copy()

    df.loc[:, 'winner'] = np.select(
        [df['home_score'] > df['away_score'], df['home_score'] < df['away_score']],
        [dict_result["HOME_WIN"], dict_result["AWAY_WIN"]],
        default=dict_result['DRAW']
    )

    df.loc[:, 'h_match_points'] = np.select(
        [df['winner'] == dict_result['HOME_WIN'], df['winner'] == dict_result['DRAW']],
        [3, 1],
        default=0
    )

    df.loc[:, 'a_match_points'] = np.select(
        [df['winner'] == dict_result['AWAY_WIN'], df['winner'] == dict_result['DRAW']],
        [3, 1],
        default=0
    )

    return df

# Função principal
def base_pre_processing(df:pd.DataFrame):
    df = df.copy()  # Evita o warning

    logger.info("Iniciando pré-processamento dos dados...")
    
    if df is None or df.empty:
        logger.error("Nenhum dado foi carregado. Encerrando pré-processamento.")
        return False

    try:
        # Criando colunas iniciais
        df['match_name'] = df['home_team'] + ' - ' + df['away_team']
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Selecionando colunas importantes
        df = df[["season", "datetime", "home_team", "away_team", 
                 "home_score", "away_score", "result", "match_name"]]

        # Convertendo colunas para inteiro
        to_int = ['season', 'home_score', 'away_score']
        df[to_int] = df[to_int].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        # Encoding dos times
        df = encode_teams(df)

        # Calculando pontos e resultado das partidas
        df = calculate_match_points(df)

        # Reordenando colunas
        cols_order = ['season', 'datetime', 'match_name', 'home_team', 'away_team',
                      'home_team_encoder', 'away_team_encoder', 'winner', 'home_score', 
                      'away_score', 'h_match_points', 'a_match_points']
        df = df[cols_order]

        # Feature Engineering
        df_storage_ranks = get_storage_ranks(df)

        logger.debug("Iniciando o feature engineering do time da casa.")
        ht_cols = [f'ht{col}' for col in get_feature_columns()]
        at_cols = [f'at{col}' for col in get_feature_columns()]

        df[ht_cols] = df.apply(lambda x: create_main_cols(x, x.home_team, df, df_storage_ranks), axis=1, result_type='expand')
        df[at_cols] = df.apply(lambda x: create_main_cols(x, x.away_team, df, df_storage_ranks), axis=1, result_type='expand')

        # Removendo colunas desnecessárias
        df.drop(columns=['match_name', 'datetime', 'home_team', 'away_team', 
                         'home_score', 'away_score', 'h_match_points', 'a_match_points'], inplace=True)

        df.fillna(-33, inplace=True)  # Preenchendo valores ausentes
        
        # Salvando resultados
        os.makedirs(FT_DIR, exist_ok=True)
        output_path = os.path.join(FT_DIR, 'ft_df.csv')
        df.to_csv(output_path, index=False)
        
        logger.info(f"Feature DataFrame salvo em {output_path}")
        return True

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}", exc_info=True)
        return False


def get_feature_columns():
    """Retorna a lista de colunas usadas na engenharia de features"""
    return ['_rank', '_ls_rank', '_days_ls_match', '_points', '_l_points', 
            '_l_wavg_points', '_goals', '_l_goals', '_l_wavg_goals', '_goals_sf', 
            '_l_goals_sf', '_l_wavg_goals_sf', '_wins', '_draws', '_losses', 
            '_win_streak', '_loss_streak', '_draw_streak']

