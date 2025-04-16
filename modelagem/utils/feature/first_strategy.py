import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from modelagem.features.match_analysis import get_storage_ranks, create_main_cols
from modelagem.utils.logs import logger
# Função para encoding dos times
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

from modelagem.settings.config import Settings
config = Settings()

# # Definindo diretórios base
# MODEL_DIR = os.path.join('database', 'models')
# # LOG_DIR = os.path.join('database', 'logs')
# FT_DIR = Path('database', "features")
# LOG_DIR = Path('database', "logs")


def encode_teams(df: pd.DataFrame, path_team_mapping: str) -> pd.DataFrame:
    """
    Codifica os times usando Label Encoding e salva o mapeamento de-para em um arquivo JSON.

    :param df: DataFrame contendo os dados das partidas.
    :param path_team_mapping: Caminho do arquivo onde o mapeamento será salvo.
    :return: DataFrame com colunas adicionais para os times codificados.
    """
    df = df.copy()

    label_encoder = LabelEncoder()
    all_teams = pd.concat([df['home_team'], df['away_team']])
    label_encoder.fit(all_teams)

    # Codifica os times
    df['home_team_encoder'] = label_encoder.transform(df['home_team'])
    df['away_team_encoder'] = label_encoder.transform(df['away_team'])

    # Cria e salva o mapeamento
    json_team_mapping = {team: int(code) for team, code in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    
    with open(path_team_mapping, 'w', encoding='utf-8') as f:
        json.dump(json_team_mapping, f, ensure_ascii=False, indent=4)

    return df


# Função para calcular vencedores e pontos
def calculate_match_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para calcular o vencedor de uma partida e os pontos correspondentes.
    Ela adiciona colunas ao DataFrame original para indicar o vencedor e os pontos ganhos por cada time.

    :param df: DataFrame contendo os dados das partidas.
    :return: DataFrame com colunas adicionais para o vencedor e os pontos.
    """
    
    df = df.copy()

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
def base_pre_processing(df:pd.DataFrame)->tuple[bool, str|pd.DataFrame]:
    """
    Função para pré-processar os dados de futebol.
    Ela realiza as seguintes etapas:
    1. Carrega os dados de um DataFrame.
    2. Cria colunas iniciais, como 'match_name' e 'datetime'.
    3. Seleciona colunas importantes para análise.
    4. Converte colunas para o tipo inteiro.
    5. Realiza o encoding dos times.
    6. Calcula os pontos e resultados das partidas.
    7. Realiza feature engineering para criar novas colunas.
    8. Salva o DataFrame resultante em um arquivo CSV.
    9. Retorna True se o pré-processamento for bem-sucedido, caso contrário, retorna False.
    
    :param df: DataFrame contendo os dados das partidas.
    :return: True se o pré-processamento for bem-sucedido, False caso contrário.
    """
    df = df.copy()  # Evita o warning

    logger.info("Iniciando pré-processamento dos dados...")
    
    if df is None or df.empty:
        logger.error("Nenhum dado foi carregado. Encerrando pré-processamento.")
        return False, 'Nenhum dado foi carregado'

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
        logger.debug("Iniciando o encoding dos times.")
        df = encode_teams(df,
                          path_team_mapping=os.path.join(config.MAPPING_DIR, "team_mapping.json"))

        # Calculando pontos e resultado das partidas
        logger.debug("Iniciando o cálculo dos pontos e resultados das partidas.")
        df = calculate_match_points(df)

        # Reordenando colunas
        cols_order = ['season', 'datetime', 'match_name', 'home_team', 'away_team',
                      'home_team_encoder', 'away_team_encoder', 'winner', 'home_score', 
                      'away_score', 'h_match_points', 'a_match_points']
        df = df[cols_order]

        # Feature Engineering
        logger.debug("Iniciando o feature engineering.")
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

        logger.debug("Salvando o DataFrame resultante.")
        os.makedirs(config.FT_DIR, exist_ok=True)
        output_path = os.path.join(config.FT_DIR, 'ft_df.csv')
        df.to_csv(output_path, index=False)
        
        logger.info(f"Feature DataFrame salvo em {output_path}")
        return True, df

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}", exc_info=True)
        return False, 'Erro no pré-processamento'


def get_feature_columns():
    """Retorna a lista de colunas usadas na engenharia de features"""
    return ['_rank', '_ls_rank', '_days_ls_match', '_points', '_l_points', 
            '_l_wavg_points', '_goals', '_l_goals', '_l_wavg_goals', '_goals_sf', 
            '_l_goals_sf', '_l_wavg_goals_sf', '_wins', '_draws', '_losses', 
            '_win_streak', '_loss_streak', '_draw_streak']

