import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import json

def encode_categorical_features(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    df = df.copy()

    df = encode_teams(df, os.path.join(path_encoder, "team_mapping.json"))
    df = encode_result(df, os.path.join(path_encoder, "result_mapping.json"))
    df = encode_match_day_of_week(df, os.path.join(path_encoder, "match_day_of_week_mapping.json"))
    df = encode_season_phase(df, os.path.join(path_encoder, "season_phase_mapping.json"))
    return df

def encode_match_day_of_week(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    df = df.copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(df['match_day_of_week'])
    df['match_day_of_week_encoder'] = label_encoder.transform(df['match_day_of_week'])

    # Cria e salva o mapeamento
    json_team_mapping = {team: int(code) for team, code in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    
    with open(path_encoder, 'w', encoding='utf-8') as f:
        json.dump(json_team_mapping, f, ensure_ascii=False, indent=4)

    return df

def encode_season_phase(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    df = df.copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(df['season_phase'])
    df['season_phase_encoder'] = label_encoder.transform(df['season_phase'])

    # Cria e salva o mapeamento
    json_team_mapping = {team: int(code) for team, code in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    
    with open(path_encoder, 'w', encoding='utf-8') as f:
        json.dump(json_team_mapping, f, ensure_ascii=False, indent=4)

    return df

# encode_categorical_features
def encode_teams(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    """
    Codifica os times usando Label Encoding e salva o mapeamento de-para em um arquivo JSON.

    :param df: DataFrame contendo os dados das partidas.
    :param path_encoder: Caminho do arquivo onde o mapeamento será salvo.
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
    
    with open(path_encoder, 'w', encoding='utf-8') as f:
        json.dump(json_team_mapping, f, ensure_ascii=False, indent=4)

    return df

def encode_result(df: pd.DataFrame, path_encoder) -> pd.DataFrame:
    # Cópia para não alterar original
    df_encoded = df.copy()

    # Label Encoding para resultado (vitória, empate, derrota)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_encoded['result'])
    df_encoded["winner"] = y

    # # One-hot encoding para as demais variáveis categóricas
    # categorical_cols = ['match_day_of_week', 'season_phase']
    # df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)

    return df_encoded
