import pandas as pd
import os
import pickle

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from pathlib import Path
from modelagem.utils.logs import logger
import pandas as pd
import numpy as np
from typing import Union, List, Dict
from modelagem.utils.metrics import metrics_per_class
from sklearn.model_selection import train_test_split
import json

BASE_DIR = os.path.dirname(Path(__file__).resolve().parent)
# DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data', 'ft_df.csv')
FT_DIR = Path("features", 'ft_df.csv')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'database', 'models')
LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'logs')

# df = pd.read_csv(FT_DIR)

def balancear_dados(X, y, mode='subamostragem', sampling_strategy='auto', random_state=42):
    """
    Realiza o balanceamento dos dados usando a técnica especificada.

    Parameters:
    - X: numpy array ou pandas DataFrame com as características.
    - y: numpy array ou pandas Series com os rótulos.
    - mode: 'subamostragem', 'superamostragem' ou 'combinado'.
    - sampling_strategy: A estratégia de balanceamento usada, 'auto' para balancear todas as classes minoritárias.
    - random_state: Valor para inicialização do gerador de números aleatórios.

    Returns:
    - X_balanced: Dados de entrada balanceados.
    - y_balanced: Rótulos balanceados.
    """

    logger.debug("Balanceando dados")
    # Verificar se o modo é válido
    if mode not in ['subamostragem', 'superamostragem', 'combinado']:
        raise ValueError("O parâmetro 'mode' deve ser 'subamostragem', 'superamostragem' ou 'combinado'.")

    if mode == 'subamostragem':
        # Aplicar subamostragem para reduzir a classe majoritária
        undersample = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        X_balanced, y_balanced = undersample.fit_resample(X, y)
    
    elif mode == 'superamostragem':
        # Aplicar superamostragem para aumentar a classe minoritária
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)

    elif mode == 'combinado':
        # Aplicar uma combinação de subamostragem e superamostragem
        # Primeiro, aplicar superamostragem, depois subamostragem
        over = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        under = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        pipeline = Pipeline(steps=[('o', over), ('u', under)])
        X_balanced, y_balanced = pipeline.fit_resample(X, y)
    
    return X_balanced.sort_index(), y_balanced.sort_index()


def get_metrics_multiclass(y_pred: Union[List[int], np.ndarray, pd.Series], 
                           y_true: Union[List[int], np.ndarray, pd.Series]) -> Dict[int, Dict[str, float]]:
    """
    Calcula múltiplas métricas de avaliação (Precisão, Recall e F1-score) para cada classe em um problema multiclasse.

    Parâmetros
    ----------
    y_pred : list | np.ndarray | pd.Series
        Lista ou array contendo as previsões do modelo.
    y_true : list | np.ndarray | pd.Series
        Lista ou array contendo os valores reais (ground truth).

    Retorno
    -------
    dict
        Dicionário onde as chaves são as classes e os valores são dicionários com métricas da classe.
        Exemplo de saída:
        {
            0: {"precision": 0.33, "recall": 0.50, "F1": 0.40},
            1: {"precision": 0.57, "recall": 0.60, "F1": 0.58},
            2: {"precision": 0.30, "recall": 0.20, "F1": 0.24}
        }
    """
    
    df_preds = pd.DataFrame({'y_pred': y_pred, 'y_true': y_true})
    
    return {int(classe): metrics_per_class(df_preds, classe) for classe in df_preds['y_true'].unique()}

def get_precision_multiclass(metrics_multiclass: dict) -> dict:
    """
    Extrai apenas os valores de precisão do dicionário de métricas multiclasse.

    Parameters
    ----------
    metrics_multiclass : dict
        Dicionário contendo métricas de cada classe.

    Returns
    -------
    dict
        Dicionário com a precisão de cada classe.
    """
    return {cls: metrics['precision'] for cls, metrics in metrics_multiclass.items()}

def show_print_precision_multiclass(precision_multiclass: dict, algorithm_name: str = '') -> None:
    """
    Exibe as métricas de precisão de forma formatada.

    Parameters
    ----------
    precision_multiclass : dict
        Dicionário com as precisões de cada classe.
    algorithm_name : str, optional
        Nome do algoritmo utilizado, por padrão ''.
    """
    home_team_precision = precision_multiclass.get(2, 0)
    draw_precision = precision_multiclass.get(0, 0)
    away_team_precision = precision_multiclass.get(1, 0)

    logger.info(f"""Resultados: {algorithm_name}
    Home Team:  {home_team_precision:.2%}
    Draw:       {draw_precision:.2%}
    Away Team:  {away_team_precision:.2%}
    ----------------------------""")

def get_precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calcula a precisão geral das previsões.

    Parameters
    ----------
    y_pred : np.ndarray
        Array com as previsões do modelo.
    y_true : np.ndarray
        Array com os valores reais.

    Returns
    -------
    float
        Precisão das previsões.
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    return np.mean(y_pred == y_true)

def get_data_transform(X: pd.DataFrame, get_y_true: bool = False):
    """
    Prepara os dados para predição aplicando MinMaxScaler.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame contendo os dados.
    get_y_true : bool, optional
        Se True, retorna também os rótulos 'winner', por padrão False.

    Returns
    -------
    np.ndarray | tuple
        Retorna os dados transformados ou (dados transformados, y_true).
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.drop(columns=['winner'], errors='ignore'))

    if get_y_true and 'winner' in X.columns:
        return X_scaled, X['winner'].values

    return X_scaled

def train(model: LogisticRegression, X_train, X_test, y_train, y_test):
    """
    Treina o modelo e exibe as métricas.

    Parameters
    ----------
    model : LogisticRegression
        Modelo de regressão logística.
    X_train : np.ndarray
        Dados de treinamento.
    X_test : np.ndarray
        Dados de teste.
    y_train : np.ndarray
        Rótulos de treinamento.
    y_test : np.ndarray
        Rótulos de teste.

    Returns
    -------
    LogisticRegression
        Modelo treinado.
    """
    logger.debug("Iniciando treinamento...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 2)

    metrics_multiclass = get_metrics_multiclass(y_pred=y_pred, y_true=y_test)
    precision_multiclass = get_precision_multiclass(metrics_multiclass)

    show_print_precision_multiclass(precision_multiclass, model.__class__.__name__)

    logger.info(f'Accuracy: {acc:.2%}')

    return model

def split_data(df: pd.DataFrame, test_size: float = 0.3, valid_year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide os dados de forma inteligente:
      - Os anos **mais recentes** são usados para **validação**.
      - Os anos **anteriores** são divididos em **treino (70%)** e **teste (30%)**.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados com a coluna 'season'.
    test_size : float, opcional (default=0.3)
        Proporção do conjunto de teste em relação ao total de treino/teste.
    valid_year : int, opcional (default=2024)
        Ano a partir do qual os dados serão usados apenas para validação.

    Retorna:
    --------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        df_train : Dados de treinamento (70% dos dados anteriores ao valid_year).
        df_test  : Dados de teste (30% dos dados anteriores ao valid_year).
        df_valid : Dados de validação (apenas anos >= valid_year).
    """
    logger.info(f"Dividindo os dados com valid_year={valid_year} e test_size={test_size}...")

    # Separa os dados futuros para validação
    df_valid = df[df['season'] >= valid_year]

    # Filtra os dados antes do ano de validação
    df_filtered = df[df['season'] < valid_year]

    # Divide treino e teste mantendo a proporção 70% / 30%
    df_train, df_test = train_test_split(df_filtered, test_size=test_size, random_state=42)

    logger.info(f"Tamanho dos conjuntos: Treino={len(df_train)}, Teste={len(df_test)}, Validação={len(df_valid)}")

    return df_train, df_test, df_valid

def main(df:pd.DataFrame):
    """
    Função principal para treinamento e validação do modelo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    """

    MODEL_DIR = Path("models")  # Diretório onde os modelos serão salvos
    MODEL_DIR.mkdir(exist_ok=True)  # Garante que o diretório exista

    # Divisão dos dados
    df_train, df_test, df_valid = split_data(df)

    # Separação de features e rótulos
    X_train, y_train = df_train.drop(columns=['winner', 'season']), df_train['winner']
    X_test, y_test = df_test.drop(columns=['winner', 'season']), df_test['winner']
    X_valid, y_valid = df_valid.drop(columns=['winner', 'season']), df_valid['winner']

    # Balanceamento dos dados
    # X_train, y_train = balancear_dados(X=X_train, y=y_train)

    # Transformação dos dados
    X_train_scaled = get_data_transform(X_train)
    X_test_scaled = get_data_transform(X_test)

    # Inicializa e treina o modelo
    model = LogisticRegression(max_iter=30000)
    model = train(model, X_train_scaled, X_test_scaled, y_train, y_test)

    # 
    with open(os.path.join(MODEL_DIR, "feature_order.json"), "w") as f:
        json.dump(list(X_train.columns), f)


    # Salvar o modelo treinado
    model_filename = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    logger.info(f"Modelo salvo com sucesso em: {model_filename}")

def load_model(model_filename):
    """
    Carrega o modelo treinado a partir do arquivo.

    :param model_filename: str
        Caminho do arquivo do modelo.
    :return: LogisticRegression
        Modelo treinado.

    """
    
    
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    logger.info(f"Modelo carregado com sucesso de: {model_filename}")
    
    return model


###### Predições

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def is_numeric(value):
    return is_float(value) or is_int(value)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_team_mappings(team_mapping_json):
    return (
        {k: v for k, v in team_mapping_json.items()},
        {v: k for k, v in team_mapping_json.items()}
    )

def get_drop_columns():
    template = [
        "{type}_rank", "{type}_ls_rank", "{type}_days_ls_match", "{type}_points",
        "{type}_l_points", "{type}_l_wavg_points", "{type}_goals", "{type}_l_goals",
        "{type}_l_wavg_goals", "{type}_goals_sf", "{type}_l_goals_sf", "{type}_l_wavg_goals_sf",
        "{type}_wins", "{type}_draws", "{type}_losses", "{type}_win_streak",
        "{type}_loss_streak", "{type}_draw_streak", "winner"
    ]
    drop_at = [col.format(type="at") for col in template] + ["away_team_encoder"]
    drop_ht = [col.format(type="ht") for col in template] + ["home_team_encoder"]
    return drop_ht, drop_at

def prepare_input(df_prep, home_team:str, away_team:str, team_mapping, feature_order):

    df_prep.sort_values(by=["season"], ascending=False, inplace=True)

    dict_team_name_id, dict_id_team_name = get_team_mappings(team_mapping)
    drop_ht, drop_at = get_drop_columns()

    if is_numeric(home_team):
        home_team = dict_id_team_name.get(int(float(home_team)))
    if is_numeric(away_team):
        away_team = dict_id_team_name.get(int(float(away_team)))
    
    if home_team is None or away_team is None:
        raise ValueError("Os times fornecidos não estão no mapeamento.")
    if home_team == away_team:
        raise ValueError("Os times fornecidos são iguais.")

    df_home = df_prep[df_prep["home_team_encoder"] == dict_team_name_id[home_team]].drop(columns=drop_at).copy()
    df_away = df_prep[df_prep["away_team_encoder"] == dict_team_name_id[away_team]].drop(columns=drop_ht).copy()

    df_merge = pd.concat([df_home.iloc[[0]].reset_index(drop=True), df_away.iloc[[0]].reset_index(drop=True)], axis=1)
    df_merge.drop(columns="season", inplace=True, errors="ignore")
    df_merge = df_merge[feature_order]

    return df_merge

def predict_per_time(home_team:str, 
                     away_team:str,
                     list_seasons:list[int]=None,
                     model=None, 
                     df_prep=None, 
                     team_mapping_json=None,
                     feature_order_json=None):
    """
    Faz previsões para um jogo específico entre dois times.

    Parameters
    ----------
    home_team : str
        Nome do time da casa.
    away_team : str
        Nome do time visitante.
    model : LogisticRegression
        Modelo treinado.
    df_prep : pd.DataFrame
        DataFrame com os dados preparados.
    team_mapping_json : str
        Caminho para o arquivo JSON com o mapeamento dos times.

    Returns
    -------
    np.ndarray
        Previsões do modelo para o jogo especificado.
    """

    if model is None:
        model = load_model(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
    if df_prep is None:
        df_prep = pd.read_csv(FT_DIR)
    if team_mapping_json is None:
        team_mapping_json = os.path.join(MODEL_DIR, "team_mapping.json")
    if feature_order_json is None:
        feature_order_json = os.path.join(MODEL_DIR, "feature_order.json")

    team_mapping = load_json(team_mapping_json)
    
    feature_order = load_json(feature_order_json)

    if list_seasons is not None:
        df_prep = df_prep[df_prep["season"].isin(list_seasons)]

    df_input = prepare_input(df_prep, home_team, away_team, team_mapping, feature_order)
    
    # df_input_scaled = get_data_transform(df_input)
    predictions, predictions_proba = predict(model, df_input)
    
    return predictions[0], predictions_proba[0]


def predict(model, df_input) -> tuple[np.ndarray, np.ndarray]:
    """
    Faz previsões usando o modelo carregado.

    Parameters
    ----------
    model : LogisticRegression
        Modelo treinado.
    X : pd.DataFrame
        Dados para previsão.

    Returns
    -------
    np.ndarray
        Previsões do modelo.
    """
    predictions = model.predict(df_input)
    predictions_proba = model.predict_proba(df_input)

    return predictions, predictions_proba

