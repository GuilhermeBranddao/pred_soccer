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
from modelagem.utils.logs import setup_logger

BASE_DIR = os.path.dirname(Path(__file__).resolve().parent)
DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data', 'ft_df.csv')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'database', 'models')
LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'logs')

logger = setup_logger(LOG_DIR, "main_train.log")

df = pd.read_csv(DATA_DIR)

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

def calc_f1(precision, recal):
    if precision + recal == 0:
        return 0
    return round(2 * ((precision * recal) / (precision + recal)), 2)

def calc_recal(tp_count, fn_count):
    if tp_count+fn_count == 0:
        return 0
    return round(tp_count / (tp_count+fn_count), 2)

def calc_precision(tp_count, fp_count):
    if tp_count+fp_count == 0:
        return 0
    return round(tp_count / (tp_count+fp_count), 2)

def metrics_per_class(df_preds, classe):
    """
        Realiza do calculos e de diversas metricas
    """
    logger.debug("Calculando metricas")

    # Filtrar as predições e os rótulos verdadeiros para a classe atual
    # True Positives (TP): Previsão correta de que a instância pertence à classe (ou seja, y_pred == classe e y_true == classe).
    true_positives = df_preds[(df_preds['y_pred'] == classe) & (df_preds['y_true'] == classe)]

    # True Negatives (TN): Previsão correta de que a instância não pertence à classe (ou seja, y_pred != classe e y_true != classe).
    true_negatives = df_preds[(df_preds['y_pred'] != classe) & (df_preds['y_true'] != classe)]

    # False Positives (FP): Previsão errada de que a instância pertence à classe, quando na verdade não pertence (ou seja, y_pred == classe e y_true != classe).
    false_positives = df_preds[(df_preds['y_pred'] != classe) & (df_preds['y_true'] == classe)]

    # False Negatives (FN): Previsão errada de que a instância não pertence à classe, quando na verdade pertence (ou seja, y_pred != classe e y_true == classe).
    false_negatives = df_preds[(df_preds['y_pred'] == classe) & (df_preds['y_true'] != classe)]

    # Contagem de cada um
    tp_count = len(true_positives)
    tn_count = len(true_negatives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    precision = calc_precision(tp_count, fp_count)
    recal = calc_recal(tp_count, fn_count)
    F1 = calc_f1(precision, recal)

    return {"precision":precision, "recal":recal, "F1":F1}

def get_metrics_multiclass(y_pred:list, y_true:list):
    """ 

    Realiza o calculo de varias metricas para cada classe

    Calcular a precisão: TP / (TP + FP)

    RETURN: {np.int64(0): 0.33, np.int64(2): 0.57, np.int64(1): 0.3}
    """

    logger.debug("get_metrics_multiclass")
    df_preds = pd.DataFrame(columns=['y_pred', 'y_true'])
    df_preds['y_pred'] = y_pred
    df_preds['y_true'] = y_true

    dict_precision = {}

    for classe in df_preds['y_true'].unique():
        dict_metrics_per_class = metrics_per_class(df_preds, classe)

        #precision, recal, F1
        dict_precision[classe] = dict_metrics_per_class

    return dict_precision

def get_precision_multiclass(metrics_multiclass):
    return {metrics:metrics_multiclass[metrics]['precision'] for metrics in metrics_multiclass}

def show_print_precision_multiclass(precision_multiclass, algorithm_name=''):
    """
        Exibe de uma baneira mais visual as metricas de precisão
    """
    home_team_precision = precision_multiclass.get(2, 0)
    draw_precision = precision_multiclass.get(0, 0)
    away_team_precision = precision_multiclass.get(1, 0)

    logger.info(f'''Reultados: ({algorithm_name})
    Home_team: ------ {round(home_team_precision*100, 2)}%
    Draw: ----------- {round(draw_precision*100, 2)}%
    Away_team: ------ {round(away_team_precision*100, 2)}%
    ----------------------------''')

def get_precision(y_pred:list, y_true:list):
    """
    Objetivo: Obtem a precisão

    Calcular a precisão: TP / (TP + FP)
    """
    all_predicted = len(y_true)
    true_positives = sum(y_pred == y_true)
    precision = true_positives / all_predicted

    return precision

def get_data_transform(X:pd.DataFrame, get_y_true=False):
    """
    Recebe dataframe e retorna o dado pronto para ser realizado a predição
    """

    #columns_pred = rfe.feature_names_in_

    #X_valid = df_valid.drop(columns=['winner', 'season'])

    #X_valid_trans = rfe.transform(df_valid[columns_pred])

    scaler = MinMaxScaler()

    X_valid_trans_scaler = scaler.fit_transform(X)

    if get_y_true:
        y_valid = X['winner']
        return X_valid_trans_scaler, y_valid

    return pd.DataFrame(X_valid_trans_scaler, columns=X.columns) 


def train_models(model:list, X_train_trans_scaler):
    logger.debug("Iniciando treinamento")

    model = model.fit(X_train_trans_scaler, y_train)

    y_pred = model.predict(X_test_trans_scaler)
    acc = round(accuracy_score(y_pred=y_pred, y_true=y_test.values), 2)
    
    metrics_multiclass = get_metrics_multiclass(y_pred=y_pred, 
                    y_true=y_test.values)
    dict_precision = get_precision_multiclass(metrics_multiclass=metrics_multiclass)
    logger.info(f'home_precision: {dict_precision.get(2, None)}%')
    logger.info(f'draw_precision: {dict_precision.get(0, None)}%')
    logger.info(f'away_precision: {dict_precision.get(1, None)}%')
    logger.info(f'accuracy: {acc}%')

    return model

df_train = df[df['season']<2021] #['season'].value_counts()
df_test = df[(df['season']>=2021) & (df['season']<2024)]
df_valid = df[df['season']>=2024]

X_train, y_train = df_train.drop(columns=['winner', 'season']), df_train['winner']
X_test, y_test = df_test.drop(columns=['winner', 'season']), df_test['winner']
X_valid, y_valid = df_valid.drop(columns=['winner', 'season']), df_valid['winner']

X_train, y_train = balancear_dados(X=X_train, y=y_train)

X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

X_train_trans_scaler = get_data_transform(X_train_copy, get_y_true=False)
X_test_trans_scaler = get_data_transform(X_test_copy, get_y_true=False)

model = LogisticRegression(max_iter=3000)

model = train_models(model, X_train_trans_scaler)

# Salvar o modelo em um arquivo pickle
model_filename = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

logger.debug(f"Modelo salvo com sucesso no arquivo: {model_filename}")