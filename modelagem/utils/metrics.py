import numpy as np
from decimal import Decimal, getcontext
from typing import Union
import pandas as pd

# Configura precisão numérica para cálculos mais exatos
getcontext().prec = 10

def calc_f1(precision: Union[float, np.ndarray], recall: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula a métrica F1-score, que é a média harmônica entre precisão e recall.

    Parâmetros
    ----------
    precision : float | np.ndarray
        Precisão da classe ou array de precisões.
    recall : float | np.ndarray
        Recall da classe ou array de recalls.

    Retorno
    -------
    float | np.ndarray
        F1-score arredondado para duas casas decimais (se entrada for escalar)
        ou array de F1-scores (se entrada for um array).
    """
    # Converte para arrays caso seja uma entrada escalar (float/int)
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)

    # Evita divisão por zero ao usar np.where()
    f1_score = np.where(
        (precision + recall) == 0,
        0.0,
        np.round(2 * (precision * recall) / (precision + recall), 2)
    )

    # Retorna escalar se a entrada for escalar, caso contrário, retorna array
    return f1_score.item() if f1_score.size == 1 else f1_score

def calc_recall(tp: Union[int, np.ndarray], fn: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula o recall (revocação), que mede a capacidade do modelo de identificar corretamente os positivos.

    Parâmetros
    ----------
    tp : int | np.ndarray
        Número de verdadeiros positivos ou array de valores.
    fn : int | np.ndarray
        Número de falsos negativos ou array de valores.

    Retorno
    -------
    float | np.ndarray
        Recall arredondado para duas casas decimais (se entrada for escalar)
        ou array de recalls (se entrada for um array).
    """
    tp, fn = np.asarray(tp, dtype=np.float64), np.asarray(fn, dtype=np.float64)
    recall = np.where((tp + fn) == 0, 0.0, np.round(tp / (tp + fn), 2))
    return recall.item() if recall.size == 1 else recall


def calc_precision(tp: Union[int, np.ndarray], fp: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula a precisão, que mede a proporção de previsões corretas entre as positivas.

    Parâmetros
    ----------
    tp : int | np.ndarray
        Número de verdadeiros positivos ou array de valores.
    fp : int | np.ndarray
        Número de falsos positivos ou array de valores.

    Retorno
    -------
    float | np.ndarray
        Precisão arredondada para duas casas decimais (se entrada for escalar)
        ou array de precisões (se entrada for um array).
    """
    tp, fp = np.asarray(tp, dtype=np.float64), np.asarray(fp, dtype=np.float64)
    precision = np.where((tp + fp) == 0, 0.0, np.round(tp / (tp + fp), 2))
    return precision.item() if precision.size == 1 else precision

def get_balanced_accuracy(tp, tn, fp, fn):
    """
    Calcula a acurácia balanceada, que é a média da sensibilidade e especificidade.

    Retorno
    -------
    float
        Acurácia balanceada.
    """
    # Exemplo de valores para TP, TN, FP e FN

    ((tp+fn) / tp) + ((tn+fp) / tn)

    # Cálculo da sensibilidade (recall)
    sensitivity = calc_recall(tp, fn)

    # Cálculo da especificidade
    specificity = calc_recall(tn, fp)

    # Cálculo da acurácia balanceada
    balanced_accuracy = (sensitivity + specificity) / 2

    return balanced_accuracy

def metrics_per_class(df_preds: pd.DataFrame, classe: Union[int, str]) -> dict:
    """
    Calcula métricas de classificação para uma determinada classe.

    Parâmetros
    ----------
    df_preds : pd.DataFrame
        DataFrame contendo as colunas `y_true` (rótulo real) e `y_pred` (previsão do modelo).
    classe : int | str
        Classe para a qual as métricas serão calculadas.

    Retorno
    -------
    dict
        Dicionário com precisão, recall e F1-score da classe.
    """
    # True Positives (TP): Previsão correta de que a instância pertence à classe (ou seja, y_pred == classe e y_true == classe).
    tp = ((df_preds['y_pred'] == classe) & (df_preds['y_true'] == classe)).sum()

    # True Negatives (TN): Previsão correta de que a instância não pertence à classe (ou seja, y_pred != classe e y_true != classe).
    tn = ((df_preds['y_pred'] != classe) & (df_preds['y_true'] != classe)).sum()


    # False Positives (FP): Previsão errada de que a instância pertence à classe, quando na verdade não pertence (ou seja, y_pred == classe e y_true != classe).
    fp = ((df_preds['y_pred'] == classe) & (df_preds['y_true'] != classe)).sum()

    # False Negatives (FN): Previsão errada de que a instância não pertence à classe, quando na verdade pertence (ou seja, y_pred != classe e y_true == classe).
    fn = ((df_preds['y_pred'] != classe) & (df_preds['y_true'] == classe)).sum()


    precision = calc_precision(tp, fp)
    recall = calc_recall(tp, fn)
    f1 = calc_f1(precision, recall)

    return {"precision": precision, 
            "recall": recall, 
            "F1": f1}



####################### 

def show_print_precision_multiclass(precision_multiclass, algorithm_name=''):
    """
        Exibe de uma baneira mais visual as metricas de precisão
    """
    home_team_precision = precision_multiclass.get(2, 0)
    draw_precision = precision_multiclass.get(0, 0)
    away_team_precision = precision_multiclass.get(1, 0)

    print(f'''Reultados: ({algorithm_name})
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

def get_precision_multiclass(metrics_multiclass):
    return {metrics:metrics_multiclass[metrics]['precision'] for metrics in metrics_multiclass}

def get_metrics_multiclass(y_pred:list, y_true:list):
    """ 

    Realiza o calculo de varias metricas para cada classe

    Calcular a precisão: TP / (TP + FP)

    RETURN: {np.int64(0): 0.33, np.int64(2): 0.57, np.int64(1): 0.3}
    """
    df_preds = pd.DataFrame(columns=['y_pred', 'y_true'])
    df_preds['y_pred'] = y_pred
    df_preds['y_true'] = y_true

    dict_precision = {}

    for classe in df_preds['y_true'].unique():
        dict_metrics_per_class = metrics_per_class(df_preds, classe)

        #precision, recal, F1
        dict_precision[classe] = dict_metrics_per_class

    return dict_precision