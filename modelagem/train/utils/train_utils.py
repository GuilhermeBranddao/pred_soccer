
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np

import time
import warnings
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

import pandas as pd
import time

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

from random import choice

warnings.filterwarnings("ignore", category=FutureWarning)

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



from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import time

def get_best_num_feature(
    x: pd.DataFrame,
    y: pd.Series,
    model,
    min_columns: int = 12,
    max_iter: int = 500,
) -> None:
    """
    Identifica o desempenho do modelo testando com vários números de features.
    
    Parâmetros:
    - x: DataFrame com as features.
    - y: Série com as labels (variável alvo).
    - model: Modelo de machine learning (exemplo: LogisticRegression()).
    - min_columns: Número mínimo de features a serem testadas.
    - max_iter: Número máximo de iterações para o otimizador (evitar warnings).
    
    Retorna:
    - Gráfico mostrando a relação entre o número de features e a acurácia.
    """
    acc_results = []
    n_features = []
    scaler = MinMaxScaler()

    # Configurando o max_iter no modelo, se aplicável
    if hasattr(model, "max_iter"):
        model.set_params(max_iter=max_iter)

    # Loop para testar diferentes quantidades de features
    for i in range(min_columns, len(x.columns) + 1):
        rfe = RFE(estimator=model, n_features_to_select=i, step=1)
        rfe.fit(x, y)
        X_temp = rfe.transform(x)

        # Dividindo dados em treino e teste
        x_train, x_test, y_train, y_test = train_test_split(
            X_temp, y, test_size=0.2, random_state=101
        )

        # Escalando os dados
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Avaliando o modelo
        start = time.time()
        scores = cross_val_score(
            model, x_train_scaled, y_train, scoring="accuracy", cv=5
        )
        end = time.time()

        print(
            f"N_features: {i}, Accuracy: {round(scores.mean(), 5)} ± {round(scores.std(), 5)} "
            f"(Time: {round(end - start, 2)}s)"
        )

        acc_results.append(scores.mean())
        n_features.append(i)

    # Plotando os resultados
    plt.figure(figsize=(10, 6))
    plt.plot(n_features, acc_results, marker="o", linestyle="-", color="b")
    plt.title("Desempenho por Número de Features", fontsize=16)
    plt.xlabel("Número de Features", fontsize=14)
    plt.ylabel("Acurácia Média (CV=5)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()



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


def run_cross_val_score(models, X, y, transform_features=False):
    """
    Executa validação cruzada com TimeSeriesSplit para avaliar modelos.
    Args:
        models (list): Lista de dicionários com os modelos e seus nomes.
        X (DataFrame): Dados de entrada.
        y (Series): Rótulos das classes.
        transform_features (bool): Se True, aplica transformação nos dados de entrada.
    Returns:
        list: Resultados contendo métricas e predições para cada modelo.
    """
    results = []

    for model_info in models:
        # Balanceamento dos dados
        X_balanced, y_balanced = balancear_dados(X=X, y=y, mode='subamostragem', )

        # Transformação dos dados, se necessário
        if transform_features:
            X_balanced = get_data_transform(X_balanced, get_y_true=False)

        # Inicializar o modelo
        model = model_info['model']
        algorithm_name = model_info['algorithm']

        # Configuração do TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)

        fold_accuracies = []
        fold_predictions = []
        fold_true_values = []
        start_time = time.time()

        # Iteração pelos folds do TimeSeriesSplit
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_balanced)):
            X_train, X_test = X_balanced.iloc[train_idx], X_balanced.iloc[test_idx]
            y_train, y_test = y_balanced.iloc[train_idx], y_balanced.iloc[test_idx]

            # Treinamento e predição
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Armazenar resultados do fold
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            fold_predictions.append(y_pred)
            fold_true_values.append(y_test.values)

        # Cálculo das métricas gerais
        mean_accuracy = round(np.mean(fold_accuracies), 2)
        std_accuracy = round(np.std(fold_accuracies), 2)
        elapsed_time = round(time.time() - start_time, 2)

        # Exibição dos resultados para o modelo
        print(f"[{algorithm_name}] Média: {mean_accuracy}% | Desvio: {std_accuracy}% | Tempo: {elapsed_time}s")

        # Armazenar resultados do modelo
        results.append({
            'algorithm': algorithm_name,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'elapsed_time': elapsed_time,
            'fold_predictions': fold_predictions[0],  # Exemplo: armazenar apenas o 1º fold
            'fold_true_values': fold_true_values[0],
            'list_accuracy': fold_accuracies,
        })

    return results
