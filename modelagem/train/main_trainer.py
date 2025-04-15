# main_trainer.py

import os
import pandas as pd
from pathlib import Path

# from database.create_connection import create_connection
from modelagem.utils.logs import logger
from modelagem.utils.get_data import get_soccer_data
from modelagem.train import model_trainer
from modelagem.utils.feature.implementing_features import create_feature
from modelagem.utils.feature.engine import base_pre_processing

# Definindo diretórios base
DATA_DIR = os.path.join('feature_eng', 'data')
MODEL_DIR = os.path.join('database', 'models')
LOG_DIR = os.path.join('database', 'logs')
# Diretórios
FT_DIR = Path("database", "features")

if __name__ == "__main__":
    df = get_soccer_data()
    success, df_prep = base_pre_processing(df)
    success, df_prep = create_feature(df, path_encoder=MODEL_DIR)
    if success:
        logger.info("Pré-processamento concluído com sucesso!")
    else:
        logger.error("Erro no pré-processamento.")

    logger.debug("Carregando base de dados trataba (csv)")
    df = pd.read_csv(os.path.join(FT_DIR, 'ft_df.csv'))
    logger.info("Iniciando treinamento")
    model_trainer.main(df)
