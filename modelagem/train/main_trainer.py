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

# from database.create_connection import create_connection
from etl.infra.database import create_connection
from modelagem.utils.preprocessing.engine import base_pre_processing
from modelagem.utils.logs import logger
from modelagem.utils.get_data import get_soccer_data
from modelagem.train import model_trainer
from modelagem.utils.feature.implementing_features import main

# Definindo diretórios base
DATA_DIR = os.path.join('feature_eng', 'data')
MODEL_DIR = os.path.join('database', 'models')
LOG_DIR = os.path.join('database', 'logs')
# Diretórios
FT_DIR = Path("database", "features")

# logger = setup_logger(LOG_DIR, "main_train.log")


if __name__ == "__main__":
    df = get_soccer_data()
    df = main(df, os.path.join(MODEL_DIR, "team_mapping.json"))
    # success = base_pre_processing(df)
    # if success:
    #     logger.info("Pré-processamento concluído com sucesso!")
    # else:
    #     logger.error("Erro no pré-processamento.")

    df = pd.read_csv(os.path.join(FT_DIR, 'ft_df.csv'))
    logger.info("Iniciando treinamento")
    model_trainer.main(df)
