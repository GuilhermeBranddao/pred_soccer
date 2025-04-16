# main_trainer.py

import os
import pandas as pd
from pathlib import Path

# from database.create_connection import create_connection
from modelagem.utils.logs import logger
from modelagem.utils.get_data import get_soccer_data
from modelagem.train import model_trainer
from modelagem.utils.feature.implementing_features import (
    create_feature_first_strategy,
    create_feature_segund_strategy,
    # strategy_basic,
    strategy_with_goal_stats,
    strategy_experimental,
    )
# from modelagem.utils.feature.encode import base_pre_processing
from modelagem.feature_eng.strategy.basic import strategy_basic
from modelagem.feature_eng.strategy.basic import strategy_basic

# Definindo diretórios base
DATA_DIR = os.path.join('feature_eng', 'data')
MODEL_DIR = os.path.join('database', 'models')
LOG_DIR = os.path.join('database', 'logs')
# Diretórios
FT_DIR = Path("database", "features")

FEATURE_STRATEGIES:list[dict] = [
    {"feature_name":"first", 
     "func_feature":create_feature_first_strategy, 
     "coluns_request":[],
     "drop_unwanted_features":[],
     },
    {"feature_name":"basic", 
    "func_feature":strategy_basic.main, 
    "coluns_request":[],
    "drop_unwanted_features":[
        "id", "country", "league", "home_team", "away_team",
        "result", "psch", "pscd", "psca", "maxch", "maxcd", "maxca",
        "avgch", "avgcd", "avgca", "bfech", "bfecd", "datetime",
        "hash", "last_updated", "match_day_of_week", "season_phase"
    ]},
    # {"feature_name":"basic", 
    # "func_feature":strategy_basic, 
    # "coluns_request":[],
    # "drop_unwanted_features":[],
    # },
    {"feature_name":"goal_stats", 
    "func_feature":strategy_with_goal_stats, 
    "coluns_request":[],
    "drop_unwanted_features":[],
    },
    {"feature_name":"experimental", 
    "func_feature":strategy_experimental, 
    "coluns_request":[],
    "drop_unwanted_features":[],
    },
]

if __name__ == "__main__":
    df = get_soccer_data()
    # success, df_prep = base_pre_processing(df)


    featute_strategies = FEATURE_STRATEGIES[1]
    func_feature = featute_strategies.get("func_feature")

    success, df_prep = func_feature(df, path_encoder=MODEL_DIR)
    if success:
        logger.info("Pré-processamento concluído com sucesso!")
    else:
        logger.error("Erro no pré-processamento.")

    logger.debug("Carregando base de dados trataba (csv)")
    df = pd.read_csv(os.path.join(FT_DIR, 'ft_df.csv'))
    logger.info("Iniciando treinamento")
    model_trainer.main(df)
