import pandas as pd
import os
import json
from sklearn.preprocessing import LabelEncoder

# 🔁 Função genérica de Label Encoding
def label_encode_column(df: pd.DataFrame, column: str, new_col: str, path_save_encoder: str) -> pd.DataFrame:
    df = df.copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(df[column])
    df[new_col] = label_encoder.transform(df[column])

    mapping = {k: int(v) for k, v in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}

    with open(path_save_encoder, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    return df

# 🎯 Funções específicas que usam a genérica
def encode_result(df, path_save_encoder): 
    return True, label_encode_column(df, "result", "result_encoded", path_save_encoder)

def encode_match_day_of_week(df, path_save_encoder):
    return True, label_encode_column(df, "match_day_of_week", "match_day_of_week_encoded", path_save_encoder)

def encode_season_phase(df, path_save_encoder):
    return True, label_encode_column(df, "season_phase", "season_phase_encoded", path_save_encoder)

def encode_teams(df, path_save_encoder):
    df = df.copy()
    encoder = LabelEncoder()
    all_teams = pd.concat([df['home_team'], df['away_team']])
    encoder.fit(all_teams)

    df['home_team_encoded'] = encoder.transform(df['home_team'])
    df['away_team_encoded'] = encoder.transform(df['away_team'])

    mapping = {team: int(code) for team, code in zip(encoder.classes_, range(len(encoder.classes_)))}

    with open(path_save_encoder, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)

    return True, df


# 📦 Mapeamento central das funções de encoding
ENCODERS = [
    {"encode_name":"teams",
    "func_encode": encode_teams,
    "columns_request":["home_team", "away_team"]
    },
    {"encode_name":"result",
    "func_encode": encode_result,
    "columns_request":['result']
    },
    {"encode_name":"match_day_of_week",
    "func_encode": encode_match_day_of_week,
    "columns_request":['match_day_of_week']
    },
    {"encode_name":"season_phase",
    "func_encode": encode_season_phase,
    "columns_request":['season_phase']
    },
]

# ⚙️ Função principal com escolha flexível
def encode_categorical_features(df: pd.DataFrame, path_save_encoder: str, features_to_encode:list[dict]=None) -> pd.DataFrame:
    df = df.copy()

    if features_to_encode is None:
        features_to_encode = ENCODERS

    for encode in features_to_encode:
        set_remaining_columns = set(encode.get("columns_request")).difference(set(df.columns))

        encode_name = encode.get('encode_name')
        func_encode = encode.get('func_encode')

        if set_remaining_columns:
            print(f"❌ DataFrame does not have column {set_remaining_columns} from encode '{encode_name}'...")
            continue

        encoder_path = os.path.join(path_save_encoder, f"{encode_name}_mapping.json")

        print(f"Encoding '{encode_name}'...")
        err, df = func_encode(df, encoder_path)

        if not err:
            print(f"❌ Encode error '{encode_name}'...")
        else:
            print(f"✅  Encode success '{encode_name}'...")

    return df
