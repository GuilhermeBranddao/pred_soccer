from etl.stages.transform.transform_csv import TransformCsv
from etl.stages.extract.extract_csv import ExtractCsv
from etl.stages.load.load_csv import LoadCsv

from etl.infra.database import create_connection
from pathlib import Path


extract_csv = ExtractCsv()
extract_contract = extract_csv = extract_csv.extract()

conn = create_connection(db_path=Path("database/soccer_data.db"))
transform_csv = TransformCsv(conn)

transform_content = transform_csv.transform(extract_contract)

load_csv = LoadCsv(conn)
load_csv.load(transform_content=transform_content)

conn.close()