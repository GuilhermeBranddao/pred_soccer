from contracts.transform_contract import TransformContract
from infra.database import insert_new_data

class LoadCsv():
    def __init__(self, conn)->None:
        self.conn = conn

    def load(self, transform_content:TransformContract)->None:
        for df in transform_content.transform_data.values():
            insert_new_data(conn=self.conn, df=df)

