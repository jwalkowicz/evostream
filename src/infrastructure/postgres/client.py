import psycopg2

from src.core.logger import logger


class DBAdmin:
    def __init__(self, dbname: str, user: str, host: str, password: str, port: int = 5432):
        self.dbname = dbname
        self.user = user
        self.host = host
        self.password = password
        self.port = port
        self.connection = None
        self._connect()

    def _connect(self):
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                host=self.host,
                password=self.password,
                port=self.port,
                connect_timeout=3,
            )
            logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.dbname}")
        except Exception as e:
            self.connection = None
            logger.debug(f"PostgreSQL connection offline ({self.host}:{self.port}/{self.dbname}): {e}")

    def is_connected(self) -> bool:
        if self.connection is None or self.connection.closed != 0:
            return False
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1;")
            return True
        except Exception:
            self.connection = None
            return False

    def create_table(
        self,
        name: str,
        schema_sql: str,
    ):
        if not self.is_connected():
            return
        query = f"CREATE TABLE IF NOT EXISTS {name} ({schema_sql});"
        with self.connection.cursor() as cursor:
            try:
                cursor.execute(query)
                self.connection.commit()
                logger.success(f"Table '{name}' checked/created.")
            except Exception as e:
                self.connection.rollback()
                logger.error(f"Failed to create table '{name}': {e}")

    def insert(self, table: str, data: dict):
        if not self.is_connected():
            return

        # Fetch table column names to prevent schema mismatch
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = %s;",
                    (table,),
                )
                valid_cols = {row[0] for row in cursor.fetchall()}
        except Exception:
            valid_cols = set()

        if valid_cols:
            filtered_data = {k: v for k, v in data.items() if k in valid_cols and k != "id"}
        else:
            filtered_data = {k: v for k, v in data.items() if k != "id"}

        if not filtered_data:
            return

        columns = list(filtered_data.keys())
        values = [filtered_data[column] for column in columns]

        placeholders = ", ".join(["%s"] * len(columns))
        columns_str = ", ".join(columns)

        query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders});"

        with self.connection.cursor() as cursor:
            try:
                cursor.execute(query, values)
                self.connection.commit()
                logger.debug(f"Data inserted into '{table}'.")
            except Exception as e:
                self.connection.rollback()
                logger.error(f"Failed to insert data into '{table}': {e}")

    def close(self):
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("PostgreSQL connection closed.")
