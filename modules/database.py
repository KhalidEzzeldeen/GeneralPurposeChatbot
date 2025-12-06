from sqlalchemy import create_engine, inspect
import pandas as pd
import os
from modules.config import ConfigManager
from modules.knowledge_base import KnowledgeBase

class DatabaseManager:
    def __init__(self):
        self.config = ConfigManager()
        self.kb = KnowledgeBase()
        
    def get_connection_string(self):
        db_conf = self.config.get("database")
        # Construct URI. Assuming PostgreSQL for enterprise, but fallback to general logic or mysql
        # Format: dialect+driver://username:password@host:port/database
        # For simplicity, let's try to detect or just assume postgres for now as per plan, 
        # but user didn't specify driver. We'll try generic sqlalchemy string.
        # If user/pass empty, might fail.
        user = db_conf.get("user")
        password = db_conf.get("password")
        host = db_conf.get("host")
        port = db_conf.get("port")
        dbname = db_conf.get("dbname")
        
        if not user or not host:
            return None
            
        base_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        
        # Neon (and many cloud PGs) require SSL. 
        # Check for neon.tech or enforce if provided.
        if "neon.tech" in host:
            return f"{base_uri}?sslmode=require"
            
        return base_uri

    def test_connection(self):
        uri = self.get_connection_string()
        if not uri:
            return False, "Invalid Configuration"
        try:
            engine = create_engine(uri)
            conn = engine.connect()
            conn.close()
            return True, "Connection Successful"
        except Exception as e:
            return False, str(e)

    def scan_schema(self):
        uri = self.get_connection_string()
        if not uri:
            return "Invalid Configuration"
            
        try:
            engine = create_engine(uri)
            inspector = inspect(engine)
            
            summary = []
            summary.append(f"Database Schema Scan for {uri.split('@')[-1]}") # Hide credentials
            
            table_names = inspector.get_table_names()
            summary.append(f"Tables: {', '.join(table_names)}\n")
            
            for table in table_names:
                summary.append(f"### Table: {table}")
                columns = inspector.get_columns(table)
                col_details = []
                for c in columns:
                    col_details.append(f"- {c['name']} ({c['type']})")
                summary.append("\n".join(col_details))
                
                # Sample Data
                try:
                    df = pd.read_sql_table(table, engine).head(3)
                    summary.append(f"Sample Data:\n{df.to_string(index=False)}")
                except:
                    summary.append("Could not fetch sample data.")
                summary.append("\n")
                
            full_text = "\n".join(summary)
            
            # Save to file
            data_dir = self.config.get("data_path")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            filename = "db_schema_scan.txt"
            filepath = os.path.join(data_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_text)
                
            # Ingest
            res = self.kb.ingest_file(filepath)
            return f"Schema scanned and ingested: {res}"
            
        except Exception as e:
            return f"Scan failed: {e}"

    def get_sql_query_engine(self, llm):
        """
        Creates a NLSQLTableQueryEngine for natural language to SQL translation.
        """
        from llama_index.core import SQLDatabase
        from llama_index.core.query_engine import NLSQLTableQueryEngine
        
        uri = self.get_connection_string()
        if not uri:
            return None
            
        try:
            engine = create_engine(uri)
            # Inspect all tables to give LLM full context
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
            sql_database = SQLDatabase(engine, include_tables=table_names)
            
            query_engine = NLSQLTableQueryEngine(
                sql_database=sql_database,
                llm=llm
            )
            return query_engine
        except Exception as e:
            print(f"Error creating SQL Engine: {e}")
            return None
