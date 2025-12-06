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

    def get_schema_summary(self):
        """
        Get a comprehensive summary of database schema including tables, columns, and sample data.
        This helps the LLM understand what data is available in the database for intelligent routing.
        Returns None if schema cannot be retrieved.
        """
        uri = self.get_connection_string()
        if not uri:
            return None
            
        try:
            engine = create_engine(uri)
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
            if not table_names:
                return None
            
            schema_summary = []
            schema_summary.append(f"Database contains {len(table_names)} table(s): {', '.join(table_names)}")
            schema_summary.append("")
            schema_summary.append("Each table contains the following data:")
            schema_summary.append("")
            
            for table in table_names:
                columns = inspector.get_columns(table)
                col_names = [c['name'] for c in columns]
                col_types = {c['name']: str(c['type']) for c in columns}
                
                # Get sample data to understand what's actually in the table
                sample_data = None
                try:
                    df = pd.read_sql_table(table, engine).head(10)  # Get 10 samples for better understanding
                    if not df.empty:
                        sample_data = df
                except Exception:
                    pass
                
                schema_summary.append(f"**Table: {table}**")
                schema_summary.append(f"Columns: {', '.join(col_names)}")
                
                # Identify text columns that likely contain descriptive content
                text_columns = [col for col in col_names if col_types.get(col, '').upper() in ['TEXT', 'VARCHAR', 'CHAR', 'STRING']]
                
                # Get DISTINCT values from key text columns to understand table content diversity
                table_description = []
                if text_columns:
                    for col in text_columns[:3]:  # Check first 3 text columns
                        try:
                            # Query for distinct values from this column
                            distinct_query = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL LIMIT 20"
                            distinct_df = pd.read_sql(distinct_query, engine)
                            if not distinct_df.empty:
                                distinct_vals = distinct_df[col].dropna().unique()
                                if len(distinct_vals) > 0:
                                    # Create a description of what this column contains
                                    val_samples = [str(v)[:60] for v in distinct_vals[:8]]  # Show up to 8 examples
                                    table_description.append(f"  {col} contains values like: {', '.join(val_samples)}")
                                    if len(distinct_vals) > 8:
                                        table_description.append(f"    ... and {len(distinct_vals) - 8} more unique values")
                        except Exception:
                            pass
                
                # Add table description based on column names and content
                if table_description:
                    schema_summary.append("This table contains:")
                    schema_summary.extend(table_description)
                    schema_summary.append("")
                
                # Add sample data examples
                if sample_data is not None and not sample_data.empty:
                    schema_summary.append("Sample rows (examples of actual data):")
                    # Show a few example rows with key information
                    for idx, row in sample_data.head(3).iterrows():
                        row_examples = []
                        for col in text_columns[:2]:  # Show first 2 text columns
                            if col in sample_data.columns:
                                val = row[col]
                                if pd.notna(val) and str(val).strip():
                                    val_str = str(val).strip()
                                    # Truncate very long values but keep enough context
                                    if len(val_str) > 80:
                                        val_str = val_str[:77] + "..."
                                    row_examples.append(f"{col}='{val_str}'")
                        
                        if row_examples:
                            schema_summary.append(f"  Row {idx+1}: {', '.join(row_examples)}")
                else:
                    schema_summary.append("(No sample data available)")
                
                schema_summary.append("")  # Empty line between tables
            
            return "\n".join(schema_summary)
        except Exception as e:
            # If schema scan fails, return None (will use fallback)
            return None

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
