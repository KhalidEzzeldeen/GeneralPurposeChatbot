import json
import os
import threading

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "llm": {
        "model_name": "qwen2.5:7b-instruct",
        "temperature": 0.2,
        "system_prompt": "You are a professional enterprise assistant. Use the provided context to answer questions strictly."
    },
    "database": {
        "host": "ep-long-sound-a9dqh1py-pooler.gwc.azure.neon.tech",
        "port": 5432,
        "user": "neondb_owner",
        "password": "npg_uZBXKw9eyt5h",
        "dbname": "neondb"
    },
    "routing": {
        "mode": "auto"
    },
    "debug": {
        "show_sql_debug": False
    },
    "data_path": "./data",
    "chroma_path": "./chroma_db",
    "ingested_files": []
}

class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._load_config()
            return cls._instance

    def _load_config(self):
        if not os.path.exists(CONFIG_FILE):
            self.config = DEFAULT_CONFIG
            self.save_config()
        else:
            try:
                with open(CONFIG_FILE, 'r') as f:
                    self.config = json.load(f)
                    # Merge with default to ensure new keys exist
                    for k, v in DEFAULT_CONFIG.items():
                        if k not in self.config:
                            self.config[k] = v
            except Exception:
                self.config = DEFAULT_CONFIG

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get(self, section, key=None):
        if key:
            return self.config.get(section, {}).get(key)
        return self.config.get(section)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
