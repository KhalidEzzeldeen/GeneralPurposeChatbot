import json
import os
import redis
from datetime import datetime
import uuid

# Redis Configuration (Assuming default localhost:6379, user needs to install Redis or we can mock it)
# For now, I will use a simple file-based json store if redis fails, to keep it "executable" offline without extra service install immediately.
# But I will write the structure for Redis.

class SessionManager:
    def __init__(self, use_redis=False):
        self.use_redis = use_redis
        if self.use_redis:
            self.r = redis.Redis(host='localhost', port=6379, db=0)
        else:
            self.file_store = os.path.join("storage", "sessions.json")
            # Ensure storage directory exists
            storage_dir = os.path.dirname(self.file_store)
            if storage_dir and not os.path.exists(storage_dir):
                os.makedirs(storage_dir, exist_ok=True)
            if not os.path.exists(self.file_store):
                with open(self.file_store, 'w') as f:
                    json.dump({}, f)

    def get_session(self, session_id):
        if self.use_redis:
            data = self.r.get(f"session:{session_id}")
            return json.loads(data) if data else []
        else:
            with open(self.file_store, 'r') as f:
                data = json.load(f)
            return data.get(session_id, [])

    def append_message(self, session_id, role, content):
        history = self.get_session(session_id)
        history.append({
            "role": role,
            "content": content,
            "timestamp": str(datetime.now())
        })
        
        if self.use_redis:
            self.r.set(f"session:{session_id}", json.dumps(history))
        else:
            with open(self.file_store, 'r') as f:
                data = json.load(f)
            data[session_id] = history
            with open(self.file_store, 'w') as f:
                json.dump(data, f)
                
    def get_summary(self, session_id):
        # Placeholder for summary retrieval
        return ""
