import os
import time

# Must be set BEFORE importing phoenix so config is picked up at module load
_storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phoenix_storage"))
os.makedirs(_storage_dir, exist_ok=True)

os.environ["PHOENIX_WORKING_DIR"] = _storage_dir
# Explicitly point the database to a persistent file; without this Phoenix
# defaults to a temporary directory and all data is lost on restart.
# Use forward slashes — Windows backslashes in the URL cause SQLAlchemy to
# silently fall back to a random temp directory.
_db_url = "sqlite:///" + _storage_dir.replace("\\", "/") + "/phoenix.db"
os.environ["PHOENIX_SQL_DATABASE_URL"] = _db_url

import phoenix as px

app = px.launch_app(use_temp_dir=False)
print(f"Phoenix running at http://localhost:6006")
print(f"Data persisted at: {_storage_dir}/phoenix.db")

#keep it alive
input("Press Enter to exit...")