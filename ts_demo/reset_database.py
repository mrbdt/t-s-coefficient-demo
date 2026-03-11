"""
Reset the SQLite database used by the T-S prototype.

What this script does:
1) Delete the existing DB file if it exists.
2) Recreate an empty DB with the current schema.

"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from ts_system import init_db

load_dotenv()


def main() -> None:
    db_path = Path(os.getenv("TS_DB_PATH", "system_db.sqlite3"))

    if db_path.exists():
        db_path.unlink()
        print(f"Deleted existing DB: {db_path}")
    else:
        print(f"No existing DB found at: {db_path}")

    conn = sqlite3.connect(db_path)
    init_db(conn)
    conn.close()
    print(f"Created fresh empty DB with current schema: {db_path}")


if __name__ == "__main__":
    main()
