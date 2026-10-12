# Long-term history and memory

The conversation engine creates two SQLite tables automatically in `nova_memory.sqlite3`:

- `conversation_archive`: user/assistant conversation entries with timestamps.
- `long_term_memories`: categorized memories, keywords, source, and timestamps.

The database path currently resolves beside `conversation_engine.py`. It is runtime user data, not source code. The `.gitignore` excludes `*.sqlite3` and `*.db` to avoid committing conversations. No real database is included in this refactor package.

Before a public release, document backup, restore, deletion, and schema migration behavior and verify that existing databases are preserved during upgrades.
