-- Run with: wrangler d1 execute second-brain-db --file=schema.sql

CREATE TABLE IF NOT EXISTS entries (
  id          TEXT PRIMARY KEY,
  content     TEXT NOT NULL,
  tags        TEXT NOT NULL DEFAULT '[]',  -- JSON array
  source      TEXT NOT NULL DEFAULT 'api', -- 'phone', 'browser', 'voice', 'claude', 'api'
  created_at  INTEGER NOT NULL,            -- Unix ms timestamp
  vector_ids  TEXT NOT NULL DEFAULT '[]'   -- JSON array of Vectorize IDs owned by this entry
);

CREATE INDEX IF NOT EXISTS idx_entries_created_at ON entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_entries_source ON entries(source);

-- Migration for existing databases:
-- ALTER TABLE entries ADD COLUMN vector_ids TEXT NOT NULL DEFAULT '[]';
