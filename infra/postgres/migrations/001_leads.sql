CREATE TABLE IF NOT EXISTS leads (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  company TEXT,
  message TEXT,
  source TEXT DEFAULT 'website_form',
  status TEXT DEFAULT 'new',
  created_at TIMESTAMPTZ DEFAULT now()
);
