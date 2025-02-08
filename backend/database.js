import sqlite3 from "sqlite3";
import { open } from "sqlite";

const dbPromise = open({
  filename: "./database.db",
  driver: sqlite3.Database,
});

(async () => {
  const db = await dbPromise;

  // ✅ Users table (now with a palette of clippings)
  await db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY, -- Example values: "admin", "P1", "P2"
      password TEXT DEFAULT '', -- Blank password by default
      clippings TEXT NOT NULL DEFAULT '[]' -- JSON-encoded list of saved clippings (favorite nodes)
    )
  `);

  // ✅ Canvases table (storing nodes as JSON)
  await db.exec(`
    CREATE TABLE IF NOT EXISTS canvases (
      id TEXT PRIMARY KEY, -- will take the form <userID>-# like so: "shm-1", "admin-2"
      user_id TEXT NOT NULL, -- Foreign key to users
      name TEXT NOT NULL, -- Name of the canvas, ex "My Cool Canvas"
      nodes TEXT NOT NULL DEFAULT '[]', -- JSON-encoded list of nodes
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
  `);

  console.log("Database initialized with Users and Canvases!");
})();

export default dbPromise;
