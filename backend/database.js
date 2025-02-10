import sqlite3 from "sqlite3";
import { open } from "sqlite";

const dbPromise = open({
  filename: "./database.db",
  driver: sqlite3.Database,
});

// ✅ Define initial nodes & viewport
const initialNodes = JSON.stringify([
  {
    id: "example",
    type: "text",
    position: { x: 100, y: 100 },
    data: { content: "bunny on the moon", loading: false, combinable: false },
  }
]);

const initialViewport = JSON.stringify({ x: 0, y: 0, zoom: 1 });

(async () => {
  const db = await dbPromise;

  // ✅ Users table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      password TEXT DEFAULT '',
      clippings TEXT NOT NULL DEFAULT '[]'
    )
  `);

  // ✅ Canvases table (now with viewport)
  await db.exec(`
    CREATE TABLE IF NOT EXISTS canvases (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      name TEXT NOT NULL,
      nodes TEXT NOT NULL DEFAULT '[]',
      viewport TEXT NOT NULL DEFAULT '${initialViewport}', -- Store viewport separately
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
  `);

  // ✅ Ensure a default user exists
  const existingUser = await db.get(`SELECT id FROM users LIMIT 1`);
  if (!existingUser) {
    await db.run(`
      INSERT INTO users (id, password, clippings)
      VALUES ('default-user', '', '[]')
    `);
    console.log("✅ Default user 'default-user' created.");
  }

  // ✅ Ensure the `new-canvas` exists for the default user
  const existingCanvas = await db.get(`SELECT id FROM canvases WHERE id = 'new-canvas'`);
  if (!existingCanvas) {
    await db.run(`
      INSERT INTO canvases (id, user_id, name, nodes, viewport)
      VALUES ('new-canvas', 'default-user', 'Untitled Canvas', ?, ?)
    `, [initialNodes, initialViewport]);
    console.log("✅ Default 'new-canvas' created with initial nodes and viewport.");
  }

  console.log("Database initialized!");
})();

export default dbPromise;
