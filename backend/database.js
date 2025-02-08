import sqlite3 from "sqlite3";
import { open } from "sqlite";

const dbPromise = open({
  filename: "./database.db",
  driver: sqlite3.Database,
});

// ✅ Define initial nodes (matches /nodes/index.ts)
const initialNodes = JSON.stringify([
  {
    id: "example",
    type: "text",
    position: { x: 100, y: 100 },
    data: { content: "bunny on the moon", loading: false, combinable: false },
  }
]);

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
      INSERT INTO canvases (id, user_id, name, nodes)
      VALUES ('new-canvas', 'default-user', 'Untitled Canvas', ?)
    `, [initialNodes]);
    console.log("✅ Default 'new-canvas' created with initial nodes.");
  }

  console.log("Database initialized with Users and Canvases!");
  const users = await db.all(`
    SELECT u.id, COUNT(c.id) as canvas_count, GROUP_CONCAT(c.id || ': ' || c.name, ', ') as canvas_details
    FROM users u
    LEFT JOIN canvases c ON u.id = c.user_id
    GROUP BY u.id
    LIMIT 5
  `);

  users.forEach(user => {
    const canvasDetails = user.canvas_details ? user.canvas_details.split(', ').slice(0, 2).join(', ') : 'No canvases';
    console.log(`User: ${user.id}, Canvases: ${user.canvas_count}, Canvas Details: [${canvasDetails}]`);
  });
})();

export default dbPromise;