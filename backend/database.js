/*
sets up the...:
the user database (list of users and their sidebar clippings)
canvas database (list of canvas names/ids)
and version history database (timestamped versions of each canvas, with json blob of the canvas data at that time)
*/
import sqlite3 from "sqlite3";
import { open } from "sqlite";
import fs from 'fs';

const defaultCanvasData = JSON.parse(fs.readFileSync('./defaultCanvasData.json', 'utf-8'));
const dbPromise = open({
  filename: "./database.db",
  driver: sqlite3.Database,
});

(async () => {
  const db = await dbPromise;

  // Create new Users table without canvases column
  await db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      userId TEXT PRIMARY KEY,
      clippings TEXT NOT NULL DEFAULT '[]' -- JSON blob of nodes in their palette
    )
  `);
  // Create Canvases table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS canvases (
      canvasId TEXT PRIMARY KEY,
      userId TEXT NOT NULL,
      canvasName TEXT NOT NULL,
      FOREIGN KEY (userId) REFERENCES users(userId) ON DELETE CASCADE
    )
  `);

  // Create Versions table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS versions (
      versionId TEXT PRIMARY KEY, -- Format: canvasId-timestamp
      canvasId TEXT NOT NULL,
      timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
      jsonBlob TEXT NOT NULL, -- JSON blob of the canvas at that point in time
      FOREIGN KEY (canvasId) REFERENCES canvases(canvasId) ON DELETE CASCADE
    )
  `);

  const admins = ['shm', 'ethan', 'elaine', 'sophia', 'bob', 'p1']; // Insert admins if they aren't there already
  const stmt = await db.prepare(`
    INSERT INTO users (userId) 
    SELECT ? WHERE NOT EXISTS (SELECT 1 FROM users WHERE userId = ?)
  `);

  for (const user of admins) {
    await stmt.run(user, user);

    // Check if the default canvas for the user already exists
    const canvasExists = await db.get("SELECT 1 FROM canvases WHERE canvasId = ?", `${user}-0`);
    if (!canvasExists) {
      // Insert new canvas data
      const canvasStmt = await db.prepare(`
        INSERT INTO canvases (canvasId, userId, canvasName) 
        VALUES (?, ?, ?)
      `);
      await canvasStmt.run(`${user}-0`, user, `${user}'s default test canvas`);

      // Insert into versions table
      const timestamp = new Date().toISOString();
      const versionStmt = await db.prepare(`
        INSERT INTO versions (versionId, canvasId, timestamp, jsonBlob) 
        VALUES (?, ?, ?, ?)
      `);
      await versionStmt.run(`${user}-0-${timestamp}`, `${user}-0`, timestamp, JSON.stringify(defaultCanvasData));
      await canvasStmt.finalize();
      await versionStmt.finalize();
    }
  }

  await stmt.finalize();
  console.log("Database initialized!");
})();

export default dbPromise;