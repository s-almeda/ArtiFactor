import sqlite3 from "sqlite3";
import { open } from "sqlite";

const dbPromise = open({
  filename: "./database.db",
  driver: sqlite3.Database,
});

const initialViewport = JSON.stringify({ x: 0, y: 0, zoom: 1 });
const defaultEdges = JSON.stringify([
  {
    id: "edge-text-1740696475448-image-1740696601239",
    source: "text-1740696475448",
    target: "image-1740696601239",
    type: "default"
  },
]);
const defaultNodeData = JSON.stringify([
  {
    id: "text-1740696475448",
    type: "text",
    zIndex: 1000,
    position: { x: 328.6810295180019, y: 39.41097388996745 },
    data: {
      words: [{ value: "testing!" }],
      provenance: "user",
      content: "testing!",
      intersections: [
        {
          id: "text-1740696475448",
          position: { x: 328.6810295180019, y: 39.41097388996745 },
          content: "testing!",
        },
      ],
      similarTexts: [
        {
          id: "4e0b9fddb0c2a40001000044",
          value: "Sparse",
          description:
            "Sparse, markedly simple or unadorned style, associated with artists Luca Loreti, Werner Haypeter, Leszek Skurski, Jonathan Binet.",
          relatedKeywordStrings: [
            "Luca Loreti",
            " Werner Haypeter",
            " Leszek Skurski",
            " Jonathan Binet",
          ],
          type: "Visual Qualities",
        },
        // ... other similarTexts
      ],
    },
    measured: { width: 200, height: 150 },
    selected: false,
    dragging: false,
  },
  {
    id: "image-1740696601239",
    type: "image",
    position: { x: 579.3232479089534, y: 107.44847647770641 },
    zIndex: 1000,
    data: {
      content: "https://uploads3.wikiart.org/images/moise-kisling/portrait-with-a-collar-1938.jpg",
      prompt: '"Portrait with a collar"(1938) by Moise Kisling',
      provenance: "user",
      artworks: [
        {
          title: "Portrait with a collar",
          date: "1938",
          artist: "Moise Kisling",
          keywords: [
            { id: "genre-1740696606896", type: "genre", value: "portrait" },
            { id: "style-1740696606896", type: "style", value: "Post-Impressionism" },
          ],
          description: "Moise Kisling / Portrait with a collar / Post-Impressionism / portrait / 1938",
          image: "https://uploads3.wikiart.org/images/moise-kisling/portrait-with-a-collar-1938.jpg",
        },
        // ... other artworks
      ],
      intersections: [
        {
          id: "image-1740696601239",
          position: { x: 579.3232479089534, y: 107.44847647770641 },
          content: "https://uploads3.wikiart.org/images/moise-kisling/portrait-with-a-collar-1938.jpg",
        },
      ],
    },
    dragHandle: ".drag-handle__invisible",
    measured: { width: 150, height: 150 },
    selected: false,
    dragging: false,
  },
]);

(async () => {
  const db = await dbPromise;

  // Create Users table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      password TEXT DEFAULT '',
      clippings TEXT NOT NULL DEFAULT '[]'
    )
  `);

  // Create Canvases table (now with viewport)
  await db.exec(`
    CREATE TABLE IF NOT EXISTS canvases (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      name TEXT NOT NULL,
      nodes TEXT NOT NULL DEFAULT '[]',
      edges TEXT NOT NULL DEFAULT '[]',
      viewport TEXT NOT NULL DEFAULT '${initialViewport}',
      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
  `);

  const admins = ['shm', 'ethan', 'elaine', 'sophia']; //insert admins if they aren't there already
  const stmt = await db.prepare("INSERT INTO users (id) SELECT ? WHERE NOT EXISTS (SELECT 1 FROM users WHERE id = ?)");

  for (const user of admins) {
    await stmt.run(user, user);

    // Delete existing canvas if it exists
    await db.run("DELETE FROM canvases WHERE id = ?", `${user}-0`);

    // Insert new canvas data
    const canvasStmt = await db.prepare(`
      INSERT INTO canvases (id, user_id, name, nodes, edges, viewport, timestamp) 
      VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    `);
    await canvasStmt.run(`${user}-0`, user, `${user}'s default test canvas`, defaultNodeData, defaultEdges, initialViewport);
    await canvasStmt.finalize();
  }

  await stmt.finalize();
  console.log("Database initialized!");
})();

export default dbPromise;
