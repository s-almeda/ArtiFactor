// Main backend server (not the knowledgebase + retrieval models flask server, which is in Python, and run in docker)
// this JS server forwards request to the flask knowledgebase+model server,
// as well as requests to the user data base (for user account and canvas version history data)
// and requests to other APIs, e.g. for Replicate based generation
import express from "express";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs"; // Import fs for file operations
import path from "path"; // Import path for file paths
import dbPromise from "./database.js"; // Import the database module
import fetch from "node-fetch"; // Ensure you have node-fetch installed
import { fileURLToPath } from "url";

// Define __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

//const flask_server = "https://data.snailbunny.site";
const flask_server = "http://localhost:8080";

// ---- get replicate access for image to text --- ///
import Replicate from "replicate";
dotenv.config();
if (!process.env.REPLICATE_API_TOKEN) { //this will fail if you don't have my replicate api key in your .env file! (like, saved to your terminal)
  console.error("REPLICATE_API_TOKEN is not set in the environment");
}
const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

// ------------------------------------------------ //

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increase the JSON payload limit 
app.use(express.urlencoded({ limit: '50mb', extended: true })); // Increase the URL-encoded payload limit 

// Global log file
let logFilePath = path.join(__dirname, `log-${new Date().toISOString()}.txt`);

// Function to log data to the global log file
const logToFile = (message) => {
  const timestamp = new Date().toISOString();
  const logEntry = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(logFilePath, logEntry, (err) => {
    if (err) {
      console.error("Error writing to log file:", err);
    }
  });
};

// Middleware to log every API call
app.use((req, res, next) => {
  const { method, url, body, query, params } = req;
  const logMessage = `API Call: ${method} ${url} | Body: ${JSON.stringify(
    body
  )} | Query: ${JSON.stringify(query)} | Params: ${JSON.stringify(params)}`;
  logToFile(logMessage);
  next();
});

// New /overview route to display the current log file
app.get("/overview", (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  try {
    const logContents = fs.readFileSync(logFilePath, "utf-8");
    res.send(`<pre>${logContents}</pre>`);
  } catch (error) {
    console.error("Error reading log file:", error);
    res.status(500).send("Error reading log file");
  }
});

// New /fresh route to start a new log file
app.get("/fresh", (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  try {
    const oldLogFilePath = logFilePath;
    logFilePath = path.join(__dirname, `log-${new Date().toISOString()}.txt`);
    logToFile(`Started new log file: ${logFilePath}`);
    res.json({
      success: true,
      message: `New log file started. Old log file: ${oldLogFilePath}`,
    });
  } catch (error) {
    console.error("Error starting new log file:", error);
    res.status(500).json({ error: "Error starting new log file" });
  }
});

// New / route
app.get("/", (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.end('Hello World! if you can see this, that means the backend server is working!');
});

app.get("/health_check", async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  
  const tests = [
    { 
      name: 'keywordCheck',
      endpoint: '/keyword_check',
      payload: { text: "dog eating a sandwich abstract expressionism portraiture cezanne", threshold: 0.3 }
    },
    { 
      name: 'textLookup',
      endpoint: '/lookup_text',
      payload: { query: "dogs", top_k: 5 }
    },
    { 
      name: 'imageLookup',
      endpoint: '/image',
      payload: { image: "https://d32dm0rphc51dk.cloudfront.net/gTPexURCjkBek6MrG7g1bg/small.jpg", top_k: 3 }
    }
  ];

  const results = await Promise.allSettled(
    tests.map(test => 
      axios.post(`${flask_server}${test.endpoint}`, test.payload, {
        headers: { "Content-Type": "application/json" },
        timeout: 10000
      })
      .then(response => ({ 
        [test.name]: {
          request: test.payload,
          response: response.data,
          status: 'success'
        }
      }))
      .catch(error => ({ 
        [test.name]: { 
          request: test.payload,
          error: error.message,
          status: 'error'
        }
      }))
    )
  );

  const healthData = Object.assign({}, ...results.map(r => r.value));
  
  let html = '<html><body><h1>ML Server Health Check</h1>';
  html += `<p>Checked at: ${new Date().toISOString()}</p>`;
  
  for (const [name, data] of Object.entries(healthData)) {
    html += `<h2>${name} - ${data.status === 'success' ? '✅' : '❌'}</h2>`;
    html += '<pre>' + JSON.stringify(data.request, null, 2) + '</pre>';
    
    if (data.status === 'success') {
      html += '<pre>' + JSON.stringify(data.response, null, 2) + '</pre>';
    } else {
      html += `<p style="color:red">Error: ${data.error}</p>`;
    }
    html += '<hr>';
  }
  
  html += '</body></html>';
  
  res.send(html);
});

// -------- Start of flask ML/Data server API calls ---------- //

app.post("/api/check-for-keywords", async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const { text, threshold = 0.3 } = req.body;

  if (!text) {
    return res.status(400).json({ error: "Missing 'text' in request body" });
  }

  try {
    const response = await axios.post(
      `${flask_server}/keyword_check`,
      { text, threshold },
      { headers: { "Content-Type": "application/json" } }
    );
    res.setHeader('Access-Control-Allow-Origin', '*');
    console.log("Received response from flask server:", response.data);

    
    res.json(response.data);


  } catch (error) {
    console.error("Error checking for keywords:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// get similar texts
app.post("/api/get-similar-texts", async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  console.log("received a get-similar-texts request:", req.body);
  const { query, top_k = 5 } = req.body;

  if (!query) {
    return res.status(400).json({ error: "Missing 'query' in request body" });
  }
  console.log("sending to flask server...");
  try {
    const response = await axios.post(
      `${flask_server}/lookup_text`,
      { query, top_k },
      { headers: { "Content-Type": "application/json" } }
    );

    res.json(response.data);
    console.log("got response from flask server:", response.data);
    
  } catch (error) {
    console.error("Error getting similar texts:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


// new /api/get-similar-images route takes an image: base64string or image: imageURL within a .json file and returns a list of similar images as urls or as base64
app.post("/api/get-similar-images", async (req, res) => {
  const { image } = req.body;
  console.log("received a get-similar-images request:", req.body);

  if (!image) {
    return res.status(400).json({ error: "Missing 'image' in request body" });
  }

  let imageData = image;

  try {
    console.log(`Sending image to ml/data server... ${flask_server}/image`);
    const response = await axios.post(
      `${flask_server}/image`,
      {
        image: imageData,
      }
    );
    console.log(`Got response from ${flask_server}:`, response.data);
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.json(response.data);
  } catch (error) {
    console.error("Error getting similar images:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// -------- End of flask ML/Data server API calls ---------- //

app.post("/api/generate-text", async (req, res) => {
  const { imageUrl = "https://uploads3.wikiart.org/images/pierre-tal-coat/en-grimpant-1962.jpg" } = req.body;
  console.log("received a generate-text request:", req.body);

  const input = {
    mode: "fast",
    image: imageUrl,
    clip_model_name: "ViT-H-14/laion2b_s32b_b79k"
  };

  try {
    const output = await replicate.run(
      "pharmapsychotic/clip-interrogator:8151e1c9f47e696fa316146a2e35812ccf79cfc9eba05b11c7f450155102af70",
      { input }
    );
    console.log("Output from replicate API:", output);
    const truncatedOutput = output.split(',').slice(0, 5).join(', ');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.json({ text: truncatedOutput });
    logToFile(`IMAGE for text generation: ${imageUrl} | Successful generated text: ${truncatedOutput}`);
  } catch (error) {
    console.error("Error in generate-text:", error);
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.status(500).json({ error: "Internal server error" });
  }
});


app.post("/api/generate-image", async (req, res) => {
  const { prompt = "lizard on saturn" } = req.body; //default prompt with no form data
  
  const formData = new URLSearchParams();
  formData.append("prompt", prompt);
  formData.append("output_format", "webp");

  console.log("MESSAGING REAGENT... with prompt:", prompt);

  try {
    // Send POST request to the Reagent API
    const response = await fetch( 
      //FOR BLACK FOREST SHNELL MODEL:
      'https://noggin.rea.gent/peaceful-spoonbill-8088',
      //FOR SDXL MODEL:
      //   'https://noggin.rea.gent/scientific-mammal-8442',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          //for sdxl:  Authorization: 'Bearer rg_v1_a19uz85azi53qm4b3vpjg6zyoud51leu8q9h_ngk',          
          Authorization: 'Bearer rg_v1_mc66rqibqakaslgktuy89n5lfjhdygcwvfj2_ngk',
        },
        body: JSON.stringify({
          "prompt": prompt,
        }),
      }
    );

    const redirectUrl = response.url;
    console.log("Redirect URL from API:", redirectUrl);
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.json({ imageUrl: redirectUrl });
    logToFile(`PROMPT for image generation: ${prompt} | Successful image generation : ${redirectUrl}`);

  } catch (error) {
    console.error("Error in generate-image:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});



// --------- USER DATA -- Adding and authenticating users! -------------- //

app.post("/api/add-user", async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const { userID = "" } = req.body;

  if (!userID) {
    return res.status(400).json({ error: "Missing required field: userID" });
  }

  try {
    const db = await dbPromise;

    // Check if user already exists
    const existingUser = await db.get(`SELECT * FROM users WHERE userId = ?`, [userID]);
    if (existingUser) {
      return res.json({ success: true, message: "User already exists", userID });
    }

    await db.run(`INSERT INTO users (userId) VALUES (?)`, [userID]);  //no passwords for now

    res.json({ success: true, userID, message: "User added successfully!" });
  } catch (error) {
    console.error("Error adding user:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


// --------- GETTING USER DATA - list users, list a user's canvasses -------------- //
// list users and their canvases
app.get("/api/list-users", async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  try {
    const db = await dbPromise;

    // Get all users
    const users = await db.all(`SELECT userId FROM users`);

    // Get each user's canvases asynchronously
    const usersWithCanvases = await Promise.all(
      users.map(async (user) => {
        const canvases = await db.all(`SELECT * FROM canvases WHERE userId = ?`, [user.userId]);
        return {
          userId: user.userId,
          canvases: canvases.map((canvas) => ({
            canvasId: canvas.canvasId,
            canvasName: canvas.canvasName,
            timestamp: canvas.timestamp,
          })),
        };
      })
    );
    console.log(`/list-users Listed ${usersWithCanvases.length} users`);
    res.json({ success: true, users: usersWithCanvases });
  } catch (error) {
    console.error("Error listing users:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


app.get("/api/list-canvases/:userID", async (req, res) => {
  const { userID } = req.params;
  res.setHeader('Access-Control-Allow-Origin', '*');
  console.log("Attempting to list canvases for user:", userID);
  try {
    const db = await dbPromise;

    // Check if user exists
    const user = await db.get(`SELECT userId FROM users WHERE userId = ?`, [userID]);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    // Get all canvases for the user
    const canvases = await db.all(`SELECT canvasId, canvasName FROM canvases WHERE userId = ?`, [userID]);

    // Format the canvases
    const formattedCanvases = canvases.map(canvas => ({
      canvasId: canvas.canvasId,
      canvasName: canvas.canvasName
    }));
    console.log(`Listed ${formattedCanvases.length} canvases for user ${userID}: ${formattedCanvases.map(c => c.name).join(", ")}`);

    res.json({ success: true, canvases: formattedCanvases });
  } catch (error) {
    console.error("Error getting canvases:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});



// --------- CANVAS DATA - Saving and loading canvases -------------- //
app.get("/api/next-canvas-id/:userID", async (req, res) => {
  const { userID } = req.params;
  console.log("Attempting to get next canvas ID for user:", userID);
  
  try {
    const db = await dbPromise;

    // Check if user exists
    const user = await db.get(`SELECT userId FROM users WHERE userId = ?`, [userID]);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    // Get the highest canvas number for this user using MAX()
    const result = await db.get(
      `SELECT MAX(CAST(SUBSTR(canvasId, INSTR(canvasId, '-') + 1) AS INTEGER)) AS maxId
       FROM canvases WHERE userId = ?`,
      [userID]
    );

    const nextCanvasId = `${userID}-${(result?.maxId ?? -1) + 1}`;

    console.log(`Next canvas ID for user ${userID} is ${nextCanvasId}`);
    res.json({ success: true, nextCanvasId });

  } catch (error) {
    console.error("Error getting next canvas ID:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


app.post("/api/save-canvas", async (req, res) => {
  console.log("Received request to save canvas.");
  const { userID, canvasID, canvasName, canvasJSONObject, timestamp: reqTimestamp } = req.body;
  const timestamp = reqTimestamp || new Date().toISOString();
  
  console.log("Attempting to save canvas:", req.body);

  if (!userID || !canvasID || !canvasName || !canvasJSONObject) {
    return res.status(400).json({ error: "Missing required fields" });
  }

  try {
    const db = await dbPromise;
    
    // Check if canvas already exists
    const existingCanvas = await db.get("SELECT canvasId FROM canvases WHERE canvasId = ?", [canvasID]);

    if (existingCanvas) {
      console.log(`Found existing canvas ${canvasID} for user ${userID}. Updating with new save.`);

      // Update the canvas name
      await db.run(
        `UPDATE canvases SET canvasName = ? WHERE canvasId = ?`,
        [canvasName, canvasID]
      );

    } else {
      console.log(`Creating new canvas ${canvasID} for user ${userID}.`);

      // Insert new canvas
      await db.run(
        `INSERT INTO canvases (canvasId, userId, canvasName) VALUES (?, ?, ?)`,
        [canvasID, userID, canvasName]
      );
    }

    // Retrieve the latest version of the canvas
    const latestVersion = await db.get(
      `SELECT * FROM versions WHERE canvasId = ? ORDER BY timestamp DESC LIMIT 1`,
      [canvasID]
    );

    let saveNewVersion = true;

    if (latestVersion) {
      const latestCanvasData = JSON.parse(latestVersion.jsonBlob);
      const latestNodeIds = latestCanvasData.nodes.map(node => node.id);
      const newNodeIds = canvasJSONObject.nodes.map(node => node.id);

      // Check if node IDs have changed
      if (JSON.stringify(latestNodeIds) === JSON.stringify(newNodeIds)) {
        saveNewVersion = false;
      }
    }

    if (saveNewVersion) {
      // Save a new version in the versions table
      await db.run(
        `INSERT INTO versions (versionId, canvasId, timestamp, jsonBlob) VALUES (?, ?, ?, ?)`,
        [`${canvasID}-${timestamp}`, `${canvasID}`, timestamp, JSON.stringify(canvasJSONObject)]
      );
      console.log(`Saved new version for canvas ${canvasID} at ${timestamp}`);
    } else {
      // Overwrite the latest version
      await db.run(
        `UPDATE versions SET jsonBlob = ?, timestamp = ? WHERE versionId = ?`,
        [JSON.stringify(canvasJSONObject), timestamp, latestVersion.versionId]
      );
      console.log(`Overwrote latest version for canvas ${canvasID} at ${timestamp}`);
    }

    res.json({ success: true });

  } catch (error) {
    console.error("Error saving canvas:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


app.delete("/api/delete-canvas/:userID/:canvasID", async (req, res) => {
  const { userID, canvasID } = req.params;
  console.log(`Attempting to delete canvas ${canvasID} for user ${userID}`);

  try {
    const db = await dbPromise;
    
    // Delete canvas from the canvases table
    const result = await db.run(
      `DELETE FROM canvases WHERE canvasId = ? AND userId = ?`, 
      [canvasID, userID]
    );

    // Check if any canvas was deleted
    if (result.changes === 0) {
      return res.status(404).json({ error: "Canvas not found or does not belong to the user" });
    }

    // Delete all versions associated with the canvas
    await db.run(
      `DELETE FROM versions WHERE canvasId = ?`, 
      [canvasID]
    );

    console.log(`Canvas ${canvasID} and its versions deleted successfully for user ${userID}.`);
    logToFile(`Canvas ${canvasID} and its versions deleted successfully for user ${userID}.`);
    res.json({ success: true, message: "Canvas and its versions deleted successfully" });
  } catch (error) {
    console.error("Error deleting canvas:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


app.get("/api/get-canvas/:canvasID", async (req, res) => {
  console.log("Attempting to pull canvas from database:", req.params);
  const { canvasID } = req.params;

  try {
    const db = await dbPromise;

    // Retrieve the canvas name from the canvases table
    const canvas = await db.get(`SELECT canvasName FROM canvases WHERE canvasId = ?`, [canvasID]);
    if (!canvas) {
      return res.status(404).json({ error: "Canvas not found" });
    }

    // Retrieve the latest version of the canvas
    const version = await db.get(`SELECT * FROM versions WHERE canvasId = ? ORDER BY timestamp DESC LIMIT 1`, [canvasID]);
    if (!version) {
      return res.status(404).json({ error: "Canvas version not found" });
    }

    // ✅ Parse JSON fields from the version data
    const canvasData = JSON.parse(version.jsonBlob);
    const { nodes, edges, viewport } = canvasData;

    console.log(`✅ Canvas ${canvasID} loaded with ${nodes.length} nodes and ${edges.length} edges from time: ${version.timestamp}`);
    logToFile(`Canvas ${canvasID} loaded with ${nodes.length} nodes and ${edges.length} edges from time: ${version.timestamp}`);
    res.json({ success: true, canvas: { canvasID, canvasName: canvas.canvasName, nodes, edges, viewport }, timestamp: version.timestamp });
  } catch (error) {
    console.error("Error loading canvas:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/api/list-versions/:canvasID", async (req, res) => {
  const { canvasID } = req.params;
  console.log(`Attempting to list versions for canvas: ${canvasID}`);

  try {
    const db = await dbPromise;

    // Retrieve all versions for the given canvas ID
    const versions = await db.all(`SELECT versionId FROM versions WHERE canvasId = ? ORDER BY timestamp DESC`, [canvasID]);

    if (versions.length === 0) {
      return res.status(404).json({ error: "No versions found for the specified canvas ID" });
    }

    console.log(`Listed ${versions.length} versions for canvas ${canvasID}`);
    res.json({ success: true, versions });
  } catch (error) {
    console.error("Error listing versions:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/api/get-version/:versionID", async (req, res) => {
  const { versionID } = req.params;
  console.log(`Attempting to get version: ${versionID}`);

  try {
    const db = await dbPromise;

    // Retrieve the specific version by version ID
    const version = await db.get(`SELECT * FROM versions WHERE versionId = ?`, [versionID]);

    if (!version) {
      return res.status(404).json({ error: "Version not found" });
    }

    // Parse the JSON blob
    const jsonBlob = JSON.parse(version.jsonBlob);

    console.log(`Retrieved version ${versionID}`);
    res.json({ success: true, version: { versionId: version.versionId, timestamp: version.timestamp, jsonBlob } });
  } catch (error) {
    console.error("Error getting version:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});


  //------------- CLIPPINGS (for saving and loading a user's Palette) ---------//
  app.get("/api/get-clippings/:userID", async (req, res) => {
    let { userID } = req.params;

    if (!userID) {
      userID = "default";
    }

    try {
      const db = await dbPromise;
      const user = await db.get(`SELECT clippings FROM users WHERE userId = ?`, [userID]);

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      res.json({ success: true, clippings: JSON.parse(user.clippings) });
    } catch (error) {
      console.error("Error loading clippings:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  app.post("/api/add-clipping", async (req, res) => {
    const { userID, clipping } = req.body;

    if (!userID || !clipping || typeof clipping !== "object") {
      return res.status(400).json({ error: "Invalid userID or clipping format" });
    }

    try {
      const db = await dbPromise;

      // Fetch current clippings
      const user = await db.get(`SELECT clippings FROM users WHERE userId = ?`, [userID]);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }

      let clippings;
      try {
        clippings = JSON.parse(user.clippings);
        if (!Array.isArray(clippings)) {
          clippings = []; // Ensure it's an array, not an object
        }
      } catch (error) {
        clippings = []; // Fallback to an empty array if parsing fails
      }
      console.log("clippings:", clippings);

      if (clippings.length === 0) {
        clippings = [clipping];
      } else {
        // Prevent duplicate clippings by checking for existing clipping
        const clippingExists = clippings.some((c) => JSON.stringify(c) === JSON.stringify(clipping));
        if (!clippingExists) {
          clippings.push(clipping);
        } else {
          return res.status(400).json({ error: "Clipping already exists" });
        }
      }

      // Update user's clippings
      await db.run(`UPDATE users SET clippings = ? WHERE userId = ?`, [JSON.stringify(clippings), userID]);

      res.json({ success: true, message: "Clipping added!", clippings });
    } catch (error) {
      console.error("Error adding clipping:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });




  app.post("/api/remove-clipping", async (req, res) => {

    const { userID, clipping } = req.body;
  
    if (!userID || !clipping || typeof clipping !== "object") {
      return res.status(400).json({ error: "Invalid userID or clipping format" });
    }
    console.log("removing clipping...", clipping);
  
    try {
      const db = await dbPromise;
  
      // Fetch current clippings
      const user = await db.get(`SELECT clippings FROM users WHERE userId = ?`, [userID]);

      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }
  
      let clippings = JSON.parse(user.clippings);
      console.log("number of clippings:", clippings.length);
  
      // Remove clipping by filtering it out based on the clipping ID
      clippings = clippings.filter((c) => c.id !== clipping.id);
  
      // Update user's clippings
      await db.run(`UPDATE users SET clippings = ? WHERE userId = ?`, [JSON.stringify(clippings), userID]);
      console.log("new num of clippings:", clippings.length);
      res.json({ success: true, message: "Clipping removed!", clippings });
    } catch (error) {
      console.error("Error removing clipping:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  
  
  
  



app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});