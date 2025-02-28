import express from "express";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";
import dbPromise from "./database.js"; // Import the database module
import fetch from 'node-fetch'; // Ensure you have node-fetch installed

const flask_server = "https://data.snailbunny.site";
//const flask_server = "http://localhost:8080";

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
app.use(express.json());

// New /overview route
app.get("/", (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.end('Hello World! if you can see this, that means the backend server is working!');
});
// New /overview route
app.get("/overview", (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.end('if you can see this, that means the backend server is working!');
});

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

  // Check if the image is a URL
  if (image.startsWith("http://") || image.startsWith("https://")) {
    try {
      const response = await fetch(image);
      const buffer = await response.buffer();
      imageData = buffer.toString('base64');
    } catch (error) {
      console.error("Error converting image URL to base64:", error);
      return res.status(500).json({ error: "Error processing image URL" });
    }
  }

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
    const response = await fetch(
      'https://noggin.rea.gent/peaceful-spoonbill-8088',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
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



  } catch (error) {
    console.error("Error in generate-image:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});



// --------- USER DATA -- Adding and authenticating users! -------------- //

app.post("/api/add-user", async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  const { userID, password = "" } = req.body;

  if (!userID) {
    return res.status(400).json({ error: "Missing required field: userID" });
  }

  try {
    const db = await dbPromise;

    // Check if user already exists
    const existingUser = await db.get(`SELECT * FROM users WHERE id = ?`, [userID]);
    if (existingUser) {
      return res.json({ success: true, message: "User already exists", userID });
    }

    // Insert new user with a blank password by default
    await db.run(`INSERT INTO users (id, password) VALUES (?, ?)`, [userID, password]);

    res.json({ success: true, userID, message: "User added successfully!" });
  } catch (error) {
    console.error("Error adding user:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});
//check a user's password -- no encryption for now lol, super simple. 
app.post("/api/authenticate-user", async (req, res) => {
  const { userID, password } = req.body;
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (!userID) {
    return res.status(400).json({ error: "Missing required field: userID" });
  }

  try {
    const db = await dbPromise;

    // Fetch the user
    const user = await db.get(`SELECT * FROM users WHERE id = ?`, [userID]);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    // Check password (blank is allowed)
    if (user.password !== password) {
      return res.status(401).json({ error: "Incorrect password" });
    }

    res.json({ success: true, userID, message: "User authenticated!" });
  } catch (error) {
    console.error("Error authenticating user:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});





// --------- GETTING USER DATA - list users, list a user's canvasses -------------- //

app.get("/api/list-canvases/:userID", async (req, res) => {
  const { userID } = req.params;
  res.setHeader('Access-Control-Allow-Origin', '*');
  console.log("Attempting to list canvases for user:", userID);
  try {
    const db = await dbPromise;

    // Check if user exists
    const user = await db.get(`SELECT id FROM users WHERE id = ?`, [userID]);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    // Get all canvases for the user
    const canvases = await db.all(`SELECT id, name FROM canvases WHERE user_id = ?`, [userID]);

    // Format the canvases
    const formattedCanvases = canvases.map(canvas => ({
      id: canvas.id,
      name: canvas.name
    }));
    console.log(`Listed ${formattedCanvases.length} canvases for user ${userID}`);

    res.json({ success: true, canvases: formattedCanvases });
  } catch (error) {
    console.error("Error getting canvases:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// list users and their canvases
app.get("/api/list-users", async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  try {
    const db = await dbPromise;

    // Get all users
    const users = await db.all(`SELECT id FROM users`);

    // Get each user's canvases asynchronously
    const usersWithCanvases = await Promise.all(
      users.map(async (user) => {
        const canvases = await db.all(`SELECT * FROM canvases WHERE user_id = ?`, [user.id]);
        return {
          id: user.id,
          canvases: canvases.map((canvas) => ({
            id: canvas.id,
            name: canvas.name,
            timestamp: canvas.timestamp,
          })),
        };
      })
    );

    console.log(`Listed ${usersWithCanvases.length} users`);
    res.json({ success: true, users: usersWithCanvases });
  } catch (error) {
    console.error("Error listing users:", error);
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
    const user = await db.get(`SELECT id FROM users WHERE id = ?`, [userID]);
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    // Get the number of canvases for the user
    const canvasCount = await db.get(`SELECT COUNT(*) as count FROM canvases WHERE user_id = ?`, [userID]);
    // Generate the next canvas ID
    console.log(`found ${canvasCount.count} canvases for user ${userID}.`);
    const nextCanvasId = `${userID}-${canvasCount.count + 1}`;
    console.log(`Next canvas ID for user ${userID} is ${nextCanvasId}`);
    res.json({ success: true, nextCanvasId });
  } catch (error) {
    console.error("Error getting next canvas ID:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

  app.post("/api/save-canvas", async (req, res) => {
    const { userID, canvasID, canvasName, nodes, viewport, timestamp } = req.body;
    console.log("Attempting to save canvas:", req.body);
    if (!userID || !canvasID || !nodes || !viewport || !timestamp) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    try {
      const db = await dbPromise;
      const existingCanvas = await db.get("SELECT id FROM canvases WHERE id = ?", [canvasID]);
      
      if (existingCanvas) {
        console.log(`found canvas ${canvasID} for user ${userID}.`);
        await db.run(
          `UPDATE canvases SET name = ?, nodes = ?, viewport = ?, timestamp = ? WHERE id = ?`,
          [canvasName, JSON.stringify(nodes), JSON.stringify(viewport), timestamp, canvasID]
        );
        console.log(`updated canvas ${canvasID} for user ${userID} with name ${canvasName}.`);
      } else {
        // ✅ Insert new canvas
        await db.run(
          `INSERT INTO canvases (id, user_id, name, nodes, viewport, timestamp) VALUES (?, ?, ?, ?, ?, ?)`,
          [canvasID, userID, canvasName, JSON.stringify(nodes), JSON.stringify(viewport), timestamp]
        );
        console.log(`New canvas ${canvasID} created for user ${userID} with name ${canvasName}.`);
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
      const result = await db.run(`DELETE FROM canvases WHERE id = ? AND user_id = ?`, [canvasID, userID]);

      if (result.changes === 0) {
        return res.status(404).json({ error: "Canvas not found or does not belong to the user" });
      }
      console.log(`Canvas ${canvasID} deleted successfully for user ${userID}.`);
      res.json({ success: true, message: "Canvas deleted successfully" });
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
      const canvas = await db.get(`SELECT * FROM canvases WHERE id = ?`, [canvasID]);

      if (!canvas) {
        return res.status(404).json({ error: "Canvas not found" });
      }

      // ✅ Parse JSON fields
      canvas.nodes = JSON.parse(canvas.nodes);
      canvas.viewport = JSON.parse(canvas.viewport);

      console.log(`✅ Canvas ${canvasID} loaded with ${canvas.nodes.length} nodes.`);
      res.json({ success: true, canvas, timestamp: canvas.timestamp });
    } catch (error) {
      console.error("Error loading canvas:", error);
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
      const user = await db.get(`SELECT clippings FROM users WHERE id = ?`, [userID]);

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
      const user = await db.get(`SELECT clippings FROM users WHERE id = ?`, [userID]);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }
  
      let clippings = JSON.parse(user.clippings);
  
      // Prevent duplicate clippings
      if (!clippings.some((c) => JSON.stringify(c) === JSON.stringify(clipping))) {
        clippings.push(clipping);
      }
  
      // Update user's clippings
      await db.run(`UPDATE users SET clippings = ? WHERE id = ?`, [JSON.stringify(clippings), userID]);
  
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
  
    try {
      const db = await dbPromise;
  
      // Fetch current clippings
      const user = await db.get(`SELECT clippings FROM users WHERE id = ?`, [userID]);
      if (!user) {
        return res.status(404).json({ error: "User not found" });
      }
  
      let clippings = JSON.parse(user.clippings);
  
      // Remove clipping by filtering it out
      clippings = clippings.filter((c) => JSON.stringify(c) !== JSON.stringify(clipping));
  
      // Update user's clippings
      await db.run(`UPDATE users SET clippings = ? WHERE id = ?`, [JSON.stringify(clippings), userID]);
  
      res.json({ success: true, message: "Clipping removed!", clippings });
    } catch (error) {
      console.error("Error removing clipping:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });
  
  
  
  



app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});