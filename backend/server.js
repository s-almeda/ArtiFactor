import express from "express";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";
import dbPromise from "./database.js"; // Import the database module

dotenv.config();

const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());

// New /overview route
app.get("/", (req, res) => {
  res.end('Hello World! if you can see this, that means the backend server is working!');
});
// New /overview route
app.get("/overview", (req, res) => {
  res.end('if you can see this, that means the backend server is working!');
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
    res.json({ imageUrl: redirectUrl });



  } catch (error) {
    console.error("Error in generate-image:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
