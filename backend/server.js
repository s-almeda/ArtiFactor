import express from "express";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";
import dbPromise from "./database.js"; // Import the database module
import fetch from 'node-fetch'; // Ensure you have node-fetch installed

import Replicate from "replicate";
const replicate = new Replicate();


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
    console.log("Sending image to data.SnailBunny...");
    const response = await axios.post(
      "https://data.snailbunny.site/image",
      {
        image: imageData,
      }
    );
    console.log("Got response from data.SnailBunny:", response.data);
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
    //todo, let the user choose how truncated the output should be? 
    const truncatedOutput = output.split(',').slice(0, 5).join(', ');
    res.json({ text: truncatedOutput });
  } catch (error) {
    console.error("Error in generate-text:", error);
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
    res.json({ imageUrl: redirectUrl });



  } catch (error) {
    console.error("Error in generate-image:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
