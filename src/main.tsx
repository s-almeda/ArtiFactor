import React from 'react';
import ReactDOM from 'react-dom/client';
import { Analytics } from "@vercel/analytics/react"

// import dotenv from 'dotenv'; //import our .env file, which includes our Stable Diffusion API key
// dotenv.config();

import App from './App';

import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <>
  <React.StrictMode> {/* NOTE-- disable this to disable the double reload at start */}
    <Analytics />
    <App />
  </React.StrictMode>
  </>
);
