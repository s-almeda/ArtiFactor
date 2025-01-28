const { join } = require('path');

module.exports = {
  important: true,
  content: [
    join(__dirname, 'src/**/*.{js,ts,jsx,tsx}'), //point tailwind to all our files

  ],
  theme: {
    extend: {},
  },
  plugins: [],
};