const path = require("path");
const webpack = require("webpack");

module.exports = {
  entry: "./src/index.js", // Entry point for your app
  output: {
    path: path.resolve(__dirname, "./static/frontend"), // Output directory
    filename: "[name].js", // Output file name
  },
  module: {
    rules: [
      {
        test: /\.js$/, // Process JS files
        exclude: /node_modules/,
        use: {
          loader: "babel-loader", // Use Babel loader for transpiling
        },
      },
      {
        test: /\.scss$/, // Process SCSS files
        use: [
          "style-loader", // Inject CSS into the DOM
          "css-loader", // Turn CSS into JavaScript
          "sass-loader", // Compile SCSS to CSS
        ],
      },
    ],
  },
  optimization: {
    minimize: true, // Minimize the output
  },
  plugins: [
    new webpack.DefinePlugin({
      "process.env": {
        // This has effect on the react lib size
        // NODE_ENV: JSON.stringify("production"),
        NODE_ENV: JSON.stringify("development"), // Set the environment to development
      },
    }),
  ],
};
