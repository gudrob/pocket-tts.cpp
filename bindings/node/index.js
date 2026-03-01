"use strict";

const fs = require("fs");
const path = require("path");
const load = require("node-gyp-build");

if (process.platform === "win32") {
  const prebuildDir = path.join(__dirname, "prebuilds", `${process.platform}-${process.arch}`);
  if (fs.existsSync(prebuildDir)) {
    process.env.PATH = `${prebuildDir};${process.env.PATH || ""}`;
  }
}

module.exports = load(path.join(__dirname));
