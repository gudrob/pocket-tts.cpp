"use strict";

const fs = require("fs");
const path = require("path");

function uniq(values) {
  return [...new Set(values.filter(Boolean))];
}

function toPosix(value) {
  return value.replace(/\\/g, "/");
}

function splitEnvPath(name) {
  const value = process.env[name];
  if (!value) {
    return [];
  }
  return value.split(path.delimiter).filter(Boolean);
}

function existingDirs(values) {
  return values.filter((candidate) => {
    try {
      return fs.statSync(candidate).isDirectory();
    } catch {
      return false;
    }
  });
}

const platform = process.platform;
const defaultPrefixes =
  platform === "darwin"
    ? ["/opt/homebrew", "/usr/local", "/usr"]
    : platform === "linux"
      ? ["/usr/local", "/usr"]
      : ["C:/vcpkg/installed/x64-windows-static", "C:/vcpkg/installed/x64-windows"];

const prefixes = uniq([
  ...splitEnvPath("POCKET_TTS_PREFIX"),
  process.env.ONNXRUNTIME_ROOT,
  process.env.SENTENCEPIECE_ROOT,
  process.env.SNDFILE_ROOT,
  process.env.SAMPLERATE_ROOT,
  ...defaultPrefixes
]);

const includeCandidates = uniq([
  path.resolve(__dirname, "../../../include"),
  ...prefixes.flatMap((prefix) => [
    path.join(prefix, "include"),
    path.join(prefix, "include", "onnxruntime"),
    path.join(prefix, "include", "onnxruntime", "core", "session")
  ])
]);

const includeDirs = (platform === "win32" ? includeCandidates : existingDirs(includeCandidates)).map(
  toPosix
);

function findLibFile(fileName) {
  for (const prefix of prefixes) {
    const candidate = path.join(prefix, "lib", fileName);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

let libraries;
if (platform === "win32") {
  const windowsLibs = [
    "onnxruntime.lib",
    "sentencepiece.lib",
    "sndfile.lib",
    "samplerate.lib"
  ];

  libraries = windowsLibs.map((libName) => {
    const discovered = findLibFile(libName);
    if (discovered) {
      return toPosix(discovered);
    }
    return toPosix(path.join(prefixes[0], "lib", libName));
  });
} else {
  const libDirs = existingDirs(uniq(prefixes.map((prefix) => path.join(prefix, "lib"))));

  libraries = uniq([
    ...libDirs.map((dir) => `-L${toPosix(dir)}`),
    "-lonnxruntime",
    "-lsentencepiece",
    "-lsndfile",
    "-lsamplerate"
  ]);
}

module.exports = {
  includeDirs,
  libraries
};
