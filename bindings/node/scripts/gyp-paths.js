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

function listFiles(dir, pattern) {
  try {
    return fs
      .readdirSync(dir, { withFileTypes: true })
      .filter((entry) => entry.isFile() && pattern.test(entry.name))
      .map((entry) => path.join(dir, entry.name));
  } catch {
    return [];
  }
}

const platform = process.platform;
const defaultPrefixes =
  platform === "darwin"
    ? ["/opt/homebrew", "/usr/local", "/usr"]
    : platform === "linux"
      ? ["/usr/local", "/usr"]
      : ["C:/vcpkg/installed/x64-windows-static", "C:/vcpkg/installed/x64-windows"];

const explicitPrefixes = uniq([
  ...splitEnvPath("POCKET_TTS_PREFIX"),
  process.env.ONNXRUNTIME_ROOT,
  process.env.SENTENCEPIECE_ROOT
]);

const prefixes = explicitPrefixes.length > 0 ? explicitPrefixes : defaultPrefixes;

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
  const libDirs = existingDirs(uniq(prefixes.map((prefix) => path.join(prefix, "lib"))));
  const primaryLibDir = libDirs[0];

  if (primaryLibDir) {
    const libsFromPrimaryDir = listFiles(primaryLibDir, /\.lib$/i)
      .filter((libPath) => !/sentencepiece_train\.lib$/i.test(libPath))
      .map(toPosix);

    libraries = uniq([
      ...libsFromPrimaryDir,
      "ws2_32.lib",
      "advapi32.lib",
      "bcrypt.lib",
      "userenv.lib",
      "shlwapi.lib",
      "ole32.lib"
    ]);
  } else {
    const windowsLibs = [
      "onnxruntime.lib",
      "sentencepiece.lib",
      "libprotobuf.lib",
      "libprotobuf-lite.lib",
      "re2.lib",
      "utf8_range.lib",
      "utf8_validity.lib",
      "abseil_dll.lib"
    ];

    const resolvedCore = windowsLibs
      .map((libName) => findLibFile(libName))
      .filter(Boolean)
      .map(toPosix);

    const abslLibs = libDirs
      .flatMap((dir) => listFiles(dir, /^absl.*\.lib$/i))
      .map(toPosix);

    libraries = uniq([...resolvedCore, ...abslLibs]);
  }
} else {
  const libDirs = existingDirs(uniq(prefixes.map((prefix) => path.join(prefix, "lib"))));

  libraries = uniq([
    ...libDirs.map((dir) => `-L${toPosix(dir)}`),
    "-lonnxruntime",
    "-lsentencepiece"
  ]);
}

module.exports = {
  includeDirs,
  libraries
};
