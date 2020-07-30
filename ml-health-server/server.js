// Requirements
const express = require("express");
const logCreator = require("./util/logger");

// Server
const port = process.env.PORT || 3000;
const logger = logCreator.createLogger("server.js");

const app = express();
app.listen(port, () => {
    logger.info("Server is running.");
});