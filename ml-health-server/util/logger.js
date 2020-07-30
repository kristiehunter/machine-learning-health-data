const winston = require('winston');
const { config } = require('winston');

const levels = {
    debug: 4,
    info: 3,
    error: 2,
    none: 1
};

const colours = {
    debug: "green",
    info: "cyan",
    error: "red",
    none: "black"
};

const createLogger = (file) => {
    const format = winston.format.printf((info) => {
        return `${info.timestamp} ${info.level}: [${info.label}] ${info.message}`;
    });
    const colourLevels = {
        colours,
        levels
    };
    const logger = winston.createLogger({
        level: 'info',
        format: winston.format.combine(
            winston.format.simple()
        ),
        transports: [
            new winston.transports.Console({
                format: winston.format.combine(
                    winston.format.colorize(),
                    winston.format.simple(),
                    winston.format.label({ label: file }),
                    winston.format.timestamp(),
                    winston.format.prettyPrint(),
                    format
                )
            })
        ],
        levels
    });
    
    winston.addColors(colours);

    return logger;
};

module.exports = {
    createLogger
};
