import logging
import json
# Define custom log levels
TRACE = 5
SUCCESS = 25

# Register them with logging
logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(SUCCESS, "SUCCESS")
class Logger(logging.Logger):
    TRACE = TRACE
    SUCCESS = SUCCESS
    def trace(self, message, *args, **kwargs):
        self.log(TRACE, message, *args, **kwargs)

    def success(self, message, *args, **kwargs):
        self.log(SUCCESS, message, *args, **kwargs)
    
    def json_dumper(self,records):
        json_string = json.dumps(records)
        return json_string

# Add methods to Logger class


logging.setLoggerClass(Logger)

# Setup basic logging
# logging.basicConfig(
#     level=TRACE,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# # Use it
# logger = logging.getLogger("carla-logger")

# logger.trace("Very detailed trace message")
# logger.debug("Debug message")
# logger.info("Info message")
# logger.success("Operation completed successfully!")
# logger.warning("Warning message")
# logger.business("Business event occurred")
# logger.error("Error message")

