import redis
import json
import subprocess
import socket
import time
import logging
from log_file import setup_logging
logger = setup_logging("redis_loader")



class RedisLoader:
    def __init__(self, logger: logging.Logger):
        self.logger = logger.getChild("redis")
        self.redis_path = r"C:\Users\Suresh Goud\Downloads\Redis-x64-5.0.14.1\redis-server.exe"
        self.port = 6379
        self.client = None

    def is_redis_running(self, host="127.0.0.1"):
        try:
            with socket.create_connection((host, self.port), timeout=1):
                return True
        except OSError:
            return False

    def start_redis_if_needed(self):
        if not self.is_redis_running():
            self.logger.info("Redis not running. Starting Redis...")
            subprocess.Popen(
                [self.redis_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)

            if self.is_redis_running():
                self.logger.info("Redis started successfully")
            else:
                self.logger.error("Failed to start Redis")
                raise RuntimeError("Redis startup failed")
        else:
            self.logger.info("Redis already running")

    def connect(self):
        self.client = redis.Redis(host="localhost", port=self.port, db=0)
        self.client.ping()
        self.logger.info("Connected to Redis")

    def store_food_items(self, food_data: dict):
        for food_name, details in food_data.items():
            key = f"food:{food_name.replace(' ', '_')}"
            self.client.set(key, json.dumps(details))
            self.logger.info(f"Stored '{food_name}' under key '{key}'")

        self.logger.info("All food items stored in Redis")

    def store_model_metrics_from_json(self, model_name: str, json_path: str):
        try:
            with open(json_path, "r") as f:
                metrics = json.load(f)

            redis_key = f"model:{model_name.lower()}:metrics"
            self.client.set(redis_key, json.dumps(metrics))

            self.logger.info(
                f"Model metrics stored successfully | model={model_name} | key={redis_key}"
            )

        except Exception as e:
            self.logger.error(f"Failed to store metrics for {model_name}: {e}")
            raise

if __name__ == "__main__":
    loader = RedisLoader(logger)

    loader.start_redis_if_needed()
    loader.connect()

    loader.store_model_metrics_from_json("cnn", "cnn_model_metrics.json")
    loader.store_model_metrics_from_json("vgg16", "vgg16_metrics .json")
    loader.store_model_metrics_from_json("resnet50", "resnet50_metrics.json")
