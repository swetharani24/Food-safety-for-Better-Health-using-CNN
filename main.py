from log_file import setup_logging
from food_items import FoodNutritionData
from food_items import JSONWriter
from redis_loader import RedisLoader
from dataset_split import DatasetSplitter
import json
from log_file import setup_logging
logger = setup_logging("main")


def main():
    # =========================
    # MAIN LOGGER
    # =========================
    logger = setup_logging("main")
    logger.info("Main program started")

    # =========================
    # 1. LOAD FOOD DATA
    # =========================
    food_logger = setup_logging("food_items")
    food_data = FoodNutritionData(food_logger)
    food_data.load_data()
    logger.info(f"Food items loaded: {len(food_data.food_items)}")

    writer = JSONWriter(logger)
    writer.save(food_data.food_items, "food_nutrition_data.json")

    # =========================
    # 2. LOAD DATA INTO REDIS
    # =========================
    redis_logger = setup_logging("redis_loader")
    redis_loader = RedisLoader(redis_logger)

    redis_loader.start_redis_if_needed()
    redis_loader.connect()
    redis_loader.store_food_items(food_data.food_items)

    logger.info("Food data stored in Redis")
    loader = RedisLoader(logger)

    loader.start_redis_if_needed()
    loader.connect()

    loader.store_model_metrics_from_json("cnn", "food_model_metrics.json")
    loader.store_model_metrics_from_json("vgg16", "vgg16_metrics.json")
    loader.store_model_metrics_from_json("resnet50", "resnet50_metrics.json")
    # =========================
    # 3. SPLIT IMAGE DATASET
    # =========================
    split_logger = setup_logging("dataset_split")
    splitter = DatasetSplitter(
        source_dir="Food Classification dataset",
        dest_dir="data_split",
        train_count=200,
        val_count=50,
        test_count=10,
        logger=split_logger
    )
    splitter.run()

    logger.info("Main program finished successfully")


if __name__ == "__main__":
    main()
    logger.info("Program started")

    food_data = FoodNutritionData(logger)
    food_data.load_data()

    writer = JSONWriter(logger)
    writer.save(food_data.food_items, "food_nutrition_data.json")

    logger.info("Program finished")