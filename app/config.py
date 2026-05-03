import os


class Config:
    GLOVE_PATH = os.environ.get("GLOVE_PATH", "/app/data/glove.6B.50d.txt")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    IS_PRODUCTION = True
