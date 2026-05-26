from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

load_dotenv()


def create_app(load_glove=True):
    app = Flask(__name__)

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    import os
    env = os.environ.get("FLASK_ENV", "development")
    if env == "production":
        from app.config import ProductionConfig
        app.config.from_object(ProductionConfig)
    else:
        from app.config import DevelopmentConfig
        app.config.from_object(DevelopmentConfig)

    if load_glove:
        from app.glove import load_glove_vectors
        path = app.config["GLOVE_PATH"]
        try:
            load_glove_vectors(path)
        except FileNotFoundError as e:
            raise RuntimeError(f"GloVe data file not found: {path}") from e
        app.logger.info(f"Loaded GloVe vectors from {path}")

    from app.routes.pages import pages_bp
    from app.routes.api import api_bp
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    return app
