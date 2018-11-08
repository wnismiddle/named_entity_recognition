from flask import Flask
import logging
import os
import configparser


import warnings
warnings.filterwarnings('ignore')

def create_app():
    app = Flask(__name__)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    app.debug = True
    return app
