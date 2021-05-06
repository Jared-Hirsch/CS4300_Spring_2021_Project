import os
from dotenv import load_dotenv

load_dotenv()
basedir = os.path.abspath(os.path.dirname(__file__))
# Different environments for the app to run in

class Config(object):
  DEBUG = False
  CSRF_ENABLED = True
  CSRF_SESSION_KEY = "secret"
  SECRET_KEY = "not_this"
  SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
  SP_USERNAME = os.environ.get('SP_USERNAME')
  SP_CLIENT_ID = os.environ.get('SP_CLIENT_ID')
  SP_CLIENT_SECRET = os.environ.get('SP_CLIENT_SECRET')
  GENUIS_TOKEN = os.environ.get('GENUIS_TOKEN')
  REDIRECT_URI = os.environ.get('REDIRECT_URI')

class ProductionConfig(Config):
  DEBUG = False

class StagingConfig(Config):
  DEVELOPMENT = True
  DEBUG = True

class DevelopmentConfig(Config):
  DEVELOPMENT = True
  DEBUG = True

class TestingConfig(Config):
  TESTING = True
