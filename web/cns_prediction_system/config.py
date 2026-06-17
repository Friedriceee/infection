import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "database.db")
SECRET_KEY = os.environ.get("CNS_SECRET_KEY", "cns-dev-secret-key-change-in-production")
DEBUG = False

