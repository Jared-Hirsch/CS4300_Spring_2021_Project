import os
from app import app, socketio

if __name__ == "__main__":
  print("Flask app running at http://0.0.0.0:5000")

  app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

  socketio.run(app, host="0.0.0.0", port=5000)
