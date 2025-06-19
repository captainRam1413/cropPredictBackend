from django.apps import AppConfig
import threading
import time
import requests


def self_ping():
    while True:
        try:
            # Use your actual domain in production
            requests.get("http://localhost:8000/ping/")
        except Exception as e:
            print("Self-ping failed:", e)
        time.sleep(300)  # 5 minutes


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        if not hasattr(self, 'self_ping_started'):
            threading.Thread(target=self_ping, daemon=True).start()
            self.self_ping_started = True
