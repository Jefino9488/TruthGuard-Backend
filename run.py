from app import create_app
from config import get_config
from apscheduler.schedulers.background import BackgroundScheduler
from app.tasks.scraper import NewsAPIFetcherTask
from app.tasks.analyzer import GeminiAnalyzerTask
from app import db
import atexit
import os

config = get_config()

app = create_app(config)

# Scheduler setup
scheduler = BackgroundScheduler()
news_api_key = os.environ.get('NEWS_API_KEY_SCRAPER')
google_api_key = os.environ.get('GOOGLE_API_KEY')

def scheduled_scrape_and_analyze():
    scraper = NewsAPIFetcherTask(db, news_api_key)
    scraper.run_scraper()
    analyzer = GeminiAnalyzerTask(db, google_api_key)
    analyzer.run_analyzer()

# Schedule every 6 hours
scheduler.add_job(scheduled_scrape_and_analyze, 'interval', hours=6, id='scrape_and_analyze')
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)