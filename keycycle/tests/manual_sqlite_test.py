import os
import sys
import time
import threading

# Add the directory containing the 'keycycle' package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from keycycle.usage.db_logic import UsageDatabase
from keycycle.usage.usage_logger import AsyncUsageLogger

def test_sqlite_concurrent():
    db_file = "test_usage_concurrent.db"
    db_path = os.path.abspath(os.path.join(current_dir, db_file))
    db_url = f"sqlite:///{db_path}"
    
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print(f"Testing Concurrent SQLite with URL: {db_url}")
    
    try:
        # Initialize DB
        db = UsageDatabase(db_url=db_url)
        
        # Initialize Logger
        logger = AsyncUsageLogger(db)
        
        provider = "openai"
        api_key = "sk-1234567890abcdef"
        
        def logger_task():
            for i in range(10):
                logger.log(provider, f"model-{i}", api_key, 100 + i)
                time.sleep(0.1)

        def reader_task():
            for i in range(5):
                history = db.load_history(provider, api_key, seconds_lookback=60)
                print(f"Reader saw {len(history)} records")
                time.sleep(0.2)

        t1 = threading.Thread(target=logger_task)
        t2 = threading.Thread(target=reader_task)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        logger.stop()
        
        final_history = db.load_history(provider, api_key, seconds_lookback=60)
        print(f"Final record count: {len(final_history)}")
        
        if len(final_history) == 10:
            print("SUCCESS: Concurrent SQLite logging works.")
        else:
            print(f"FAILURE: Expected 10 records, got {len(final_history)}")

    except Exception as e:
        print(f"FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sqlite_concurrent()