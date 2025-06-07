from concurrent.futures import ThreadPoolExecutor
import random
import time

import concurrent

def process_task(task_id):
    print(f"Statr task {task_id}")
    time.sleep(random.uniform(0.1,0.5))

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_task,i) for i in range(5)]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
        