import concurrent.futures
import threading
from urllib.parse import urlparse
import concurrent
import requests
import time
import asyncio
import aiohttp

# 1 同步版本
def sync_fetch(url):
    """同步方式获取网页内容"""
    response = requests.get(url)
    return response.text[:100]

def run_sync(urls):
    """顺序执行"""
    results = []
    start_time = time.monotonic()

    for url in urls:
        try:
            result = sync_fetch(url)
            results.append(result)
            print(f"Sync: Fetched {urlparse(url).netloc} in thread {threading.current_thread().name}")
        except Exception as e:
            print(f"Error fecting {url}:{e}")
    duration = time.monotonic() - start_time
    return len(results),duration

# 2 线程池版本
def thread_fetch(url):
    return sync_fetch(url)

def run_threaded(urls,max_workers=5):
    """使用线程池并发请求"""
    results = []
    start_time = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(thread_fetch,url): url for url in urls}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Threaded: Fetched {urlparse(url).netloc} in thread {threading.current_thread().name}")
            except Exception as e:
                print(f"Error fetching {url}: {e}")
    duration = time.monotonic() - start_time
    return len(results),duration

# 3 协程版本
async def async_fetch(session,url):
    """协程方式获取网页内容"""
    async with session.get(url) as response:
        content = await response.text()
        return content[:100] #返回前100个字符
    
async def run_async(urls):
    """使用AsyncIO并发请求"""
    results = []
    start_time = time.monotonic()

    async with aiohttp.ClientSession() as session:
        tasks = [async_fetch(session,url) for url in urls]
        results = await asyncio.gather(*tasks,return_exceptions=True)

        # 处理结果
        valid_results = 0
        for i,result in enumerate(results):
            url = urls[i]
            if isinstance(result,Exception):
                print(f"Error fetching:{url}:{str(result)}")
            else:
                print(f"Async:Fetched {urlparse(url).netloc} in thread {threading.current_thread().name}")
                valid_results+1
        duration = time.monotonic() - start_time
        return valid_results,duration

# 性能对比函数
def benchmark(urls, runs=3):
    """运行多次测试取平均性能"""
    print(f"=== Starting benchmark with {len(urls)} URLs, {runs} runs ===")
    
    # 结果存储
    sync_times = []
    thread_times = []
    async_times = []
    
    for run in range(runs):
        print(f"\n--- Run {run+1}/{runs} ---")
        
        # 同步测试
        sync_count, sync_dur = run_sync(urls)
        sync_times.append(sync_dur)
        print(f"Sync completed: {sync_count}/{len(urls)} in {sync_dur:.2f}s")
        
        # 线程测试
        thread_count, thread_dur = run_threaded(urls)
        thread_times.append(thread_dur)
        print(f"Threaded completed: {thread_count}/{len(urls)} in {thread_dur:.2f}s")
        
        # 异步测试
        # 注意：协程需要在事件循环中运行
        loop = asyncio.get_event_loop()
        async_count, async_dur = loop.run_until_complete(run_async(urls))
        async_times.append(async_dur)
        print(f"Async completed: {async_count}/{len(urls)} in {async_dur:.2f}s")
    
    # 计算平均值
    avg_sync = sum(sync_times) / len(sync_times)
    avg_thread = sum(thread_times) / len(thread_times)
    avg_async = sum(async_times) / len(async_times)
    
    # 生成性能报告
    report = f"""
=== Performance Comparison Report ===
Test Parameters:
- URLs: {len(urls)}
- Runs: {runs}
- Average Latency: {simulated_latency:.2f}s per request

Results (average time):
1. Synchronous: {avg_sync:.4f} seconds
   • Slower by {avg_sync/avg_async:.1f}x than Async
   
2. ThreadPool ({threading.active_count()} max workers): {avg_thread:.4f} seconds
   • {avg_sync/avg_thread:.1f}x faster than Sync
   
3. AsyncIO: {avg_async:.4f} seconds
   • {avg_sync/avg_async:.1f}x faster than Sync
   • {avg_thread/avg_async:.1f}x faster than Threaded

Conclusion:
For I/O-bound tasks like HTTP requests:
- AsyncIO provides the best performance
- ThreadPool is simpler but less efficient
- Synchronous should only be used for few/simple tasks
"""
    print(report)
    return report

if __name__ == "__main__":
    # 使用安全、快速的测试URL（避免触发反爬虫机制）
    test_urls = [
        "https://httpbin.org/delay/0.1",  # 模拟0.1秒延迟
        "https://httpbin.org/delay/0.15",
        "https://httpbin.org/delay/0.2",
        "https://httpbin.org/delay/0.1",
        "https://httpbin.org/delay/0.25",
        "https://httpbin.org/delay/0.2",
        "https://httpbin.org/delay/0.15",
        "https://httpbin.org/delay/0.3",
        "https://httpbin.org/delay/0.1",
        "https://httpbin.org/delay/0.2",
    ]
    
    # 计算平均延迟用于报告
    simulated_latency = sum(float(url.split("/")[-1]) for url in test_urls) / len(test_urls)
    
    # 执行性能测试 (3次取平均)
    performance_report = benchmark(test_urls, runs=3)
    
    # 保存结果到文件
    with open("day1/concurrency_benchmark.txt", "w") as f:
        f.write(performance_report)
    
    print("Benchmark completed. Results saved to concurrency_benchmark.txt")