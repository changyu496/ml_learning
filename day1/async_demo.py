import asyncio
import aiohttp

async def fetch_page(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content =  await response.text()        
            return content

async def main():
    urls = [
        "http://www.baidu.com",
        "http://www.google.com"
    ]  
    tasks = [fetch_page(url) for url in urls]
    pages = await asyncio.gather(*tasks)
    print(f"Fetched {len(pages)} pages")

asyncio.run(main())    