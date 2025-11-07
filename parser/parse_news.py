import time, json, re, os, traceback, logging
from datetime import date, timedelta
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count
import requests

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_news_info(url):
    info = dict()
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/118.0.0.0 Safari/537.36"
        })
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Ошибка при запросе {url}: {e}")
        info['title'] = None
        info['subtitle'] = None
        info['author'] = None
        info['url'] = url
        info['text'] = None
        info['time'] = None
        info['date'] = None
        info['difficulty'] = None
        info['tags'] = None
        return info

    content = BeautifulSoup(resp.text, 'html.parser')
    
    # Заголовок
    title_tag = content.find("h1")
    info['title'] = title_tag.text.strip() if title_tag else None

    # Подзаголовок
    subtitle_tag = content.find_all("p", {'class': 'text-main-gray'})
    info['subtitle'] = subtitle_tag[0].text.strip() if len(subtitle_tag) == 2 else None

    # Автор
    author_tag = content.find("a", {'class': 'underline'})
    info['author'] = author_tag.text.strip() if author_tag else None

    text_blocks = content.find_all("div", {"class": "n1_material text-18"})
    paragraphs = []
    for block in text_blocks:
        p = block.find("p")
        if p:
            paragraphs.append(p.get_text())
    info['url'] = url
    info['text'] = "\n".join(paragraphs)

    meta_info = content.find("div", {"class": "flex flex-wrap lg:mb-10 gap-2 text-tags xl:pr-9"})
    spans = [tag.text for tag in meta_info.find_all("span", {"class": "group-hover:text-main transition-colors duration-75"})]
    info['time'] = spans[0]
    info['date'] = spans[1]
    info['difficulty'] = spans[2]
    tags = spans[3:]
    info['tags'] = tags if tags else None

    return info


def process_batch(urls):
    """Обработка пачки ссылок"""
    results = []
    total = len(urls)
    logger.info(f"Начата обработка батча из {total} ссылок")

    for i, url in enumerate(urls, start=1):
        try:
            info = get_news_info(url)
            results.append(info)
        except Exception as e:
            logger.warning(f"Ошибка при обработке {url}: {e}")

        if i % 100 == 0:  # прогресс каждые 100 ссылок
            logger.info(f"Обработано {i}/{total} ссылок в батче")

    logger.info(f"Завершена обработка батча ({total} ссылок)")
    return results


if __name__ == '__main__':
    df = pd.read_csv('news_links.csv')
    urls = df["url"].tolist()
    num_workers = min(8, cpu_count())
    chunk_size = len(urls) // num_workers

    with Pool(processes=num_workers) as pool:
        batches = [urls[i:i+chunk_size] for i in range(0, len(urls), chunk_size)]
        all_results = pool.map(process_batch, batches)

    flat_results = [item for sublist in all_results for item in sublist]

    out_df = pd.DataFrame(flat_results)
    out_df.to_csv("news_data_extra.csv", index=False, encoding="utf-8")
    logger.info(f"Сохранено {len(flat_results)} записей в news_data.csv")
