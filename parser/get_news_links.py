import time, json, re, os, traceback
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

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
prefixs = ['https://nplus1.ru/news/']
def accept(link: str):
  if link.startswith('https://nplus1.ru/material/difficulty/'):
    return False
  for prefix in prefixs:
    if link.startswith(prefix) and len(link) > len(prefix) + 11:
      return True
  return False

def search():
  result = set()
  step = 0
  start_date = date(2016, 1, 1)
  end_date = date.today()
  for prefix in prefixs:
    cur_date = start_date
    while cur_date <= end_date:
      url = f"{prefix}{cur_date.year}/{cur_date.month:02d}/{cur_date.day:02d}/"
      print(f"date: {cur_date}; len: {len(result)}; url: {url}")
      driver.get(url)
      while True:
        refs = driver.find_elements(By.TAG_NAME, "a")
        links = [ref.get_attribute('href') for ref in refs]
        links = [link for link in links if accept(link)]
        if not links:
          break
        for link in links:
          result.add(link)
        try:
          btn = driver.find_element(By.XPATH, '//button[span[text()="Ещё"]]')
          driver.execute_script("arguments[0].scrollIntoView(true);", btn)  # скроллим к кнопке
          time.sleep(0.3)
          driver.execute_script("arguments[0].click();", btn)  # клик через JS
          time.sleep(0.8)
        except Exception as e:
          print("Ошибка при попытке нажать кнопку 'Ещё':")
          traceback.print_exc()
          break
      cur_date += timedelta(days=1)
  return result

if __name__ == '__main__':
  news_urls = search()
  df = pd.DataFrame(news_urls, columns=["url"])
  df.to_csv("news_links.csv", index=False, encoding="utf-8")
  print(f"Сохранено {len(news_urls)} ссылок в файл news_links.csv")