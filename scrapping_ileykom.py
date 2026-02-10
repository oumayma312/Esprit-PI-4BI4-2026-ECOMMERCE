import time
import random
import logging
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# ==================== CONFIGURATION ====================
class ScraperConfig:
    """Centralized configuration - Clean & Simple"""
    CSV_FILENAME = "ileycom_products_clean.csv"
    JSON_FILENAME = "ileycom_products.json"
    LOG_FILENAME = f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Scraping behavior
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    SCROLL_ITERATIONS = 3
    RATE_LIMIT_MIN = 2.0
    RATE_LIMIT_MAX = 5.0
    PAGE_LOAD_TIMEOUT = 20
    CAPTCHA_SOLVE_TIMEOUT = 180
    
    # Multi-URL features
    BETWEEN_CATEGORY_DELAY_MIN = 3.0
    BETWEEN_CATEGORY_DELAY_MAX = 8.0
    
    # Features
    SAVE_JSON = True
    VERBOSE_LOGGING = True

# ==================== LOGGING SETUP ====================
def setup_logging():
    """Configure logging to both file and console"""
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    Path("logs").mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(f"logs/{ScraperConfig.LOG_FILENAME}", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if ScraperConfig.VERBOSE_LOGGING else logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if ScraperConfig.VERBOSE_LOGGING else logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ==================== UTILITIES ====================
class ScraperStats:
    """Track scraping statistics"""
    def __init__(self):
        self.start_time = time.time()
        self.total_products = 0
        self.total_pages = 0
        self.failed_products = 0
        self.retries_used = 0
        self.categories_scraped = 0
        self.categories_failed = 0
        self.category_breakdown = {}  # Track products per category
        
    def add_category_result(self, url: str, products_count: int, success: bool):
        """Track results for each category"""
        self.category_breakdown[url] = {
            'products': products_count,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        if success:
            self.categories_scraped += 1
        else:
            self.categories_failed += 1
    
    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        return f"""
╔════════════════════════════════════════╗
║        SCRAPING SUMMARY                ║
╠════════════════════════════════════════╣
║ Total Products: {self.total_products:>19} ║
║ Categories Success: {self.categories_scraped:>16} ║
║ Categories Failed: {self.categories_failed:>17} ║
║ Pages Scraped: {self.total_pages:>20} ║
║ Failed Items: {self.failed_products:>21} ║
║ Retries Used: {self.retries_used:>21} ║
║ Time Elapsed: {elapsed/60:>19.1f} min ║
╚════════════════════════════════════════╝
"""
    
    def category_summary(self) -> str:
        """Detailed breakdown per category"""
        lines = ["\n" + "="*60, "CATEGORY BREAKDOWN:", "="*60]
        for idx, (url, data) in enumerate(self.category_breakdown.items(), 1):
            status = "✓" if data['success'] else "✗"
            short_url = url.split('product-category/')[-1][:40] if 'product-category/' in url else url[-40:]
            lines.append(f"{status} [{idx}] {short_url}: {data['products']} products")
        lines.append("="*60)
        return "\n".join(lines)

def clean_price(price_text: str) -> tuple:
    """
    Parse price text and extract sale_price and regular_price.
    Price format: "د.ت60.00 د.ت80.00" where first is SALE, second is REGULAR
    Returns: (sale_price, regular_price)
    """
    if not price_text or price_text == "N/A":
        return ("", "")
    
    # Extract all prices using regex
    prices = re.findall(r'د\.ت\s*[\d,]+\.?\d*', price_text)
    
    if len(prices) == 0:
        return ("", "")
    elif len(prices) == 1:
        # Only one price = no sale, this is regular price
        return ("", prices[0].strip())
    else:
        # Two prices: first is SALE (lower), second is REGULAR (higher)
        return (prices[0].strip(), prices[1].strip())

def clean_category(category_text: str) -> str:
    """
    Clean category text by removing newlines and formatting properly.
    Input: "/\nBeauty Products\n/\nHair care\n/\nHair care accessories"
    Output: "Beauty Products > Hair care > Hair care accessories"
    """
    if not category_text:
        return ""
    
    # Remove leading/trailing slashes and whitespace
    cleaned = category_text.strip().strip('/')
    
    # Split by newlines and slashes, filter empty strings
    parts = [p.strip() for p in re.split(r'[\n/]+', cleaned) if p.strip()]
    
    # Join with " > " separator
    return " > ".join(parts)

def clean_rating(rating_text: str) -> str:
    """
    Extract numeric rating from text.
    Input: "Rated 4.5 out of 5"
    Output: "4.5"
    """
    if not rating_text:
        return ""
    
    match = re.search(r'(\d+\.?\d*)\s*out of', rating_text)
    if match:
        return match.group(1)
    return ""

# ==================== MAIN SCRAPER CLASS ====================
class IleycomScraperClean:
    def __init__(self, headless=False):
        self.base_url = "https://ileycom.com"
        self.headless = headless
        self.driver = self._create_driver()
        self.stats = ScraperStats()
        self.all_products = []
        self._prepare_csv()

    def _create_driver(self):
        """Initialize Chrome driver with anti-detection measures"""
        options = Options()
        
        if self.headless:
            options.add_argument("--headless=new")
            logger.info("Running in headless mode")
        
        # Anti-bot detection
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Performance & compatibility
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-gpu")
        
        # Realistic user agent
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/132.0.0.0 Safari/537.36"
        )
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), 
            options=options
        )
        
        # Remove webdriver flag
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        logger.info("✓ Chrome driver initialized")
        return driver

    def _prepare_csv(self):
        """Create CSV with clean headers for analysis"""
        if not os.path.exists(ScraperConfig.CSV_FILENAME):
            with open(ScraperConfig.CSV_FILENAME, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Product Name",
                    "Regular Price",
                    "Sale Price",
                    "Vendor",
                    "Category",
                    "Rating",
                    "Review Count",
                    "Availability",
                    "Product URL",
                    "Scraped At"
                ])
            logger.info(f"✓ Created clean CSV: {ScraperConfig.CSV_FILENAME}")

    def detect_captcha(self):
        """Enhanced CAPTCHA detection"""
        page_source = self.driver.page_source.lower()
        
        if "cf-challenge" in page_source and "breadcrumb-container" not in page_source:
            logger.warning("🤖 Cloudflare detected! Solve the challenge in the browser...")
            logger.warning("⏳ Waiting up to 3 minutes...")
            
            timeout = ScraperConfig.CAPTCHA_SOLVE_TIMEOUT
            start = time.time()
            
            while time.time() - start < timeout:
                if "breadcrumb-container" in self.driver.page_source.lower():
                    logger.info("✓ Challenge cleared!")
                    return True
                time.sleep(2)
            
            raise TimeoutError("❌ CAPTCHA timeout")
        
        return True

    def rate_limit(self):
        """Polite delay between requests"""
        delay = random.uniform(ScraperConfig.RATE_LIMIT_MIN, ScraperConfig.RATE_LIMIT_MAX)
        time.sleep(delay)

    def scroll_to_load(self):
        """Progressive scrolling for lazy-loaded content"""
        for i in range(ScraperConfig.SCROLL_ITERATIONS):
            scroll_position = (i + 1) * (100 / ScraperConfig.SCROLL_ITERATIONS)
            self.driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {scroll_position/100});")
            time.sleep(1.5)
        
        self.driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

    def extract_product_data(self, item) -> Optional[Dict]:
        """Extract and clean product data"""
        try:
            # Product name
            name = item.find_element(By.CSS_SELECTOR, ".woocommerce-loop-product__title").text.strip()
            
            # Price extraction (RAW TEXT FIRST)
            try:
                price_elem = item.find_element(By.CSS_SELECTOR, "span.price")
                raw_price_text = price_elem.text.strip()
                sale_price, regular_price = clean_price(raw_price_text)
            except:
                sale_price, regular_price = "", ""
            
            # Vendor
            try:
                vendor = item.find_element(By.CSS_SELECTOR, ".wolmart-sold-by-container").text.replace("Vendu par", "").strip()
            except:
                vendor = "ILEYCOM"
            
            # Product URL
            try:
                product_url = item.find_element(By.CSS_SELECTOR, "a.woocommerce-LoopProduct-link").get_attribute("href")
            except:
                product_url = ""
            
            # Rating
            try:
                rating_elem = item.find_element(By.CSS_SELECTOR, ".star-rating")
                raw_rating = rating_elem.get_attribute("aria-label")
                rating = clean_rating(raw_rating)
            except:
                rating = ""
            
            # Review count
            review_count = ""  # Usually requires visiting product page
            
            # Availability
            try:
                stock_elem = item.find_element(By.CSS_SELECTOR, ".stock")
                availability = stock_elem.text.strip()
            except:
                availability = "In Stock"
            
            return {
                "name": name,
                "regular_price": regular_price,
                "sale_price": sale_price,
                "vendor": vendor,
                "category": "",  # Filled from breadcrumb
                "rating": rating,
                "review_count": review_count,
                "availability": availability,
                "url": product_url,
                "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract product: {e}")
            self.stats.failed_products += 1
            return None

    def scrape_page_with_retry(self, url: str) -> List[Dict]:
        """Scrape a single page with retry logic"""
        for attempt in range(ScraperConfig.MAX_RETRIES):
            try:
                logger.info(f"📄 Scraping: {url} (Attempt {attempt + 1}/{ScraperConfig.MAX_RETRIES})")
                
                self.driver.get(url)
                self.detect_captcha()
                
                # Wait for products
                WebDriverWait(self.driver, ScraperConfig.PAGE_LOAD_TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "li.product-wrap"))
                )
                
                self.scroll_to_load()
                self.rate_limit()
                
                # Extract CLEAN category from breadcrumbs
                category = ""
                try:
                    breadcrumb = self.driver.find_element(By.CSS_SELECTOR, ".breadcrumb-container")
                    raw_category = breadcrumb.text.strip()
                    category = clean_category(raw_category)
                except:
                    pass
                
                # Get all products
                product_elements = self.driver.find_elements(By.CSS_SELECTOR, "li.product-wrap")
                page_results = []
                
                logger.info(f"Found {len(product_elements)} products on page")
                
                for idx, item in enumerate(product_elements, 1):
                    product_data = self.extract_product_data(item)
                    if product_data:
                        product_data['category'] = category
                        page_results.append(product_data)
                        logger.debug(f"  [{idx}/{len(product_elements)}] ✓ {product_data['name'][:50]}")
                
                self.stats.total_products += len(page_results)
                self.stats.total_pages += 1
                
                return page_results
                
            except TimeoutException:
                logger.warning(f"⚠ Timeout on attempt {attempt + 1}")
                if attempt < ScraperConfig.MAX_RETRIES - 1:
                    self.stats.retries_used += 1
                    logger.info(f"Retrying in {ScraperConfig.RETRY_DELAY}s...")
                    time.sleep(ScraperConfig.RETRY_DELAY)
                else:
                    logger.error(f"❌ Failed after {ScraperConfig.MAX_RETRIES} attempts")
                    return []
            
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                if attempt < ScraperConfig.MAX_RETRIES - 1:
                    self.stats.retries_used += 1
                    time.sleep(ScraperConfig.RETRY_DELAY)
                else:
                    return []
        
        return []

    def append_to_csv(self, products: List[Dict]):
        """Append clean products to CSV"""
        if not products:
            return
        
        with open(ScraperConfig.CSV_FILENAME, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "regular_price", "sale_price", "vendor", "category",
                "rating", "review_count", "availability", "url", "scraped_at"
            ])
            for p in products:
                writer.writerow(p)

    def save_json(self):
        """Save all products as JSON"""
        if not ScraperConfig.SAVE_JSON or not self.all_products:
            return
        
        output = {
            "scrape_date": datetime.now().isoformat(),
            "total_products": len(self.all_products),
            "statistics": {
                "pages_scraped": self.stats.total_pages,
                "categories_scraped": self.stats.categories_scraped,
                "categories_failed": self.stats.categories_failed,
                "failed_items": self.stats.failed_products,
                "retries_used": self.stats.retries_used
            },
            "category_breakdown": self.stats.category_breakdown,
            "products": self.all_products
        }
        
        with open(ScraperConfig.JSON_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Saved JSON: {ScraperConfig.JSON_FILENAME}")

    def scrape_category(self, start_url: str) -> int:
        """Scrape a single category with pagination"""
        current_url = start_url
        category_product_count = 0
        
        while current_url:
            # Scrape current page
            page_products = self.scrape_page_with_retry(current_url)
            
            if page_products:
                self.append_to_csv(page_products)
                self.all_products.extend(page_products)
                category_product_count += len(page_products)
                logger.info(f"✓ Saved {len(page_products)} products (Category total: {category_product_count})")
            
            # Check for next page
            try:
                next_button = self.driver.find_element(By.CSS_SELECTOR, "a.next.page-numbers")
                current_url = next_button.get_attribute("href")
                logger.info(f"➡ Next page: {current_url}")
                time.sleep(random.uniform(1, 2))
            except NoSuchElementException:
                logger.info("✓ Reached last page of category")
                current_url = None
        
        return category_product_count

    def scrape_multiple_categories(self, url_list: List[str]) -> int:
        """
        Main method to scrape multiple URLs iteratively
        
        Args:
            url_list: List of category URLs to scrape
            
        Returns:
            Total number of products scraped
        """
        logger.info("="*60)
        logger.info(f"🚀 Starting multi-category scrape: {len(url_list)} categories")
        logger.info("="*60)
        
        for idx, url in enumerate(url_list, 1):
            logger.info("")
            logger.info("🔹" + "="*58 + "🔹")
            logger.info(f"📂 CATEGORY {idx}/{len(url_list)}: {url}")
            logger.info("🔹" + "="*58 + "🔹")
            
            try:
                # Scrape this category
                category_products = self.scrape_category(url)
                
                # Track success
                self.stats.add_category_result(url, category_products, success=True)
                
                logger.info(f"✅ Category {idx} complete: {category_products} products")
                
                # Rate limiting between categories (be extra polite!)
                if idx < len(url_list):
                    delay = random.uniform(
                        ScraperConfig.BETWEEN_CATEGORY_DELAY_MIN,
                        ScraperConfig.BETWEEN_CATEGORY_DELAY_MAX
                    )
                    logger.info(f"⏸ Cooling down {delay:.1f}s before next category...")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"❌ Category {idx} FAILED: {e}")
                self.stats.add_category_result(url, 0, success=False)
                # Continue to next category instead of crashing
                continue
        
        # Final exports
        self.save_json()
        
        logger.info("")
        logger.info("="*60)
        logger.info(self.stats.summary())
        logger.info(self.stats.category_summary())
        logger.info("="*60)
        
        return self.stats.total_products

    def shutdown(self):
        """Clean shutdown"""
        try:
            self.driver.quit()
            logger.info("✓ Browser closed")
        except Exception as e:
            logger.warning(f"Shutdown warning: {e}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║  🧹 ILEYCOM MULTI-URL SCRAPER 🧹                     ║
    ║  Iterate Through Multiple Categories                  ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    # ==================== CONFIGURE YOUR URLs HERE ====================
    TARGET_URLS = [
        "https://ileycom.com/en/product-category/gifts-ideas/",
        "https://ileycom.com/en/product-category/clothing/",
        "https://ileycom.com/en/product-category/beauty-products/",
        "https://ileycom.com/en/product-category/accessories/",
        "https://ileycom.com/en/product-category/home-decor/",
        "https://ileycom.com/en/product-category/gastronomy/"
        
        # Add as many URLs as you want!
    ]
    
    HEADLESS_MODE = False  # Set to True for background scraping
    
    # ==================== RUN THE SCRAPER ====================
    scraper = IleycomScraperClean(headless=HEADLESS_MODE)
    
    try:
        total = scraper.scrape_multiple_categories(TARGET_URLS)
        
        print("\n" + "="*60)
        print(f"🎉 SUCCESS! Scraped {total} total products")
        print(f"📁 Clean CSV: {ScraperConfig.CSV_FILENAME}")
        if ScraperConfig.SAVE_JSON:
            print(f"📁 JSON: {ScraperConfig.JSON_FILENAME}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Interrupted by user")
        logger.info(f"Partial results: {scraper.stats.total_products} products")
    
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
    
    finally:
        scraper.shutdown()
        print("\n👋 Check logs/ for details")