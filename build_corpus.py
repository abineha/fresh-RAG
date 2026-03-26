"""
Mediterranean Cuisine — RAG Background Corpus Builder
======================================================
Deliverable 1: Background Corpus

Ethically compliant scraper:
  - Checks robots.txt before scraping each domain (fetched with our
    custom User-Agent so the server returns the real robots.txt file)
  - Identifies itself honestly via User-Agent (Wikimedia-recommended format)
  - Respects rate limits (configurable delay between requests)
  - Retries with back-off on HTTP 429 Too Many Requests
  - Only scrapes the three permitted public sources
  - Attributes every document with source URL and title
  - No login walls, paywalls, or copyrighted commercial content
  - Skips already-scraped files so the run is resumable

Allowed sources (per coursework specification):
  1. Wikipedia  — https://en.wikipedia.org/wiki/List_of_cuisines
  2. Wikibooks  — https://en.wikibooks.org/wiki/Cookbook:Cuisines
  3. Blog       — https://aroundtheworldin80cuisinesblog.wordpress.com/

robots.txt note:
  Wikipedia's User-agent:* section allows /wiki/ pages and explicitly
  disallows /w/ (API) and /api/.  We therefore scrape /wiki/ HTML only.
  The robots.txt is fetched via requests (with our custom UA) so that
  the server returns the real directives instead of an error splash page.

  
Usage:
    pip install requests beautifulsoup4 lxml
    python build_corpus.py

Outputs:
    corpus/              ← one .txt file per scraped page
    corpus_combined.txt  ← single merged file ready for chunking
    corpus_manifest.csv  ← metadata table (title, url, source, words, status)
"""

import os
import re
import csv
import time
import urllib.robotparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR      = "corpus"
COMBINED_FILE   = "corpus_combined.txt"
MANIFEST_FILE   = "corpus_manifest.csv"
DELAY_SECONDS   = 2.0          # polite crawl delay between requests

# Wikimedia-recommended User-Agent format: AppName/version (contact info)
HEADERS = {
    "User-Agent": (
        "RAGCorpusBuilder/1.0 "
        "(university NLP research; Mediterranean cuisine QA project; "
        "not for commercial use; contact: student-project)"
    )
}

# ──────────────────────────────────────────────────────────────
# ROBOTS.TXT COMPLIANCE
# ──────────────────────────────────────────────────────────────
_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

def can_fetch(url: str) -> bool:
    """Return True if robots.txt permits fetching this URL.

    IMPORTANT: We fetch robots.txt via `requests` with our custom User-Agent.
    Without this, Python's urllib sends 'Python-urllib/x.y', which causes
    Wikipedia (and Wikimedia sites) to return an error splash page instead of
    the real robots.txt, making RobotFileParser block all URLs incorrectly.
    """
    from urllib.parse import urlparse
    parsed   = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    if base_url not in _robots_cache:
        robots_url = f"{base_url}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            resp = requests.get(robots_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            rp.parse(resp.text.splitlines())
        except Exception as e:
            print(f"  [ROBOTS] Could not fetch {robots_url}: {e} — assuming allowed")
            rp = None
        _robots_cache[base_url] = rp
    rp = _robots_cache[base_url]
    if rp is None:
        return True
    return rp.can_fetch(HEADERS["User-Agent"], url)


# ──────────────────────────────────────────────────────────────
# TARGET PAGES
# ──────────────────────────────────────────────────────────────
WIKIPEDIA_PAGES = [
    # Overview
    ("Mediterranean cuisine", "https://en.wikipedia.org/wiki/Mediterranean_cuisine"),
    ("Mediterranean diet", "https://en.wikipedia.org/wiki/Mediterranean_diet"),
    ("Mediterranean Diet Pyramid", "https://en.wikipedia.org/wiki/Mediterranean_Diet_Pyramid"),

    # Regional & Country cuisines
    ("Greek cuisine", "https://en.wikipedia.org/wiki/Greek_cuisine"),
    ("Italian cuisine", "https://en.wikipedia.org/wiki/Italian_cuisine"),
    ("Spanish cuisine", "https://en.wikipedia.org/wiki/Spanish_cuisine"),
    ("Turkish cuisine", "https://en.wikipedia.org/wiki/Turkish_cuisine"),
    ("Lebanese cuisine", "https://en.wikipedia.org/wiki/Lebanese_cuisine"),
    ("Moroccan cuisine", "https://en.wikipedia.org/wiki/Moroccan_cuisine"),
    ("Egyptian cuisine", "https://en.wikipedia.org/wiki/Egyptian_cuisine"),
    ("Tunisian cuisine", "https://en.wikipedia.org/wiki/Tunisian_cuisine"),
    ("Libyan cuisine", "https://en.wikipedia.org/wiki/Libyan_cuisine"),
    ("Syrian cuisine", "https://en.wikipedia.org/wiki/Syrian_cuisine"),
    ("Algerian cuisine", "https://en.wikipedia.org/wiki/Algerian_cuisine"),
    ("Maltese cuisine", "https://en.wikipedia.org/wiki/Maltese_cuisine"),
    ("Cypriot cuisine", "https://en.wikipedia.org/wiki/Cypriot_cuisine"),
    ("Croatian cuisine", "https://en.wikipedia.org/wiki/Croatian_cuisine"),
    ("Palestinian cuisine", "https://en.wikipedia.org/wiki/Palestinian_cuisine"),
    ("Ottoman cuisine", "https://en.wikipedia.org/wiki/Ottoman_cuisine"),
    ("Albanian cuisine", "https://en.wikipedia.org/wiki/Albanian_cuisine"),
    ("Berber cuisine", "https://en.wikipedia.org/wiki/Berber_cuisine"),
    ("Monegasque cuisine", "https://en.wikipedia.org/wiki/Monegasque_cuisine"),
    ("Montenegrin cuisine", "https://en.wikipedia.org/wiki/Montenegrin_cuisine"),
    ("Occitan cuisine", "https://en.wikipedia.org/wiki/Occitan_cuisine"),
    ("Levantine cuisine", "https://en.wikipedia.org/wiki/Levantine_cuisine"),
    ("Maghrebi cuisine", "https://en.wikipedia.org/wiki/Maghrebi_cuisine"),
    ("Israeli cuisine", "https://en.wikipedia.org/wiki/Israeli_cuisine"),
    ("Kurdish cuisine", "https://en.wikipedia.org/wiki/Kurdish_cuisine"),
    ("Catalan cuisine", "https://en.wikipedia.org/wiki/Catalan_cuisine"),
    ("Cuisine of Provence", "https://en.wikipedia.org/wiki/Cuisine_of_Provence"),
    ("Cuisine of Corsica", "https://en.wikipedia.org/wiki/Cuisine_of_Corsica"),
    ("Bosnia and Herzegovina cuisine", "https://en.wikipedia.org/wiki/Bosnia_and_Herzegovina_cuisine"),
    ("Slovenian cuisine", "https://en.wikipedia.org/wiki/Slovenian_cuisine"),
    ("Gibraltarian cuisine", "https://en.wikipedia.org/wiki/Gibraltarian_cuisine"),
    ("Assyrian cuisine", "https://en.wikipedia.org/wiki/Assyrian_cuisine"),
    ("Aromanian cuisine", "https://en.wikipedia.org/wiki/Aromanian_cuisine"),
    ("Mizrahi Jewish cuisine", "https://en.wikipedia.org/wiki/Mizrahi_Jewish_cuisine"),
    ("Sephardic Jewish cuisine", "https://en.wikipedia.org/wiki/Sephardic_Jewish_cuisine"),
    ("Syrian Jewish cuisine", "https://en.wikipedia.org/wiki/Syrian_Jewish_cuisine"),
    ("Arab cuisine", "https://en.wikipedia.org/wiki/Arab_cuisine"),
    ("Balkan cuisine", "https://en.wikipedia.org/wiki/Balkan_cuisine"),
    ("French cuisine", "https://en.wikipedia.org/wiki/French_cuisine"),
    ("Middle Eastern cuisine", "https://en.wikipedia.org/wiki/Middle_Eastern_cuisine"),

    # Key dishes (core + extended)
    ("Hummus", "https://en.wikipedia.org/wiki/Hummus"),
    ("Falafel", "https://en.wikipedia.org/wiki/Falafel"),
    ("Shawarma", "https://en.wikipedia.org/wiki/Shawarma"),
    ("Baklava", "https://en.wikipedia.org/wiki/Baklava"),
    ("Moussaka", "https://en.wikipedia.org/wiki/Moussaka"),
    ("Paella", "https://en.wikipedia.org/wiki/Paella"),
    ("Pizza", "https://en.wikipedia.org/wiki/Pizza"),
    ("Tagine", "https://en.wikipedia.org/wiki/Tagine"),
    ("Couscous", "https://en.wikipedia.org/wiki/Couscous"),
    ("Tabbouleh", "https://en.wikipedia.org/wiki/Tabbouleh"),
    ("Tzatziki", "https://en.wikipedia.org/wiki/Tzatziki"),
    ("Dolma", "https://en.wikipedia.org/wiki/Dolma"),
    ("Baba ghanoush", "https://en.wikipedia.org/wiki/Baba_ghanoush"),
    ("Pita bread", "https://en.wikipedia.org/wiki/Pita"),
    ("Halloumi", "https://en.wikipedia.org/wiki/Halloumi"),
    ("Feta cheese", "https://en.wikipedia.org/wiki/Feta"),
    ("Shakshuka", "https://en.wikipedia.org/wiki/Shakshouka"),
    ("Fattoush", "https://en.wikipedia.org/wiki/Fattoush"),
    ("Mansaf", "https://en.wikipedia.org/wiki/Mansaf"),
    ("Kibbeh", "https://en.wikipedia.org/wiki/Kibbeh"),
    ("Lahmacun", "https://en.wikipedia.org/wiki/Lahmacun"),
    ("Spanakopita", "https://en.wikipedia.org/wiki/Spanakopita"),
    ("Borek", "https://en.wikipedia.org/wiki/B%C3%B6rek"),
    ("Ratatouille", "https://en.wikipedia.org/wiki/Ratatouille_(food)"),
    ("Gazpacho", "https://en.wikipedia.org/wiki/Gazpacho"),
    ("Risotto", "https://en.wikipedia.org/wiki/Risotto"),
    ("Pasta", "https://en.wikipedia.org/wiki/Pasta"),
    ("Souvlaki", "https://en.wikipedia.org/wiki/Souvlaki"),
    ("Gyros", "https://en.wikipedia.org/wiki/Gyros"),
    ("Tapas", "https://en.wikipedia.org/wiki/Tapas"),
    ("Bouillabaisse", "https://en.wikipedia.org/wiki/Bouillabaisse"),
    ("Greek salad", "https://en.wikipedia.org/wiki/Greek_salad"),
    ("Salade nicoise", "https://en.wikipedia.org/wiki/Salade_ni%C3%A7oise"),
    ("Labneh", "https://en.wikipedia.org/wiki/Labneh"),
    ("Ful medames", "https://en.wikipedia.org/wiki/Ful_medames"),
    ("Musakhan", "https://en.wikipedia.org/wiki/Musakhan"),
    ("Knafeh", "https://en.wikipedia.org/wiki/Knafeh"),
    ("Loukoumades", "https://en.wikipedia.org/wiki/Loukoumades"),
    ("Caponata", "https://en.wikipedia.org/wiki/Caponata"),
    ("Bottarga", "https://en.wikipedia.org/wiki/Bottarga"),
    ("Brik", "https://en.wikipedia.org/wiki/Brik"),
    ("Cacciucco", "https://en.wikipedia.org/wiki/Cacciucco"),
    ("Brudet", "https://en.wikipedia.org/wiki/Brudet"),
    ("Doner kebab", "https://en.wikipedia.org/wiki/Doner_kebab"),
    ("Kleftiko", "https://en.wikipedia.org/wiki/Kleftiko"),
    ("Koshari", "https://en.wikipedia.org/wiki/Kushari"),
    ("Mulukhiyah", "https://en.wikipedia.org/wiki/Mulukhiyah"),
    ("Pastilla", "https://en.wikipedia.org/wiki/Pastilla"),
    ("Harira", "https://en.wikipedia.org/wiki/Harira"),
    ("Maqluba", "https://en.wikipedia.org/wiki/Maqluba"),

    # Ingredients
    ("Olive oil", "https://en.wikipedia.org/wiki/Olive_oil"),
    ("Olives", "https://en.wikipedia.org/wiki/Olive"),
    ("Tahini", "https://en.wikipedia.org/wiki/Tahini"),
    ("Chickpea", "https://en.wikipedia.org/wiki/Chickpea"),
    ("Za'atar", "https://en.wikipedia.org/wiki/Za%27atar"),
    ("Harissa", "https://en.wikipedia.org/wiki/Harissa"),
    ("Sumac", "https://en.wikipedia.org/wiki/Sumac"),
    ("Saffron", "https://en.wikipedia.org/wiki/Saffron"),
    ("Pomegranate", "https://en.wikipedia.org/wiki/Pomegranate"),
    ("Cumin", "https://en.wikipedia.org/wiki/Cumin"),
    ("Yogurt", "https://en.wikipedia.org/wiki/Yogurt"),
    ("Eggplant", "https://en.wikipedia.org/wiki/Eggplant"),
    ("Bulgur", "https://en.wikipedia.org/wiki/Bulgur"),
    ("Semolina", "https://en.wikipedia.org/wiki/Semolina"),
    ("Filo pastry", "https://en.wikipedia.org/wiki/Phyllo"),
    ("Oregano", "https://en.wikipedia.org/wiki/Oregano"),
    ("Thyme", "https://en.wikipedia.org/wiki/Thyme"),

    # Culture / concepts
    ("Mezze", "https://en.wikipedia.org/wiki/Meze"),
    ("Halal food", "https://en.wikipedia.org/wiki/Halal"),
    ("Kosher food", "https://en.wikipedia.org/wiki/Kosher_foods"),
]

WIKIBOOKS_PAGES = [
    # Core Mediterranean + base pages
    ("Cookbook: Mediterranean cuisines",    "https://en.wikibooks.org/wiki/Cookbook:Cuisines/Mediterranean"),
    ("Cookbook: Hummus",                    "https://en.wikibooks.org/wiki/Cookbook:Hummus"),
    ("Cookbook: Falafel",                   "https://en.wikibooks.org/wiki/Cookbook:Falafel"),
    ("Cookbook: Tabbouleh",                 "https://en.wikibooks.org/wiki/Cookbook:Tabbouleh"),
    ("Cookbook: Tabbouleh I",               "https://en.wikibooks.org/wiki/Cookbook:Tabbouleh_I"),
    ("Cookbook: Tabbouleh II",              "https://en.wikibooks.org/wiki/Cookbook:Tabbouleh_II"),
    ("Cookbook: Tabbouleh III",             "https://en.wikibooks.org/wiki/Cookbook:Tabbouleh_III"),
    ("Cookbook: Baba ghanoush",             "https://en.wikibooks.org/wiki/Cookbook:Baba_Ghanoush"),
    ("Cookbook: Baklava",                   "https://en.wikibooks.org/wiki/Cookbook:Baklava"),
    ("Cookbook: Baklava I",                 "https://en.wikibooks.org/wiki/Cookbook:Baklava_I"),
    ("Cookbook: Baklava II",                "https://en.wikibooks.org/wiki/Cookbook:Baklava_II"),
    ("Cookbook: Baklava Pistachio",         "https://en.wikibooks.org/wiki/Cookbook:Baklava_with_Pistachio_Nuts"),
    ("Cookbook: Couscous",                  "https://en.wikibooks.org/wiki/Cookbook:Couscous"),
    ("Cookbook: Paella",                    "https://en.wikibooks.org/wiki/Cookbook:Paella"),
    ("Cookbook: Pizza dough",               "https://en.wikibooks.org/wiki/Cookbook:Pizza_Dough"),
    ("Cookbook: Pasta",                     "https://en.wikibooks.org/wiki/Cookbook:Pasta"),
    ("Cookbook: Moussaka",                  "https://en.wikibooks.org/wiki/Cookbook:Moussaka"),
    ("Cookbook: Moussaka Bulgarian",        "https://en.wikibooks.org/wiki/Cookbook:Moussaka_(Bulgarian)"),
    ("Cookbook: Greek Moussaka",            "https://en.wikibooks.org/wiki/Cookbook:Greek_Moussaka"),
    ("Cookbook: Tzatziki",                  "https://en.wikibooks.org/wiki/Cookbook:Tzatziki"),
    ("Cookbook: Shakshuka",                 "https://en.wikibooks.org/wiki/Cookbook:Shakshouka"),
    ("Cookbook: Tagine",                    "https://en.wikibooks.org/wiki/Cookbook:Chicken_Tagine"),
    ("Cookbook: Risotto",                   "https://en.wikibooks.org/wiki/Cookbook:Risotto"),
    ("Cookbook: Risotto Basic",             "https://en.wikibooks.org/wiki/Cookbook:Risotto_(Basic)"),
    ("Cookbook: Risotto II",                "https://en.wikibooks.org/wiki/Cookbook:Risotto_II"),
    ("Cookbook: Risotto ai Funghi",         "https://en.wikibooks.org/wiki/Cookbook:Risotto_ai_Funghi"),
    ("Cookbook: Risotto alla Milanese",     "https://en.wikibooks.org/wiki/Cookbook:Risotto_alla_Milanese"),
    ("Cookbook: Gazpacho",                  "https://en.wikibooks.org/wiki/Cookbook:Gazpacho"),
    ("Cookbook: Dolma",                     "https://en.wikibooks.org/wiki/Cookbook:Dolma"),
    ("Cookbook: Pita bread",                "https://en.wikibooks.org/wiki/Cookbook:Pita"),
    ("Cookbook: Olive oil",                 "https://en.wikibooks.org/wiki/Cookbook:Olive_Oil"),
    ("Cookbook: Tahini",                    "https://en.wikibooks.org/wiki/Cookbook:Tahini"),
    ("Cookbook: Halvah",                    "https://en.wikibooks.org/wiki/Cookbook:Halvah"),

    # Cypriot
    ("Cookbook: Cypriot Salad", "https://en.wikibooks.org/wiki/Cookbook:Cypriot_Salad"),

    # Greek extended
    ("Cookbook: Frappé Coffee", "https://en.wikibooks.org/wiki/Cookbook:Frapp%C3%A9_Coffee"),
    ("Cookbook: Galaktoboureko", "https://en.wikibooks.org/wiki/Cookbook:Galaktoboureko_(Greek_Semolina_Custard_Pastry)"),
    ("Cookbook: Greek Chicken Wrap", "https://en.wikibooks.org/wiki/Cookbook:Greek_Chicken_Wrap"),
    ("Cookbook: Greek Salad", "https://en.wikibooks.org/wiki/Cookbook:Greek_Salad"),
    ("Cookbook: Greek Yogurt Sauce", "https://en.wikibooks.org/wiki/Cookbook:Greek_Yogurt_Sauce"),
    ("Cookbook: Greek-Style Grilled Chicken", "https://en.wikibooks.org/wiki/Cookbook:Greek-Style_Grilled_Chicken"),
    ("Cookbook: Hummus Greek", "https://en.wikibooks.org/wiki/Cookbook:Hummus_(Greek)"),
    ("Cookbook: Loukoumas I", "https://en.wikibooks.org/wiki/Cookbook:Loukoumas_(Greek_Donuts_with_Honey)_I"),
    ("Cookbook: Loukoumas II", "https://en.wikibooks.org/wiki/Cookbook:Loukoumas_(Greek_Donuts_with_Honey)_II"),
    ("Cookbook: Saganaki", "https://en.wikibooks.org/wiki/Cookbook:Pan-Fried_White_Brined_Cheese_(Saganaki)"),
    ("Cookbook: Pastitsio", "https://en.wikibooks.org/wiki/Cookbook:Pastitsio"),
    ("Cookbook: Pasta Casserole Pastitsio", "https://en.wikibooks.org/wiki/Cookbook:Pasta_Casserole_(Pastitsio)"),
    ("Cookbook: Rosemary Garlic Fish", "https://en.wikibooks.org/wiki/Cookbook:Rosemary_Garlic_Fish"),
    ("Cookbook: Spanakopita", "https://en.wikibooks.org/wiki/Cookbook:Spanakopita"),
    ("Cookbook: Tyropita", "https://en.wikibooks.org/wiki/Cookbook:Tyropita_(Greek_Cheese_Pie)"),

    # Maltese
    ("Cookbook: Arjoli", "https://en.wikibooks.org/wiki/Cookbook:Arjoli_(Maltese_Herbed_Bread_Dip)"),
    ("Cookbook: Kawlata", "https://en.wikibooks.org/wiki/Cookbook:Kawlata_(Maltese_Vegetable_and_Pork_Soup)"),
    ("Cookbook: Maltese Rabbit Stew", "https://en.wikibooks.org/wiki/Cookbook:Maltese_Rabbit_Stew_(Stuffat_tal-Fenek)"),

    # Turkish
    ("Cookbook: Ayran", "https://en.wikibooks.org/wiki/Cookbook:Ayran_(Turkish_Yogurt_Drink)"),
    ("Cookbook: Beef and Vegetable Kabobs", "https://en.wikibooks.org/wiki/Cookbook:Beef_and_Vegetable_Kabobs"),
    ("Cookbook: Börek", "https://en.wikibooks.org/wiki/Cookbook:B%C3%B6rek_(Turkish_Filled_Pastries)"),
    ("Cookbook: Cheese Filling Börek", "https://en.wikibooks.org/wiki/Cookbook:Cheese_Filling_for_Peynirli_B%C3%B6rek"),
    ("Cookbook: Döner Kebab", "https://en.wikibooks.org/wiki/Cookbook:D%C3%B6ner_Kebab"),
    ("Cookbook: Turkish Sausage Eggs", "https://en.wikibooks.org/wiki/Cookbook:Egg_with_Cabbage_and_Turkish_Sausage"),
    ("Cookbook: Gözleme", "https://en.wikibooks.org/wiki/Cookbook:G%C3%B6zleme_(Turkish_Flatbread)"),
    ("Cookbook: Kebab", "https://en.wikibooks.org/wiki/Cookbook:Kebab"),
    ("Cookbook: Menemen", "https://en.wikibooks.org/wiki/Cookbook:Menemen_(Eggs_with_Onion,_Green_Pepper,_and_Tomato)"),
    ("Cookbook: Muhallebi", "https://en.wikibooks.org/wiki/Cookbook:Pudding_with_Pistachios_and_Rose_Water_(Muhallebi)"),
    ("Cookbook: Shakshuka I", "https://en.wikibooks.org/wiki/Cookbook:Shakshuka_I"),
    ("Cookbook: Tarhana", "https://en.wikibooks.org/wiki/Cookbook:Tarhana_(Turkish_Yogurt_Soup)"),
    ("Cookbook: Turkish Delight", "https://en.wikibooks.org/wiki/Cookbook:Turkish_Delight"),
    ("Cookbook: Dolma Pilav", "https://en.wikibooks.org/wiki/Cookbook:Turkish_Dolma_Pilav"),
    ("Cookbook: Yuvarlak", "https://en.wikibooks.org/wiki/Cookbook:Yuvarlak_(Greek_Meatballs)"),

    # French Mediterranean
    ("Cookbook: Bouillabaisse", "https://en.wikibooks.org/wiki/Cookbook:Bouillabaisse"),
    ("Cookbook: Ratatouille", "https://en.wikibooks.org/wiki/Cookbook:Ratatouille_I"),
    ("Cookbook: Niçoise Salad", "https://en.wikibooks.org/wiki/Cookbook:Ni%C3%A7oise_Salad"),

    # Italian
    ("Cookbook: Carbonara Pasta", "https://en.wikibooks.org/wiki/Cookbook:Carbonara_Pasta"),
    ("Cookbook: Focaccia", "https://en.wikibooks.org/wiki/Cookbook:Focaccia_II"),
    ("Cookbook: Ossobuco", "https://en.wikibooks.org/wiki/Cookbook:Ossobuco_Alla_Milanese"),
    ("Cookbook: Pesto", "https://en.wikibooks.org/wiki/Cookbook:Pesto_I"),
    ("Cookbook: Spaghetti alla Puttanesca", "https://en.wikibooks.org/wiki/Cookbook:Spaghetti_alla_Puttanesca"),

    # Spanish
    ("Cookbook: Arroz Negro", "https://en.wikibooks.org/wiki/Cookbook:Arroz_Negro_(Valencian_Squid_Rice)"),
    ("Cookbook: Paella de Marisco", "https://en.wikibooks.org/wiki/Cookbook:Paella_de_Marisco"),
    ("Cookbook: Paella Roja", "https://en.wikibooks.org/wiki/Cookbook:Paella_Roja"),
    ("Cookbook: Paella Valenciana", "https://en.wikibooks.org/wiki/Cookbook:Paella_Valenciana"),
    ("Cookbook: Valencian Paella", "https://en.wikibooks.org/wiki/Cookbook:Valencian-Inspired_Paella"),

    # Levantine
    ("Cookbook: Baba Ganoush", "https://en.wikibooks.org/wiki/Cookbook:Baba_Ganoush"),
    ("Cookbook: Musakhan", "https://en.wikibooks.org/wiki/Cookbook:Baked_Chicken_with_Onions,_Sumac_and_Allspice_(Musakhan)"),
    ("Cookbook: Moutabbal", "https://en.wikibooks.org/wiki/Cookbook:Eggplant_and_Tahini_(Moutabbal)"),
    ("Cookbook: Mediterranean Beef Stew", "https://en.wikibooks.org/wiki/Cookbook:Mediterranean_Beef_Stew"),
    ("Cookbook: Mediterranean Green Chicken", "https://en.wikibooks.org/wiki/Cookbook:Mediterranean_Green_Chicken"),
    ("Cookbook: Mediterranean Grilled Tuna", "https://en.wikibooks.org/wiki/Cookbook:Mediterranean_Grilled_Tuna"),
    ("Cookbook: Stuffed Grape Leaves", "https://en.wikibooks.org/wiki/Cookbook:Stuffed_Grape_Leaves"),

    # North Africa
    ("Cookbook: Harissa bil Djaj", "https://en.wikibooks.org/wiki/Cookbook:Harissa_bil_Djaj_(Libyan_Chicken_Harissa)"),
    ("Cookbook: Algerian Couscous", "https://en.wikibooks.org/wiki/Cookbook:Algerian_Couscous_with_Meat_and_Vegetables"),
    ("Cookbook: Algerian Lemonade", "https://en.wikibooks.org/wiki/Cookbook:Algerian_Lemonade"),
    ("Cookbook: Algerian Mint Tea", "https://en.wikibooks.org/wiki/Cookbook:Algerian_Mint_Tea"),
]

BLOG_PAGES = [
    # Direct blog pages (clean articles)
    ("Blog: Southern France and Monaco", "https://aroundtheworldin80cuisinesblog.wordpress.com/2017/04/14/1-france-and-monaco/"),
    ("Blog: Turkey and Cyprus",          "https://aroundtheworldin80cuisinesblog.wordpress.com/2017/04/20/4-turkey-and-cyprus/"),
    ("Blog: North Africa",               "https://aroundtheworldin80cuisinesblog.wordpress.com/2017/05/07/13-north-africa/"),
    ("Blog: Former Yugoslavia",          "https://aroundtheworldin80cuisinesblog.wordpress.com/2017/09/12/27-the-balkan-peninsula/"),
    ("Blog: Greece",                     "https://aroundtheworldin80cuisinesblog.wordpress.com/2017/10/05/33-greece/"),
    ("Blog: Southern Italy",             "https://aroundtheworldin80cuisinesblog.wordpress.com/2017/11/02/37-southern-italy/"),
    ("Blog: Central Italy",              "https://aroundtheworldin80cuisinesblog.wordpress.com/2018/02/18/47-centralitaly/"),
    ("Blog: Spain",                      "https://aroundtheworldin80cuisinesblog.wordpress.com/2018/04/29/53-spain/"),
    ("Blog: Fertile Crescent",           "https://aroundtheworldin80cuisinesblog.wordpress.com/2018/07/17/58-the-fertile-crescent/"),
    ("Blog: Northern Italy",             "https://aroundtheworldin80cuisinesblog.wordpress.com/2018/10/14/69-northern-italy/"),

    # Category pages (for broader coverage)
    ("Category: Southern France and Monaco", "https://aroundtheworldin80cuisinesblog.wordpress.com/category/01-southern-france-and-monaco/"),
    ("Category: Turkey and Cyprus",          "https://aroundtheworldin80cuisinesblog.wordpress.com/category/04-turkey-and-cyprus/"),
    ("Category: North Africa",               "https://aroundtheworldin80cuisinesblog.wordpress.com/category/13-north-africa/"),
    ("Category: Portugal",                   "https://aroundtheworldin80cuisinesblog.wordpress.com/category/15-portugal/"),
    ("Category: Former Yugoslavia",          "https://aroundtheworldin80cuisinesblog.wordpress.com/category/27-former-yugoslavia/"),
    ("Category: Greece",                     "https://aroundtheworldin80cuisinesblog.wordpress.com/category/33-greece/"),
    ("Category: Southern Italy",             "https://aroundtheworldin80cuisinesblog.wordpress.com/category/37-southern-italy/"),
    ("Category: Northern France",            "https://aroundtheworldin80cuisinesblog.wordpress.com/category/41-northern-france/"),
    ("Category: Central Italy",              "https://aroundtheworldin80cuisinesblog.wordpress.com/category/47-central-italy/"),
    ("Category: Spain",                      "https://aroundtheworldin80cuisinesblog.wordpress.com/category/53-spain/"),
    ("Category: Fertile Crescent",           "https://aroundtheworldin80cuisinesblog.wordpress.com/category/58-the-fertile-crescent/"),
    ("Category: East Balkan Peninsula",      "https://aroundtheworldin80cuisinesblog.wordpress.com/category/60-east-balkan-peninsula/"),
    ("Category: Northern Italy",             "https://aroundtheworldin80cuisinesblog.wordpress.com/category/69-northern-italy/"),
]

# ──────────────────────────────────────────────────────────────
# FETCH HELPERS
# ──────────────────────────────────────────────────────────────

def fetch_soup(url: str, max_retries: int = 3) -> BeautifulSoup | None:
    """Fetch URL (after robots.txt check) and return parsed HTML.

    Retries up to `max_retries` times on HTTP 429 (rate-limit), honouring
    the Retry-After header when present.
    """
    if not can_fetch(url):
        print(f"  [ROBOTS] Disallowed by robots.txt: {url}")
        return None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  [429] Rate-limited — waiting {wait}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.HTTPError as e:
            print(f"  [HTTP {e.response.status_code}] {url}")
            return None
        except Exception as e:
            print(f"  [ERROR] {url}: {e}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
    return None


def clean(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)          # remove citation markers [1]
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# SOURCE-SPECIFIC SCRAPERS
# ──────────────────────────────────────────────────────────────

def scrape_wikipedia(url: str) -> str:
    soup = fetch_soup(url)
    if not soup:
        return ""
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return ""
    for tag in content.find_all([
        "table", "sup", "div", "span"
    ], class_=re.compile(
        r"navbox|reflist|mw-references|hatnote|sistersitebox|mw-editsection"
    )):
        tag.decompose()
    paras = [p.get_text() for p in content.find_all("p") if p.get_text(strip=True)]
    return clean("\n\n".join(paras))


def scrape_wikibooks(url: str) -> str:
    soup = fetch_soup(url)
    if not soup:
        return ""
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return ""
    for tag in content.find_all(["table", "sup"]):
        tag.decompose()
    elements = content.find_all(["p", "li", "h2", "h3"])
    text = "\n\n".join(e.get_text() for e in elements if e.get_text(strip=True))
    return clean(text)


def scrape_blog(url: str) -> str:
    """Scrape a WordPress blog category page.

    Strategy:
    1. Grab any full post bodies visible on the category page.
    2. If the category page only shows excerpts (or is empty), follow every
       post permalink found in the listing and scrape each individual post.
    3. Fall back to all <p> tags if nothing else is found.
    """
    soup = fetch_soup(url)
    if not soup:
        return ""

    # -- Step 1: extract full post bodies present on the category page -------
    texts = []
    for article in soup.find_all("article"):
        body = article.get_text(separator="\n").strip()
        if len(body) > 200:
            texts.append(body)

    # -- Step 2: if few/no bodies, follow individual post links --------------
    if len(texts) < 2:
        post_links = set()
        # WordPress post permalinks inside the listing area
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Only follow links within the same domain that look like posts
            if (href.startswith("https://aroundtheworldin80cuisinesblog.wordpress.com/")
                    and "/category/" not in href
                    and len(href) > len("https://aroundtheworldin80cuisinesblog.wordpress.com/") + 5):
                post_links.add(href.rstrip("/"))
        for post_url in list(post_links)[:12]:  # cap at 12 posts per category
            time.sleep(DELAY_SECONDS)
            post_soup = fetch_soup(post_url)
            if not post_soup:
                continue
            article = post_soup.find("article") or post_soup.find(
                "div", class_=re.compile(r"entry-content|post-content|hentry")
            )
            if article:
                body = article.get_text(separator="\n").strip()
                if len(body) > 200:
                    texts.append(body)

    # -- Step 3: paragraph fallback ------------------------------------------
    if not texts:
        texts = [p.get_text() for p in soup.find_all("p") if p.get_text(strip=True)]

    return clean("\n\n".join(texts))


# ──────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower())[:60]


def build_corpus():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []
    combined = []

    all_pages = (
        [(t, u, "wikipedia", scrape_wikipedia) for t, u in WIKIPEDIA_PAGES]
        + [(t, u, "wikibooks", scrape_wikibooks) for t, u in WIKIBOOKS_PAGES]
        + [(t, u, "blog",      scrape_blog)      for t, u in BLOG_PAGES]
    )

    for idx, (title, url, source, scraper) in enumerate(all_pages, 1):
        filename = f"{slugify(source)}_{slugify(title)}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        print(f"[{idx:02d}/{len(all_pages)}] {source.upper():10s} {title}")

        # Resume: reload already-scraped file without hitting the network
        if os.path.exists(filepath):
            with open(filepath, encoding="utf-8") as f:
                existing = f.read()
            wc = len(existing.split())
            print(f"           -> RESUMED  {wc:,} words  (file exists)")
            manifest.append(dict(title=title, url=url, source=source,
                                 word_count=wc, status="ok"))
            # Still add to combined
            text_body = existing.split("=" * 60 + "\n\n", 1)[-1]
            combined.append(
                f"\n\n{'='*60}\n"
                f"TITLE: {title}\nSOURCE: {source}\nURL: {url}\n"
                f"{'='*60}\n\n{text_body}"
            )
            continue

        text = scraper(url)

        if not text or len(text) < 80:
            print("           -> SKIPPED (no content)")
            manifest.append(dict(title=title, url=url, source=source,
                                 word_count=0, status="failed"))
            time.sleep(DELAY_SECONDS)
            continue

        wc       = len(text.split())
        filename = f"{slugify(source)}_{slugify(title)}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"TITLE:      {title}\n")
            f.write(f"SOURCE:     {source}\n")
            f.write(f"URL:        {url}\n")
            f.write(f"SCRAPED:    {datetime.utcnow().strftime('%Y-%m-%d')}\n")
            f.write(f"LICENCE:    CC BY-SA (Wikipedia/Wikibooks) / blog author (blog)\n")
            f.write("=" * 60 + "\n\n")
            f.write(text)

        combined.append(
            f"\n\n{'='*60}\n"
            f"TITLE: {title}\nSOURCE: {source}\nURL: {url}\n"
            f"{'='*60}\n\n{text}"
        )
        manifest.append(dict(title=title, url=url, source=source,
                             word_count=wc, status="ok"))
        print(f"           -> {wc:,} words  ->  {filename}")
        time.sleep(DELAY_SECONDS)

    # ── Combined file ──────────────────────────────────────────
    with open(COMBINED_FILE, "w", encoding="utf-8") as f:
        f.write("MEDITERRANEAN CUISINE — RAG BACKGROUND CORPUS\n")
        f.write(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d')}\n")
        f.write("Sources: Wikipedia (CC BY-SA), Wikibooks (CC BY-SA), "
                "Around the World in 80 Cuisines blog\n")
        f.write("=" * 60 + "\n")
        f.write("".join(combined))

    # ── Manifest CSV ───────────────────────────────────────────
    with open(MANIFEST_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["title", "url", "source", "word_count", "status"]
        )
        w.writeheader()
        w.writerows(manifest)

    # ── Summary ────────────────────────────────────────────────
    ok    = [r for r in manifest if r["status"] == "ok"]
    total = sum(r["word_count"] for r in ok)
    print(f"\n{'-'*50}")
    print(f"  Pages successfully scraped : {len(ok)} / {len(all_pages)}")
    print(f"  Total word count           : {total:,}")
    print(f"  Individual files           : ./{OUTPUT_DIR}/")
    print(f"  Combined corpus            : {COMBINED_FILE}")
    print(f"  Manifest                   : {MANIFEST_FILE}")
    print(f"{'-'*50}")


if __name__ == "__main__":
    build_corpus()
