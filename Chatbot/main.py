import asyncio
from collections import Counter, defaultdict
import http
import shutil
import time
import traceback
from openai import OpenAI
from fastapi import FastAPI, Form, Request, File, UploadFile, WebSocket, Depends, HTTPException, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from slugify import slugify
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, Integer, String, Float, case, desc, func, or_
from databases import Database
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.future import select
from fastapi.middleware.cors import CORSMiddleware
from markdown import markdown
from datetime import datetime
from typing import List
from pydantic import BaseModel
from langdetect import detect, lang_detect_exception
from spacy_langdetect import LanguageDetector
from spacy.language import Language
from langdetect import detect
from libretranslatepy import LibreTranslateAPI
from deep_translator import MyMemoryTranslator
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import os
import re
import uuid
import requests
import pymysql
import aiomysql
import logging
from scipy import spatial
import pandas as pd
import spacy
import cloudinary
import cloudinary.uploader
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import pickle
import os
import hashlib
from datetime import datetime, timedelta
import numpy as np
import json
import atexit
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ Sentence Transformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ Sentence Transformers not available. Install with: pip install sentence-transformers")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available. Install with: pip install scikit-learn")

openai = OpenAI(
    api_key="sk-V4UGt-FNIde85D7t0zrvZbVG5eolZDsE8awXTAuJYgT3BlbkFJToj--_okCjQAcwzYu4ZC6JDX8kznTJhzruBhL9Q5YA"
)

client = OpenAI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Enhanced Fashion Chatbot with Hybrid Intelligence...")
    
    # Create upload directory
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Connect to database
    await database.connect()
    await create_tables()
    
    # Initialize NLP
    global nlp
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ Spacy English model loaded")
    except OSError:
        print("⚠️ Spacy English model not found. Install with: python -m spacy download en_core_web_sm")
        nlp = None
    
    # Initialize hybrid system
    print("🧠 Hybrid LLM + Vector system ready!")
    print("✅ Enhanced Fashion Chatbot is ready!")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down application...")
    try:
        await database.disconnect()
        print("✅ Database disconnected properly")
        
        # Close the engine
        await engine.dispose()
        print("✅ Database engine disposed")
        
    except Exception as e:
        print(f"⚠️ Error during shutdown: {e}")

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="Chatbot/static"), name="static")

templates = Jinja2Templates(directory="Chatbot/templates")

UPLOAD_DIR = "static/uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "jfif"}

Base = declarative_base()

DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/ecommerce"

engine = create_async_engine(DATABASE_URL, echo=True)

database = Database(DATABASE_URL)

# ================================
# FASHION CATEGORIES CONSTANTS
# ================================

class FashionCategories:
    """
    Centralized fashion categories used throughout the application
    for keyword extraction, preference detection, and consultation summaries
    Based on existing HybridKeywordExtractor fashion_categories
    """
    
    # CORE CLOTHING ITEMS (Priority 400)
    CLOTHING_TERMS = [
        # TOPS
        'kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'tshirt',
        'sweater', 'cardigan', 'hoodie', 'tank top', 'crop top', 'tube top',
        'halter top', 'camisole', 'singlet', 'vest', 'rompi', 'polo shirt',
        'henley', 'turtleneck', 'off shoulder', 'cold shoulder', 'wrap top',
        
        # BOTTOMS  
        'celana', 'pants', 'trousers', 'jeans', 'denim', 'rok', 'skirt',
        'shorts', 'leggings', 'jeggings', 'palazzo pants',
        'wide leg pants', 'skinny jeans', 'straight jeans', 'bootcut',
        'flare pants', 'culottes', 'palazzo', 'cargo pants', 'joggers',
        'track pants', 'sweatpants', 'chinos', 'capri', 'bermuda',
        
        # DRESSES
        'dress', 'gaun', 'terusan', 'maxi dress', 'mini dress', 'midi dress',
        'bodycon dress', 'a-line dress', 'shift dress', 'wrap dress',
        'slip dress', 'shirt dress', 'sweater dress', 'sundress',
        'cocktail dress', 'evening dress',
        
        # OUTERWEAR 
        'jaket', 'jacket', 'blazer', 'coat', 'mantel',
        'bomber jacket', 'denim jacket', 'leather jacket', 'varsity jacket',
        'puffer jacket', 'windbreaker', 'raincoat', 'trench coat',
        'peacoat', 'parka', 'cape', 'poncho',

        # ACCESSORIES 
        'shawl', 'pashmina', 'scarf', 'belt', 'bag', 'purse', 'jewelry',
        'necklace', 'earrings', 'bracelet', 'ring', 'watch',
    ]
    
    # SLEEVE TERMS (Priority 350 - extracted from fits_and_styles)
    SLEEVE_TERMS = [
        'lengan panjang', 'lengan pendek', 'long sleeve', 'long sleeves',
        'short sleeve', 'short sleeves', 'sleeveless', 'tanpa lengan',
        '3/4 sleeve', '3/4 sleeves', 'quarter sleeve', 'quarter sleeves',
        'cap sleeve', 'cap sleeves', 'bell sleeve', 'bell sleeves',
        'puff sleeve', 'puff sleeves', 'balloon sleeve', 'balloon sleeves',
        'bishop sleeve', 'bishop sleeves', 'dolman sleeve', 'dolman sleeves',
        'raglan sleeve', 'raglan sleeves', 'flutter sleeve', 'flutter sleeves'
    ]
    
    # FIT TERMS (Priority 350 - extracted from fits_and_styles)
    FIT_TERMS = [
        'oversized', 'oversize', 'longgar', 'loose', 'baggy', 'relaxed',
        'fitted', 'ketat', 'tight', 'slim', 'skinny', 'regular fit',
        'tailored', 'structured', 'flowy', 'draped', 'a-line', 'straight'
    ]
    
    # LENGTH TERMS (Priority 350 - extracted from fits_and_styles)
    LENGTH_TERMS = [
        'maxi', 'midi', 'mini', 'ankle length', 'knee length', 'thigh length',
        'floor length', 'tea length', 'above knee', 'below knee', 'cropped length',
        'cropped', 'crop', 'panjang', 'pendek', 'long', 'short'
    ]
    
    # NECKLINE TERMS (Priority 350 - extracted from fits_and_styles)
    NECKLINE_TERMS = [
        'v-neck', 'scoop neck', 'crew neck', 'boat neck', 'off shoulder',
        'one shoulder', 'strapless', 'halter neck', 'high neck',
        'mock neck', 'cowl neck', 'square neck', 'sweetheart neck'
    ]
    
    # STYLE CATEGORIES (Priority 300)
    STYLE_TERMS = [
        'casual', 'santai', 'formal', 'resmi', 'elegant', 'elegan',
        'minimalis', 'minimalist', 'vintage', 'retro', 'bohemian', 'boho',
        'ethnic', 'etnik', 'modern', 'contemporary', 'classic', 'klasik',
        'trendy', 'fashionable', 'chic', 'sophisticated', 'edgy',
        'feminine', 'masculine', 'androgynous', 'romantic', 'sporty',
        'preppy', 'grunge', 'punk', 'gothic', 'kawaii', 'streetwear'
    ]
    
    # COLOR TERMS (Priority 250 - from colors_and_materials)
    COLOR_TERMS = [
        # COLOR CATEGORIES (most important for user preferences)
        'neutral', 'neutral colors', 'bright colors', 'pastel', 'pastels',
        'mixed', 'mixed colors', 'colorful', 'earth tones', 'natural colors',
        'warm colors', 'cool colors', 'monochrome', 'vibrant colors',
        
        # BASIC COLORS
        'hitam', 'black', 'putih', 'white', 'merah', 'red', 'biru', 'blue',
        'hijau', 'green', 'kuning', 'yellow', 'orange', 'oranye',
        'ungu', 'purple', 'pink', 'merah muda', 'coklat', 'brown',
        'abu-abu', 'grey', 'gray', 'navy', 'biru tua', 'maroon',
        'burgundy', 'wine', 'cream', 'krem', 'beige', 'khaki',
        'gold', 'emas', 'silver', 'perak', 'rose gold', 'copper',
        'mint', 'turquoise', 'coral', 'salmon', 'lavender', 'lilac',
        
        # GENERIC COLOR TERMS
        'color', 'colors', 'warna', 'tone', 'tones', 'shade', 'shades'
    ]
    
    # MATERIAL TERMS (Priority 250 - from colors_and_materials)
    MATERIAL_TERMS = [
        'cotton', 'katun', 'silk', 'sutra', 'satin', 'chiffon',
        'lace', 'renda', 'denim', 'leather', 'kulit', 'faux leather',
        'velvet', 'beludru', 'corduroy', 'tweed', 'wool', 'wol',
        'cashmere', 'linen', 'polyester', 'spandex', 'elastane',
        'viscose', 'rayon', 'modal', 'bamboo', 'organic cotton'
    ]
    
    # PATTERN TERMS (Priority 250 - from colors_and_materials)
    PATTERN_TERMS = [
        'polos', 'solid', 'plain', 'striped', 'garis-garis', 'polka dot',
        'floral', 'bunga-bunga', 'geometric', 'abstract', 'animal print',
        'leopard', 'zebra', 'snake print', 'plaid', 'checkered',
        'houndstooth', 'paisley', 'tribal', 'ethnic print', 'batik',
        'tie dye', 'ombre', 'gradient', 'metallic', 'glitter', 'sequin'
    ]
    
    # OCCASION TERMS (Priority 200)
    OCCASION_TERMS = [
        'office', 'kantor', 'work', 'kerja', 'business', 'professional',
        'party', 'pesta', 'clubbing', 'nightout', 'date', 'kencan',
        'wedding', 'pernikahan', 'formal event', 'graduation', 'wisuda',
        'beach', 'pantai', 'vacation', 'liburan', 'travel', 'weekend',
        'everyday', 'sehari-hari', 'casual outing', 'shopping', 'hangout',
        'gym', 'workout', 'sport', 'olahraga', 'yoga', 'running',
        'casual everyday', 'work office', 'special events', 'mixed occasions'
    ]
    
    # SPECIAL FEATURES (Priority 180)
    SPECIAL_FEATURE_TERMS = [
        'backless', 'open back', 'cut out', 'mesh', 'sheer', 'transparent',
        'embroidered', 'bordir', 'beaded', 'rhinestone', 'studded',
        'fringe', 'tassel', 'tie', 'wrap around', 'convertible',
        'reversible', 'pocket', 'kantong', 'zipper', 'resleting',
        'button', 'kancing', 'snap', 'hook', 'drawstring', 'elastic'
    ]
    
    # BODY TYPE TERMS (Priority 150 - from body_and_sizes, excluding clothing sizes)
    BODY_TERMS = [
        # BODY SHAPES
        'pear shape', 'bentuk pir', 'apple shape', 'bentuk apel',
        'hourglass', 'jam pasir', 'rectangle', 'persegi panjang',
        'inverted triangle', 'athletic build', 'curvy', 'lekuk tubuh',
        'petite', 'mungil', 'tall', 'tinggi', 'plus size', 'ukuran besar',
        'maternity', 'hamil', 'nursing', 'menyusui', 'pear',
        
        # MEASUREMENTS (only actual measurements)
        'cm', 'kg', 'height', 'weight', 'tinggi', 'berat'
    ]
    
    # SIZE TERMS (Priority 150 - from body_and_sizes)
    SIZE_TERMS = [
        'xs', 'extra small', 'sangat kecil', 's', 'small', 'kecil',
        'm', 'medium', 'sedang', 'l', 'large', 'besar', 'xl', 'extra large',
        'xxl', 'xxxl', 'plus size', 'big size', 'ukuran besar',
        'free size', 'one size', 'ukuran bebas', 'all size'
    ]
    
    # SEASONAL TERMS (Priority 120)
    SEASONAL_TERMS = [
        'summer', 'musim panas', 'winter', 'musim dingin', 'spring', 'musim semi',
        'fall', 'autumn', 'musim gugur', 'rainy season', 'musim hujan',
        'hot weather', 'cuaca panas', 'cold weather', 'cuaca dingin',
        'humid', 'lembab', 'tropical', 'tropis'
    ]

    ACTIVITY_TERMS = [
        'office worker', 'student', 'teacher', 'work from home', 'remote work',
        'travel', 'active lifestyle', 'gym', 'sports', 'outdoor activities',
        'sedentary', 'desk job', 'standing job', 'retail', 'healthcare',
        'business meetings', 'social events', 'freelancer', 'entrepreneur',
        'stay at home', 'retired', 'part time', 'full time', 'shift work',
        'pekerja kantoran', 'mahasiswa', 'guru', 'kerja dari rumah',
        'sering traveling', 'aktif bergerak', 'olahraga', 'aktivitas outdoor',
        'banyak duduk', 'kerja berdiri', 'sering meeting', 'acara sosial'
    ]
    
    # BLACKLISTED TERMS (accessories and problematic terms)
    BLACKLISTED_TERMS = [       
        # META TERMS
        'style preference', 'clothing preference', 'fashion choice',
        'recommendation', 'suggestion', 'option', 'choice',
        'confirmation', 'question', 'summary', 'information'
    ]
    
    # CONFLICT GROUPS for preference change detection
    CONFLICT_GROUPS = {
        'sleeve_length': {
            'patterns': [r'long sleeve', r'short sleeve', r'sleeveless', r'3/4 sleeve', r'lengan panjang', r'lengan pendek'],
            'keywords': SLEEVE_TERMS,
            # NEW: Detailed conflict mapping for precise conflict detection
            'conflicts': {
                'long': ['long sleeve', 'long sleeves', 'lengan panjang'],
                'short': ['short sleeve', 'short sleeves', 'lengan pendek'],
                'sleeveless': ['sleeveless', 'tanpa lengan', 'tank top'],
                '3/4': ['3/4 sleeve', '3/4 sleeves', 'quarter sleeve', 'quarter sleeves'],
                'cap': ['cap sleeve', 'cap sleeves'],
                'bell': ['bell sleeve', 'bell sleeves'],
                'puff': ['puff sleeve', 'puff sleeves']
            }
        },
        'color_preference': {
            'patterns': [r'neutral', r'bright', r'pastel', r'mixed', r'netral', r'terang', r'colorful'],
            'keywords': COLOR_TERMS,
            # NEW: Color category conflicts
            'conflicts': {
                'neutral': ['neutral', 'neutral colors', 'netral', 'warna netral'],
                'bright': ['bright', 'bright colors', 'cerah', 'warna cerah', 'vibrant', 'vibrant colors'],
                'pastel': ['pastel', 'pastels', 'pastel colors'],
                'dark': ['dark', 'dark colors', 'gelap', 'warna gelap'],
                'light': ['light', 'light colors', 'terang', 'warna terang'],
                'warm': ['warm', 'warm colors', 'hangat', 'warna hangat'],
                'cool': ['cool', 'cool colors', 'sejuk', 'warna sejuk'],
                'monochrome': ['monochrome', 'monokrom', 'black and white', 'hitam putih'],
                'colorful': ['colorful', 'mixed', 'mixed colors', 'warna-warni', 'beragam warna']
            }
        },
        'fit_style': {
            'patterns': [r'fitted', r'loose', r'oversized', r'slim', r'relaxed', r'tight', r'ketat', r'longgar'],
            'keywords': FIT_TERMS,
            # NEW: Fit style conflicts
            'conflicts': {
                'fitted': ['fitted', 'slim', 'tight', 'ketat', 'tailored'],
                'loose': ['loose', 'oversized', 'baggy', 'longgar', 'relaxed'],
                'regular': ['regular fit', 'standard fit'],
                'structured': ['structured', 'tailored'],
                'flowy': ['flowy', 'draped', 'flowing']
            }
        },
        'clothing_length': {
            'patterns': [r'mini', r'midi', r'maxi', r'short', r'long', r'crop', r'pendek', r'panjang'],
            'keywords': LENGTH_TERMS,
            # NEW: Length conflicts
            'conflicts': {
                'crop': ['crop', 'cropped', 'short length', 'cropped length'],
                'regular': ['regular length', 'standard length'],
                'long': ['long length', 'maxi', 'ankle length', 'floor length', 'panjang'],
                'mini': ['mini', 'very short', 'above knee'],
                'midi': ['midi', 'knee length', 'below knee'],
                'maxi': ['maxi', 'ankle length', 'floor length', 'full length'],
                'short': ['short', 'pendek', 'above knee', 'thigh length'],
                'knee': ['knee length', 'at knee'],
                'ankle': ['ankle length', 'full length']
            }
        },
        # NEW: Additional conflict groups
        'neckline_style': {
            'patterns': [r'v-neck', r'crew neck', r'scoop neck', r'high neck', r'off shoulder'],
            'keywords': NECKLINE_TERMS,
            'conflicts': {
                'high': ['high neck', 'mock neck', 'turtleneck'],
                'low': ['v-neck', 'scoop neck', 'deep v'],
                'off_shoulder': ['off shoulder', 'one shoulder', 'strapless'],
                'crew': ['crew neck', 'round neck'],
                'boat': ['boat neck', 'wide neck']
            }
        },
        'style_formality': {
            'patterns': [r'casual', r'formal', r'elegant', r'sporty'],
            'keywords': STYLE_TERMS,
            'conflicts': {
                'casual': ['casual', 'santai', 'relaxed', 'everyday'],
                'formal': ['formal', 'resmi', 'business', 'professional'],
                'elegant': ['elegant', 'elegan', 'sophisticated', 'classy'],
                'sporty': ['sporty', 'athletic', 'active'],
                'trendy': ['trendy', 'fashionable', 'modern'],
                'classic': ['classic', 'klasik', 'timeless', 'traditional']
            }
        }
    }

    COLOR_CATEGORY_MAPPING = {
        # Neutral colors
        'neutral': ['black', 'white', 'grey', 'gray', 'beige', 'cream', 'ivory', 'khaki', 'taupe', 'charcoal', 'navy'],
        'neutral colors': ['black', 'white', 'grey', 'gray', 'beige', 'cream', 'ivory', 'khaki', 'taupe', 'charcoal', 'navy'],
        'netral': ['black', 'white', 'grey', 'gray', 'beige', 'cream', 'ivory', 'khaki', 'taupe', 'charcoal', 'navy'],
        'warna netral': ['black', 'white', 'grey', 'gray', 'beige', 'cream', 'ivory', 'khaki', 'taupe', 'charcoal', 'navy'],
        
        # Dark colors
        'dark': ['black', 'navy', 'dark blue', 'charcoal', 'dark grey', 'dark gray', 'maroon', 'burgundy', 'dark green', 'dark brown'],
        'dark colors': ['black', 'navy', 'dark blue', 'charcoal', 'dark grey', 'dark gray', 'maroon', 'burgundy', 'dark green', 'dark brown'],
        'gelap': ['black', 'navy', 'dark blue', 'charcoal', 'dark grey', 'dark gray', 'maroon', 'burgundy', 'dark green', 'dark brown'],
        'warna gelap': ['black', 'navy', 'dark blue', 'charcoal', 'dark grey', 'dark gray', 'maroon', 'burgundy', 'dark green', 'dark brown'],
        
        # Light colors
        'light': ['white', 'cream', 'ivory', 'beige', 'light blue', 'light pink', 'light grey', 'light gray', 'pale yellow', 'mint'],
        'light colors': ['white', 'cream', 'ivory', 'beige', 'light blue', 'light pink', 'light grey', 'light gray', 'pale yellow', 'mint'],
        'terang': ['white', 'cream', 'ivory', 'beige', 'light blue', 'light pink', 'light grey', 'light gray', 'pale yellow', 'mint'],
        'warna terang': ['white', 'cream', 'ivory', 'beige', 'light blue', 'light pink', 'light grey', 'light gray', 'pale yellow', 'mint'],
        
        # Bright colors
        'bright': ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'purple', 'magenta', 'cyan', 'lime'],
        'bright colors': ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'purple', 'magenta', 'cyan', 'lime'],
        'cerah': ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'purple', 'magenta', 'cyan', 'lime'],
        'warna cerah': ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'purple', 'magenta', 'cyan', 'lime'],
        'vibrant': ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'purple', 'magenta', 'cyan', 'lime'],
        'vibrant colors': ['red', 'blue', 'yellow', 'green', 'orange', 'pink', 'purple', 'magenta', 'cyan', 'lime'],
        
        # Pastel colors
        'pastel': ['pastel pink', 'pastel blue', 'pastel yellow', 'pastel green', 'lavender', 'mint', 'peach', 'baby blue', 'powder pink'],
        'pastels': ['pastel pink', 'pastel blue', 'pastel yellow', 'pastel green', 'lavender', 'mint', 'peach', 'baby blue', 'powder pink'],
        'pastel colors': ['pastel pink', 'pastel blue', 'pastel yellow', 'pastel green', 'lavender', 'mint', 'peach', 'baby blue', 'powder pink'],
        
        # Earth tones
        'earth tones': ['brown', 'tan', 'beige', 'khaki', 'olive', 'rust', 'terracotta', 'camel', 'sand', 'moss green'],
        'earth tone': ['brown', 'tan', 'beige', 'khaki', 'olive', 'rust', 'terracotta', 'camel', 'sand', 'moss green'],
        'natural colors': ['brown', 'tan', 'beige', 'khaki', 'olive', 'rust', 'terracotta', 'camel', 'sand', 'moss green'],
        
        # Warm colors
        'warm': ['red', 'orange', 'yellow', 'pink', 'coral', 'peach', 'gold', 'burgundy', 'rust', 'terracotta'],
        'warm colors': ['red', 'orange', 'yellow', 'pink', 'coral', 'peach', 'gold', 'burgundy', 'rust', 'terracotta'],
        'hangat': ['red', 'orange', 'yellow', 'pink', 'coral', 'peach', 'gold', 'burgundy', 'rust', 'terracotta'],
        'warna hangat': ['red', 'orange', 'yellow', 'pink', 'coral', 'peach', 'gold', 'burgundy', 'rust', 'terracotta'],
        
        # Cool colors
        'cool': ['blue', 'green', 'purple', 'navy', 'teal', 'mint', 'lavender', 'turquoise', 'cyan', 'indigo'],
        'cool colors': ['blue', 'green', 'purple', 'navy', 'teal', 'mint', 'lavender', 'turquoise', 'cyan', 'indigo'],
        'sejuk': ['blue', 'green', 'purple', 'navy', 'teal', 'mint', 'lavender', 'turquoise', 'cyan', 'indigo'],
        'warna sejuk': ['blue', 'green', 'purple', 'navy', 'teal', 'mint', 'lavender', 'turquoise', 'cyan', 'indigo'],
        
        # Monochrome
        'monochrome': ['black', 'white', 'grey', 'gray'],
        'monokrom': ['black', 'white', 'grey', 'gray'],
        'black and white': ['black', 'white'],
        'hitam putih': ['black', 'white'],
        
        # Mixed/Colorful
        'mixed': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white'],
        'mixed colors': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white'],
        'colorful': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'magenta', 'cyan'],
        'warna-warni': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'magenta', 'cyan'],
        'beragam warna': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'magenta', 'cyan'],
        
        # Metallic colors
        'metallic': ['gold', 'silver', 'rose gold', 'copper', 'bronze', 'platinum'],
        'metalik': ['gold', 'silver', 'rose gold', 'copper', 'bronze', 'platinum'],
        
        # Seasonal colors
        'spring colors': ['pastel pink', 'mint', 'lavender', 'peach', 'light green', 'baby blue'],
        'summer colors': ['bright blue', 'coral', 'yellow', 'turquoise', 'lime', 'hot pink'],
        'autumn colors': ['burgundy', 'rust', 'gold', 'brown', 'orange', 'olive'],
        'winter colors': ['navy', 'black', 'white', 'grey', 'burgundy', 'emerald'],
        
        # Professional/Business colors
        'professional': ['black', 'navy', 'grey', 'gray', 'white', 'beige', 'burgundy'],
        'business': ['black', 'navy', 'grey', 'gray', 'white', 'beige', 'burgundy'],
        'formal colors': ['black', 'navy', 'grey', 'gray', 'white', 'beige', 'burgundy']
    }
    
    # NEW: Color synonyms for better matching
    COLOR_SYNONYMS = {
        'grey': ['gray', 'grey'],
        'gray': ['grey', 'gray'],
        'beige': ['tan', 'sand', 'camel', 'nude'],
        'navy': ['dark blue', 'navy blue'],
        'burgundy': ['wine', 'maroon', 'deep red'],
        'mint': ['mint green', 'light green'],
        'coral': ['orange pink', 'salmon'],
        'turquoise': ['teal', 'cyan'],
        'lavender': ['light purple', 'pale purple'],
        'rose gold': ['pink gold', 'copper'],
        'charcoal': ['dark grey', 'dark gray'],
        'ivory': ['off white', 'cream white'],
        'khaki': ['olive', 'army green'],
        'magenta': ['hot pink', 'bright pink'],
        'indigo': ['dark blue', 'deep blue']
    }
    
    # NEW: Indonesian to English color translations
    COLOR_TRANSLATIONS = {
        'hitam': 'black',
        'putih': 'white', 
        'merah': 'red',
        'biru': 'blue',
        'hijau': 'green',
        'kuning': 'yellow',
        'ungu': 'purple',
        'coklat': 'brown',
        'abu-abu': 'grey',
        'pink': 'pink',
        'orange': 'orange',
        'emas': 'gold',
        'perak': 'silver',
        'krem': 'cream',
        'navy': 'navy'
    }
    
    # COMPLETE TERM CATEGORIES for summary generation
    TERM_CATEGORIES = {
        'clothing_terms': CLOTHING_TERMS,
        'sleeve_terms': SLEEVE_TERMS,
        'fit_terms': FIT_TERMS,
        'length_terms': LENGTH_TERMS,
        'neckline_terms': NECKLINE_TERMS,
        'style_terms': STYLE_TERMS,
        'color_terms': COLOR_TERMS,
        'material_terms': MATERIAL_TERMS,
        'pattern_terms': PATTERN_TERMS,
        'occasion_terms': OCCASION_TERMS,
        'special_feature_terms': SPECIAL_FEATURE_TERMS,
        'body_terms': BODY_TERMS,
        'size_terms': SIZE_TERMS,
        'seasonal_terms': SEASONAL_TERMS,
        'blacklisted_terms': BLACKLISTED_TERMS
    }
    
    # PRIORITY MAPPING (from your original HybridKeywordExtractor)
    PRIORITY_SCORES = {
        'clothing_terms': 400,
        'sleeve_terms': 350,
        'fit_terms': 350,
        'length_terms': 350,
        'neckline_terms': 350,
        'style_terms': 300,
        'color_terms': 250,
        'material_terms': 250,
        'pattern_terms': 250,
        'occasion_terms': 200,
        'special_feature_terms': 180,
        'body_terms': 150,
        'size_terms': 150,
        'seasonal_terms': 120
    }
    
    @classmethod
    def is_blacklisted(cls, keyword):
        """Check if a keyword is blacklisted"""
        keyword_lower = str(keyword).lower()
        return any(blacklisted in keyword_lower for blacklisted in cls.BLACKLISTED_TERMS)
    
    @classmethod
    def get_category(cls, keyword):
        """Get the category of a keyword"""
        keyword_lower = str(keyword).lower()
        
        for category_name, terms in cls.TERM_CATEGORIES.items():
            if category_name == 'blacklisted_terms':
                continue
                
            for term in terms:
                if (term.lower() == keyword_lower or 
                    term.lower() in keyword_lower or 
                    keyword_lower in term.lower()):
                    return category_name
        
        return 'unknown'
    
    @classmethod
    def get_priority_score(cls, keyword):
        """Get priority score for a keyword based on its category"""
        category = cls.get_category(keyword)
        return cls.PRIORITY_SCORES.get(category, 50)  # Default score for unknown
    
    @classmethod
    def is_color_preference(cls, keyword):
        """Check if keyword is a color preference"""
        keyword_lower = str(keyword).lower()
        return any(color_term in keyword_lower for color_term in 
                  ['color', 'warna', 'neutral', 'bright', 'mixed', 'pastel', 'colorful'])
    
    @classmethod
    def is_sleeve_preference(cls, keyword):
        """Check if keyword is a sleeve preference"""
        keyword_lower = str(keyword).lower()
        return any(sleeve_term in keyword_lower for sleeve_term in cls.SLEEVE_TERMS)
    
    @classmethod
    def get_all_fashion_concepts(cls):
        """Get all fashion concepts for vector embedding (used by HybridKeywordExtractor)"""
        all_concepts = []
        for category_name, terms in cls.TERM_CATEGORIES.items():
            if category_name != 'blacklisted_terms':
                all_concepts.extend(terms)
        return all_concepts
    
    @classmethod
    def get_color_mapping(cls, color_category):
        """Get colors for a given color category"""
        return cls.COLOR_CATEGORY_MAPPING.get(color_category.lower(), [])
    
    @classmethod
    def is_color_category(cls, keyword):
        """Check if a keyword is a color category that maps to specific colors"""
        return keyword.lower() in cls.COLOR_CATEGORY_MAPPING
    
    @classmethod
    def get_color_synonyms(cls, color):
        """Get synonyms for a given color"""
        return cls.COLOR_SYNONYMS.get(color.lower(), [])
    
    @classmethod
    def translate_color(cls, color):
        """Translate color between Indonesian and English"""
        color_lower = color.lower()
        
        # Indonesian to English
        if color_lower in cls.COLOR_TRANSLATIONS:
            return cls.COLOR_TRANSLATIONS[color_lower]
        
        # English to Indonesian
        for indo_color, eng_color in cls.COLOR_TRANSLATIONS.items():
            if eng_color == color_lower:
                return indo_color
        
        return color  # Return original if no translation found
    
    @classmethod
    def get_all_color_mappings(cls):
        """Get the complete color category mapping dictionary"""
        return cls.COLOR_CATEGORY_MAPPING
    
    @classmethod
    def get_conflict_groups(cls):
        """Get all conflict groups"""
        return cls.CONFLICT_GROUPS
    
    @classmethod
    def get_conflict_group(cls, group_name):
        """Get a specific conflict group"""
        return cls.CONFLICT_GROUPS.get(group_name, {})
    
    @classmethod
    def get_conflict_mapping(cls, group_name):
        """Get the detailed conflict mapping for a group"""
        group = cls.CONFLICT_GROUPS.get(group_name, {})
        return group.get('conflicts', {})
    
    @classmethod
    def detect_conflicts_in_group(cls, group_name, user_preferences, product_text):
        """
        Detect conflicts between user preferences and product for a specific group
        Returns: (has_conflict, conflict_details)
        """
        conflict_mapping = cls.get_conflict_mapping(group_name)
        if not conflict_mapping or not user_preferences:
            return False, {}
        
        product_text_lower = product_text.lower()
        conflicts_found = []
        
        # Find what the user prefers in this group
        user_pref_type = None
        user_pref_weight = 0
        user_pref_keyword = None
        
        for pref_type, terms in conflict_mapping.items():
            for term in terms:
                for pref_keyword, weight, _ in user_preferences:
                    if term in pref_keyword.lower() or pref_keyword.lower() in term:
                        if weight > user_pref_weight:
                            user_pref_type = pref_type
                            user_pref_weight = weight
                            user_pref_keyword = pref_keyword
        
        if not user_pref_type:
            return False, {}
        
        # Check if product conflicts with user preference
        conflicting_terms = []
        for other_type, terms in conflict_mapping.items():
            if other_type != user_pref_type:
                for term in terms:
                    if term in product_text_lower:
                        conflicting_terms.append(term)
        
        if conflicting_terms:
            return True, {
                'user_preference': user_pref_type,
                'user_keyword': user_pref_keyword,
                'user_weight': user_pref_weight,
                'conflicting_terms': conflicting_terms,
                'group': group_name
            }
        
        return False, {}
    
# ================================
# NEW HYBRID LLM + VECTOR CLASSES
# ================================

class HybridKeywordExtractor:
    """
    INTEGRATED: HybridKeywordExtractor using FashionCategories class
    """
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.vector_enabled = False
        self.sentence_model = None
        self.concept_embeddings = None
        
        # Try to initialize vector capabilities
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_enabled = True
            print("✅ Sentence Transformers initialized successfully")
            
        except ImportError:
            self.vector_enabled = False
            print("⚠️ Sentence Transformers not available. Install with: pip install sentence-transformers")
        except Exception as e:
            self.vector_enabled = False
            print(f"⚠️ Vector initialization failed: {e}")

        # Use FashionCategories instead of local fashion_categories
        self.fashion_categories = FashionCategories()
        
        # Get all fashion concepts for vector embedding
        self.fashion_concepts = self.fashion_categories.get_all_fashion_concepts()
        print(f"📚 Loaded {len(self.fashion_concepts)} fashion concepts from FashionCategories")
        
        # Initialize concept embeddings if vector is enabled
        if self.vector_enabled:
            self._initialize_concept_embeddings()
    
    def _initialize_concept_embeddings(self):
        """
        FIXED: Safe initialization of concept embeddings using FashionCategories
        """
        if not self.vector_enabled or not self.sentence_model:
            return
        
        try:
            print("🔄 Initializing fashion concept embeddings...")
            
            if len(self.fashion_concepts) == 0:
                print("⚠️ No fashion concepts found")
                return
            
            # Generate embeddings
            self.concept_embeddings = self.sentence_model.encode(self.fashion_concepts)
            print(f"✅ Generated embeddings shape: {self.concept_embeddings.shape}")
            
        except Exception as e:
            print(f"❌ Failed to initialize concept embeddings: {e}")
            self.vector_enabled = False
            self.concept_embeddings = None

    def _is_relevant_keyword(self, keyword, user_input=None):
        """
        INTEGRATED: Using FashionCategories for relevance checking
        """
        if not keyword or len(str(keyword)) < 2:
            return False
        
        keyword_lower = str(keyword).lower()
        
        # STEP 1: Check if it's blacklisted
        if self.fashion_categories.is_blacklisted(keyword_lower):
            print(f"   ❌ BLACKLISTED: '{keyword}' → NOT RELEVANT")
            return False
        
        # STEP 2: Check if it's a known fashion term
        category = self.fashion_categories.get_category(keyword_lower)
        if category != 'unknown':
            print(f"   ✅ FASHION TERM: '{keyword}' → RELEVANT ({category})")
            return True
        
        # STEP 3: Check for fashion context in user input
        if user_input and self._has_fashion_context(keyword, user_input):
            print(f"   ✅ FASHION CONTEXT: '{keyword}' → RELEVANT")
            return True
        
        # STEP 4: Allow compound fashion terms
        if self._is_compound_fashion_term(keyword):
            print(f"   ✅ COMPOUND FASHION: '{keyword}' → RELEVANT")
            return True
        
        # STEP 5: Allow descriptive terms that commonly appear with fashion
        if self._is_fashion_descriptive(keyword):
            print(f"   ✅ FASHION DESCRIPTIVE: '{keyword}' → RELEVANT")
            return True
        
        # STEP 6: Be more conservative - reject unclear terms
        print(f"   ❌ UNCLEAR TERM: '{keyword}' → NOT RELEVANT")
        return False
    
    def _is_directly_relevant(self, keyword, user_input):
        """INTEGRATED: Check if keyword is directly relevant using FashionCategories"""
        keyword_lower = keyword.lower()
        user_input_lower = user_input.lower()
        
        # Check if blacklisted first
        if self.fashion_categories.is_blacklisted(keyword_lower):
            return False
        
        # Must be explicitly mentioned
        if keyword_lower in user_input_lower:
            return True
        
        # Check if any word from the keyword appears in user input
        keyword_words = keyword_lower.split()
        if any(word in user_input_lower for word in keyword_words):
            return True
        
        # If user mentions a clothing item, allow only directly related style/fit terms
        clothing_mentioned = any(clothing in user_input_lower 
                               for clothing in self.fashion_categories.CLOTHING_TERMS)
        
        if clothing_mentioned:
            # Only allow basic style terms when clothing is mentioned
            basic_style_terms = ['casual', 'formal', 'santai', 'resmi', 'elegant', 'elegan']
            if keyword_lower in basic_style_terms:
                return True
        
        return False
    
    def _is_compound_fashion_term(self, keyword):
        """Check for compound fashion terms using FashionCategories patterns"""
        keyword_lower = str(keyword).lower()
        
        # Check against conflict groups which contain compound patterns
        for group_name, group_data in self.fashion_categories.CONFLICT_GROUPS.items():
            patterns = group_data.get('patterns', [])
            for pattern in patterns:
                if re.search(pattern, keyword_lower):
                    return True
        
        # Additional compound patterns
        compound_patterns = [
            # General compound patterns not in conflict groups
            r'\b(plus|big|free|one)\s+(size|sizes)\b',
            r'\b(earth|natural)\s+(tone|tones|color|colors)\b',
            r'\b(warm|cool)\s+(color|colors|tone|tones)\b'
        ]
        
        for pattern in compound_patterns:
            if re.search(pattern, keyword_lower):
                return True
        
        return False
    
    def _is_fashion_descriptive(self, keyword):
        """Check for descriptive terms commonly used in fashion"""
        keyword_lower = str(keyword).lower()
        
        # Common fashion descriptive terms
        descriptive_terms = [
            'comfortable', 'stylish', 'trendy', 'classic', 'modern', 'vintage',
            'flattering', 'versatile', 'chic', 'sophisticated', 'edgy',
            'feminine', 'masculine', 'youthful', 'mature', 'professional',
            'breathable', 'stretchy', 'soft', 'lightweight', 'durable',
            'washable', 'wrinkle-free', 'easy-care', 'low-maintenance'
        ]
        
        return any(desc in keyword_lower for desc in descriptive_terms)
    
    def _has_fashion_context(self, keyword, user_input):
        """INTEGRATED: Better fashion context detection using FashionCategories"""
        if not user_input:
            return False
        
        user_input_lower = user_input.lower()
        keyword_lower = keyword.lower()
        
        # Fashion context indicators from clothing terms
        fashion_indicators = self.fashion_categories.CLOTHING_TERMS + [
            'style', 'fashion', 'gaya', 'outfit', 'clothing',
            'wear', 'pakai', 'carikan', 'recommendation', 'suggest', 'cocok'
        ]
        
        # Check if keyword appears in a fashion context
        for indicator in fashion_indicators:
            if indicator.lower() in user_input_lower and keyword_lower in user_input_lower:
                return True
        
        return False

    async def _extract_with_llm_improved(self, user_input):
        """INTEGRATED: LLM extraction using FashionCategories validation"""
        prompt = f"""
        Extract ONLY the most relevant fashion keywords from this text. Be selective and focused.
        
        Text: "{user_input}"
        
        Rules:
        1. Extract ONLY terms that are directly mentioned or clearly implied
        2. Focus on: specific clothing items, styles, fits, colors mentioned
        3. Don't add generic terms unless explicitly stated
        4. Maximum 5-6 keywords for short inputs like this
        
        Return a JSON array with the most important keywords:
        Example: ["kemeja", "casual"] for "bisa carikan kemeja casual"
        
        Be SELECTIVE - only include what's actually relevant to the specific request.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.find(']') + 1
                json_text = response_text[start:end]
                
                keywords_list = json.loads(json_text)
                
                result = []
                for i, keyword in enumerate(keywords_list):
                    if isinstance(keyword, str) and len(keyword) > 1:
                        # Use FashionCategories for validation
                        if self._is_directly_relevant(keyword.lower(), user_input):
                            score = self.fashion_categories.get_priority_score(keyword.lower())
                            position_score = score - (i * 5)
                            result.append((keyword.lower(), max(position_score, 50)))
                            print(f"   ✅ LLM ACCEPTED: '{keyword}' → {position_score:.1f}")
                        else:
                            print(f"   🚫 LLM REJECTED: '{keyword}' (not directly relevant)")
                
                return result
            else:
                return self._extract_manual_keywords_improved(user_input)
            
        except Exception as e:
            print(f"   ❌ LLM extraction error: {e}")
            return self._extract_manual_keywords_improved(user_input)
    
    def _extract_with_vectors_improved(self, user_input):
        """INTEGRATED: Vector extraction using FashionCategories"""
        if not self.vector_enabled:
            return []
        
        try:
            input_embedding = self.sentence_model.encode([user_input])
            
            # FIXED: Ensure proper vector dimensions
            if len(input_embedding.shape) == 2:
                input_embedding = input_embedding[0]
            
            # FIXED: Ensure both vectors are 1D and have same shape
            if len(self.concept_embeddings.shape) == 2:
                similarities = cosine_similarity([input_embedding], self.concept_embeddings)[0]
            else:
                concept_embeddings_2d = self.concept_embeddings.reshape(1, -1)
                similarities = cosine_similarity([input_embedding], concept_embeddings_2d)[0]
            
            keywords = []
            user_input_lower = user_input.lower()
            
            for i, similarity in enumerate(similarities):
                concept = self.fashion_concepts[i]
                
                # Skip blacklisted terms immediately
                if self.fashion_categories.is_blacklisted(concept):
                    continue
                
                # IMPROVED: Much more selective filtering
                should_include = False
                
                # HIGH similarity threshold for automatic inclusion (very confident matches)
                if similarity > 0.6:
                    should_include = True
                    print(f"   🎯 HIGH SIMILARITY: '{concept}' → {similarity:.3f}")
                
                # MEDIUM similarity with additional validation
                elif similarity > 0.5:
                    # Must be explicitly mentioned OR be a core clothing item
                    if (concept in user_input_lower or 
                        any(word in user_input_lower for word in concept.split()) or
                        concept in self.fashion_categories.CLOTHING_TERMS):
                        should_include = True
                        print(f"   ✅ MEDIUM + CONTEXT: '{concept}' → {similarity:.3f}")
                    else:
                        print(f"   ❌ MEDIUM NO CONTEXT: '{concept}' → {similarity:.3f}")
                
                # LOW similarity - only if explicitly mentioned
                elif similarity > 0.4:
                    if concept in user_input_lower:
                        should_include = True
                        print(f"   ✅ LOW BUT EXPLICIT: '{concept}' → {similarity:.3f}")
                    else:
                        print(f"   ❌ LOW NO MENTION: '{concept}' → {similarity:.3f}")
                
                if should_include and self._is_relevant_keyword(concept, user_input):
                    score = similarity * 100  # Convert to score
                    keywords.append((concept, score))
            
            # Sort by similarity and limit results
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            # IMPROVED: Return fewer but more relevant results
            top_keywords = keywords[:3]
            
            print(f"   🎯 Vector extraction found {len(top_keywords)} high-quality keywords")
            return top_keywords
            
        except Exception as e:
            print(f"   ❌ Vector extraction error: {e}")
            return []

    async def extract_keywords_hybrid(self, ai_response=None, translated_input=None, accumulated_keywords=None):
        """
        INTEGRATED: Hybrid extraction using FashionCategories
        """
        print("\n🧠 INTEGRATED HYBRID KEYWORD EXTRACTION")
        print("="*60)
        
        all_keywords = {}
        
        # STEP 1: LLM Analysis (primary source)
        if translated_input:
            llm_keywords = await self._extract_with_llm_improved(translated_input)
            for keyword, score in llm_keywords:
                all_keywords[keyword] = score
                print(f"   🤖 LLM: '{keyword}' → {score:.1f}")
        
        # STEP 2: Vector Analysis (secondary, more conservative)
        if self.vector_enabled and translated_input:
            vector_keywords = self._extract_with_vectors_improved(translated_input)
            for keyword, score in vector_keywords:
                if keyword in all_keywords:
                    all_keywords[keyword] = max(all_keywords[keyword], score)
                else:
                    all_keywords[keyword] = score
                print(f"   🔍 Vector: '{keyword}' → {score:.1f}")
        
        # STEP 3: Process AI response (minimal weight)
        if ai_response:
            ai_keywords = self._extract_from_ai_response_improved(ai_response)
            for keyword, score in ai_keywords:
                if keyword in all_keywords:
                    all_keywords[keyword] += score * 0.2
                else:
                    all_keywords[keyword] = score * 0.2
                print(f"   💬 AI Response: '{keyword}' → {score * 0.2:.1f}")
        
        # STEP 4: Include only relevant accumulated keywords
        if accumulated_keywords:
            for keyword, old_score in accumulated_keywords[:5]:
                if (self._is_valid_accumulated_keyword(keyword) and 
                    self._is_directly_relevant(keyword, translated_input or "")):
                    decay_score = old_score * 0.1
                    if keyword in all_keywords:
                        all_keywords[keyword] += decay_score
                    else:
                        all_keywords[keyword] = decay_score
                    print(f"   📚 Accumulated: '{keyword}' → {decay_score:.1f}")
                else:
                    print(f"   🗑️ Skipped accumulated: '{keyword}' (invalid/irrelevant)")
        
        # STEP 5: More aggressive filtering using FashionCategories
        filtered_keywords = {}
        for keyword, score in all_keywords.items():
            if (not self.fashion_categories.is_blacklisted(keyword) and 
                self._is_relevant_keyword(keyword, translated_input) and 
                score > 10):
                filtered_keywords[keyword] = score
            else:
                print(f"   🗑️ Filtered out: '{keyword}' (score: {score:.1f})")
        
        # STEP 6: Post-process to handle conflicts
        final_keywords = self._post_process_keywords(filtered_keywords, translated_input)
        
        # Sort and return fewer results
        ranked_keywords = sorted(final_keywords.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 INTEGRATED RESULTS:")
        for i, (keyword, score) in enumerate(ranked_keywords[:8]):
            category = self.fashion_categories.get_category(keyword)
            print(f"   {i+1:2d}. '{keyword}' → {score:.1f} ({category})")
        
        return ranked_keywords[:8]
    
    def _extract_from_ai_response_improved(self, ai_response):
        """INTEGRATED: AI response extraction using FashionCategories"""
        bold_headings = re.findall(r'\*\*(.*?)\*\*', ai_response)
        
        keywords = []
        for heading in bold_headings:
            cleaned = re.sub(r'[^\w\s-]', '', heading.lower()).strip()
            
            # More conservative filtering using FashionCategories
            if (len(cleaned) > 2 and 
                not re.match(r'^\d+', cleaned) and
                len(cleaned.split()) <= 2 and
                not self.fashion_categories.is_blacklisted(cleaned) and
                self.fashion_categories.get_category(cleaned) != 'unknown'):
                
                score = self.fashion_categories.get_priority_score(cleaned)
                keywords.append((cleaned, score * 0.5))
        
        return keywords
    
    def _is_valid_accumulated_keyword(self, keyword):
        """INTEGRATED: Validation using FashionCategories"""
        if not keyword or len(str(keyword)) < 2:
            return False
        
        keyword_str = str(keyword).lower()
        
        # Use FashionCategories blacklist
        if self.fashion_categories.is_blacklisted(keyword_str):
            return False
        
        # Additional checks for problematic terms
        if (re.match(r'^\d+', keyword_str) or
            len(keyword_str) > 30 or
            any(bad in keyword_str for bad in 
                ['rb', 'ribu', '000', 'idr', 'price', 'chart', 'spec'])):
            return False
        
        return True
    
    def _post_process_keywords(self, keywords_dict, user_input):
        """INTEGRATED: Post-process using FashionCategories conflict detection"""
        if not user_input:
            return keywords_dict
        
        user_input_lower = user_input.lower()
        processed_keywords = {}
        
        for keyword, score in keywords_dict.items():
            # Handle conflicts using FashionCategories conflict groups
            conflict_detected = False
            
            for group_name, group_data in self.fashion_categories.CONFLICT_GROUPS.items():
                patterns = group_data.get('patterns', [])
                
                # Check if this keyword matches any pattern in the group
                keyword_matches_group = False
                for pattern in patterns:
                    if re.search(pattern, keyword):
                        keyword_matches_group = True
                        break
                
                if keyword_matches_group:
                    # Check if user input contains conflicting terms
                    for pattern in patterns:
                        if (re.search(pattern, user_input_lower) and 
                            not re.search(pattern, keyword)):
                            # Conflict detected
                            score *= 0.1
                            conflict_detected = True
                            print(f"   ⚖️ Conflict in {group_name}: reduced '{keyword}' score")
                            break
                
                if conflict_detected:
                    break
            
            # Boost terms that are explicitly mentioned
            if keyword in user_input_lower:
                score *= 1.5
                print(f"   🚀 Boosted '{keyword}' (explicitly mentioned)")
            
            processed_keywords[keyword] = score
        
        return processed_keywords
    
    def extract_color_preferences_enhanced(accumulated_keywords):
        """
        ENHANCED: Extract color preferences with better detection and priority boost
        """
        color_preferences = []
        
        print(f"\n🎨 ENHANCED COLOR PREFERENCE EXTRACTION:")
        print("=" * 60)
        
        # Get all color-related keywords with significant weights
        for keyword, data in accumulated_keywords.items():
            if keyword is None:
                continue
                
            weight = data.get("weight", 0) if isinstance(data, dict) else data
            source = data.get("source", "unknown") if isinstance(data, dict) else "unknown"
            keyword_lower = str(keyword).lower()
            
            # Check if it's a color-related keyword with lower threshold to catch more
            if (FashionCategories.get_category(keyword) == 'color_terms' and 
                weight > 100):  # Much lower threshold to catch more colors
                
                # Skip only the most generic terms
                skip_terms = ['color', 'colors', 'warna', 'tone', 'tones', 'shade', 'shades']
                if keyword_lower in skip_terms:
                    continue
                
                # ENHANCED: Apply massive boost based on source and user input patterns
                final_weight = weight
                
                # Check if this was from user input (high priority)
                if source == "user_input":
                    final_weight = weight * 5.0  # 5x boost for user input colors
                    print(f"   🎨🔥 USER INPUT COLOR: '{keyword_lower}' → {weight:.0f} → BOOSTED TO {final_weight:.0f}")
                
                # Check if this was from consultation (very high priority)
                elif source == "consultation":
                    final_weight = weight * 8.0  # 8x boost for consultation colors
                    print(f"   🎨🔥 CONSULTATION COLOR: '{keyword_lower}' → {weight:.0f} → BOOSTED TO {final_weight:.0f}")
                
                # Check if it's a color category that maps to specific colors
                if FashionCategories.is_color_category(keyword_lower):
                    # Add all colors from the category mapping with boosted weight
                    mapped_colors = FashionCategories.get_color_mapping(keyword_lower)
                    for mapped_color in mapped_colors:
                        mapped_weight = final_weight * 0.9  # Keep most of the boost for mapped colors
                        color_preferences.append((mapped_color, mapped_weight))
                        print(f"   🎨 MAPPED COLOR: '{keyword_lower}' → '{mapped_color}' → {mapped_weight:.0f}")
                else:
                    # Direct color mention
                    color_preferences.append((keyword_lower, final_weight))
                    print(f"   🎨 DIRECT COLOR: '{keyword_lower}' → {final_weight:.0f}")
        
        # Remove duplicates and combine weights for same colors
        color_dict = {}
        for color, weight in color_preferences:
            if color in color_dict:
                color_dict[color] += weight  # Combine weights for duplicate colors
            else:
                color_dict[color] = weight
        
        # Convert back to list and sort by weight
        final_color_preferences = [(color, weight) for color, weight in color_dict.items()]
        final_color_preferences.sort(key=lambda x: x[1], reverse=True)
        
        # Return top color preferences
        top_colors = final_color_preferences[:10]  # Increased from 8 to 10
        
        print(f"\n🎨 FINAL ENHANCED COLOR PREFERENCES:")
        for i, (color, weight) in enumerate(top_colors):
            print(f"   {i+1}. '{color}' → {weight:.0f}")
        
        return top_colors
    
    def _extract_manual_keywords_improved(self, text):
        """INTEGRATED: Manual extraction using FashionCategories"""
        text_lower = text.lower()
        found_keywords = []
        
        # Extract only clearly mentioned keywords using all fashion concepts
        for concept in self.fashion_concepts:
            # Skip blacklisted terms
            if self.fashion_categories.is_blacklisted(concept):
                continue
                
            # CONSERVATIVE matching - exact presence required
            if (f" {concept} " in f" {text_lower} " or 
                text_lower.startswith(f"{concept} ") or
                text_lower.endswith(f" {concept}") or
                concept == text_lower):
                
                score = self.fashion_categories.get_priority_score(concept)
                found_keywords.append((concept, score))
        
        # Sort by score and return top keywords
        found_keywords.sort(key=lambda x: x[1], reverse=True)
        return found_keywords[:5]
    
    def _get_keyword_priority_score(self, keyword):
        """INTEGRATED: Use FashionCategories priority scoring"""
        return self.fashion_categories.get_priority_score(keyword)
    
    def _get_keyword_category(self, keyword):
        """INTEGRATED: Use FashionCategories category detection"""
        return self.fashion_categories.get_category(keyword)
                
class SmartConsultationManager:
    """
    UPDATED: Consultation manager with daily activity question included
    """
    
    def __init__(self, openai_client, hybrid_extractor=None):
        self.openai_client = openai_client
        self.hybrid_extractor = hybrid_extractor
        
        # Use the FashionCategories if available
        if hybrid_extractor and hasattr(hybrid_extractor, 'fashion_categories'):
            self.fashion_categories = hybrid_extractor.fashion_categories
        else:
            self.fashion_categories = self._get_fallback_categories()
        
        # Track what questions have been asked per session
        self.session_questions_asked = {}
    
    def _get_fallback_categories(self):
        """Fallback categories matching the FashionCategories"""
        return {
            'style_categories': ['casual', 'formal', 'elegant', 'minimalis', 'trendy', 'vintage'],
            'core_clothing': ['kemeja', 'shirt', 'dress', 'celana', 'pants', 'rok', 'blazer'],
            'fits_and_styles': ['oversized', 'fitted', 'loose', 'slim', 'relaxed', 'tailored', 'long sleeves', 'short sleeves'],
            'colors_and_materials': ['black', 'white', 'red', 'blue', 'hitam', 'putih', 'netral', 'neutral colors'],
            'occasions': ['work', 'casual', 'party', 'wedding', 'everyday', 'sehari-hari', 'casual everyday']
        }
    
    def _get_session_questions(self, session_id):
        """Get or initialize questions asked for this session - UPDATED with daily activity"""
        if session_id not in self.session_questions_asked:
            self.session_questions_asked[session_id] = {
                'gender_asked': False,
                'body_info_asked': False,
                'daily_activity_asked': False,  
                'style_asked': False,
                'clothing_asked': False,
                'fit_asked': False,
                'sleeves_asked': False,
                'colors_asked': False,
                'occasions_asked': False,
                'budget_asked': False,
                'summary_shown': False
            }
        return self.session_questions_asked[session_id]
    
    def _mark_question_asked(self, session_id, question_type):
        """Mark a question type as asked"""
        questions = self._get_session_questions(session_id)
        questions[question_type] = True
    
    def _detect_clothing_from_keywords(self, accumulated_keywords):
        """
        Detect clothing items from accumulated keywords
        """
        if not accumulated_keywords:
            return None
        
        # Use FashionCategories if available
        if hasattr(FashionCategories, 'CLOTHING_TERMS'):
            clothing_terms = FashionCategories.CLOTHING_TERMS
        else:
            # Fallback clothing terms
            clothing_terms = [
                'kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'tshirt',
                'sweater', 'cardigan', 'hoodie', 'tank top', 'crop top', 'polo shirt',
                'dress', 'gaun', 'terusan', 'maxi dress', 'mini dress', 'midi dress',
                'celana', 'pants', 'trousers', 'jeans', 'denim', 'shorts', 'leggings',
                'rok', 'skirt', 'culottes', 'palazzo',
                'blazer', 'jaket', 'jacket', 'coat', 'bomber jacket', 'denim jacket',
                'top', 'vest', 'rompi'
            ]
        
        detected_clothing = []
        
        # Check accumulated keywords for clothing items
        for keyword, data in accumulated_keywords.items():
            if keyword is None:
                continue
                
            weight = data.get("weight", 0) if isinstance(data, dict) else data
            keyword_lower = str(keyword).lower()
            
            # More flexible matching - exact match OR contains
            clothing_match = False
            for clothing_term in clothing_terms:
                if (clothing_term == keyword_lower or 
                    clothing_term in keyword_lower or 
                    keyword_lower in clothing_term):
                    clothing_match = True
                    break
            
            # If keyword matches clothing and has significant weight
            if clothing_match and weight > 1000:
                detected_clothing.append((keyword_lower, weight))
                print(f"🎯 DETECTED CLOTHING FROM KEYWORDS: '{keyword_lower}' (weight: {weight:.0f})")
        
        if detected_clothing:
            # Sort by weight and return the highest weighted clothing item
            detected_clothing.sort(key=lambda x: x[1], reverse=True)
            top_clothing = detected_clothing[0][0]
            print(f"✅ PRIMARY CLOTHING DETECTED: '{top_clothing}'")
            return top_clothing
        
        return None
    
    def _categorize_keywords(self, accumulated_keywords):
        """
        Better categorization using FashionCategories if available
        """
        known_info = {
            'gender': False,
            'body_info': False,
            'daily_activity': False,  
            'style': False,
            'clothing': False,
            'fit': False,
            'colors': False,
            'occasions': False,
            'sleeves': False,
            'budget': False
        }
        
        detected_preferences = []
        
        # Use FashionCategories if available, otherwise fallback
        if hasattr(FashionCategories, 'TERM_CATEGORIES'):
            term_categories = {
                'sleeve_terms': FashionCategories.SLEEVE_TERMS,
                'body_terms': FashionCategories.BODY_TERMS,
                'style_terms': FashionCategories.STYLE_TERMS,
                'clothing_terms': FashionCategories.CLOTHING_TERMS,
                'fit_terms': FashionCategories.FIT_TERMS,
                'color_terms': FashionCategories.COLOR_TERMS,
                'occasion_terms': FashionCategories.OCCASION_TERMS,
                'activity_terms': FashionCategories.ACTIVITY_TERMS
            }
        else:
            # Fallback term categories
            term_categories = {
                'sleeve_terms': [
                    'lengan panjang', 'lengan pendek', 'long sleeve', 'long sleeves',
                    'short sleeve', 'short sleeves', 'sleeveless', 'tanpa lengan',
                    '3/4 sleeve', '3/4 sleeves', 'quarter sleeve', 'quarter sleeves',
                    'cap sleeve', 'cap sleeves', 'bell sleeve', 'bell sleeves',
                    'puff sleeve', 'puff sleeves'
                ],
                'body_terms': [
                    'petite', 'mungil', 'tall', 'tinggi', 'short', 'pendek',
                    'pear shape', 'bentuk pir', 'apple shape', 'bentuk apel',
                    'hourglass', 'jam pasir', 'rectangle', 'persegi panjang',
                    'inverted triangle', 'athletic build', 'curvy', 'lekuk tubuh',
                    'slim build', 'average build', 'plus size', 'ukuran besar',
                    'cm', 'kg', 'height', 'weight', 'tinggi', 'berat', 'body type', 'pear'
                ],
                'style_terms': [
                    'casual', 'formal', 'elegant', 'minimalis', 'trendy', 'vintage', 
                    'classic', 'modern', 'chic', 'sophisticated', 'edgy', 'bohemian'
                ],
                'clothing_terms': [
                    'kemeja', 'shirt', 'blouse', 'dress', 'gaun', 'celana', 'pants', 
                    'rok', 'skirt', 'blazer', 'jaket', 'sweater', 'cardigan', 'hoodie',
                    'top', 'kaos', 't-shirt', 'tshirt'
                ],
                'fit_terms': [
                    'oversized', 'fitted', 'loose', 'slim', 'relaxed', 'tailored', 
                    'tight', 'baggy', 'regular fit', 'structured', 'flowy'
                ],
                'color_terms': [
                    'black', 'white', 'red', 'blue', 'hitam', 'putih', 'netral', 
                    'beige', 'navy', 'pink', 'warna', 'neutral', 'neutral colors',
                    'bright colors', 'pastel', 'pastels', 'earth tones'
                ],
                'occasion_terms': [
                    'work', 'casual', 'party', 'wedding', 'everyday', 'sehari-hari', 
                    'office', 'formal event', 'casual everyday', 'work office',
                    'special events', 'mixed occasions'
                ],
                'activity_terms': [
                    'office worker', 'student', 'teacher', 'work from home', 'remote work',
                    'travel', 'active lifestyle', 'gym', 'sports', 'outdoor activities',
                    'sedentary', 'desk job', 'standing job', 'retail', 'healthcare',
                    'business meetings', 'social events', 'freelancer', 'entrepreneur',
                    'stay at home', 'retired', 'part time', 'full time', 'shift work',
                    'pekerja kantoran', 'mahasiswa', 'guru', 'kerja dari rumah',
                    'sering traveling', 'aktif bergerak', 'olahraga', 'aktivitas outdoor',
                    'banyak duduk', 'kerja berdiri', 'sering meeting', 'acara sosial'
                ]
            }
        
        # Check accumulated keywords with more flexible matching
        if accumulated_keywords:
            for keyword, data in accumulated_keywords.items():
                if keyword is None:
                    continue
                    
                weight = data.get("weight", 0) if isinstance(data, dict) else data
                if weight > 300:  # Threshold for significant preferences
                    keyword_lower = str(keyword).lower()
                    
                    # More flexible matching - exact match OR contains
                    for category, terms in term_categories.items():
                        category_match = False
                        for term in terms:
                            if (term == keyword_lower or 
                                term in keyword_lower or 
                                keyword_lower in term):
                                category_match = True
                                break
                        
                        if category_match:
                            if category == 'body_terms':
                                known_info['body_info'] = True
                                detected_preferences.append(f"body: {keyword}")
                            elif category == 'activity_terms':  # NEW: Activity detection
                                known_info['daily_activity'] = True
                                detected_preferences.append(f"activity: {keyword}")
                            elif category == 'sleeve_terms':
                                known_info['sleeves'] = True
                                detected_preferences.append(f"sleeves: {keyword}")
                            elif category == 'style_terms':
                                known_info['style'] = True
                                detected_preferences.append(f"style: {keyword}")
                            elif category == 'clothing_terms':
                                known_info['clothing'] = True
                                detected_preferences.append(f"clothing: {keyword}")
                            elif category == 'fit_terms':
                                known_info['fit'] = True
                                detected_preferences.append(f"fit: {keyword}")
                            elif category == 'color_terms':
                                known_info['colors'] = True
                                detected_preferences.append(f"colors: {keyword}")
                            elif category == 'occasion_terms':
                                known_info['occasions'] = True
                                detected_preferences.append(f"occasions: {keyword}")
                            break  # Stop checking other categories once matched
        
        return known_info, detected_preferences
    
    def _all_essential_questions_asked(self, session_id):
        """UPDATED: Check if all essential questions including daily activity have been asked"""
        questions_asked = self._get_session_questions(session_id)
        
        essential_questions = [
            'gender_asked',
            'body_info_asked',
            'daily_activity_asked',  
            'style_asked',
            'clothing_asked',
            'fit_asked',
            'sleeves_asked',
            'colors_asked',
            'occasions_asked',
            'budget_asked'
        ]
        
        return all(questions_asked.get(q, False) for q in essential_questions)
    
    def should_reask_length_preferences(self, new_clothing_items, user_context):
        """
        Check if we should re-ask length preferences for new clothing types
        """
        
        # Get existing length preferences
        accumulated_keywords = user_context.get("accumulated_keywords", {})
        existing_length_prefs = []
        
        for keyword, data in accumulated_keywords.items():
            if keyword and FashionCategories.get_category(keyword) == 'length_terms':
                weight = data.get("weight", 0) if isinstance(data, dict) else data
                if weight > 1000:
                    existing_length_prefs.append(keyword)
        
        # Check if new clothing types need different length considerations
        clothing_length_relevance = {
            'celana': ['panjang', 'pendek', 'long', 'short', 'ankle length', 'knee length'],
            'pants': ['long', 'short', 'ankle length', 'knee length'],
            'rok': ['mini', 'midi', 'maxi', 'knee length', 'ankle length'],
            'skirt': ['mini', 'midi', 'maxi', 'knee length', 'ankle length'], 
            'dress': ['mini', 'midi', 'maxi', 'knee length', 'ankle length'],
            'gaun': ['mini', 'midi', 'maxi', 'knee length', 'ankle length'],
            'kemeja': ['long', 'short', 'panjang', 'pendek'],
            'shirt': ['long', 'short']
        }
        
        for clothing_item in new_clothing_items:
            relevant_lengths = clothing_length_relevance.get(clothing_item.lower(), [])
            
            # Check if existing length preferences are relevant for this clothing
            has_relevant_length = any(
                existing_length.lower() in [rl.lower() for rl in relevant_lengths]
                for existing_length in existing_length_prefs
            )
            
            if not has_relevant_length and relevant_lengths:
                print(f"   📏 Should re-ask length for '{clothing_item}' - no relevant existing preferences")
                return True, clothing_item
        
        return False, None
    
    async def analyze_consultation_state(self, user_input, user_context, session_id):
        """
        UPDATED: Analyze consultation state with daily activity question included
        """
        user_input_lower = user_input.lower().strip()
        
        # Get session tracking
        questions_asked = self._get_session_questions(session_id)
        
        # PRIORITY 1: Check for explicit product requests
        product_triggers = [
            'show me products', 'tampilkan produk', 'carikan produk', 'show products',
            'bisa saya lihat produk', 'lihat produk', 'rekomendasinya', 'rekomendasi',
            'yes show me', 'ya tunjukkan', 'show me some products',
            'ready for products', 'siap lihat produk', 'lanjut ke produk'
        ]
        
        for trigger in product_triggers:
            if trigger in user_input_lower:
                print(f"🎯 PRODUCT TRIGGER: '{trigger}' → Going to products")
                return "products"
        
        # PRIORITY 2: Check if user is responding to summary 
        if user_context.get("awaiting_confirmation", False):
            confirmation_type = user_context.get("confirmation_type", "summary")
            
            # More specific confirmation words
            summary_confirmation_words = [
                'this is correct', 'looks good', 'that\'s right', 'accurate', 
                'yes this is right', 'benar', 'tepat', 'sesuai', 'iya benar',
                'proceed', 'lanjut', 'looks accurate', 'this looks good', 
                'terlihat bagus', 'bagus', 'cocok', 'betul', 'oke', 'sudah benar'
            ]
            
            # Check for corrections/changes
            correction_indicators = [
                'change', 'correction', 'wrong', 'not right', 'actually', 
                'should be', 'i meant', 'correct that', 'ubah', 'salah',
                'seharusnya', 'maksudnya', 'bukan', 'koreksi'
            ]
            
            # If user wants to make corrections
            if any(indicator in user_input_lower for indicator in correction_indicators):
                print("✏️ User wants to make corrections → Stay in consultation")
                user_context["awaiting_confirmation"] = False
                return "consultation"
            
            # Check what type of confirmation this is
            elif any(word in user_input_lower for word in summary_confirmation_words) or user_input_lower in ['yes', 'ya', 'iya', 'ok', 'okay']:
                if confirmation_type == "summary":
                    print("✅ User confirmed summary → Going to style recommendations")
                    return "style_recommendations"
                elif confirmation_type == "style_recommendations":
                    print("✅ User confirmed style recommendations → Going to products")
                    return "products"
        
        # PRIORITY 3: Check if all questions including daily activity have been asked
        if self._all_essential_questions_asked(session_id) and not questions_asked.get('summary_shown', False):
            print("📋 All questions including daily activity asked → Auto-show summary")
            self._mark_question_asked(session_id, 'summary_shown')
            return "summary"
        
        # PRIORITY 4: Continue with consultation if more questions needed
        print("🔄 Continue consultation - more questions needed")
        return "consultation"
    
    async def generate_consultation_response(self, user_input, user_context, session_id):
        """
        UPDATED: Added daily activity question after body info
        """
        
        # Get current information
        gender_data = user_context.get("user_gender", {})
        gender = gender_data.get("category") if gender_data else None
        accumulated_keywords = user_context.get("accumulated_keywords", {})
        
        # Get session-specific questions tracking
        questions_asked = self._get_session_questions(session_id)
        
        # Categorize what we know
        known_info, detected_preferences = self._categorize_keywords(accumulated_keywords)
        known_info['gender'] = gender is not None and gender != "not specified"
        
        # Build acknowledgment for user's response
        acknowledgment = ""
        if detected_preferences and len(detected_preferences) > 0:
            acknowledgment = "Perfect! "
        
        # Handle correction requests
        correction_indicators = [
            'change', 'correction', 'wrong', 'not right', 'actually', 
            'should be', 'i meant', 'correct that', 'ubah', 'salah'
        ]
        
        if any(indicator in user_input.lower() for indicator in correction_indicators):
            return """I'd be happy to help you correct any information! What would you like to change?

You can say things like:
- "Change my style to casual" 
- "Actually I prefer fitted clothes"
- "My body type is hourglass"
- "I want long sleeves instead"
- "My budget is actually 200rb"
- "I'm a student, not an office worker"

What would you like to update?"""
        
        # Auto-detect clothing from keywords
        if not questions_asked.get('clothing_asked', False):
            detected_clothing = self._detect_clothing_from_keywords(accumulated_keywords)
            if detected_clothing:
                self._mark_question_asked(session_id, 'clothing_asked')
                print(f"✅ Auto-marked clothing_asked=True due to detected clothing: {detected_clothing}")
                known_info['clothing'] = True
        
        # ASK ALL QUESTIONS IN SEQUENCE - INCLUDING DAILY ACTIVITY
        
        # 1. GENDER (ALWAYS FIRST)
        if not known_info['gender'] and not questions_asked['gender_asked']:
            self._mark_question_asked(session_id, 'gender_asked')
            return "Hi! I'd love to help you find the perfect fashion items. First, could you let me know your gender? This helps me suggest the right fits and styles for you. (You can say 'male', 'female', 'man', 'woman', etc.)"
        
        # 2. BODY INFORMATION
        elif known_info['gender'] and not questions_asked['body_info_asked']:
            self._mark_question_asked(session_id, 'body_info_asked')
            return f"""{acknowledgment}Now to recommend the perfect fit, could you share some body information?

**Body Type** (if comfortable): 
- Pear shape (hips wider than shoulders)
- Apple shape (fuller midsection) 
- Hourglass (balanced shoulders and hips)
- Rectangle (similar measurements)
- Athletic build (broader shoulders)

**OR Height & Weight** (for size guidance):
- Example: "I'm 160cm, 55kg" or "I'm petite" or "I'm tall"

This helps me suggest clothes that will look amazing on you! 😊"""
        
        # 3. DAILY ACTIVITY (NEW - AFTER BODY INFO)
        elif not questions_asked['daily_activity_asked']:
            self._mark_question_asked(session_id, 'daily_activity_asked')
            return f"""{acknowledgment}What's your daily activity or lifestyle like?

- **Office worker** (desk job, business meetings)
- **Student** (classes, campus life)
- **Work from home** (remote work, video calls)
- **Active lifestyle** (gym, sports, outdoor activities)
- **Retail/Service** (standing job, customer interaction)
- **Healthcare/Teaching** (professional but active)
- **Freelancer/Creative** (flexible schedule, varied activities)
- **Stay at home** (household activities, errands)
- **Mixed activities** (combination of the above)

This helps me recommend clothes that suit your daily routine! 🏃‍♀️💼"""
        
        # 4. STYLE
        elif not questions_asked['style_asked']:
            self._mark_question_asked(session_id, 'style_asked')
            return f"{acknowledgment}What's your overall style preference?\n- **Casual** (everyday, comfortable)\n- **Formal** (professional, dressy)\n- **Elegant** (sophisticated, classy)\n- **Minimalist** (simple, clean lines)\n- **Trendy** (fashionable, current styles)\n- **Vintage** (classic, retro styles)"
        
        # 5. CLOTHING TYPE
        elif not questions_asked['clothing_asked']:
            self._mark_question_asked(session_id, 'clothing_asked')
            return f"{acknowledgment}What type of clothing are you most interested in?\n- **Shirts/Kemeja** (button-up, blouses)\n- **Dresses/Gaun** (casual, formal)\n- **Pants/Celana** (jeans, trousers)\n- **Skirts/Rok** (mini, midi, maxi)\n- **Blazers/Jackets** (formal, casual)\n- **Mixed** (various types)"
        
        # 6. FIT
        elif not questions_asked['fit_asked']:
            self._mark_question_asked(session_id, 'fit_asked')
            return f"{acknowledgment}How do you like your clothes to fit?\n- **Oversized** (loose, roomy)\n- **Fitted** (follows your body shape)\n- **Relaxed** (comfortable with some room)\n- **Slim** (tailored, not too tight)\n- **Loose** (flowing, not form-fitting)"
        
        # 7. SLEEVES
        elif not questions_asked['sleeves_asked']:
            self._mark_question_asked(session_id, 'sleeves_asked')
            return f"{acknowledgment}What sleeve length do you prefer?\n- **Long sleeves** (lengan panjang)\n- **Short sleeves** (lengan pendek)\n- **Sleeveless** (tank tops)\n- **3/4 sleeves** (between elbow and wrist)\n- **Mixed** (I like different lengths)"
        
        # 8. COLORS
        elif not questions_asked['colors_asked']:
            self._mark_question_asked(session_id, 'colors_asked')
            return f"{acknowledgment}What colors do you usually prefer?\n- **Neutral colors** (black, white, beige, navy)\n- **Bright colors** (red, blue, pink)\n- **Earth tones** (brown, green, khaki)\n- **Pastels** (light colors)\n- **Mixed** (I like various colors)"
        
        # 9. OCCASIONS
        elif not questions_asked['occasions_asked']:
            self._mark_question_asked(session_id, 'occasions_asked')
            return f"{acknowledgment}What occasions will you wear this for?\n- **Work/Office** (professional settings)\n- **Casual everyday** (shopping, meeting friends)\n- **Special events** (parties, dates)\n- **Mixed occasions** (various situations)"
        
        # 10. BUDGET (FINAL QUESTION)
        elif not questions_asked['budget_asked']:
            self._mark_question_asked(session_id, 'budget_asked')
            return f"""{acknowledgment}Finally, what's your budget range for clothing items?

- **Under 100rb** (budget-friendly options)
- **100rb - 300rb** (affordable range)
- **300rb - 500rb** (mid-range)
- **500rb - 1jt** (premium range)
- **Above 1jt** (luxury range)
- **No specific budget** (show me all options)

This helps me show you products within your comfortable price range! 💰"""
        
        # 11. ALL QUESTIONS ASKED - Trigger summary
        else:
            return "TRIGGER_SUMMARY"
                
# FIXED: Generate consultation summary (simpler and clearer)
async def generate_consultation_summary_llm(user_context, user_language, session_id):
    """
    ENHANCED: Summary generation with proper keyword categorization including daily activity
    """
    
    # Get the CURRENT gender from user_context
    gender_data = user_context.get("user_gender", {})
    current_gender = gender_data.get("category", "Not specified")
    
    # Log gender information for debugging
    print(f"\n📊 SUMMARY GENERATION - GENDER CHECK:")
    print(f"   Current gender in context: {current_gender}")
    print(f"   Gender data: {gender_data}")
    
    accumulated_keywords = user_context.get("accumulated_keywords", {})
    budget_range = user_context.get("budget_range", None)
    
    # Initialize preference lists INCLUDING daily activities
    style_preferences = []
    clothing_interests = []
    fit_preferences = []
    sleeve_preferences = []
    color_preferences = []
    occasion_needs = []
    body_info = []
    length_preferences = []
    neckline_preferences = []
    material_preferences = []
    daily_activities = []  # NEW: Daily activity list
    
    print(f"\n🎨 ENHANCED SUMMARY GENERATION WITH DAILY ACTIVITY:")
    print("=" * 60)
    
    # ENHANCED: Better keyword categorization logic including daily activities
    for keyword, data in accumulated_keywords.items():
        if keyword is None:
            continue
            
        weight = data.get("weight", 0) if isinstance(data, dict) else data
        source = data.get("source", "unknown") if isinstance(data, dict) else "unknown"
        
        # Use lower threshold for summary to capture more details
        if source == "consultation":
            threshold = 100
        elif source == "user_input":
            threshold = 500
        else:
            threshold = 1000
        
        print(f"🔍 Analyzing: '{keyword}' → Weight: {weight:.0f}, Source: {source}")
        
        if weight > threshold:
            keyword_lower = str(keyword).lower()
            
            # ENHANCED: More precise categorization logic including daily activities
            categorized = False
            
            # 1. BODY TERMS - Check first (most specific)
            if not categorized:
                for body_term in FashionCategories.BODY_TERMS:
                    if (body_term.lower() == keyword_lower or 
                        body_term.lower() in keyword_lower or 
                        keyword_lower in body_term.lower()):
                        body_info.append(keyword)
                        print(f"   ✅ BODY: '{keyword}' (matched: {body_term})")
                        categorized = True
                        break
            
            # 2. DAILY ACTIVITY TERMS - NEW: Check for activity terms
            if not categorized:
                for activity_term in FashionCategories.ACTIVITY_TERMS:
                    if (activity_term.lower() == keyword_lower or 
                        activity_term.lower() in keyword_lower or 
                        keyword_lower in activity_term.lower()):
                        daily_activities.append(keyword)
                        print(f"   ✅ DAILY ACTIVITY: '{keyword}' (matched: {activity_term})")
                        categorized = True
                        break
            
            # 3. SLEEVE TERMS
            if not categorized:
                for sleeve_term in FashionCategories.SLEEVE_TERMS:
                    if (sleeve_term.lower() == keyword_lower or 
                        sleeve_term.lower() in keyword_lower or 
                        keyword_lower in sleeve_term.lower()):
                        sleeve_preferences.append(keyword)
                        print(f"   ✅ SLEEVES: '{keyword}' (matched: {sleeve_term})")
                        categorized = True
                        break
            
            # 4. CLOTHING TERMS
            if not categorized:
                for clothing_term in FashionCategories.CLOTHING_TERMS:
                    if (clothing_term.lower() == keyword_lower or 
                        clothing_term.lower() in keyword_lower or 
                        keyword_lower in clothing_term.lower()):
                        clothing_interests.append(keyword)
                        print(f"   ✅ CLOTHING: '{keyword}' (matched: {clothing_term})")
                        categorized = True
                        break
            
            # 5. FIT TERMS
            if not categorized:
                for fit_term in FashionCategories.FIT_TERMS:
                    if (fit_term.lower() == keyword_lower or 
                        fit_term.lower() in keyword_lower or 
                        keyword_lower in fit_term.lower()):
                        fit_preferences.append(keyword)
                        print(f"   ✅ FIT: '{keyword}' (matched: {fit_term})")
                        categorized = True
                        break
            
            # 6. LENGTH TERMS
            if not categorized:
                for length_term in FashionCategories.LENGTH_TERMS:
                    if (length_term.lower() == keyword_lower or 
                        length_term.lower() in keyword_lower or 
                        keyword_lower in length_term.lower()):
                        length_preferences.append(keyword)
                        print(f"   ✅ LENGTH: '{keyword}' (matched: {length_term})")
                        categorized = True
                        break
            
            # 7. STYLE TERMS
            if not categorized:
                for style_term in FashionCategories.STYLE_TERMS:
                    if (style_term.lower() == keyword_lower or 
                        style_term.lower() in keyword_lower or 
                        keyword_lower in style_term.lower()):
                        style_preferences.append(keyword)
                        print(f"   ✅ STYLE: '{keyword}' (matched: {style_term})")
                        categorized = True
                        break
            
            # 8. OCCASION TERMS - Check before color terms (important!)
            if not categorized:
                for occasion_term in FashionCategories.OCCASION_TERMS:
                    if (occasion_term.lower() == keyword_lower or 
                        occasion_term.lower() in keyword_lower or 
                        keyword_lower in occasion_term.lower()):
                        occasion_needs.append(keyword)
                        print(f"   ✅ OCCASIONS: '{keyword}' (matched: {occasion_term})")
                        categorized = True
                        break
                
                # SPECIAL: Handle compound occasion terms
                if not categorized and ('occasion' in keyword_lower or 'event' in keyword_lower):
                    occasion_needs.append(keyword)
                    print(f"   ✅ OCCASIONS: '{keyword}' (compound occasion term)")
                    categorized = True
            
            # 9. COLOR TERMS - Check after occasions to avoid conflicts
            if not categorized:
                # First check specific color terms
                for color_term in FashionCategories.COLOR_TERMS:
                    if (color_term.lower() == keyword_lower or 
                        color_term.lower() in keyword_lower or 
                        keyword_lower in color_term.lower()):
                        # FIXED: Avoid occasion terms being categorized as colors
                        if 'occasion' not in keyword_lower and 'event' not in keyword_lower and 'activity' not in keyword_lower:
                            color_preferences.append(keyword)
                            print(f"   ✅ COLORS: '{keyword}' (matched: {color_term})")
                            categorized = True
                            break
                
                # Check color categories from mapping
                if not categorized and FashionCategories.is_color_category(keyword_lower):
                    color_preferences.append(keyword)
                    print(f"   ✅ COLORS: '{keyword}' (color category)")
                    categorized = True
            
            # 10. NECKLINE TERMS
            if not categorized:
                for neckline_term in FashionCategories.NECKLINE_TERMS:
                    if (neckline_term.lower() == keyword_lower or 
                        neckline_term.lower() in keyword_lower or 
                        keyword_lower in neckline_term.lower()):
                        neckline_preferences.append(keyword)
                        print(f"   ✅ NECKLINE: '{keyword}' (matched: {neckline_term})")
                        categorized = True
                        break
            
            # 11. MATERIAL TERMS
            if not categorized:
                for material_term in FashionCategories.MATERIAL_TERMS:
                    if (material_term.lower() == keyword_lower or 
                        material_term.lower() in keyword_lower or 
                        keyword_lower in material_term.lower()):
                        material_preferences.append(keyword)
                        print(f"   ✅ MATERIAL: '{keyword}' (matched: {material_term})")
                        categorized = True
                        break
            
            # If still not categorized, log it
            if not categorized:
                print(f"   ❓ UNCATEGORIZED: '{keyword}' - unable to classify")
        else:
            print(f"   ⏭️ BELOW THRESHOLD: '{keyword}' (weight: {weight})")
    
    # Clean and deduplicate preferences
    def clean_preferences(pref_list, max_items=3):
        """Remove duplicates and limit to max items"""
        seen = set()
        cleaned = []
        for item in pref_list:
            item_lower = item.lower()
            if item_lower not in seen and len(cleaned) < max_items:
                seen.add(item_lower)
                cleaned.append(item)
        return cleaned
    
    # Apply cleaning INCLUDING daily activities
    body_info = clean_preferences(body_info, 2)
    daily_activities = clean_preferences(daily_activities, 3)  # NEW: Clean daily activities
    sleeve_preferences = clean_preferences(sleeve_preferences, 2)
    style_preferences = clean_preferences(style_preferences, 3)
    clothing_interests = clean_preferences(clothing_interests, 3)
    fit_preferences = clean_preferences(fit_preferences, 2)
    color_preferences = clean_preferences(color_preferences, 3)
    occasion_needs = clean_preferences(occasion_needs, 3)
    length_preferences = clean_preferences(length_preferences, 2)
    neckline_preferences = clean_preferences(neckline_preferences, 2)
    material_preferences = clean_preferences(material_preferences, 2)
    
    # DEBUGGING: Print final categorized lists INCLUDING daily activities
    print(f"\n📋 FINAL CATEGORIZED PREFERENCES:")
    print(f"   Body Info: {body_info}")
    print(f"   Daily Activities: {daily_activities}")  # NEW: Debug daily activities
    print(f"   Style: {style_preferences}")
    print(f"   Clothing: {clothing_interests}")
    print(f"   Fit: {fit_preferences}")
    print(f"   Sleeves: {sleeve_preferences}")
    print(f"   Length: {length_preferences}")
    print(f"   Colors: {color_preferences}")
    print(f"   Occasions: {occasion_needs}")
    print(f"   Necklines: {neckline_preferences}")
    print(f"   Materials: {material_preferences}")
    
    # Format budget information
    budget_text = "Not specified"
    if budget_range:
        if isinstance(budget_range, tuple):
            min_price, max_price = budget_range
            if min_price and max_price:
                budget_text = f"IDR {min_price:,} - IDR {max_price:,}"
            elif max_price:
                budget_text = f"Under IDR {max_price:,}"
            elif min_price:
                budget_text = f"Above IDR {min_price:,}"
        elif budget_range == "NO_LIMIT":
            budget_text = "No specific budget limit"
    
    # Create summary with proper categorization INCLUDING daily activities
    print(f"📊 FINAL GENDER FOR SUMMARY: '{current_gender}'")
    
    summary = f"""**YOUR FASHION PROFILE** ✨

Based on our conversation, here's your complete style profile:

**Personal Information:**
- **Gender:** {current_gender.title() if current_gender != "Not specified" else "Not specified"}
- **Body Information:** {', '.join(body_info) if body_info else 'Not specified'}
- **Daily Activity/Lifestyle:** {', '.join(daily_activities) if daily_activities else 'Not specified'}
- **Budget Range:** {budget_text}

**Style Preferences:**
- **Overall Style:** {', '.join(style_preferences) if style_preferences else 'Not specified'}
- **Clothing Types:** {', '.join(clothing_interests) if clothing_interests else 'Not specified'}
- **Preferred Fit:** {', '.join(fit_preferences) if fit_preferences else 'Not specified'}
- **Sleeve Preferences:** {', '.join(sleeve_preferences) if sleeve_preferences else 'Not specified'}
- **Length Preferences:** {', '.join(length_preferences) if length_preferences else 'Not specified'}
- **Color Preferences:** {', '.join(color_preferences) if color_preferences else 'Not specified'}
- **Occasions:** {', '.join(occasion_needs) if occasion_needs else 'Not specified'}

---

**WHAT'S NEXT:**

🔧 **To make corrections:** Say "Change [item]" or "Actually I prefer [preference]" or "I am [gender]" or "I work as [job]"

✅ **If this looks accurate:** Say "This looks good" or "Show me products" to see your personalized recommendations within your budget!

I'm ready to find products that perfectly match your style, body type, daily activities, and budget! 🎯💰"""
    
    # Translate if needed
    if user_language != "en":
        try:
            summary = translate_text(summary, user_language, session_id)
        except Exception as e:
            print(f"Translation error: {e}")
    
    return summary

# FIXED: Handle message function with better logic
async def handle_message_with_hybrid_intelligence(user_input, session_id, user_context, db, user_language):
    """
    ENHANCED: Better gender change handling in message processing
    """
    
    print(f"\n🌍 MESSAGE PROCESSING WITH GENDER CHANGE DETECTION")
    print("="*60)
    print(f"   User language detected: '{user_language}'")
    print(f"   User input: '{user_input[:100]}...'")
    
    # Always translate first
    if user_language != "en":
        print(f"🔄 Translating user input from '{user_language}' to English...")
        translated_input = translate_text(user_input, "en", session_id)
        print(f"   Translated input: '{translated_input[:100]}...'")
    else:
        translated_input = user_input
        print(f"✅ Input already in English, no translation needed")

    # PRIORITY: Check for gender changes FIRST before other processing
    old_gender = user_context.get("user_gender", {}).get("category", None)
    detected_gender = detect_and_update_gender(translated_input, user_context)
    new_gender = user_context.get("user_gender", {}).get("category", None)
    
    # Check if gender actually changed
    if old_gender != new_gender and new_gender is not None:
        print(f"🔄 GENDER CHANGE DETECTED: {old_gender} → {new_gender}")
        
        # Reset certain consultation questions if gender changed
        questions_asked = consultation_manager._get_session_questions(session_id)
        questions_asked['summary_shown'] = False  # Allow new summary
        
        # Optional: Reset body info question to get gender-appropriate guidance
        if old_gender is not None:  # Only reset if this was a change, not first detection
            questions_asked['body_info_asked'] = False
            print(f"   🔄 Reset body_info_asked due to gender change")
        
        # Create acknowledgment message for gender change
        if old_gender is not None:
            if user_language != "en":
                gender_change_msg = translate_text(
                    f"Thank you for the correction! I've updated your gender to {new_gender}. This helps me provide better fashion recommendations for you.",
                    user_language, session_id
                )
            else:
                gender_change_msg = f"Thank you for the correction! I've updated your gender to {new_gender}. This helps me provide better fashion recommendations for you."
            
            return gender_change_msg, False

    # Check for clothing changes
    is_clothing_change, new_clothing_list = detect_clothing_change_in_input(user_input, user_context)

    if is_clothing_change and new_clothing_list:
        print(f"🔄 CLOTHING CHANGE DETECTED: {new_clothing_list}")
        
        # Handle clothing changes (existing logic)
        user_input_lower = user_input.lower()
        is_only_request = any(only_word in user_input_lower for only_word in 
                             ['only', 'just', 'hanya', 'cuma', 'aja', 'saja'])
        
        accumulated_keywords = user_context.get("accumulated_keywords", {})
        
        if is_only_request:
            print(f"   🎯 ONLY REQUEST: Replacing ALL clothing with {new_clothing_list}")
            clothing_items_to_remove = []
            for keyword, data in list(accumulated_keywords.items()):
                if keyword and FashionCategories.get_category(keyword) == 'clothing_terms':
                    clothing_items_to_remove.append(keyword)
            
            for old_clothing in clothing_items_to_remove:
                if old_clothing in accumulated_keywords:
                    del accumulated_keywords[old_clothing]
                    print(f"   🗑️ REMOVED OLD CLOTHING: '{old_clothing}' (only request)")
        else:
            print(f"   ➕ ADDITION REQUEST: Adding {new_clothing_list} to existing")
            clothing_items_to_remove = []
            for keyword, data in list(accumulated_keywords.items()):
                if keyword and FashionCategories.get_category(keyword) == 'clothing_terms':
                    if not any(keyword.lower() == new_item.lower() for new_item in new_clothing_list):
                        clothing_items_to_remove.append(keyword)
            
            for old_clothing in clothing_items_to_remove:
                if old_clothing in accumulated_keywords:
                    del accumulated_keywords[old_clothing]
                    print(f"   🗑️ REMOVED OLD CLOTHING: '{old_clothing}' (not in new list)")
        
        user_context["accumulated_keywords"] = accumulated_keywords
        
        # Add new clothing items
        for new_clothing in new_clothing_list:
            new_clothing_keywords = [(new_clothing, 50000)]
            update_accumulated_keywords(
                new_clothing_keywords, 
                user_context, 
                user_input,
                is_user_input=True, 
                is_consultation=True
            )
            print(f"   ✅ ADDED NEW CLOTHING: '{new_clothing}'")
        
        # Reset summary
        questions_asked = consultation_manager._get_session_questions(session_id)
        questions_asked['summary_shown'] = False
        
        print(f"   ✅ Updated clothing list - continuing with consultation")
    
    # Extract and update keywords
    accumulated_keywords = [(k, v.get("weight", 0) if isinstance(v, dict) else v) 
                          for k, v in user_context.get("accumulated_keywords", {}).items()]
    
    # Mark consultation phase keywords specifically
    is_in_consultation = not consultation_manager._all_essential_questions_asked(session_id)
    
    new_keywords = await extract_ranked_keywords("", translated_input, accumulated_keywords)
    
    print(f"🔄 EXTRACTED KEYWORDS: {[(kw, score) for kw, score in new_keywords[:5]]}")
    
    # Mark as consultation keywords if we're still in consultation phase
    update_accumulated_keywords(new_keywords, user_context, user_input, 
                              is_user_input=True, is_consultation=is_in_consultation)
    
    # Analyze consultation state
    consultation_state = await consultation_manager.analyze_consultation_state(
        user_input, user_context, session_id
    )
    
    print(f"🧠 Consultation state: {consultation_state}")
    print(f"🔍 User input: '{user_input}'")
    print(f"📋 Awaiting confirmation: {user_context.get('awaiting_confirmation', False)}")
    
    # Handle consultation responses
    if consultation_state == "consultation":
        consultation_response = await consultation_manager.generate_consultation_response(
            user_input, user_context, session_id
        )
        
        if consultation_response == "TRIGGER_SUMMARY":
            summary_response = await generate_consultation_summary_llm(user_context, user_language, session_id)
            user_context["awaiting_confirmation"] = True
            user_context["confirmation_type"] = "summary"
            
            summary_html = render_markdown(summary_response)
            return summary_html, False
        
        if user_language != "en":
            print(f"🔄 Translating consultation response from English to '{user_language}'...")
            try:
                translated_consultation = translate_text(consultation_response, user_language, session_id)
                print(f"✅ Consultation translation successful")
                consultation_response = translated_consultation
            except Exception as e:
                print(f"❌ Consultation translation failed: {e}")
        
        consultation_html = render_markdown(consultation_response)
        return consultation_html, False
    
    elif consultation_state == "style_recommendations":
        print("🎨 STYLE RECOMMENDATIONS STATE: Generating AI style advice")
        
        style_response = await generate_style_recommendations_from_consultation(user_context, user_language, session_id)
        
        style_keywords = await extract_ranked_keywords(style_response, "", [])
        update_accumulated_keywords(style_keywords, user_context, user_input, 
                                  is_ai_response=True, is_consultation=False)
        
        user_context["confirmation_type"] = "style_recommendations"
        user_context["awaiting_confirmation"] = True
        
        style_html = render_markdown(style_response)
        return style_html, False
    
    elif consultation_state == "summary":
        summary_response = await generate_consultation_summary_llm(user_context, user_language, session_id)
        user_context["confirmation_type"] = "summary"
        user_context["awaiting_confirmation"] = True
        
        summary_html = render_markdown(summary_response)
        return summary_html, False
    
    elif consultation_state == "products":
        user_context["awaiting_confirmation"] = False
        print("🛍️ PRODUCTS STATE: Triggering product search with consultation priority")
        return "TRIGGER_PRODUCT_SEARCH", True
    
    else:
        # Fallback to consultation
        consultation_response = await consultation_manager.generate_consultation_response(
            user_input, user_context, session_id
        )
        
        if user_language != "en":
            try:
                translated_consultation = translate_text(consultation_response, user_language, session_id)
                consultation_response = translated_consultation
            except Exception as e:
                print(f"❌ Fallback consultation translation failed: {e}")
        
        consultation_html = render_markdown(consultation_response)
        return consultation_html, False
                            
def get_enhanced_keywords_for_product_search(user_context):
    """
    NEW: Get keywords prioritized for product search with consultation preferences boosted
    """
    accumulated_keywords = user_context.get("accumulated_keywords", {})
    
    enhanced_keywords = []
    
    for keyword, data in accumulated_keywords.items():
        if isinstance(data, dict):
            base_weight = data.get("weight", 0)
            is_consultation = data.get("consultation_priority", False)
            source = data.get("source", "unknown")
            
            # MASSIVE boost for consultation keywords
            if is_consultation or source == "consultation":
                final_weight = base_weight * 5.0  # 5x boost for consultation
                print(f"   🔥 CONSULTATION BOOST: '{keyword}' → {final_weight:.0f} (was {base_weight:.0f})")
            else:
                final_weight = base_weight
            
            enhanced_keywords.append((keyword, final_weight))
        else:
            # Handle legacy format
            enhanced_keywords.append((keyword, data))
    
    # Sort by enhanced weights
    enhanced_keywords.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 ENHANCED KEYWORDS FOR PRODUCT SEARCH:")
    for i, (kw, weight) in enumerate(enhanced_keywords[:15]):
        source_info = accumulated_keywords.get(kw, {})
        if isinstance(source_info, dict):
            source = source_info.get("source", "unknown")
            is_consultation = source_info.get("consultation_priority", False)
            priority_marker = "🔥" if is_consultation else "⚪"
        else:
            priority_marker = "⚪"
            source = "legacy"
        
        print(f"   {i+1:2d}. {priority_marker} '{kw}' → {weight:.0f} ({source})")
    
    return enhanced_keywords[:15]

async def enhanced_product_search_in_websocket(user_context, db, user_language, session_id):
    """
    ENHANCED: Product search with consultation keyword prioritization
    """
    try:
        # Get enhanced keywords with consultation boost
        enhanced_keywords = get_enhanced_keywords_for_product_search(user_context)
        
        # Translate keywords if needed
        if user_language != "en":
            translated_keywords = []
            for kw, score in enhanced_keywords:
                try:
                    translated_kw = translate_text(kw, "en", session_id)
                    translated_keywords.append((translated_kw, score))
                except:
                    translated_keywords.append((kw, score))
        else:
            translated_keywords = enhanced_keywords

        # Get user gender and budget for filtering
        user_gender = user_context.get("user_gender", {}).get("category", None)
        budget_range = user_context.get("budget_range", None)
        
        # Enhanced confirmation message
        positive_response = "🧠 **Enhanced AI Search Results** - Here are products that perfectly match your consultation preferences and style recommendations:"
        
        if budget_range:
            min_price, max_price = budget_range
            if min_price and max_price:
                budget_text = f" (within your budget of IDR {min_price:,} - IDR {max_price:,})"
            elif max_price:
                budget_text = f" (under IDR {max_price:,})"
            else:
                budget_text = ""
            positive_response += budget_text

        if user_language != "en":
            positive_response = translate_text(positive_response, user_language, session_id)
        
        # Use enhanced product search with consultation-boosted keywords
        recommended_products = await fetch_products_from_db(
            db=db,
            top_keywords=translated_keywords,
            max_results=15,
            gender_category=user_gender,
            budget_range=budget_range
        )
        
        return recommended_products, positive_response
        
    except Exception as e:
        logging.error(f"Error in enhanced product search: {str(e)}")
        return pd.DataFrame(), "I'm sorry, I couldn't fetch product recommendations."
                
# Make sure initialization is correct
hybrid_extractor = HybridKeywordExtractor(openai)
consultation_manager = SmartConsultationManager(openai, hybrid_extractor)

# ================================
# EXISTING CLASSES
# ================================

class FashionSemanticSystem:
    """
    Enhanced semantic understanding system for fashion recommendations
    Integrates with your existing keyword system
    """
    
    def __init__(self):
        self.model = None
        self.product_embeddings = {}
        self.product_cache = {}
        self.embedding_cache = {}
        self.cultural_context = self._build_indonesian_fashion_context()
        self.fashion_vocabulary = self._build_fashion_vocabulary()
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the best available model for fashion understanding"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Try multilingual model first (supports Indonesian)
                self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                print("✅ Loaded multilingual sentence transformer for Indonesian support")
                return
            except Exception as e:
                print(f"⚠️ Failed to load multilingual model: {e}")
                try:
                    # Fallback to English model
                    self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    print("✅ Loaded English sentence transformer")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load sentence transformer: {e}")
        
        print("⚠️ Using fallback TF-IDF approach")
        self.model = None
    
    def _build_indonesian_fashion_context(self):
        """Build Indonesian-specific fashion context"""
        return {
            'traditional_wear': {
                'batik': ['traditional', 'cultural', 'patterned', 'formal', 'Indonesian'],
                'kebaya': ['traditional', 'formal', 'elegant', 'cultural', 'Indonesian'],
                'sarong': ['traditional', 'casual', 'comfortable', 'wrap']
            },
            'climate_appropriate': {
                'tropical': ['lightweight', 'breathable', 'cotton', 'linen', 'airy'],
                'humid': ['moisture-wicking', 'quick-dry', 'loose-fit', 'ventilated'],
                'hot_weather': ['light colors', 'short sleeves', 'sun protection', 'UV resistant']
            },
            'body_types': {
                'petite': ['asian fit', 'small frame', 'proportional', 'fitted'],
                'average': ['standard fit', 'regular', 'balanced'],
                'curvy': ['flattering cut', 'comfortable fit', 'accentuating']
            },
            'local_preferences': {
                'modest': ['covered', 'conservative', 'appropriate', 'respectful'],
                'hijab_friendly': ['loose fit', 'long sleeves', 'high neck', 'modest'],
                'work_appropriate': ['professional', 'modest', 'formal', 'office-suitable']
            }
        }
    
    def _build_fashion_vocabulary(self):
        """Enhanced Indonesian-English fashion vocabulary mapping"""
        return {
            # Clothing types
            'kaos': ['shirt', 't-shirt', 'top', 'tee', 'casual shirt'],
            'kemeja': ['shirt', 'blouse', 'dress shirt', 'button-up', 'formal shirt'],
            'celana': ['pants', 'trousers', 'bottoms', 'slacks'],
            'jaket': ['jacket', 'blazer', 'coat', 'outerwear'],
            'gaun': ['dress', 'gown', 'frock'],
            'rok': ['skirt', 'bottom'],
            'sepatu': ['shoes', 'footwear', 'sneakers', 'boots'],
            
            # Styles
            'kasual': ['casual', 'relaxed', 'informal', 'everyday'],
            'formal': ['formal', 'business', 'professional', 'dressy'],
            'santai': ['comfortable', 'leisure', 'relaxed', 'easy-going'],
            'elegan': ['elegant', 'sophisticated', 'classy', 'refined'],
            'tradisional': ['traditional', 'cultural', 'ethnic', 'heritage'],
            'modern': ['modern', 'contemporary', 'current', 'trendy'],
            
            # Colors
            'hitam': ['black', 'dark'],
            'putih': ['white', 'cream', 'ivory'],
            'merah': ['red', 'crimson', 'scarlet'],
            'biru': ['blue', 'navy', 'azure'],
            'hijau': ['green', 'emerald', 'olive'],
            'kuning': ['yellow', 'gold', 'amber'],
            
            # Fits and styles
            'longgar': ['loose', 'oversized', 'relaxed fit', 'baggy'],
            'ketat': ['tight', 'fitted', 'slim', 'body-hugging'],
            'pas': ['perfect fit', 'just right', 'well-fitted'],
        }
    
    def preprocess_fashion_text(self, text: str, user_profile: Optional[Dict] = None) -> str:
        """
        CLEANER preprocessing - don't over-expand for semantic search
        """
        if not text:
            return ""
        
        # Keep the original text clean
        enhanced_text = text.lower().strip()
        
        # Only add direct translations, not expansions
        main_translations = {
            'blazer': 'blazer jas',
            'jas': 'blazer jas', 
            'kemeja': 'kemeja shirt',
            'shirt': 'kemeja shirt',
            'celana': 'celana pants',
            'pants': 'celana pants',
            'rok': 'rok skirt',
            'skirt': 'rok skirt',
            'dress': 'dress gaun',
            'gaun': 'dress gaun'
        }
        
        # Check if text contains any main clothing items
        for indonesian, translation in main_translations.items():
            if indonesian in enhanced_text:
                enhanced_text = translation
                break
        
        # DON'T add cultural context for semantic search - keep it focused
        print(f"   🧹 CLEAN PREPROCESSING: '{text}' → '{enhanced_text}'")
        return enhanced_text

    def get_semantic_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Get semantic embedding for text with caching"""
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        if self.model and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use sentence transformer
            embedding = self.model.encode([text], convert_to_tensor=False)[0]
        elif SKLEARN_AVAILABLE:
            # Fallback to TF-IDF
            if not hasattr(self, 'tfidf_vectorizer'):
                self.tfidf_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
                # Initialize with some fashion terms
                sample_texts = [
                    "casual shirt comfortable style",
                    "formal dress elegant evening",
                    "traditional batik indonesian culture",
                    "modern minimalist design simple"
                ]
                self.tfidf_vectorizer.fit(sample_texts)
            
            embedding = self.tfidf_vectorizer.transform([text]).toarray()[0]
        else:
            # Very basic fallback - word overlap
            words = set(text.lower().split())
            fashion_terms = set(['shirt', 'dress', 'pants', 'casual', 'formal', 'style'])
            overlap = len(words.intersection(fashion_terms))
            embedding = np.array([overlap / max(len(words), 1)])
        
        if use_cache:
            self.embedding_cache[text] = embedding
        
        return embedding
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        emb1 = self.get_semantic_embedding(text1)
        emb2 = self.get_semantic_embedding(text2)
        
        if len(emb1) != len(emb2):
            return 0.0
        
        # Calculate cosine similarity
        if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

# Initialize the semantic system
semantic_system = FashionSemanticSystem()

class EnhancedSemanticProductMatcher:
    """
    Improved version with persistent embedding storage
    """
    
    def __init__(self, semantic_system: FashionSemanticSystem):
        self.semantic_system = semantic_system
        self.product_embeddings = {}
        self.products_df = None
        self.embeddings_cache_dir = "embeddings_cache"
        self.embeddings_file = os.path.join(self.embeddings_cache_dir, "product_embeddings.pkl")
        self.products_file = os.path.join(self.embeddings_cache_dir, "products_data.pkl")
        self.metadata_file = os.path.join(self.embeddings_cache_dir, "metadata.pkl")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.embeddings_cache_dir, exist_ok=True)

    def clean_semantic_query(self, query):
        """
        Clean semantic query from noise
        """
        # Remove numbers and numbered items
        cleaned = re.sub(r'\b\d+\b', '', query)
        
        # Remove action words that don't help matching
        noise_words = ['carikan', 'tunjukkan', 'show', 'find', 'cari', 'aja', 'saja', 'only']
        words = cleaned.split()
        
        filtered_words = []
        for word in words:
            if word.lower() not in noise_words and len(word) > 1:
                filtered_words.append(word)
        
        # Keep only the first 3-4 relevant words to avoid confusion
        result = ' '.join(filtered_words[:4])
        return result.strip()

    def matches_main_category(self, query, product):
        """
        Check if product matches the main category in query
        """
        query_lower = query.lower()
        product_name_lower = product['product_name'].lower()
        product_detail_lower = product['product_detail'].lower()
        
        # Define category matching rules
        category_rules = {
            'blazer': ['blazer', 'jas'],
            'jas': ['blazer', 'jas'],
            'shirt': ['shirt', 'kemeja'],
            'kemeja': ['shirt', 'kemeja'],  
            'pants': ['pants', 'celana', 'trouser'],
            'celana': ['pants', 'celana', 'trouser'],
            'skirt': ['skirt', 'rok'],
            'rok': ['skirt', 'rok'],
            'dress': ['dress', 'gaun'],
            'gaun': ['dress', 'gaun']
        }
        
        for main_term, valid_terms in category_rules.items():
            if main_term in query_lower:
                # Product must contain at least one valid term
                if any(term in product_name_lower or term in product_detail_lower 
                    for term in valid_terms):
                    return True
                else:
                    return False  # Wrong category
        
        return True  # No specific category detected, allow all

    async def preprocess_products(self, db: AsyncSession, force_refresh=False):
        """
        Create enhanced product descriptions for embedding with intelligent caching
        """
        print("\n🔄 SMART EMBEDDING PREPROCESSING")
        print("="*50)
        
        try:
            # Get current products from database
            variant_subquery = (
                select(
                    ProductVariant.product_id,
                    func.min(ProductVariant.product_price).label('min_price'),
                    func.group_concat(ProductVariant.size.distinct()).label('available_sizes'),
                    func.group_concat(ProductVariant.color.distinct()).label('available_colors'),
                    func.sum(ProductVariant.stock).label('total_stock')
                )
                .where(ProductVariant.stock > 0)
                .group_by(ProductVariant.product_id)
                .subquery()
            )
            
            query = (
                select(
                    Product.product_id, 
                    Product.product_name, 
                    Product.product_detail, 
                    Product.product_seourl,
                    Product.product_gender,
                    variant_subquery.c.min_price,
                    variant_subquery.c.available_sizes,
                    variant_subquery.c.available_colors,
                    variant_subquery.c.total_stock,
                    ProductPhoto.productphoto_path
                )
                .select_from(Product)
                .join(variant_subquery, Product.product_id == variant_subquery.c.product_id)
                .join(ProductPhoto, Product.product_id == ProductPhoto.product_id)
                .where(variant_subquery.c.total_stock > 0)
            )
            
            result = await db.execute(query)
            all_products = result.fetchall()
            
            if not all_products:
                print("❌ No products found for preprocessing")
                return
            
            print(f"📦 Found {len(all_products)} products in database")
            
            # Create DataFrame for products
            product_data = []
            for product_row in all_products:
                product_data.append({
                    "product_id": product_row[0],
                    "product_name": product_row[1],
                    "product_detail": product_row[2],
                    "price": product_row[5],
                    "gender": product_row[4],
                    "sizes": product_row[6],
                    "colors": product_row[7],
                    "stock": product_row[8],
                    "photo": product_row[9],
                    "seourl": product_row[3]
                })
            
            self.products_df = pd.DataFrame(product_data)
            print(f"📋 Created product DataFrame with {len(self.products_df)} products")
            
        except Exception as e:
            logging.error(f"Error in preprocess_products: {str(e)}")
            print(f"❌ Error preprocessing products: {str(e)}")

# Initialize enhanced product matcher
enhanced_matcher = EnhancedSemanticProductMatcher(semantic_system)

# ================================
# DATABASE MODELS (UNCHANGED)
# ================================

class Product(Base):
    __tablename__ = "product"
    product_id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False)
    product_detail = Column(String(255), nullable=False)
    product_price = Column(Float, nullable=False)
    product_seourl = Column(String(255), nullable=False)
    product_gender = Column(String(255), nullable=False)

class ProductPhoto(Base):  
    __tablename__ = "product_photo"
    productphoto_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, nullable=False)
    productphoto_path = Column(String(255), nullable=False)
    productphoto_order = Column(Integer, nullable=False)

class ProductVariant(Base):
    __tablename__ = "product_variant"
    variant_id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False)
    size = Column(String(50), nullable=False)
    color = Column(String(50), nullable=False)
    stock = Column(Integer, nullable=False)
    product_price = Column(Float, nullable=False)

class ChatHistoryDB(Base):
    __tablename__ = "chat_history"
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False)
    message_type = Column(String(20), nullable=False)
    content = Column(String(2000), nullable=False)
    timestamp = Column(DateTime, server_default=func.now())

class ChatMessage(BaseModel):
    session_id: str
    message_type: str
    content: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]

# ================================
# UTILITY FUNCTIONS (KEEP EXISTING)
# ================================

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSession(engine) as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], engine=model)['data'][0]['embedding']

def cosine_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@Language.factory("language_detector")
def get_language_detector(nlp, name):
    return LanguageDetector()

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import sys
        print("Please install 'en_core_web_sm' or 'en_core_web_lg' spacy model.")
        sys.exit()

if not spacy.tokens.Doc.has_extension("language"):
    spacy.tokens.Doc.set_extension("language", default={})

nlp.add_pipe("language_detector", last=True)

def get_mymemory_language_code(detected_language):
    """
    Map detected language codes to MyMemoryTranslator supported formats
    """
    language_mapping = {
        # Common language codes to MyMemoryTranslator format
        'id': 'indonesian',      # Indonesian
        'en': 'english',         # English  
        'es': 'spanish',         # Spanish
        'fr': 'french',          # French
        'de': 'german',          # German
        'it': 'italian',         # Italian
        'pt': 'portuguese',      # Portuguese
        'ru': 'russian',         # Russian
        'ja': 'japanese',        # Japanese
        'ko': 'korean',          # Korean
        'zh': 'chinese simplified',  # Chinese
        'ar': 'arabic',          # Arabic
        'hi': 'hindi',           # Hindi
        'th': 'thai',            # Thai
        'vi': 'vietnamese',      # Vietnamese
        'ms': 'malay',           # Malay
        'tl': 'tagalog',         # Tagalog/Filipino
        'nl': 'dutch',           # Dutch
        'sv': 'swedish',         # Swedish
        'da': 'danish',          # Danish
        'no': 'norwegian bokmål', # Norwegian
        'fi': 'finnish',         # Finnish
        'pl': 'polish',          # Polish
        'cs': 'czech',           # Czech
        'sk': 'slovak',          # Slovak
        'hu': 'hungarian',       # Hungarian
        'ro': 'romanian',        # Romanian
        'bg': 'bulgarian',       # Bulgarian
        'hr': 'croatian',        # Croatian
        'sr': 'serbian latin',   # Serbian
        'sl': 'slovenian',       # Slovenian
        'et': 'estonian',        # Estonian
        'lv': 'latvian',         # Latvian
        'lt': 'lithuanian',      # Lithuanian
        'el': 'greek',           # Greek
        'tr': 'turkish',         # Turkish
        'he': 'hebrew',          # Hebrew
        'fa': 'persian',         # Persian
        'ur': 'urdu',            # Urdu
        'bn': 'bengali',         # Bengali
        'ta': 'tamil india',     # Tamil
        'te': 'telugu',          # Telugu
        'ml': 'malayalam',       # Malayalam
        'kn': 'kannada',         # Kannada
        'gu': 'gujarati',        # Gujarati
        'pa': 'punjabi',         # Punjabi
        'mr': 'marathi',         # Marathi
        'ne': 'nepali',          # Nepali
        'si': 'sinhala',         # Sinhala
        'my': 'burmese',         # Burmese
        'km': 'khmer',           # Khmer
        'lo': 'lao',             # Lao
        'ka': 'georgian',        # Georgian
        'hy': 'armenian',        # Armenian
        'az': 'azerbaijani',     # Azerbaijani
        'kk': 'kazakh',          # Kazakh
        'ky': 'kyrgyz',          # Kyrgyz
        'uz': 'uzbek',           # Uzbek
        'tg': 'tajik',           # Tajik
        'mn': 'mongolian',       # Mongolian
        'bo': 'tibetan',         # Tibetan
        'ug': 'uyghur ug',       # Uyghur
        'am': 'amharic',         # Amharic
        'om': 'west central oromo',  # Oromo
        'ti': 'tigrinya',        # Tigrinya
        'so': 'somali',          # Somali
        'sw': 'swahili',         # Swahili
        'af': 'afrikaans',       # Afrikaans
    }
    
    return language_mapping.get(detected_language.lower(), 'english')  # Default to English

def translate_text(text, target_language, session_id=None):
    """
    FIXED: Enhanced translation with proper language mapping and error handling
    """
    try:
        if not text or not text.strip():
            print("⚠️ Translation skipped: Empty text")
            return text
        
        # Detect source language
        if session_id and session_id in session_manager.session_languages:
            source_language_code = session_manager.session_languages[session_id]
        else:
            source_language_code = detect_language(text)
            if session_id:
                session_manager.session_languages[session_id] = source_language_code
        
        print(f"🔍 Language detection: '{source_language_code}' → target: '{target_language}'")
        
        # Skip translation if same language
        if source_language_code == target_language:
            print("✅ Same language detected, skipping translation")
            return text
        
        # Map language codes to MyMemoryTranslator format
        source_lang_name = get_mymemory_language_code(source_language_code)
        target_lang_name = get_mymemory_language_code(target_language)
        
        print(f"🗺️ Language mapping: '{source_language_code}' → '{source_lang_name}', '{target_language}' → '{target_lang_name}'")
        
        # Attempt translation
        translator = MyMemoryTranslator(source=source_lang_name, target=target_lang_name)
        translated_text = translator.translate(text)
        
        print(f"✅ Translation successful: '{text[:50]}...' → '{translated_text[:50]}...'")
        return translated_text

    except Exception as e:
        print(f"❌ Translation error: {e}")
        print(f"   Source: '{source_language_code}' → Target: '{target_language}'")
        print(f"   Mapped: '{source_lang_name}' → '{target_lang_name}'")
        print(f"   Returning original text: '{text[:100]}...'")
        return text  # Return original text if translation fails

def generate_length_question_for_clothing(clothing_item, user_language="en"):
    """
    Generate specific length questions based on clothing type
    """
    
    clothing_lower = clothing_item.lower()
    
    if clothing_lower in ['celana', 'pants', 'trousers']:
        question = f"""Perfect! I see you're interested in **{clothing_item}**. 

What length do you prefer for pants?
- **Long/Full length** (ankle length)
- **Short** (above knee, like shorts)
- **Capri** (mid-calf length)
- **Mixed** (I like different lengths)"""
        
    elif clothing_lower in ['rok', 'skirt']:
        question = f"""Great choice! For **{clothing_item}**, what length do you prefer?
- **Mini** (above knee, shorter)
- **Midi** (knee to mid-calf)
- **Maxi** (ankle length, longer)
- **Mixed** (I like different lengths)"""
        
    elif clothing_lower in ['dress', 'gaun']:
        question = f"""Wonderful! For **{clothing_item}**, what length works best for you?
- **Mini dress** (above knee)
- **Midi dress** (knee to mid-calf) 
- **Maxi dress** (ankle length)
- **Mixed** (I like different lengths)"""
        
    elif clothing_lower in ['kemeja', 'shirt']:
        question = f"""Great! For **{clothing_item}**, what sleeve length do you prefer?
- **Long sleeves** (full arm coverage)
- **Short sleeves** (t-shirt style)
- **3/4 sleeves** (between elbow and wrist)
- **Mixed** (I like different lengths)"""
        
    else:
        question = f"""I see you're interested in **{clothing_item}**. What length preferences do you have for this type of clothing?
- **Long** 
- **Short**
- **Medium/Midi**
- **Mixed** (I like different lengths)"""
    
    return question

# ================================
# NEW HYBRID FUNCTIONS (REPLACEMENTS)
# ================================

async def extract_ranked_keywords(ai_response=None, translated_input=None, accumulated_keywords=None):
    """
    UPDATED: Uses the integrated HybridKeywordExtractor
    """
    return await hybrid_extractor.extract_keywords_hybrid(ai_response, translated_input, accumulated_keywords)
   
# ================================
# EXISTING FUNCTIONS (KEEP AS-IS)
# ================================

def update_accumulated_keywords(keywords, user_context, user_input, is_user_input=False, is_ai_response=False, is_consultation=False):
    """
    UPDATED: Enhanced keyword management with budget detection
    """
    if "accumulated_keywords" not in user_context:
        user_context["accumulated_keywords"] = {}
    
    # STEP 1: Detect budget information from user input
    if is_user_input:
        detected_budget = detect_and_extract_budget(user_input, user_context)
        if detected_budget:
            print(f"💰 Budget information detected and stored: {detected_budget}")
    
    # STEP 2: Detect preference changes (only for user input)
    preference_changes = []
    if is_user_input:
        preference_changes = detect_preference_changes_with_constants(user_input, user_context["accumulated_keywords"])
        
        # STEP 3: Apply preference changes with COMPLETE REMOVAL
        if preference_changes:
            apply_preference_changes_with_constants(preference_changes, user_context["accumulated_keywords"])
    
    updates_made = 0
    new_keywords_added = 0
    
    for keyword, score in keywords:
        if not keyword or len(keyword) < 2:
            continue

        keyword_lower = keyword.lower()
        
        # BLACKLIST CHECK using constants
        if hasattr(FashionCategories, 'is_blacklisted') and FashionCategories.is_blacklisted(keyword):
            print(f"   🚫 SKIPPING BLACKLISTED KEYWORD: '{keyword}'")
            continue
        
        # Enhanced boosting logic
        if is_consultation:
            frequency_boost = 50.0
            estimated_frequency = max(1, score / 10)
            source_label = "consultation"
        elif is_user_input:
            base_boost = 15.0
            
            # Extra boost for preference changes
            is_preference_change = any(change['new_preference'] == keyword_lower for change in preference_changes)
            
            # Extra boost for important preferences using constants
            is_color_preference = (hasattr(FashionCategories, 'is_color_preference') and 
                                 FashionCategories.is_color_preference(keyword))
            is_sleeve_preference = (hasattr(FashionCategories, 'is_sleeve_preference') and 
                                  FashionCategories.is_sleeve_preference(keyword))
            
            if is_preference_change:
                frequency_boost = 30.0
                print(f"   🔄 PREFERENCE CHANGE: '{keyword}' → change boost")
            elif is_color_preference:
                frequency_boost = 25.0
                print(f"   🎨 COLOR PREFERENCE: '{keyword}' → color boost")
            elif is_sleeve_preference:
                frequency_boost = 22.0
                print(f"   👕 SLEEVE PREFERENCE: '{keyword}' → sleeve boost")
            else:
                frequency_boost = base_boost
            
            estimated_frequency = max(1, score / 20)
            source_label = "user_input"
        else:
            frequency_boost = 1.0
            estimated_frequency = max(1, score / 100)
            source_label = "ai_response"
        
        # Keyword processing logic
        if keyword_lower in user_context["accumulated_keywords"]:
            # Update existing keyword
            data = user_context["accumulated_keywords"][keyword_lower]
            old_weight = get_weight_compatible(data)
            old_source = data.get("source", "unknown")
            
            if old_source == "consultation":
                if is_consultation:
                    new_weight = old_weight + (score * frequency_boost)
                else:
                    new_weight = old_weight + (score * 0.2)
                new_count = data.get("count", 1) + frequency_boost
                final_source = "consultation"
            elif is_consultation:
                new_weight = old_weight + (score * frequency_boost * 5)
                new_count = data.get("count", 1) + frequency_boost
                final_source = "consultation"
                print(f"   ⬆️ UPGRADING TO CONSULTATION: '{keyword}'")
            else:
                boost_multiplier = 3.0 if any(change['new_preference'] == keyword_lower for change in preference_changes) else 1.0
                new_weight = old_weight + (score * frequency_boost * boost_multiplier)
                new_count = data.get("count", 1) + frequency_boost
                final_source = source_label
                
                if boost_multiplier > 1.0:
                    print(f"   🚀 PREFERENCE UPDATE: '{keyword}' → {boost_multiplier}x boost")
            
            # Get category using FashionCategories if available
            if hasattr(FashionCategories, 'get_category'):
                category = FashionCategories.get_category(keyword_lower)
            else:
                category = get_keyword_category(keyword_lower)  # Fallback
            
            user_context["accumulated_keywords"][keyword_lower] = {
                "weight": new_weight,
                "total_frequency": new_count,
                "mention_count": new_count,
                "count": new_count,
                "first_seen": data.get("first_seen", datetime.now().isoformat()),
                "last_seen": datetime.now().isoformat(),
                "source": final_source,
                "category": category,
                "consultation_priority": old_source == "consultation" or is_consultation
            }
            
            updates_made += 1
            
        else:
            # Add new keyword
            initial_frequency = estimated_frequency * frequency_boost
            base_weight = initial_frequency * 100
            
            weight_multiplier = 1.0
            
            if is_consultation:
                weight_multiplier = 10.0
                print(f"   🆕 NEW CONSULTATION: '{keyword}'")
            elif any(change['new_preference'] == keyword_lower for change in preference_changes):
                weight_multiplier = 8.0
                print(f"   🆕 NEW PREFERENCE: '{keyword}'")
            elif is_user_input and (is_color_preference or is_sleeve_preference):
                weight_multiplier = 5.0
                print(f"   🆕 NEW IMPORTANT: '{keyword}'")
            
            final_weight = base_weight * weight_multiplier
            
            # Get category using FashionCategories if available
            if hasattr(FashionCategories, 'get_category'):
                category = FashionCategories.get_category(keyword_lower)
            else:
                category = get_keyword_category(keyword_lower)  # Fallback
            
            user_context["accumulated_keywords"][keyword_lower] = {
                "weight": final_weight,
                "total_frequency": initial_frequency,
                "mention_count": 1,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source": source_label,
                "category": category,
                "consultation_priority": is_consultation
            }
            new_keywords_added += 1

def detect_preference_changes_with_constants(user_input, accumulated_keywords):
    """UPDATED: Using FashionCategories conflict groups for consistent preference change detection"""
    if not user_input or not accumulated_keywords:
        return []
    
    user_input_lower = user_input.lower()
    preference_changes = []
    
    # Change patterns
    change_patterns = [
        r'change.*(sleeve|color|fit|style|length).*to\s+([^.!?]+)',
        r'actually.*prefer\s+([^.!?]+)',
        r'instead.*want\s+([^.!?]+)',
        r'switch.*to\s+([^.!?]+)',
        r'now.*want\s+([^.!?]+)',
        r'ganti.*(lengan|warna|model).*jadi\s+([^.!?]+)',
        r'sebenarnya.*prefer\s+([^.!?]+)',
        r'lebih suka\s+([^.!?]+)',
    ]
    
    # NEW: Use FashionCategories conflict groups
    conflict_groups = FashionCategories.get_conflict_groups()
    
    # Check for explicit change patterns
    for pattern in change_patterns:
        matches = re.finditer(pattern, user_input_lower)
        for match in matches:
            if len(match.groups()) >= 1:
                new_preference = match.group(-1).strip()
                
                for group_name, group_data in conflict_groups.items():
                    conflict_mapping = group_data.get('conflicts', {})
                    
                    for pref_type, terms in conflict_mapping.items():
                        if any(term in new_preference for term in terms):
                            # Get all conflicting keywords from this group
                            conflicting_keywords = []
                            for other_type, other_terms in conflict_mapping.items():
                                if other_type != pref_type:
                                    conflicting_keywords.extend(other_terms)
                            
                            preference_changes.append({
                                'type': 'explicit_change',
                                'group': group_name,
                                'new_preference': new_preference,
                                'conflicting_keywords': conflicting_keywords
                            })
                            print(f"🔄 DETECTED EXPLICIT CHANGE: {group_name} → '{new_preference}'")
                            break
    
    # Check for implicit conflicts using FashionCategories
    current_preferences = {}
    for group_name, group_data in conflict_groups.items():
        patterns = group_data.get('patterns', [])
        for pattern in patterns:
            if re.search(pattern, user_input_lower):
                matched_preference = re.search(pattern, user_input_lower).group()
                current_preferences[group_name] = matched_preference
                break
    
    for group_name, current_pref in current_preferences.items():
        conflict_mapping = FashionCategories.get_conflict_mapping(group_name)
        conflicting_accumulated = []
        
        for keyword, data in accumulated_keywords.items():
            if isinstance(data, dict) and data.get('weight', 0) > 1000:
                # Check if this keyword conflicts with current preference
                for pref_type, terms in conflict_mapping.items():
                    if any(term in keyword.lower() for term in terms):
                        if current_pref.lower() not in keyword.lower():
                            conflicting_accumulated.append(keyword)
                            break
        
        if conflicting_accumulated:
            preference_changes.append({
                'type': 'implicit_conflict',
                'group': group_name,
                'new_preference': current_pref,
                'conflicting_keywords': conflicting_accumulated
            })
            print(f"🔄 DETECTED IMPLICIT CONFLICT: {group_name} → '{current_pref}' conflicts with {conflicting_accumulated}")
    
    return preference_changes

def apply_preference_changes_with_constants(preference_changes, accumulated_keywords):
    """Using constants for consistent preference change application"""
    for change in preference_changes:
        print(f"\n🔄 APPLYING PREFERENCE CHANGE:")
        print(f"   Type: {change['type']}")
        print(f"   Group: {change['group']}")
        print(f"   New preference: '{change['new_preference']}'")
        
        # Complete removal of conflicting keywords
        conflicting_keywords = change['conflicting_keywords']
        removed_count = 0
        
        for keyword in list(accumulated_keywords.keys()):
            keyword_lower = str(keyword).lower()
            
            # Check if this keyword conflicts with the new preference
            for conflict in conflicting_keywords:
                if (conflict in keyword_lower and 
                    change['new_preference'].lower() not in keyword_lower):
                    print(f"   ❌ COMPLETELY REMOVING: '{keyword}' (conflicts with '{change['new_preference']}')")
                    del accumulated_keywords[keyword]
                    removed_count += 1
                    break
        
        print(f"   🗑️ Completely removed {removed_count} conflicting keywords")

def detect_clothing_change_in_input(user_input, user_context):
    """
    ENHANCED: Detect clothing changes including "only" requests and multiple items
    """
    user_input_lower = user_input.lower().strip()
    
    print(f"\n🔍 ENHANCED CLOTHING CHANGE DETECTION:")
    print(f"   Input: '{user_input}'")
    
    # Get current primary clothing from accumulated keywords
    current_clothing = []
    accumulated_keywords = user_context.get("accumulated_keywords", {})
    
    # Find ALL current clothing items
    for keyword, data in accumulated_keywords.items():
        if keyword and FashionCategories.get_category(keyword) == 'clothing_terms':
            weight = data.get("weight", 0) if isinstance(data, dict) else data
            if weight > 1000:
                current_clothing.append(keyword)
                print(f"   Current clothing: '{keyword}' (weight: {weight})")
    
    if not current_clothing:
        print(f"   No current clothing found")
    
    # ENHANCED: Check for "only" patterns that indicate exclusivity
    only_patterns = [
        r'\b(only|just|hanya|cuma|aja)\s+([^.!?]+)',
        r'\b([^.!?]+)\s+(only|aja|saja|doang|hanya)',
        r'\b(i want only|saya mau)\s+([^.!?]+)',
        r'\b(just|cuma)\s+(show me|carikan)\s+([^.!?]+)'
    ]
    
    is_only_request = False
    only_clothing_items = []
    
    # Check for "only" patterns first
    for pattern in only_patterns:
        matches = re.finditer(pattern, user_input_lower)
        for match in matches:
            groups = match.groups()
            # Extract the clothing part from the match
            for group in groups:
                if group and group not in ['only', 'just', 'hanya', 'cuma', 'aja', 'saja', 'doang', 'i want only', 'saya mau', 'show me', 'carikan']:
                    phrase = group.strip()
                    print(f"   Found 'only' phrase: '{phrase}'")
                    
                    # Check if this phrase contains clothing items
                    for clothing_item in FashionCategories.CLOTHING_TERMS:
                        if (clothing_item.lower() in phrase or 
                            any(word.strip('.,!?()[]{}":;') == clothing_item.lower() for word in phrase.split())):
                            if clothing_item not in only_clothing_items:
                                only_clothing_items.append(clothing_item)
                                is_only_request = True
                                print(f"   ✅ Found ONLY clothing: '{clothing_item}'")
    
    # Enhanced patterns for multiple clothing detection
    clothing_change_patterns = [
        r'\b(?:bagaimana|gimana|how about|what about)\s+(?:dengan\s+)?([^.!?]+)',
        r'\b(?:ada|do you have|punya)\s+([^.!?]+)',
        r'\b(?:carikan|cari|find|show)\s+(?:me\s+)?([^.!?]+)',
        r'\b(?:sekarang|now)\s+(?:saya\s+)?(?:mau|want|ingin)\s+([^.!?]+)',
        r'\b(?:untuk|for)\s+([^.!?]+)',
        r'\b([^.!?]+)\s+(?:dong|aja|saja)',
        r'\b(?:ganti|change|switch)\s+(?:ke|to)\s+([^.!?]+)',
        # NEW: Handle "dan/and" combinations
        r'\b([a-zA-Z]+)\s+(?:dan|and)\s+([a-zA-Z]+)',
        r'\b([a-zA-Z]+)\s*,\s*([a-zA-Z]+)'
    ]
    
    # Collect ALL new clothing mentions (if not an "only" request)
    new_clothing_items = only_clothing_items.copy() if is_only_request else []
    detected_phrases = set()
    
    if not is_only_request:
        # First try pattern matching for context-aware detection
        for pattern in clothing_change_patterns:
            matches = re.finditer(pattern, user_input_lower)
            for match in matches:
                groups = match.groups()
                
                # Handle "dan/and" patterns specially
                if len(groups) == 2 and 'dan|and' in pattern:
                    for word in groups:
                        word = word.strip('.,!?()[]{}":;')
                        for clothing_item in FashionCategories.CLOTHING_TERMS:
                            if (clothing_item.lower() == word or 
                                word in clothing_item.lower() or 
                                clothing_item.lower() in word):
                                if clothing_item not in new_clothing_items:
                                    new_clothing_items.append(clothing_item)
                                    print(f"   ✅ Found clothing via dan/and: '{clothing_item}' from word '{word}'")
                else:
                    # Handle other patterns
                    for group in groups:
                        phrase = group.strip()
                        if phrase in detected_phrases:
                            continue
                        detected_phrases.add(phrase)
                        
                        print(f"   Analyzing phrase: '{phrase}'")
                        
                        # Check each word in the phrase for clothing items
                        words = phrase.split()
                        for word in words:
                            word = word.strip('.,!?()[]{}":;')
                            
                            # Check if it's a valid clothing item
                            for clothing_item in FashionCategories.CLOTHING_TERMS:
                                if (clothing_item.lower() == word or 
                                    word in clothing_item.lower() or 
                                    clothing_item.lower() in word):
                                    if clothing_item not in new_clothing_items:
                                        new_clothing_items.append(clothing_item)
                                        print(f"   ✅ Found clothing via pattern: '{clothing_item}' from word '{word}'")
        
        # Also try direct matching for any missed items
        for clothing_item in FashionCategories.CLOTHING_TERMS:
            if clothing_item.lower() in user_input_lower and clothing_item not in new_clothing_items:
                new_clothing_items.append(clothing_item)
                print(f"   ✅ Found clothing via direct match: '{clothing_item}'")
    
    if not new_clothing_items:
        print(f"   ❌ No clothing items detected")
        return False, []
    
    # ENHANCED: Handle "only" requests - this means REPLACE all current clothing
    if is_only_request:
        print(f"   🎯 ONLY REQUEST: User wants ONLY {only_clothing_items}")
        print(f"   🔄 Will replace ALL current clothing with: {only_clothing_items}")
        return True, only_clothing_items
    
    # Check if any new clothing is different from current clothing
    truly_new_items = []
    
    for new_item in new_clothing_items:
        is_new = True
        for current_item in current_clothing:
            if new_item.lower() == current_item.lower():
                is_new = False
                print(f"   ⚪ Already have: '{new_item}'")
                break
        
        if is_new:
            truly_new_items.append(new_item)
    
    if truly_new_items:
        print(f"   ✅ NEW CLOTHING DETECTED: {truly_new_items}")
        print(f"   📝 Will ADD to existing: {current_clothing}")
        return True, new_clothing_items  # Return ALL clothing items (existing + new)
    elif not current_clothing and new_clothing_items:
        print(f"   🆕 FIRST CLOTHING DETECTED: {new_clothing_items}")
        return True, new_clothing_items
    else:
        print(f"   ⚪ No new clothing items")
        return False, []
    
def get_weight_compatible(data):
    """Extract weight from data in a compatible way"""
    if isinstance(data, dict):
        return data.get("weight", 0)
    else:
        return float(data) if data else 0

def get_keyword_category(keyword_lower):
    """Get the category of a keyword for smart boosting"""
    categories = {
        'clothing_items': ['kemeja', 'shirt', 'blouse', 'celana', 'pants', 'rok', 'skirt', 'dress', 'gaun'],
        'style_attributes': ['casual', 'formal', 'elegant', 'vintage', 'modern'],
        'colors': ['black', 'white', 'red', 'blue', 'hitam', 'putih'],
        'user_identity': ['female', 'male', 'woman', 'man', 'perempuan'],
        'occasions': ['office', 'party', 'wedding', 'kantor', 'pesta']
    }
    
    for category, terms in categories.items():
        if any(term in keyword_lower for term in terms):
            return category
    
    return 'other'

def detect_and_update_gender(user_input, user_context, force_update=False):
    """
    ENHANCED: More responsive gender detection that handles changes during conversation
    """
    print(f"\n👤 ENHANCED GENDER DETECTION")
    print("="*50)
    print(f"📝 Input: '{user_input}'")
    
    current_gender = user_context.get("user_gender", {})
    has_existing_gender = current_gender.get("category") is not None
    
    # Enhanced gender detection patterns with change detection
    gender_patterns = {
        'male': [
            # Direct statements
            r'\b(i am|i\'m|saya|aku)\s+(a\s+)?(male|man|pria|laki-laki|cowok|cowo|laki)\b',
            r'\b(male|man|pria|laki-laki|cowok|cowo|laki)\b',
            r'\b(as a|sebagai)\s+(male|man|pria|laki-laki)\b',
            r'\b(gender|jenis kelamin).*(male|man|pria|laki-laki)\b',
            r'\b(i\'m a|saya)\s+(guy|boy|male)\b',
            
            # Change patterns
            r'\b(change|ganti|ubah).*gender.*(to|jadi|menjadi)\s+(male|man|pria|laki-laki)\b',
            r'\b(actually|sebenarnya).*i.*(am|\'m)\s+(male|man|pria|laki-laki)\b',
            r'\b(not|bukan|tidak).*(female|woman|perempuan).*i.*(am|\'m)\s+(male|man|pria)\b',
            r'\b(i.*(am|\'m)\s+a\s+)?(male|man|pria|laki-laki).*not\s+(female|woman|perempuan)\b'
        ],
        'female': [
            # Direct statements
            r'\b(i am|i\'m|saya|aku)\s+(a\s+)?(female|woman|perempuan|wanita|cewek|cewe)\b',
            r'\b(female|woman|perempuan|wanita|cewek|cewe)\b',
            r'\b(as a|sebagai)\s+(female|woman|perempuan|wanita)\b',
            r'\b(gender|jenis kelamin).*(female|woman|perempuan|wanita)\b',
            r'\b(i\'m a|saya)\s+(girl|lady|female)\b',
            
            # Change patterns
            r'\b(change|ganti|ubah).*gender.*(to|jadi|menjadi)\s+(female|woman|perempuan|wanita)\b',
            r'\b(actually|sebenarnya).*i.*(am|\'m)\s+(female|woman|perempuan|wanita)\b',
            r'\b(not|bukan|tidak).*(male|man|pria).*i.*(am|\'m)\s+(female|woman|perempuan)\b',
            r'\b(i.*(am|\'m)\s+a\s+)?(female|woman|perempuan|wanita).*not\s+(male|man|pria)\b'
        ]
    }
    
    user_input_lower = user_input.lower()
    detected_gender = None
    detected_term = None
    confidence = 0
    is_correction = False
    
    # Check for gender change/correction patterns first
    change_indicators = [
        'change', 'ganti', 'ubah', 'actually', 'sebenarnya', 'not', 'bukan', 'tidak',
        'correction', 'koreksi', 'salah', 'wrong', 'should be', 'seharusnya'
    ]
    
    if any(indicator in user_input_lower for indicator in change_indicators):
        is_correction = True
        print(f"   🔄 CORRECTION/CHANGE DETECTED in input")
    
    # Check for gender patterns
    for gender, patterns in gender_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                detected_gender = gender
                detected_term = match.group()
                
                # Higher confidence for corrections or when existing gender is different
                if is_correction or (has_existing_gender and current_gender.get("category") != gender):
                    confidence = 20.0  # Higher confidence for corrections
                    print(f"   🔄 GENDER CHANGE/CORRECTION: '{pattern}' → '{detected_term}'")
                else:
                    confidence = 15.0
                    print(f"   🎯 GENDER DETECTION: '{pattern}' → '{detected_term}'")
                break
        if detected_gender:
            break
    
    # Update gender logic
    should_update = False
    
    if detected_gender:
        if not has_existing_gender:
            # No existing gender - always update
            should_update = True
            print(f"   ✅ FIRST TIME GENDER DETECTION: {detected_gender}")
            
        elif current_gender.get("category") != detected_gender:
            # Different gender detected - always update (user is changing)
            should_update = True
            old_gender = current_gender.get("category")
            print(f"   🔄 GENDER CHANGE: {old_gender} → {detected_gender}")
            
        elif is_correction or force_update:
            # Correction or forced update
            should_update = True
            print(f"   🔧 FORCED GENDER UPDATE: {detected_gender}")
            
        elif current_gender.get("confidence", 0) < confidence:
            # Higher confidence detection
            should_update = True
            print(f"   📈 HIGHER CONFIDENCE UPDATE: {detected_gender} (conf: {confidence})")
        
        else:
            print(f"   ⚪ SAME GENDER CONFIRMED: {detected_gender}")
    
    # Update gender if needed
    if should_update and detected_gender:
        old_gender = current_gender.get("category", "None")
        user_context["user_gender"] = {
            "category": detected_gender,
            "term": detected_term,
            "confidence": confidence,
            "last_updated": datetime.now().isoformat(),
            "is_correction": is_correction,
            "previous_gender": old_gender if old_gender != "None" else None
        }
        
        print(f"   ✅ GENDER UPDATED: {old_gender} → {detected_gender}")
        print(f"   📊 Confidence: {confidence}, Is Correction: {is_correction}")
        
        return detected_gender
    
    # Return existing gender if available
    if has_existing_gender:
        existing_gender = current_gender["category"]
        print(f"   📋 Using existing gender: {existing_gender}")
        return existing_gender
    
    print(f"   ❌ No gender detected in input")
    return None

async def fetch_products_from_db(db: AsyncSession, top_keywords: list, max_results=15, gender_category=None, budget_range=None):
    """
    UPDATED: Product fetching with enhanced color preference prioritization
    """
    print(f"\n🔍 ENHANCED MULTI-CLOTHING PRODUCT SEARCH WITH COLOR PRIORITY")
    print(f"📊 Keywords: {len(top_keywords)}")
    
    # Extract clothing types from keywords
    clothing_types = []
    for i, (kw, score) in enumerate(top_keywords[:10]):
        if FashionCategories.get_category(kw) == 'clothing_terms':
            clothing_types.append(kw)
        print(f"   {i+1:2d}. '{kw}' → Score: {score:.2f}")
    
    print(f"🎯 Detected clothing types: {clothing_types}")
    
    # ENHANCED: Extract color preferences using the new function
    # First, we need to reconstruct accumulated_keywords from top_keywords for color extraction
    accumulated_keywords_for_colors = {}
    for keyword, score in top_keywords:
        # Simulate the accumulated keywords structure for color extraction
        accumulated_keywords_for_colors[keyword] = {
            "weight": score,
            "source": "user_input" if score > 50000 else "unknown"  # High scores likely from user input
        }
    
    # Extract enhanced color preferences
    enhanced_color_preferences = HybridKeywordExtractor.extract_color_preferences_enhanced(accumulated_keywords_for_colors)
    
    print(f"🎨 Enhanced color preferences: {[(color, weight) for color, weight in enhanced_color_preferences[:5]]}")
    
    try:
        # Get products with variants (same query as before)
        variant_subquery = (
            select(
                ProductVariant.product_id,
                func.min(ProductVariant.product_price).label('min_price'),
                func.group_concat(ProductVariant.size.distinct()).label('available_sizes'),
                func.group_concat(ProductVariant.color.distinct()).label('available_colors'),
                func.sum(ProductVariant.stock).label('total_stock')
            )
            .where(ProductVariant.stock > 0)
            .group_by(ProductVariant.product_id)
            .subquery()
        )
        
        base_query = (
            select(
                Product.product_id, 
                Product.product_name, 
                Product.product_detail, 
                Product.product_seourl,
                Product.product_gender,
                variant_subquery.c.min_price,
                variant_subquery.c.available_sizes,
                variant_subquery.c.available_colors,
                variant_subquery.c.total_stock,
                ProductPhoto.productphoto_path
            )
            .select_from(Product)
            .join(variant_subquery, Product.product_id == variant_subquery.c.product_id)
            .join(ProductPhoto, Product.product_id == ProductPhoto.product_id)
            .where(variant_subquery.c.total_stock > 0)
        )
        
        # Apply filters (same as before)
        if gender_category:
            if gender_category.lower() in ['female', 'woman', 'perempuan', 'wanita']:
                base_query = base_query.where(Product.product_gender == 'female')
            elif gender_category.lower() in ['male', 'man', 'pria', 'laki-laki']:
                base_query = base_query.where(Product.product_gender == 'male')
        
        if budget_range and isinstance(budget_range, (tuple, list)) and len(budget_range) == 2:
            min_price, max_price = budget_range
            if min_price and max_price:
                base_query = base_query.where(variant_subquery.c.min_price.between(min_price, max_price))
            elif max_price:
                base_query = base_query.where(variant_subquery.c.min_price <= max_price)
            elif min_price:
                base_query = base_query.where(variant_subquery.c.min_price >= min_price)
        
        # Execute query
        result = await db.execute(base_query)
        all_products = result.fetchall()
        
        if not all_products:
            print("❌ No products found")
            return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
        
        print(f"\n📦 SCORING {len(all_products)} PRODUCTS WITH ENHANCED COLOR PRIORITY...")
        print("=" * 60)
        
        # ENHANCED: Calculate relevance scores with color preferences
        scored_products = []
        products_by_type = {}
        
        for product_row in all_products:
            # UPDATED: Pass enhanced color preferences to relevance calculation
            relevance_score = calculate_relevance_score(product_row, top_keywords, enhanced_color_preferences)
            
            if relevance_score > 0:
                # Determine which clothing type this product matches
                product_name_lower = product_row[1].lower()
                product_detail_lower = product_row[2].lower()
                search_text = f"{product_name_lower} {product_detail_lower}"
                
                matched_types = []
                for clothing_type in clothing_types:
                    if clothing_type.lower() in search_text:
                        matched_types.append(clothing_type)
                
                product_data = {
                    "product_id": product_row[0],
                    "product": product_row[1],
                    "description": product_row[2],
                    "price": product_row[5],
                    "size": ", ".join(product_row[6].split(',')) if product_row[6] else "N/A",
                    "color": ", ".join(product_row[7].split(',')) if product_row[7] else "N/A", 
                    "stock": product_row[8],
                    "link": f"http://localhost/e-commerce-main/product-{product_row[3]}-{product_row[0]}",
                    "photo": product_row[9],
                    "relevance": relevance_score,
                    "matched_types": matched_types
                }
                
                scored_products.append(product_data)
                
                # Group by clothing type for balanced distribution
                for clothing_type in matched_types:
                    if clothing_type not in products_by_type:
                        products_by_type[clothing_type] = []
                    products_by_type[clothing_type].append(product_data)
        
        if not scored_products:
            print("\n❌ NO PRODUCTS PASSED COLOR AND CONFLICT FILTERING")
            return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
        
        # ENHANCED: Ensure balanced representation but prioritize high color matches
        final_products = []
        
        # First, add products with highest relevance scores (likely good color matches)
        all_scored = sorted(scored_products, key=lambda x: x['relevance'], reverse=True)
        high_score_products = all_scored[:max_results//2]  # Take top half
        final_products.extend(high_score_products)
        
        # Then, ensure clothing type distribution for remaining slots
        remaining_slots = max_results - len(final_products)
        if remaining_slots > 0 and clothing_types:
            max_per_type = max(2, remaining_slots // len(clothing_types))
            existing_ids = {p['product_id'] for p in final_products}
            
            print(f"\n🎯 DISTRIBUTING REMAINING {remaining_slots} PRODUCTS ACROSS {len(clothing_types)} CLOTHING TYPES:")
            for clothing_type in clothing_types:
                if clothing_type in products_by_type and remaining_slots > 0:
                    type_products = sorted(products_by_type[clothing_type], 
                                         key=lambda x: x['relevance'], reverse=True)
                    
                    added_for_type = 0
                    for product in type_products:
                        if (product['product_id'] not in existing_ids and 
                            remaining_slots > 0 and added_for_type < max_per_type):
                            final_products.append(product)
                            existing_ids.add(product['product_id'])
                            remaining_slots -= 1
                            added_for_type += 1
                    
                    print(f"   📦 {clothing_type}: {added_for_type} additional products")
        
        # Create DataFrame and sort by relevance (color-prioritized)
        products_df = pd.DataFrame(final_products)
        products_df = products_df.sort_values(by=['relevance'], ascending=False).reset_index(drop=True)
        
        # Take top results
        final_results = products_df[:max_results]
        
        print(f"\n✅ FINAL COLOR-PRIORITIZED RESULTS: {len(final_results)} PRODUCTS")
        for i, row in final_results.iterrows():
            matched_types = row.get('matched_types', ['unknown'])
            colors = row['color']
            print(f"   {i+1:2d}. '{row['product']}' | Colors: {colors} | Score: {row['relevance']:.0f} | Types: {matched_types}")
        
        # Remove the matched_types column before returning
        if 'matched_types' in final_results.columns:
            final_results = final_results.drop('matched_types', axis=1)
        
        return final_results
        
    except Exception as e:
        logging.error(f"Error in enhanced color-priority fetch_products_from_db: {str(e)}")
        return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
        
def calculate_relevance_score(product_row, keywords, color_preferences=None):
    """
    ENHANCED: Calculate relevance score using FashionCategories conflict groups
    """
    product_name = product_row[1].lower()
    product_detail = product_row[2].lower()
    available_colors = product_row[7].lower() if product_row[7] else ""
    
    search_text = f"{product_name} {product_detail} {available_colors}"
    
    total_score = 0
    conflict_penalty = 0
    matched_keywords = []  # Track which keywords matched
    color_bonus = 0
    matched_colors = []
    
    # NEW: Use FashionCategories conflict groups
    conflict_groups = FashionCategories.get_conflict_groups()
    
    # Extract user preferences from keywords for each conflict group
    user_preferences_by_group = {}
    high_score_keywords = []
    
    for i, (keyword, weight) in enumerate(keywords[:15]):
        keyword_lower = keyword.lower()
        
        # Track high-scoring keywords for preference detection
        if weight > 5000000:  # High priority keywords
            high_score_keywords.append(keyword_lower)
        
        # NEW: Detect user preferences using FashionCategories conflict groups
        for group_name, group_data in conflict_groups.items():
            conflict_mapping = group_data.get('conflicts', {})
            
            for pref_type, terms in conflict_mapping.items():
                if any(term in keyword_lower or keyword_lower in term for term in terms):
                    if group_name not in user_preferences_by_group:
                        user_preferences_by_group[group_name] = []
                    user_preferences_by_group[group_name].append((keyword_lower, weight, pref_type))
    
    print(f"🔍 ANALYZING PRODUCT: '{product_row[1]}'")
    print(f"   📋 User preferences by group: {user_preferences_by_group}")
    print(f"   🎯 High score keywords: {high_score_keywords}")
    if color_preferences:
        print(f"   🎨 Color preferences: {[color for color, weight in color_preferences]}")
    
    # NEW: Check for conflicts using FashionCategories
    for group_name in conflict_groups.keys():
        if group_name not in user_preferences_by_group:
            continue
        
        user_prefs = user_preferences_by_group[group_name]
        has_conflict, conflict_details = FashionCategories.detect_conflicts_in_group(
            group_name, user_prefs, search_text
        )
        
        if has_conflict:
            user_weight = conflict_details['user_weight']
            conflicting_terms = conflict_details['conflicting_terms']
            user_preference = conflict_details['user_preference']
            user_keyword = conflict_details['user_keyword']
            
            print(f"   🎯 {group_name.upper()}: User prefers '{user_preference}' ({user_keyword}) - weight: {user_weight:.0f}")
            
            # Apply penalty based on weight and conflict severity
            if user_weight > 5000000:  # High priority preference
                penalty = user_weight * 2.0  # Double penalty
                conflict_penalty += penalty
                print(f"   ❌ MAJOR CONFLICT: Product has {conflicting_terms} but user wants {user_preference}")
                print(f"   💥 Applied penalty: {penalty:.0f}")
            else:
                penalty = user_weight * 0.5  # Smaller penalty for lower priority
                conflict_penalty += penalty
                print(f"   ⚠️ Minor conflict: Product has {conflicting_terms} but user wants {user_preference}")
                print(f"   💥 Applied penalty: {penalty:.0f}")
    
    # Calculate positive scores for matching keywords (same as before)
    for i, (keyword, weight) in enumerate(keywords[:15]):
        keyword_lower = keyword.lower()
        
        # Position importance
        position_weight = (15 - i) / 15
        
        match_score = 0
        
        # Exact match bonus
        if keyword_lower in search_text:
            matched_keywords.append(keyword_lower)  # Track matched keywords
            
            if keyword_lower in product_name:
                match_score = weight * position_weight * 3.0
                print(f"   ✅ EXACT NAME MATCH: '{keyword_lower}' → +{match_score:.0f}")
            elif keyword_lower in product_detail:
                match_score = weight * position_weight * 2.0
                print(f"   ✅ DETAIL MATCH: '{keyword_lower}' → +{match_score:.0f}")
            else:
                match_score = weight * position_weight * 1.0
                print(f"   ✅ COLOR MATCH: '{keyword_lower}' → +{match_score:.0f}")
            
            total_score += match_score
        
        # Partial match
        elif any(word in search_text for word in keyword_lower.split()):
            partial_score = weight * position_weight * 0.5
            total_score += partial_score
            print(f"   ⚪ PARTIAL MATCH: '{keyword_lower}' → +{partial_score:.0f}")
    
    # Integrated color preference handling (same as before)
    if color_preferences:
        print(f"   🎨 ANALYZING COLOR PREFERENCES:")
        
        for color, weight in color_preferences:
            color_lower = color.lower()
            
            # Check for color matches with different strategies
            exact_match = False
            partial_match = False
            synonym_match = False
            translation_match = False
            
            # Strategy 1: Exact word match
            if f" {color_lower} " in f" {search_text} ":
                exact_match = True
            elif search_text.startswith(f"{color_lower} ") or search_text.endswith(f" {color_lower}"):
                exact_match = True
            elif color_lower == search_text:
                exact_match = True
            
            # Strategy 2: Synonym matching using FashionCategories
            if not exact_match:
                synonyms = FashionCategories.get_color_synonyms(color_lower)
                for synonym in synonyms:
                    if synonym in search_text:
                        synonym_match = True
                        print(f"     🔄 SYNONYM MATCH: '{color_lower}' matched via '{synonym}'")
                        break
            
            # Strategy 3: Translation matching using FashionCategories
            if not exact_match and not synonym_match:
                translated_color = FashionCategories.translate_color(color_lower)
                if translated_color != color_lower and translated_color in search_text:
                    translation_match = True
                    print(f"     🌍 TRANSLATION MATCH: '{color_lower}' → '{translated_color}'")
            
            # Strategy 4: Partial match for compound colors
            if not exact_match and not synonym_match and not translation_match:
                color_parts = color_lower.split()
                if len(color_parts) > 1:
                    # For compound colors like "dark blue", "light pink"
                    if all(part in search_text for part in color_parts if len(part) > 2):
                        partial_match = True
                        print(f"     🔗 COMPOUND MATCH: '{color_lower}' matched via parts")
                elif any(color_part in search_text for color_part in color_lower.split() if len(color_part) > 2):
                    partial_match = True
                    print(f"     🔗 PARTIAL MATCH: '{color_lower}' matched partially")
            
            # Calculate color bonus based on match type
            if exact_match:
                color_bonus += weight * 0.4  # 40% bonus for exact color match
                matched_colors.append(color_lower)
                print(f"     ✅ EXACT COLOR MATCH: '{color_lower}' → +{weight * 0.4:.0f}")
            elif translation_match:
                color_bonus += weight * 0.38  # 38% bonus for translation match
                matched_colors.append(f"{color_lower} (translated)")
                print(f"     ✅ TRANSLATION COLOR MATCH: '{color_lower}' → +{weight * 0.38:.0f}")
            elif synonym_match:
                color_bonus += weight * 0.35  # 35% bonus for synonym match
                matched_colors.append(f"{color_lower} (synonym)")
                print(f"     ✅ SYNONYM COLOR MATCH: '{color_lower}' → +{weight * 0.35:.0f}")
            elif partial_match:
                color_bonus += weight * 0.2  # 20% bonus for partial color match
                matched_colors.append(f"{color_lower} (partial)")
                print(f"     ⚪ PARTIAL COLOR MATCH: '{color_lower}' → +{weight * 0.2:.0f}")
            else:
                print(f"     ❌ NO COLOR MATCH: '{color_lower}'")
    
    # Multi-match bonus calculation (same as before)
    num_matches = len(matched_keywords)
    multi_match_bonus = 0
    
    if num_matches >= 2:
        # Exponential bonus for products matching multiple criteria
        if num_matches == 2:
            multi_match_bonus = total_score * 0.10
        elif num_matches == 3:
            multi_match_bonus = total_score * 0.25
        elif num_matches >= 4:
            multi_match_bonus = total_score * 0.50
        
        print(f"   🎯 MULTI-MATCH BONUS: {num_matches} matches → +{multi_match_bonus:.0f} ({(multi_match_bonus/total_score*100):.1f}% bonus)")
        total_score += multi_match_bonus
    
    # Apply conflict penalty and add color bonus
    final_score = total_score + color_bonus - conflict_penalty
    
    print(f"   📊 INTEGRATED SCORING SUMMARY:")
    print(f"      Base positive score: {total_score - multi_match_bonus:.0f}")
    print(f"      Multi-match bonus: {multi_match_bonus:.0f}")
    print(f"      Color bonus: {color_bonus:.0f} ({len(matched_colors)} color matches)")
    print(f"      Conflict penalty: {conflict_penalty:.0f}")
    print(f"      Final score: {final_score:.0f}")
    print(f"      Matched keywords: {matched_keywords}")
    if matched_colors:
        print(f"      Matched colors: {matched_colors}")
    print(f"   {'🚫 REJECTED' if final_score < 0 else '✅ ACCEPTED'}")
    print("-" * 60)
    
    # Return 0 if conflicts outweigh positives
    return max(final_score, 0)

def get_paginated_products(all_products_df, page=0, products_per_page=5):
    """Get paginated products"""
    if all_products_df.empty:
        return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"]), False
    
    start_idx = page * products_per_page
    end_idx = start_idx + products_per_page
    
    paginated_products = all_products_df.iloc[start_idx:end_idx]
    has_more = end_idx < len(all_products_df)
    
    return paginated_products, has_more

def detect_more_products_request(user_input: str) -> bool:
    """Detect if user is asking for more products"""
    more_patterns = [
        r'\b(more|other|another|additional|different|else)\s+(product|item|option|choice|recommendation)',
        r'\b(show|give|find|get)\s+(me\s+)?(more|other|another|additional)',
        r'\b(what|anything)\s+else',
        r'\b(lain|lainnya|yang lain|lagi)\b',
        r'\b(tunjukkan|carikan|kasih|coba)\s+(yang\s+)?(lain|lainnya)',
    ]
    
    user_input_lower = user_input.lower().strip()
    
    # Don't trigger on simple confirmations
    simple_responses = ["yes", "ya", "iya", "ok", "okay", "sure", "tentu", "no", "tidak"]
    if user_input_lower in simple_responses:
        return False
    
    for pattern in more_patterns:
        if re.search(pattern, user_input_lower):
            return True
    
    return False

def detect_and_extract_budget(user_input, user_context):
    """
    NEW: Detect and extract budget information from user input
    """
    if not user_input:
        return None
    
    user_input_lower = user_input.lower()
    
    # Budget range patterns
    budget_patterns = [
        # Range patterns
        (r'(?:antara|between)?\s*(\d+)(?:rb|ribu|k)?\s*(?:-|sampai|hingga|to)\s*(\d+)(?:rb|ribu|k)?', "RANGE"),
        (r'(\d+)\s*(?:-|to|sampai)\s*(\d+)\s*(?:rb|ribu|k|jt|juta)', "RANGE"),
        
        # Maximum patterns
        (r'(?:under|dibawah|maksimal|max|kurang\s+dari|less\s+than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta)?', "MAX"),
        (r'(?:below|dibawah)\s*(\d+)(?:rb|ribu|k)', "MAX"),
        
        # Minimum patterns
        (r'(?:above|diatas|over|minimal|min|lebih\s+dari|more\s+than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta)?', "MIN"),
        
        # Exact/Around patterns
        (r'(?:around|sekitar|budget|anggaran)?\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta)', "EXACT"),
        
        # Predefined ranges
        (r'budget.?friendly|murah|cheap', "BUDGET_FRIENDLY"),
        (r'mid.?range|menengah|affordable', "MID_RANGE"),
        (r'premium|mahal|expensive|luxury', "PREMIUM"),
        (r'no.?budget|unlimited|bebas|tidak.?ada.?batas', "NO_LIMIT")
    ]
    
    def convert_to_rupiah(amount_str, unit):
        """Convert amount string with unit to rupiah"""
        try:
            amount = int(amount_str)
            if unit and ('rb' in unit or 'ribu' in unit or 'k' in unit):
                return amount * 1000
            elif unit and ('jt' in unit or 'juta' in unit):
                return amount * 1000000
            else:
                # Assume thousands if no unit and amount is reasonable
                if amount < 1000:
                    return amount * 1000
                return amount
        except:
            return None
    
    # Check patterns
    for pattern, pattern_type in budget_patterns:
        matches = list(re.finditer(pattern, user_input_lower))
        
        for match in matches:
            groups = match.groups()
            match_text = match.group(0)
            
            print(f"🔍 Budget pattern matched: '{match_text}' → Type: {pattern_type}")
            
            if pattern_type == "RANGE" and len(groups) >= 2 and groups[0] and groups[1]:
                unit = None
                if any(x in match_text for x in ['rb', 'ribu', 'k']):
                    unit = 'rb'
                elif any(x in match_text for x in ['jt', 'juta']):
                    unit = 'jt'
                
                min_price = convert_to_rupiah(groups[0], unit)
                max_price = convert_to_rupiah(groups[1], unit)
                
                if min_price and max_price:
                    budget_range = (min(min_price, max_price), max(min_price, max_price))
                    print(f"💰 Budget range detected: IDR {budget_range[0]:,} - IDR {budget_range[1]:,}")
                    user_context["budget_range"] = budget_range
                    return budget_range
            
            elif pattern_type in ["MAX", "MIN", "EXACT"] and len(groups) >= 1 and groups[0]:
                unit = None
                if any(x in match_text for x in ['rb', 'ribu', 'k']):
                    unit = 'rb'
                elif any(x in match_text for x in ['jt', 'juta']):
                    unit = 'jt'
                
                amount = convert_to_rupiah(groups[0], unit)
                
                if amount:
                    if pattern_type == "MAX":
                        budget_range = (None, amount)
                        print(f"💰 Budget max detected: Under IDR {amount:,}")
                    elif pattern_type == "MIN":
                        budget_range = (amount, None)
                        print(f"💰 Budget min detected: Above IDR {amount:,}")
                    elif pattern_type == "EXACT":
                        # Create range around the amount (±20%)
                        min_range = int(amount * 0.8)
                        max_range = int(amount * 1.2)
                        budget_range = (min_range, max_range)
                        print(f"💰 Budget exact detected: IDR {min_range:,} - IDR {max_range:,}")
                    
                    user_context["budget_range"] = budget_range
                    return budget_range
            
            elif pattern_type == "BUDGET_FRIENDLY":
                budget_range = (None, 200000)  # Under 200rb
                print(f"💰 Budget-friendly detected: Under IDR 200,000")
                user_context["budget_range"] = budget_range
                return budget_range
                
            elif pattern_type == "MID_RANGE":
                budget_range = (200000, 500000)  # 200rb - 500rb
                print(f"💰 Mid-range detected: IDR 200,000 - IDR 500,000")
                user_context["budget_range"] = budget_range
                return budget_range
                
            elif pattern_type == "PREMIUM":
                budget_range = (500000, None)  # Above 500rb
                print(f"💰 Premium detected: Above IDR 500,000")
                user_context["budget_range"] = budget_range
                return budget_range
                
            elif pattern_type == "NO_LIMIT":
                user_context["budget_range"] = None
                print(f"💰 No budget limit detected")
                return "NO_LIMIT"
    
    return None

class SessionLanguageManager:
    """
    ENHANCED: Better session language management
    """
    def __init__(self):
        self.session_languages = {}
        
    def detect_or_retrieve_language(self, text, session_id):
        """
        ENHANCED: Better language detection and caching
        """
        if session_id in self.session_languages:
            cached_lang = self.session_languages[session_id]
            print(f"🔄 Using cached language for session {session_id}: '{cached_lang}'")
            return cached_lang
        
        try:
            if text and text.strip():
                lang = detect(text)
                self.session_languages[session_id] = lang
                print(f"🆕 New language detected for session {session_id}: '{lang}'")
                return lang
            print(f"⚠️ Empty text for session {session_id}, defaulting to 'en'")
            return "en"
        except Exception as e:
            print(f"❌ Language detection error for session {session_id}: {e}")
            return "en"
            
    def reset_session(self, session_id):
        if session_id in self.session_languages:
            del self.session_languages[session_id]
            print(f"🗑️ Reset language for session {session_id}")

def detect_language(text):
    """
    ENHANCED: Better language detection with fallback handling
    """
    try:
        if not text or not text.strip():
            raise ValueError("Input text is empty or invalid.")
        
        # Use langdetect library
        detected = detect(text)
        print(f"🔍 Language detected: '{detected}' for text: '{text[:50]}...'")
        return detected
        
    except Exception as e:
        print(f"❌ Language detection error: {e}")
        print(f"   Text: '{text[:100]}...'")
        return "en"  # Default to English

session_manager = SessionLanguageManager()

def translate_text(text, target_language, session_id=None):
    try:
        if session_id and session_id in session_manager.session_languages:
            source_language = session_manager.session_languages[session_id]
        else:
            source_language = detect_language(text)
            if session_id:
                session_manager.session_languages[session_id] = source_language
        
        if source_language == target_language:
            return text

        translated_text = MyMemoryTranslator(source=source_language, target=target_language).translate(text)
        return translated_text

    except Exception as e:
        print(f"Error during translation: {e}")
        return text

def render_markdown(text: str) -> str:
    extensions = [
        'tables',
        'nl2br',
        'fenced_code',
        'smarty'
    ]
    html_content = markdown(text, extensions=extensions)
    return html_content

cloudinary.config(
    cloud_name="dn0xl1q3g",
    api_key="252519847388784",
    api_secret="pzLNZgLzfMQ9bmwiIRoyjRFqqkU"
)

def upload_to_cloudinary(file_location):
    try:
        if not os.path.exists(file_location):
            logging.error(f"File not found at location: {file_location}")
            return None
            
        file_size = os.path.getsize(file_location)
        if file_size == 0:
            logging.error(f"File exists but is empty (0 bytes): {file_location}")
            return None
            
        transformation = {
            'quality': 'auto',
            'fetch_format': 'auto',
        }

        response = cloudinary.uploader.upload(
            file_location, 
            folder="uploads/",
            transformation=transformation
        )
        return response['url']
    except Exception as e:
        logging.error(f"Cloudinary upload error: {e}")
        return None

async def is_small_talk(input_text):
    greetings = ["hello", "hi", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening", "selamat pagi", "pagi", "selamat siang", "siang", "malam", "selamat malam"]
    return input_text.lower() in greetings or re.match(r"^\s*(hi|hello|hey)\s*$", input_text, re.IGNORECASE)

async def analyze_uploaded_image(image_url: str):
    try:
        if not image_url:
            return "Error: No image URL provided."
        
        logging.info(f"Analyzing image at URL: {image_url}")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI fashion consultant specializing in detailed analysis of clothing and appearance. "
                                "Provide comprehensive, structured analysis focusing on the following aspects:\n\n"
                                
                                "FOR CLOTHING ITEMS:\n"
                                "1. Type & Category: Precisely identify the specific garment type(s)\n"
                                "2. Color & Pattern: Describe dominant and accent colors, pattern types\n"
                                "3. Fabric & Texture: Identify material composition if visible\n"
                                "4. Style Classification: Casual, formal, business, streetwear, etc.\n"
                                "5. Design Elements: Note distinctive features, cuts, shapes\n"
                                "6. Fit Profile: Loose, tight, oversized, tailored\n\n"
                                
                                "FOR PEOPLE IN IMAGES:\n"
                                "1. Body Structure: Height range, build type (slender, athletic, etc.)\n"
                                "2. Proportions: Shoulder width, waist-to-hip ratio, limb length\n"
                                "3. Physical Features: Skin tone (using neutral descriptors)\n\n"
                                
                                "FORMAT YOUR RESPONSE IN THESE DISTINCT SECTIONS:\n"
                                "CLOTHING ANALYSIS: [clothing details organized by category]\n"
                                "PHYSICAL ATTRIBUTES: [objective body structure information]\n"
                                "KEY STYLE ELEMENTS: [3-5 most distinctive features as simple bullet points]\n\n"
                                
                                "Important rules:\n"
                                "- Use objective, technical language and avoid subjective assessments\n"
                                "- Never comment on attractiveness or make value judgments\n"
                                "- Avoid gender assumptions unless clearly evident\n"
                                "- Be specific and detailed with the clothing analysis\n"
                                "- Focus on details that would be relevant for finding similar items"
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this image in detail, focusing on clothing items and relevant physical attributes for fashion recommendations.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                    },
                                },
                            ] ,
                        },
                    ],
                    max_tokens=600,
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                return analysis
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.info(f"Failed to analyze image at URL: {image_url}. Error: {str(e)}")
                    return f"Error: Unable to analyse image. Please try again or use text description instead."
                else:
                    await asyncio.sleep(2)

    except Exception as e:
        print(f"Error during image analysis: {e}")
        return f"Error: {str(e)}"

# ================================
# FASTAPI ROUTES (UNCHANGED)
# ================================

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with smart embedding initialization"""
    print("🚀 Starting Enhanced Fashion Chatbot with Hybrid Intelligence...")
    
    # Create upload directory
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Connect to database
    await database.connect()
    await create_tables()
    
    # Initialize NLP
    global nlp
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ Spacy English model loaded")
    except OSError:
        print("⚠️ Spacy English model not found. Install with: python -m spacy download en_core_web_sm")
        nlp = None
    
    # Initialize hybrid system
    print("🧠 Hybrid LLM + Vector system ready!")
    print("✅ Enhanced Fashion Chatbot is ready!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload(user_input: str = Form(None), file: UploadFile = None):
    if not file and not user_input:
        return JSONResponse(content={"success": False, "error": "No input or file received"})

    try:
        if file:
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Invalid file type.")

            file_content = await file.read()
            file_size = len(file_content)
            
            if file_size == 0:
                return JSONResponse(content={"success": False, "error": "Uploaded file is empty."})
                
            if file_size > 5 * 1024 * 1024:
                return JSONResponse(content={"success": False, "error": "File size exceeds 5MB limit."})
            
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
            
            unique_id = uuid.uuid4()
            sanitized_filename = slugify(file.filename.rsplit(".", 1)[0], lowercase=False)
            unique_filename = f"{unique_id}_{sanitized_filename}.{file_extension}"
            file_location = os.path.join(UPLOAD_DIR, unique_filename)
            
            with open(file_location, "wb") as file_object:
                file_object.write(file_content)
            
            image_url = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    image_url = upload_to_cloudinary(file_location)
                    if image_url:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to upload to Cloudinary after {max_retries} attempts: {str(e)}")
                    else:
                        time.sleep(1)

            if image_url:
                return JSONResponse(content={"success": True, "file_url": image_url})
            else:
                return JSONResponse(content={"success": False, "error": "Failed to upload image to Cloudinary."})

        elif user_input:
            return JSONResponse(content={"success": True})

        return JSONResponse(content={"success": False, "error": "No input or file received"})

    except Exception as e:
        logging.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during file upload.")

@app.post("/chat/save")
async def save_message(message: ChatMessage, db: AsyncSession = Depends(get_db)):
    try:
        new_message = ChatHistoryDB(
            session_id=message.session_id,
            message_type=message.message_type,
            content=message.content
        )
        db.add(new_message)
        await db.commit()
        return {"success": True}
    except Exception as e:
        await db.rollback()
        logging.error(f"Error saving message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        query = select(ChatHistoryDB).where(
            ChatHistoryDB.session_id == session_id
        ).order_by(ChatHistoryDB.timestamp)

        result = await db.execute(query)
        messages = result.scalars().all()

        return ChatHistoryResponse(
            messages=[
                ChatMessage(
                    session_id=msg.session_id,
                    message_type=msg.message_type,
                    content=msg.content
                ) for msg in messages
            ]
        )
            
    except Exception as e:
        await db.rollback()
        logging.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# ENHANCED WEBSOCKET HANDLER WITH HYBRID INTELLIGENCE
# ================================

# Add this function before the websocket handler
async def generate_style_recommendations_from_consultation(user_context, user_language, session_id):
    """
    ENHANCED: Generate AI-powered style recommendations with specific clothing type guidance
    """
    
    gender = user_context.get("user_gender", {}).get("category", "Not specified")
    accumulated_keywords = user_context.get("accumulated_keywords", {})
    
    # Initialize preference lists including daily activities
    style_preferences = []
    clothing_interests = []
    fit_preferences = []
    sleeve_preferences = []
    color_preferences = []
    occasion_needs = []
    body_info = []
    length_preferences = []
    neckline_preferences = []
    material_preferences = []
    daily_activities = []  # NEW: Daily activity preferences
    
    print(f"\n🎨 ENHANCED STYLE RECOMMENDATIONS - INCLUDING SPECIFIC CLOTHING TYPES:")
    print("=" * 60)
    
    # ENHANCED: Use FashionCategories for consistent categorization including activities
    for keyword, data in accumulated_keywords.items():
        if keyword is None:
            continue
            
        weight = data.get("weight", 0) if isinstance(data, dict) else data
        source = data.get("source", "unknown") if isinstance(data, dict) else "unknown"
        
        # Use lower threshold for style recommendations to capture more details
        if source == "consultation":
            threshold = 100
        elif source == "user_input":
            threshold = 500
        else:
            threshold = 1000
        
        print(f"🔍 Analyzing: '{keyword}' → Weight: {weight:.0f}, Source: {source}")
        
        if weight > threshold:
            # ENHANCED: Use FashionCategories.get_category with activity support
            category = FashionCategories.get_category(keyword)
            
            if category == 'body_terms':
                body_info.append(keyword)
                print(f"   ✅ BODY: '{keyword}'")
            elif category == 'activity_terms':  # NEW: Daily activity detection
                daily_activities.append(keyword)
                print(f"   ✅ DAILY ACTIVITY: '{keyword}'")
            elif category == 'sleeve_terms':
                sleeve_preferences.append(keyword)
                print(f"   ✅ SLEEVES: '{keyword}'")
            elif category == 'style_terms':
                style_preferences.append(keyword)
                print(f"   ✅ STYLE: '{keyword}'")
            elif category == 'clothing_terms':
                clothing_interests.append(keyword)
                print(f"   ✅ CLOTHING: '{keyword}'")
            elif category == 'fit_terms':
                fit_preferences.append(keyword)
                print(f"   ✅ FIT: '{keyword}'")
            elif category == 'color_terms':
                color_preferences.append(keyword)
                print(f"   ✅ COLORS: '{keyword}'")
            elif category == 'occasion_terms':
                occasion_needs.append(keyword)
                print(f"   ✅ OCCASIONS: '{keyword}'")
            elif category == 'length_terms':
                length_preferences.append(keyword)
                print(f"   ✅ LENGTH: '{keyword}'")
            elif category == 'neckline_terms':
                neckline_preferences.append(keyword)
                print(f"   ✅ NECKLINE: '{keyword}'")
            elif category == 'material_terms':
                material_preferences.append(keyword)
                print(f"   ✅ MATERIAL: '{keyword}'")
            elif FashionCategories.is_color_preference(keyword):
                # Special color detection for terms not in exact list
                color_preferences.append(keyword)
                print(f"   ✅ SPECIAL COLOR: '{keyword}'")
            else:
                print(f"   ❓ UNCATEGORIZED: '{keyword}' (category: {category})")
        else:
            print(f"   ⏭️ BELOW THRESHOLD: '{keyword}'")
    
    # Clean and deduplicate preferences
    def clean_preferences(pref_list, max_items=3):
        """Remove duplicates and limit to max items"""
        seen = set()
        cleaned = []
        for item in pref_list:
            item_lower = item.lower()
            if item_lower not in seen and len(cleaned) < max_items:
                seen.add(item_lower)
                cleaned.append(item)
        return cleaned
    
    # Apply cleaning to all preference lists including activities
    body_info = clean_preferences(body_info, 2)
    daily_activities = clean_preferences(daily_activities, 3)  # NEW: Clean daily activities
    sleeve_preferences = clean_preferences(sleeve_preferences, 2)
    style_preferences = clean_preferences(style_preferences, 3)
    clothing_interests = clean_preferences(clothing_interests, 3)
    fit_preferences = clean_preferences(fit_preferences, 2)
    color_preferences = clean_preferences(color_preferences, 3)
    occasion_needs = clean_preferences(occasion_needs, 3)
    length_preferences = clean_preferences(length_preferences, 2)
    neckline_preferences = clean_preferences(neckline_preferences, 2)
    material_preferences = clean_preferences(material_preferences, 2)
    
    print(f"\n📋 CLEANED PREFERENCES FOR ENHANCED STYLE RECOMMENDATIONS:")
    print(f"   Body Info: {body_info}")
    print(f"   Daily Activities: {daily_activities}")  # NEW: Debug daily activities
    print(f"   Sleeves: {sleeve_preferences}")
    print(f"   Style: {style_preferences}")
    print(f"   Clothing: {clothing_interests}")
    print(f"   Fit: {fit_preferences}")
    print(f"   Colors: {color_preferences}")
    print(f"   Occasions: {occasion_needs}")
    print(f"   Length: {length_preferences}")
    print(f"   Necklines: {neckline_preferences}")
    print(f"   Materials: {material_preferences}")
    
    # ENHANCED: Create comprehensive prompt with specific clothing type guidance
    style_prompt = f"""You are an expert fashion consultant with deep understanding of how clothing interacts with different body types, proportions, and personal attributes. Your mission is to provide personalized, thoughtful fashion recommendations that enhance each user's unique features.\n\n

**Complete User Profile:**
- **Gender:** {gender}
- **Body Information:** {', '.join(body_info) if body_info else 'Not specified'}
- **Daily Activity/Lifestyle:** {', '.join(daily_activities) if daily_activities else 'Not specified'}
- **Overall Style:** {', '.join(style_preferences) if style_preferences else 'Not specified'}
- **Clothing Types Interested In:** {', '.join(clothing_interests) if clothing_interests else 'Not specified'}
- **Preferred Fit:** {', '.join(fit_preferences) if fit_preferences else 'Not specified'}
- **Sleeve Preferences:** {', '.join(sleeve_preferences) if sleeve_preferences else 'Not specified'}
- **Length Preferences:** {', '.join(length_preferences) if length_preferences else 'Not specified'}
- **Neckline Preferences:** {', '.join(neckline_preferences) if neckline_preferences else 'Not specified'}
- **Color Preferences:** {', '.join(color_preferences) if color_preferences else 'Not specified'}
- **Material Preferences:** {', '.join(material_preferences) if material_preferences else 'Not specified'}
- **Occasions:** {', '.join(occasion_needs) if occasion_needs else 'Not specified'}

**Instructions:**
Please provide detailed, actionable style recommendations organized as follows:

**1. Specific Clothing Type Recommendations**
For EACH clothing type the user is interested in, provide detailed guidance:

*If they want SHIRTS/KEMEJA:*
- Specific shirt types (button-down, blouse, polo, henley, etc.)
- Best necklines for their body type and preferences
- Sleeve styles that work best for them
- Fit recommendations (fitted, relaxed, oversized) based on their body
- Colors and patterns that complement their skin tone
- Styling tips for their daily activities

*If they want DRESSES/GAUN:*
- Dress silhouettes that flatter their body type (A-line, bodycon, shift, wrap, etc.)
- Best lengths for their preferences and body proportions
- Neckline styles that work for them
- Sleeve options that suit their preferences
- Colors and patterns for their style
- Occasion-appropriate styling

*If they want PANTS/CELANA:*
- Pant styles that flatter their body type (straight, wide-leg, skinny, bootcut, etc.)
- Best rise (high-waist, mid-rise, low-rise) for their body
- Length recommendations
- Fit preferences that work for their lifestyle
- Colors and styling tips

*If they want SKIRTS/ROK:*
- Skirt styles that complement their body type (A-line, pencil, pleated, etc.)
- Best lengths for their body proportions
- Waistline styles that flatter them
- Colors and styling suggestions

*If they want BLAZERS/JACKETS:*
- Blazer styles that work for their body type and activities
- Best fit and structure for their preferences
- Styling for both casual and formal occasions
- Color recommendations

**2. Body Type Specific Recommendations**
Based on their body information, explain:
- Silhouettes that enhance their natural proportions
- Clothing details that create the most flattering look
- What to emphasize and what to balance
- Specific cuts and styles to look for

**3. Lifestyle Integration**
Based on their daily activities and occasions:
- How to style each clothing type for their work/lifestyle
- Versatile pieces that work for multiple activities
- Practical considerations for their daily routine
- Transition pieces from day to night or casual to formal

**4. Color & Styling Strategy**
- Specific color recommendations for each clothing type
- Color combinations that work with their preferences
- How to incorporate their preferred colors into different pieces
- Pattern suggestions that suit their style and body type

**5. Complete Outfit Ideas** (3-4 outfit suggestions)
Create complete outfits using their preferred clothing types:
- Exact pieces (specify the recommended clothing types from above)
- How each piece flatters their body type
- Why it matches their style and activity needs
- Styling details and accessories

**6. Shopping Tips**
- Key details to look for when shopping for each clothing type
- Fit checkpoints specific to their body type
- Quality indicators for their preferred materials
- How to spot pieces that match their style preferences

Format with clear markdown headers and bullet points. Be very specific with clothing names, cuts, and styling details. Make sure to address each clothing type they're interested in with detailed, personalized guidance.

End with asking if they'd like to see actual product recommendations that match these specific clothing type suggestions."""

    try:
        print(f"🤖 Generating enhanced clothing-specific style recommendations for user language: {user_language}")
        
        # Generate style recommendations using OpenAI
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": style_prompt}],
            temperature=0.7,
            max_tokens=1500  # Increased for more detailed clothing-specific recommendations
        )
        
        style_recommendations = response.choices[0].message.content.strip()
        
        # Add enhanced call-to-action with clothing-specific language
        style_recommendations += "\n\n---\n\n**Ready to Find Your Perfect Pieces?** 🛍️\n\nSay **'yes'**, **'looks good'**, or **'show me products'** to see actual clothing items that match these specific recommendations for your body type, style, and lifestyle needs!"
        
        # UPDATED: Better translation handling
        if user_language != "en":
            print(f"🌍 Translating enhanced clothing-specific recommendations from English to {user_language}")
            try:
                # Use the centralized translation function
                translated_recommendations = translate_text(style_recommendations, user_language, session_id)
                print(f"✅ Enhanced clothing-specific recommendations translation successful")
                return translated_recommendations
            except Exception as e:
                print(f"❌ Style recommendations translation failed: {e}")
                print(f"   Keeping original English version")
                # Return English version if translation fails
                return style_recommendations
        
        return style_recommendations
        
    except Exception as e:
        print(f"❌ Error generating enhanced clothing-specific style recommendations: {e}")
        
        # ENHANCED: Better fallback message with clothing-specific guidance
        fallback_sections = []
        
        if clothing_interests:
            fallback_sections.append(f"**Clothing Types:** {', '.join(clothing_interests[:3])}")
        
        if style_preferences:
            fallback_sections.append(f"**Your Style:** {', '.join(style_preferences[:2])}")
        
        if body_info:
            fallback_sections.append(f"**Body Type:** {', '.join(body_info[:2])}")
        
        if fit_preferences:
            fallback_sections.append(f"**Preferred Fit:** {', '.join(fit_preferences[:2])}")
        
        if daily_activities:
            fallback_sections.append(f"**Daily Activities:** {', '.join(daily_activities[:2])}")
        
        if sleeve_preferences:
            fallback_sections.append(f"**Sleeve Style:** {', '.join(sleeve_preferences[:2])}")
        
        if color_preferences:
            fallback_sections.append(f"**Colors:** {', '.join(color_preferences[:2])}")
        
        if occasion_needs:
            fallback_sections.append(f"**Occasions:** {', '.join(occasion_needs[:2])}")
        
        # Create personalized fallback message with clothing-specific guidance
        if fallback_sections:
            personalized_info = "\n".join([f"• {section}" for section in fallback_sections])
            
            # Create clothing-specific recommendations based on detected interests
            clothing_specific_tips = ""
            if clothing_interests:
                clothing_specific_tips = "\n\n**Specific Recommendations for Your Clothing Interests:**\n"
                for clothing in clothing_interests[:3]:
                    if 'shirt' in clothing.lower() or 'kemeja' in clothing.lower():
                        clothing_specific_tips += f"• **{clothing.title()}**: Look for fits that complement your body type, choose necklines that flatter you, and select colors from your preferred palette\n"
                    elif 'dress' in clothing.lower() or 'gaun' in clothing.lower():
                        clothing_specific_tips += f"• **{clothing.title()}**: Choose silhouettes that enhance your proportions, select lengths that work for your lifestyle, and pick styles suitable for your occasions\n"
                    elif 'pants' in clothing.lower() or 'celana' in clothing.lower():
                        clothing_specific_tips += f"• **{clothing.title()}**: Select cuts that flatter your body shape, choose rises that work for your torso, and pick styles that match your activity level\n"
                    elif 'skirt' in clothing.lower() or 'rok' in clothing.lower():
                        clothing_specific_tips += f"• **{clothing.title()}**: Choose styles that complement your hip and waist proportions, select lengths that work for your height, and pick fits that match your comfort level\n"
                    elif 'blazer' in clothing.lower() or 'jaket' in clothing.lower():
                        clothing_specific_tips += f"• **{clothing.title()}**: Look for structures that enhance your silhouette, choose lengths that proportion well with your body, and select styles that work for your professional needs\n"
                    else:
                        clothing_specific_tips += f"• **{clothing.title()}**: Choose pieces that align with your style preferences, fit well with your body type, and work for your daily activities\n"
            
            fallback_message = f"""**Your Personalized Style Recommendations** ✨

Based on your consultation, here's what works best for you:

{personalized_info}{clothing_specific_tips}

**General Guidelines:**
• Choose clothing that complements your body type and enhances your best features
• Stick to your preferred fits and styles for confidence and comfort
• Incorporate your favorite colors into each clothing type you're interested in
• Select pieces suitable for your lifestyle and daily activities
• Focus on quality pieces that can be mixed and matched across occasions

**Next Steps:**
Say **'yes'** or **'show me products'** to see clothing recommendations that match your specific style and body type needs! 🛍️"""
        else:
            # Generic fallback if no preferences detected
            fallback_message = """**Your Personalized Style Recommendations** ✨

I'd love to provide more specific clothing recommendations, but I need a bit more information about your preferences.

**General Style Tips:**
• **For Shirts**: Choose necklines and fits that flatter your body type
• **For Dresses**: Select silhouettes that enhance your natural proportions  
• **For Pants**: Pick cuts and rises that work best for your body shape
• **For Skirts**: Choose lengths and styles that complement your figure
• **For Blazers**: Look for structures that create a polished, flattering silhouette

**Next Steps:**
Say **'yes'** or **'show me products'** to explore clothing options, or share more about your style preferences for better personalized recommendations! 🛍️"""
        
        # UPDATED: Translate fallback message if needed
        if user_language != "en":
            print(f"🌍 Translating clothing-specific fallback message from English to {user_language}")
            try:
                translated_fallback = translate_text(fallback_message, user_language, session_id)
                print(f"✅ Fallback message translation successful")
                return translated_fallback
            except Exception as e:
                print(f"❌ Fallback message translation failed: {e}")
                print(f"   Keeping original English version")
        
        return fallback_message
        
@app.websocket("/ws")
async def chat(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    try:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        await websocket.send_text(f"{session_id}|🧠 Enhanced Fashion Assistant with Hybrid Intelligence Ready!\n\nSelamat Datang! Bagaimana saya bisa membantu Anda hari ini?\n\nWelcome! How can I help you today?")

        # Initialize enhanced system for this session
        print(f"✅ Session {session_id}: Hybrid LLM + Vector system ENABLED")

        # Enhanced system prompt (keep your existing one)
        message_objects = [{
            "role": "system",
            "content": (
                "You are an expert fashion consultant with deep understanding of how clothing interacts with different body types, proportions, and personal attributes. Your mission is to provide personalized, thoughtful fashion recommendations that enhance each user's unique features.\n\n"
                
                "CONVERSATION CONTEXT & MEMORY:\n"
                "- Always remember and reference information the user has shared throughout our conversation\n"
                "- Build upon previous recommendations and acknowledge their preferences or concerns\n"
                "- If they mention budget, lifestyle, or specific needs, keep these in mind for all future suggestions\n"
                "- Reference their previous questions or comments to show you're actively listening\n\n"
                
                "CONSULTATION PROCESS:\n"
                "Gather information through natural, conversational questions. Ask 2-3 questions at a time to avoid overwhelming the user. Make sure to cover ALL categories below before providing recommendations.\n\n"
                
                "ESSENTIAL INFORMATION TO GATHER:\n\n"
                
                "**Basic Personal Information:**\n"
                "- Gender identity and preferred clothing styles\n"
                "- Height and weight (for proportion considerations)\n"
                "- Body shape/type if comfortable sharing (apple, pear, hourglass, rectangle, inverted triangle)\n"
                "- Skin tone and undertones (warm, cool, or neutral)\n"
                "- Lifestyle (professional, casual, active, student, etc.)\n"
                "- Budget range if relevant\n"
                "- Any body areas they want to highlight or feel more confident about\n"
                "- Any areas they prefer to minimize or feel less confident about\n\n"
                
                "**Detailed Style Preferences (Ask with Clear Examples):**\n\n"
                
                "**Sleeve Preferences** - Ask: 'What sleeve styles do you prefer?'\n"
                "Provide examples and ask them to choose:\n"
                "- Sleeveless (tank tops, sleeveless blouses)\n"
                "- Cap sleeves (very short sleeves that just cover the shoulder)\n"
                "- Short sleeves (t-shirt style, ending mid-upper arm)\n"
                "- 3/4 sleeves (ending between elbow and wrist)\n"
                "- Long sleeves (full arm coverage)\n"
                "- Bell sleeves (fitted at shoulder, flaring out)\n"
                "- Fitted sleeves vs. loose/flowing sleeves\n"
                "- Comfort level with showing arms\n\n"
                
                "**Fit Preferences** - Ask: 'How do you prefer your clothes to fit your body?'\n"
                "Provide examples and ask them to choose:\n"
                "- Slim/tailored fit (follows body shape closely, like fitted blazers or skinny jeans)\n"
                "- Regular fit (comfortable fit with some room, like straight-leg jeans or classic t-shirts)\n"
                "- Relaxed/oversized fit (loose and roomy, like boyfriend jeans or oversized sweaters)\n"
                "- Body-hugging (form-fitting like bodycon dresses or compression tops)\n"
                "- Flowing/loose (drapes away from body like A-line dresses or palazzo pants)\n"
                "- Different preferences for tops vs. bottoms\n\n"
                
                "**Length Preferences** - Ask: 'What lengths do you prefer for different clothing items?'\n"
                "Provide examples for each category:\n\n"
                
                "*Top Lengths:*\n"
                "- Cropped (ends above waist, showing midriff)\n"
                "- Regular (ends at waist or hip bone)\n"
                "- Tunic length (ends mid-thigh, longer than regular)\n"
                "- Long/oversized (ends at or below mid-thigh)\n\n"
                
                "*Bottom Lengths:*\n"
                "- Mini (very short, ends mid-thigh)\n"
                "- Above knee (ends 2-4 inches above knee)\n"
                "- Knee length (ends at the knee)\n"
                "- Below knee/midi (ends between knee and ankle)\n"
                "- Maxi/full length (ends at or near ankle)\n\n"
                
                "*Dress Lengths:*\n"
                "- Mini dress (ends mid-thigh)\n"
                "- Knee-length dress\n"
                "- Midi dress (ends between knee and ankle)\n"
                "- Maxi dress (ends at or near ankle)\n\n"
                
                "**Additional Style Details:**\n"
                "- Neckline preferences (V-neck, crew neck, scoop neck, off-shoulder, etc.)\n"
                "- Pattern preferences (solid colors, stripes, florals, geometric, etc.)\n"
                "- Color preferences and colors to avoid\n"
                "- Fabric preferences (cotton, silk, denim, knits, etc.)\n"
                "- Occasion focus (work, casual, special events, etc.)\n\n"
                
                "RECOMMENDATION FORMAT:\n"
                "Only provide recommendations AFTER gathering ALL necessary information. Provide at least 3-5 detailed recommendations using this structure:\n\n"
                
                "**[Clothing Item Name]**\n"
                "- *Why it works for you:* Detailed explanation addressing their body type, proportions, and skin tone\n"
                "- *Perfect fit for your preferences:* How it matches their stated sleeve, fit, and length preferences\n"
                "- *Styling tips:* Specific suggestions for colors, accessories, and how to wear it\n"
                "- *Lifestyle compatibility:* How it fits their daily life and occasions\n\n"
                
                "CONSULTATION CONFIRMATION PROCESS:\n"
                "Before providing final recommendations, always provide this comprehensive summary:\n\n"
                
                "**CONSULTATION SUMMARY**\n"
                "Let me confirm all the information I've gathered to ensure my recommendations will be perfectly tailored for you:\n\n"
                
                "**Personal Details:**\n"
                "- Gender Identity & Style: [user's response]\n"
                "- Height & Weight: [user's response]\n"
                "- Body Shape: [user's response]\n"
                "- Skin Tone & Undertones: [user's response]\n"
                "- Lifestyle: [user's response]\n"
                "- Budget: [user's response if mentioned]\n"
                "- Areas to highlight: [user's response]\n"
                "- Areas to minimize: [user's response]\n\n"
                
                "**Style Preferences:**\n"
                "- Sleeve Preferences: [user's specific choices from the options provided]\n"
                "- Fit Preferences: [user's specific choices from the options provided]\n"
                "- Length Preferences: [user's specific choices for tops, bottoms, and dresses]\n"
                "- Neckline Preferences: [user's response]\n"
                "- Pattern & Color Preferences: [user's response]\n"
                "- Fabric Preferences: [user's response]\n"
                "- Primary Occasions: [user's response]\n\n"
                
                "**Confirmation Question:**\n"
                "'Is all of this information accurate and complete? Please let me know if anything needs to be corrected or if you'd like to add any additional preferences. Once you confirm this summary is correct, I'll provide you with 3-5 personalized style recommendations that perfectly match your body type, lifestyle, and all your specific preferences!'\n\n"
                
                "IMPORTANT GUIDELINES:\n"
                "- Never make assumptions about preferences - always ask directly with clear examples\n"
                "- If a user seems unsure about any category, provide more examples and context\n"
                "- Always explain WHY certain styles work for their specific combination of body type and preferences\n"
                "- Be encouraging and positive while being honest about what will work best\n"
                "- Remember that confidence is the best accessory - help them feel great in their choices\n"
            )
        }]
        
        last_ai_response = ""
        
        # Enhanced user context with hybrid capabilities
        user_context = {
            "current_image_url": None,
            "current_text_input": None,
            "pending_image_analysis": False,
            "has_shared_image": False,
            "has_shared_preferences": False,
            "last_query_type": None,
            "awaiting_confirmation": False,
            "confirmation_type": None,  # NEW: Track what type of confirmation we're waiting for
            "accumulated_keywords": {},
            "preferences": {},
            "known_attributes": {},
            "user_gender": {
                "category": None,
                "term": None, 
                "confidence": 0,
                "last_updated": None
            },
            "product_cache": {
                "all_results": pd.DataFrame(),
                "current_page": 0,
                "products_per_page": 5,
                "last_search_params": {},
                "has_more": False
            },
            "semantic_enabled": True,  # Always enabled with hybrid system
            "cultural_context": {},
        }

        while True:            
            try:
                data = await websocket.receive_text()
                logging.info(f"Received Websocket data: {data}")
                
                if "|" not in data:
                    await websocket.send_text(f"{session_id}|Invalid message format.")
                    continue
                    
                session_id, user_input = data.split("|", 1)
                
                # Save user message to database
                new_user_message = ChatHistoryDB(
                    session_id=session_id,
                    message_type="user",
                    content=user_input
                )
                db.add(new_user_message)
                await db.commit()
                
                # Detect language
                try:
                    user_language = session_manager.detect_or_retrieve_language(user_input, session_id)
                    logging.info(f"User language '{user_language}' for session {session_id}")
                except Exception as e:
                    logging.error(f"Language detection error: {str(e)}")
                    user_language = "en"
                
                # UPDATED: Enhanced confirmation handling with 3-step flow
                if user_context["awaiting_confirmation"]:
                    # Process confirmation response
                    is_positive = user_input.strip().lower() in ["yes", "ya", "iya", "sure", "tentu", "ok", "okay"]
                    is_negative = user_input.strip().lower() in ["no", "tidak", "nope", "nah", "tidak usah"]
                    is_more_request = detect_more_products_request(user_input)
                    
                    # Enhanced confirmation detection
                    confirmation_phrases = [
                        'this is correct', 'looks good', 'that\'s right', 'accurate', 
                        'yes this is right', 'benar', 'tepat', 'sesuai', 'iya benar',
                        'proceed', 'lanjut', 'looks accurate', 'this looks good', 
                        'terlihat bagus', 'bagus', 'cocok', 'betul', 'oke', 'sudah benar'
                    ]
                    
                    is_confirming = is_positive or any(phrase in user_input.lower() for phrase in confirmation_phrases)
                    
                    # STEP 1: Handle confirmation based on confirmation type
                    if is_confirming and user_context.get("confirmation_type") == "summary":
                        # User confirmed consultation summary → Generate AI style recommendations
                        print("✅ User confirmed summary → Generating style recommendations")
                        
                        try:
                            # Generate style recommendations using the new function
                            style_recommendations = await generate_style_recommendations_from_consultation(
                                user_context, user_language, session_id
                            )
                            
                            # Update confirmation type for next step
                            user_context["confirmation_type"] = "style_recommendations"
                            user_context["awaiting_confirmation"] = True
                            
                            # Save and send style recommendations
                            new_ai_message = ChatHistoryDB(
                                session_id=session_id,
                                message_type="assistant",
                                content=style_recommendations
                            )
                            db.add(new_ai_message)
                            await db.commit()
                            
                            ai_response_html = render_markdown(style_recommendations)
                            await websocket.send_text(f"{session_id}|{ai_response_html}")
                            
                        except Exception as e:
                            logging.error(f"Error generating style recommendations: {str(e)}")
                            error_msg = "I'm sorry, I had trouble generating style recommendations. Let me try again."
                            if user_language != "en":
                                error_msg = translate_text(error_msg, user_language, session_id)
                            await websocket.send_text(f"{session_id}|{error_msg}")
                        
                        continue
                    
                    # STEP 2: Handle style recommendations confirmation → Product search
                    elif is_confirming and user_context.get("confirmation_type") == "style_recommendations":
                        print("✅ User confirmed style recommendations → Searching products")
                        try:
                            # Use enhanced product search with consultation keyword boost
                            recommended_products, positive_response = await enhanced_product_search_in_websocket(
                                user_context, db, user_language, session_id
                            )
                            
                            # Cache results for pagination
                            user_context["product_cache"]["all_results"] = recommended_products
                            user_context["product_cache"]["current_page"] = 0
                            user_context["product_cache"]["has_more"] = len(recommended_products) > 5

                            first_page_products, has_more = get_paginated_products(
                                recommended_products,
                                page=0, 
                                products_per_page=5
                            )

                            user_context["product_cache"]["has_more"] = has_more

                            # Create enhanced response using your existing product card format
                            if not first_page_products.empty:
                                complete_response = positive_response + "\n\n"
                                
                                for _, row in first_page_products.iterrows():
                                    product_card = (
                                        "<div class='product-card'>\n"
                                        f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                        f"<div class='product-info'>\n"
                                        f"<h3>🎯 **Enhanced Match** | {row['product']}</h3>\n"
                                        f"<p class='price'>IDR {row['price']:,.0f}</p>\n"
                                        f"<p class='description'>{row['description']}</p>\n"
                                        f"<p class='available'>Available in size: {row['size']}, Color: {row['color']}</p>\n"
                                        f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                        "</div>\n"
                                        "</div>\n"
                                    )
                                    complete_response += product_card
                                
                                if has_more:
                                    complete_response += "\n\n🧠 **Want to see more AI-matched products?** Just ask for 'more products' or 'lainnya'!"
                            else:
                                complete_response = positive_response + "\n\nI'm sorry, but I couldn't find specific product recommendations at the moment. Would you like me to help you with something else?"

                            # Handle translation while protecting HTML using your existing logic
                            if user_language != "en":
                                try:
                                    def encode_html_blocks(text):
                                        pattern = r'(<div class=\'product-card\'>.*?</div>\n)'
                                        blocks = []
                                        
                                        def replace_block(match):
                                            nonlocal blocks
                                            placeholder = f"__HTML_BLOCK_{len(blocks)}__"
                                            blocks.append(match.group(0))
                                            return placeholder
                                        
                                        protected_text = re.sub(pattern, replace_block, text, flags=re.DOTALL)
                                        return protected_text, blocks
                                    
                                    def decode_html_blocks(text, blocks):
                                        for i, block in enumerate(blocks):
                                            text = text.replace(f"__HTML_BLOCK_{i}__", block)
                                        return text
                                    
                                    protected_text, html_blocks = encode_html_blocks(complete_response)
                                    translated_protected = translate_text(protected_text, user_language, session_id)
                                    translated_response = decode_html_blocks(translated_protected, html_blocks)
                                    
                                except Exception as e:
                                    logging.error(f"Error in HTML protection during translation: {str(e)}")
                                    translated_response = complete_response
                            else:
                                translated_response = complete_response
                            
                            # Save response to database
                            new_ai_message = ChatHistoryDB(
                                session_id=session_id,
                                message_type="assistant",
                                content=complete_response
                            )
                            db.add(new_ai_message)
                            await db.commit()
                            
                            # Send the enhanced response
                            complete_response_html = render_markdown(translated_response)
                            await websocket.send_text(f"{session_id}|{complete_response_html}")
                            
                            # Reset confirmation after showing products
                            user_context["awaiting_confirmation"] = True
                            user_context["confirmation_type"] = "products"
                            
                        except Exception as e:
                            logging.error(f"Error during enhanced product recommendation: {str(e)}")
                            error_msg = "I'm sorry, I couldn't fetch product recommendations. Is there something else you'd like to know about fashion?"
                            if user_language != "en":
                                error_msg = translate_text(error_msg, user_language, session_id)
                            await websocket.send_text(f"{session_id}|{error_msg}")
                        
                        continue
                    # STEP 3: Handle existing product pagination and other responses
                    elif is_more_request:
                        # Use your existing "more products" logic
                        if not user_context["product_cache"]["all_results"].empty:
                            current_page = user_context["product_cache"]["current_page"]
                            next_page = current_page + 1
                            
                            next_page_products, has_more = get_paginated_products(
                                user_context["product_cache"]["all_results"],
                                page=next_page,
                                products_per_page=5
                            )
                            
                            if not next_page_products.empty:
                                user_context["product_cache"]["current_page"] = next_page
                                user_context["product_cache"]["has_more"] = has_more
                                
                                more_response = "🧠 Here are more AI-matched options for you:"
                                if user_language != "en":
                                    more_response = translate_text(more_response, user_language, session_id)
                                
                                complete_response = more_response + "\n\n"
                                
                                for _, row in next_page_products.iterrows():
                                    product_card = (
                                        "<div class='product-card'>\n"
                                        f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                        f"<div class='product-info'>\n"
                                        f"<h3>🎯 **Enhanced Match** | {row['product']}</h3>\n"
                                        f"<p class='price'>IDR {row['price']:,.0f}</p>\n"
                                        f"<p class='description'>{row['description']}</p>\n"
                                        f"<p class='available'>Available in size: {row['size']}, Color: {row['color']}</p>\n"
                                        f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                        "</div>\n"
                                        "</div>\n"
                                    )
                                    complete_response += product_card
                                
                                if has_more:
                                    more_hint = "\n\n🧠 I have even more AI-matched options! Just let me know if you want to continue exploring."
                                    if user_language != "en":
                                        more_hint = translate_text(more_hint, user_language, session_id)
                                    complete_response += more_hint
                                
                                # Save and send response
                                new_ai_message = ChatHistoryDB(
                                    session_id=session_id,
                                    message_type="assistant",
                                    content=complete_response
                                )
                                db.add(new_ai_message)
                                await db.commit()
                                
                                complete_response_html = render_markdown(complete_response)
                                await websocket.send_text(f"{session_id}|{complete_response_html}")
                                
                                user_context["awaiting_confirmation"] = True
                            else:
                                no_more_msg = "🎯 I've shown you all the best AI matches I could find. Would you like to try a different style or adjust your preferences?"
                                if user_language != "en":
                                    no_more_msg = translate_text(no_more_msg, user_language, session_id)
                                await websocket.send_text(f"{session_id}|{no_more_msg}")
                        
                        continue
                        
                    elif is_negative:
                        # User declined
                        negative_response = "I understand. What specific styles or fashion advice would you prefer instead? I'm here to help you find the perfect look."
                        if user_language != "en":
                            negative_response = translate_text(negative_response, user_language, session_id)
                        await websocket.send_text(f"{session_id}|{negative_response}")
                        user_context["awaiting_confirmation"] = False
                        continue
                    
                    else:
                        # Continue with regular processing
                        user_context["awaiting_confirmation"] = False
                
                # Check if input contains an image URL (keep your existing image processing)
                url_pattern = re.compile(r'(https?://\S+\.(?:jpg|jpeg|png|gif|bmp|webp))', re.IGNORECASE)
                image_url_match = url_pattern.search(user_input)
                
                if not user_context["awaiting_confirmation"] and image_url_match:
                    # Use your existing image processing logic
                    image_url = image_url_match.group(1)
                    text_content = user_input.replace(image_url, "").strip()
                    
                    try:
                        user_context["has_shared_image"] = True
                        user_context["last_query_type"] = "mixed" if text_content else "image"
                        user_context["current_image_url"] = image_url
                        user_context["current_text_input"] = text_content

                        # Call image analysis
                        clothing_features = await analyze_uploaded_image(image_url)

                        if clothing_features.startswith("Error:"):
                            await websocket.send_text(f"{session_id}|{clothing_features}")
                            continue

                        # Enhanced gender detection from text content
                        if text_content:
                            detect_and_update_gender(text_content, user_context)

                        # Prepare enhanced prompt
                        user_gender_info = user_context.get("user_gender", {})
                        gender_context = ""
                        if user_gender_info.get("category"):
                            gender_context = f" I am {user_gender_info['category']}."
                        
                        semantic_context = " Please provide detailed style recommendations that I can use for enhanced AI-powered product matching."
                            
                        if text_content:
                            prompt = f"I've shared an image with the following request: '{text_content}'.{gender_context}{semantic_context} Here's what the image shows: {clothing_features}. Please give me style recommendations based on this image and my specific request, but DO NOT offer product recommendations yet."
                        else:
                            prompt = f"I've shared an image.{gender_context}{semantic_context} Here's what the image shows: {clothing_features}. Please give me style recommendations based on this image, but DO NOT offer product recommendations yet."
                        
                        # Generate styling recommendations
                        message_objects.append({
                            "role": "user",
                            "content": prompt
                        })

                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=message_objects,
                            temperature=0.5
                        )
                        
                        ai_response = response.choices[0].message.content.strip()
                        last_ai_response = ai_response
                        user_context["last_ai_response"] = ai_response
                        
                        message_objects.append({
                            "role": "assistant",
                            "content": ai_response
                        })

                        # Enhanced keyword extraction from image analysis
                        image_keywords = await extract_ranked_keywords(clothing_features, "", [])
                        update_accumulated_keywords(image_keywords, user_context, user_input, is_user_input=True)
                        
                        # Process text content keywords if available
                        if text_content:
                            text_keywords = await extract_ranked_keywords("", text_content, [])
                            update_accumulated_keywords(text_keywords, user_context, user_input, is_user_input=True)
                        
                        # Extract keywords from AI style suggestions
                        style_keywords = await extract_ranked_keywords(ai_response, "", [])
                        update_accumulated_keywords(style_keywords, user_context, user_input, is_ai_response=True)
                        
                        # Translate if needed
                        if user_language != "en":
                            translated_ai_response = translate_text(ai_response, user_language, session_id)
                        else:
                            translated_ai_response = ai_response

                        # Save and send styling recommendations
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=ai_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        ai_response_html = render_markdown(translated_ai_response)
                        await websocket.send_text(f"{session_id}|{ai_response_html}")
                        
                        user_context["awaiting_confirmation"] = True
                        user_context["confirmation_type"] = "style_recommendations"  # Skip consultation for images

                    except Exception as input_error:
                        logging.error(f"Error during enhanced image processing: {str(input_error)}")
                        error_msg = "Sorry, there was an issue processing your image. Could you try again?"
                        await websocket.send_text(f"{session_id}|{error_msg}")
                
                # Handle normal text input with hybrid intelligence
                elif not user_context["awaiting_confirmation"]:
                    # Check for small talk
                    if await is_small_talk(user_input):
                        ai_response = "🧠 Hello! I'm your enhanced AI fashion assistant. How can I help you with personalized fashion recommendations today? Feel free to share information about your style preferences or upload an image for AI-powered suggestions."
                        
                        if user_language != "en":
                            ai_response = translate_text(ai_response, user_language, session_id)
                            
                        await websocket.send_text(f"{session_id}|{ai_response}")
                        continue
                    
                    # USE HYBRID INTELLIGENCE HANDLER
                    hybrid_response, should_search_products = await handle_message_with_hybrid_intelligence(
                        user_input, session_id, user_context, db, user_language
                    )
                    
                    if hybrid_response and hybrid_response != "TRIGGER_PRODUCT_SEARCH":
                        # Save AI message
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=hybrid_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        # Set confirmation type based on the hybrid response
                        if "**YOUR FASHION PROFILE**" in hybrid_response:
                            # This is a consultation summary
                            user_context["confirmation_type"] = "summary"
                            user_context["awaiting_confirmation"] = True
                        elif "**Personalized Style Recommendations**" in hybrid_response:
                            # This is style recommendations
                            user_context["confirmation_type"] = "style_recommendations"
                            user_context["awaiting_confirmation"] = True
                        
                        # Send response
                        await websocket.send_text(f"{session_id}|{hybrid_response}")
                        continue
                    
                    elif should_search_products or hybrid_response == "TRIGGER_PRODUCT_SEARCH":
                        # This should not happen in the new flow, but keep as fallback
                        user_context["awaiting_confirmation"] = True
                        user_context["confirmation_type"] = "summary"
                        
                        # Generate consultation summary
                        summary_response = await generate_consultation_summary_llm(user_context, user_language, session_id)
                        
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=summary_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        await websocket.send_text(f"{session_id}|{summary_response}")
                        continue
                    
                    else:
                        # Fall back to existing OpenAI chat logic
                        user_context["last_query_type"] = "text"
                        user_context["current_text_input"] = user_input

                        # Translate if needed
                        if user_language != "en":
                            translated_input = translate_text(user_input, "en", session_id)
                        else:
                            translated_input = user_input
                        
                        # Enhanced gender detection
                        detect_and_update_gender(translated_input, user_context)
                            
                        # Enhanced keyword extraction
                        accumulated_keywords = [(k, v.get("weight", 0) if isinstance(v, dict) else v) 
                                              for k, v in user_context.get("accumulated_keywords", {}).items()]
                        
                        input_keywords = await extract_ranked_keywords("", translated_input, accumulated_keywords)
                        update_accumulated_keywords(input_keywords, user_context, user_input, is_user_input=True)
                        
                        # Enhanced conversation context
                        enhanced_context = " I have advanced AI capabilities to provide highly personalized fashion recommendations."
                        
                        # Add to message history with enhanced context
                        message_objects.append({
                            "role": "user",
                            "content": translated_input + enhanced_context,
                        })
                        
                        # Get AI response with enhanced styling recommendations
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=message_objects,
                            temperature=0.5
                        )
                        
                        ai_response = response.choices[0].message.content.strip()
                        last_ai_response = ai_response
                        user_context["last_ai_response"] = ai_response
                        
                        message_objects.append({
                            "role": "assistant",
                            "content": ai_response
                        })
                        
                        # Enhanced keyword extraction from AI response
                        response_keywords = await extract_ranked_keywords(ai_response, "", [])
                        update_accumulated_keywords(response_keywords, user_context, user_input, is_ai_response=True)
                        
                        # Translate back if needed
                        if user_language != "en":
                            translated_response = translate_text(ai_response, user_language, session_id)
                        else:
                            translated_response = ai_response
                        
                        # Save response
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=ai_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        # Render and send enhanced response
                        ai_response_html = render_markdown(translated_response)
                        await websocket.send_text(f"{session_id}|{ai_response_html}")

                        # Enhanced context management
                        user_context["current_text_input"] = user_input
                        
                        # Set awaiting confirmation flag - but let hybrid system determine type
                        user_context["awaiting_confirmation"] = True
                
            except WebSocketDisconnect:
                logging.info(f"Enhanced websocket disconnected for session {session_id}")
                session_manager.reset_session(session_id)
                break
                
            except Exception as e:
                logging.error(f"Error processing enhanced message: {str(e)}\n{traceback.format_exc()}")
                error_message = "I'm sorry, I encountered an error while processing your request. Please try again."
                if user_language != "en":
                    try:
                        error_message = translate_text(error_message, user_language, session_id)
                    except:
                        pass
                await websocket.send_text(f"{session_id}|{error_message}")
                
    except Exception as e:
        logging.error(f"Enhanced websocket error: {str(e)}\n{traceback.format_exc()}")
        try:
            await websocket.close()
        except:
            pass