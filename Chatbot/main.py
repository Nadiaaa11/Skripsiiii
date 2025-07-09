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
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from ast import expr_context

# ================================
# FASHION CATEGORIES CONSTANTS
# ================================

class FashionCategories:
    """
    Centralized fashion categories used throughout the application
    for keyword extraction, preference detection, and consultation summaries
    """
    
    # CORE CLOTHING ITEMS (Priority 400)
    CLOTHING_TERMS = [
        # TOPS
        'kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'tshirt',
        'sweater', 'cardigan', 'hoodie', 'tank top', 'crop top', 'tube top',
        'halter top', 'camisole', 'singlet', 'vest', 'rompi', 'polo shirt',
        'henley', 'turtleneck', 'off shoulder', 'cold shoulder', 'wrap top',
        
        # BOTTOMS - separate specific styles from base terms
        'celana', 'pants', 'trousers', 'jeans', 'denim', 'rok', 'skirt',
        'shorts', 'leggings', 'jeggings', 
        
        # SPECIFIC PANT STYLES - only include when specifically mentioned
        'palazzo pants', 'wide leg pants', 'skinny jeans', 'straight jeans', 'bootcut',
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
    
    # SLEEVE TERMS (Priority 350)
    SLEEVE_TERMS = [
        'lengan panjang', 'lengan pendek', 'long sleeve', 'long sleeves',
        'short sleeve', 'short sleeves', 'sleeveless', 'tanpa lengan',
        '3/4 sleeve', '3/4 sleeves', 'quarter sleeve', 'quarter sleeves',
        'cap sleeve', 'cap sleeves', 'bell sleeve', 'bell sleeves',
        'puff sleeve', 'puff sleeves', 'balloon sleeve', 'balloon sleeves',
        'bishop sleeve', 'bishop sleeves', 'dolman sleeve', 'dolman sleeves',
        'raglan sleeve', 'raglan sleeves', 'flutter sleeve', 'flutter sleeves'
    ]
    
    # FIT TERMS (Priority 350)
    FIT_TERMS = [
        'oversized', 'oversize', 'longgar', 'loose', 'baggy', 'relaxed',
        'fitted', 'ketat', 'tight', 'slim', 'skinny', 'regular fit',
        'tailored', 'structured', 'flowy', 'draped', 'a-line', 'straight'
    ]
    
    # LENGTH TERMS (Priority 350)
    LENGTH_TERMS = [
        'maxi', 'midi', 'mini', 'ankle length', 'knee length', 'thigh length',
        'floor length', 'tea length', 'above knee', 'below knee', 'cropped length',
        'cropped', 'crop', 'panjang', 'pendek', 'long', 'short'
    ]
    
    # NECKLINE TERMS (Priority 350)
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
    
    # COLOR TERMS (Priority 250)
    COLOR_TERMS = [
        # COLOR CATEGORIES
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
    ]
    
    # MATERIAL TERMS (Priority 250)
    MATERIAL_TERMS = [
        'cotton', 'katun', 'silk', 'sutra', 'satin', 'chiffon',
        'lace', 'renda', 'denim', 'leather', 'kulit', 'faux leather',
        'velvet', 'beludru', 'corduroy', 'tweed', 'wool', 'wol',
        'cashmere', 'linen', 'polyester', 'spandex', 'elastane',
        'viscose', 'rayon', 'modal', 'bamboo', 'organic cotton'
    ]
    
    # PATTERN TERMS (Priority 250)
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
    ]
    
    # GENDER TERMS for detection
    GENDER_TERMS = [
        'perempuan', 'wanita', 'female', 'woman', 'cewek', 'cewe',
        'pria', 'laki-laki', 'male', 'man', 'cowok', 'cowo'
    ]
    
    # BLACKLISTED TERMS (Updated)
    BLACKLISTED_TERMS = [
        # Budget/price terms
        'rb', 'ribu', 'jt', 'juta', '000', 'budget', 'anggaran', 'harga', 'price',
        'rupiah', 'rp', 'idr', 'cost', 'biaya',
        
        # Physical measurements
        'cm', 'kg', 'height', 'weight', 'tinggi', 'berat', 'kulit', 'skin',
        
        # Generic conversation & filler words (expanded)
        'yang', 'dan', 'atau', 'dengan', 'untuk', 'dari', 'pada', 'akan',
        'dapat', 'adalah', 'ini', 'itu', 'saya', 'anda', 'kamu', 'mereka',
        'dia', 'sangat', 'lebih', 'kurang', 'baik', 'cocok', 'bisa', 'tolong', # Added bisa, tolong
        
        # Generic recommendation/query terms (expanded)
        'recommendation', 'rekomendasi', 'suggestion', 'saran', 'preferensi', 'ukuran',
        'ada', 'carikan', 'tunjukkan', 'ingin', 'mau', 'cari', 'mencari', 'looking',
        'for', 'untuk', 'by', 'dengan', 'about', 'tentang',
        'other', 'lain', 'lainnya', 'additional', 'tambahan', 'semua', 'all',
        
        # Generic attribute/clothing descriptors (often better handled by direct category checks)
        'panjang', 'lengan', 'pakaian', 'sleeve', 'length', 'fit', 'style', 'gaya',
        'type', 'jenis', 'item', 'barang', 'bagus', 'great', 'nice', 'mantap', # Added bagus, great, nice, mantap
    ]
    
    # CLOTHING CATEGORIES MAPPING
    CLOTHING_CATEGORIES = {
        'tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'tshirt', 'sweater', 'hoodie', 'cardigan', 'blazer', 'tank', 'top'],
        'bottoms_pants': ['celana', 'pants', 'jeans', 'trousers', 'leggings'],
        'bottoms_skirts': ['rok', 'skirt'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'coat', 'mantel'],
        'shorts': ['shorts', 'celana pendek']
    }
    
    @classmethod
    def get_clothing_category(cls, keyword):
        """Get clothing category for a keyword"""
        keyword_lower = keyword.lower()
        for category, terms in cls.CLOTHING_CATEGORIES.items():
            if any(term in keyword_lower for term in terms):
                return category
        return None
    
    @classmethod
    def is_clothing_item(cls, keyword):
        """Check if keyword is a clothing item"""
        return any(term in keyword.lower() for term in cls.CLOTHING_TERMS)
    
    @classmethod
    def is_style_term(cls, keyword):
        """Check if keyword is a style term"""
        return any(term in keyword.lower() for term in cls.STYLE_TERMS)
    
    @classmethod
    def is_color_term(cls, keyword):
        """Check if keyword is a color term"""
        return any(term in keyword.lower() for term in cls.COLOR_TERMS)
    
    @classmethod
    def is_blacklisted(cls, keyword):
        """Check if keyword is blacklisted"""
        return any(term in keyword.lower() for term in cls.BLACKLISTED_TERMS)
    
    @classmethod
    def is_gender_term(cls, keyword):
        """Check if keyword is a gender term"""
        return any(term in keyword.lower() for term in cls.GENDER_TERMS)
    
    @classmethod
    def get_all_fashion_terms(cls):
        """Get all fashion terms combined"""
        return (cls.CLOTHING_TERMS + cls.SLEEVE_TERMS + cls.FIT_TERMS + 
                cls.LENGTH_TERMS + cls.NECKLINE_TERMS + cls.STYLE_TERMS + 
                cls.COLOR_TERMS + cls.MATERIAL_TERMS + cls.PATTERN_TERMS + 
                cls.OCCASION_TERMS)
    
    @classmethod
    def get_category_priority(cls, keyword):
        """Get priority score based on category"""
        if cls.is_clothing_item(keyword):
            return 400
        elif any(term in keyword.lower() for term in cls.SLEEVE_TERMS + cls.FIT_TERMS + cls.LENGTH_TERMS + cls.NECKLINE_TERMS):
            return 350
        elif cls.is_style_term(keyword):
            return 300
        elif cls.is_color_term(keyword) or any(term in keyword.lower() for term in cls.MATERIAL_TERMS + cls.PATTERN_TERMS):
            return 250
        elif any(term in keyword.lower() for term in cls.OCCASION_TERMS):
            return 200
        else:
            return 100

# Initialize FashionCategories
fashion_categories = FashionCategories()

tfidf_vectorizer = None
TFIDF_MODEL_FITTED = False
product_tfidf_matrix = None

openai = OpenAI(
    api_key="sk-V4UGt-FNIde85D7t0zrvZbVG5eolZDsE8awXTAuJYgT3BlbkFJToj--_okCjQAcwzYu4ZC6JDX8kznTJhzruBhL9Q5YA"
)

client = OpenAI()

app = FastAPI()

app.mount("/static", StaticFiles(directory="Chatbot/static"), name="static")

templates = Jinja2Templates(directory="Chatbot/templates")

UPLOAD_DIR = "static/uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "jfif"}

Base = declarative_base()

DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/ecommerce"

engine = create_async_engine(DATABASE_URL, echo=True)

database = Database(DATABASE_URL)

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

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSession(engine) as session:  # Updated to use AsyncSession
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

# Updated stop_words using FashionCategories blacklisted terms
stop_words = set(fashion_categories.BLACKLISTED_TERMS)

def enhance_keywords_with_mapping(keywords_list):
    """
    Enhance a list of keywords using the complete mapping function.
    This can be called from extract_ranked_keywords to expand the keyword coverage.
    """
    enhanced_keywords = []
    
    for keyword in keywords_list:
        if isinstance(keyword, tuple):
            # If it's already a (keyword, weight) tuple
            keyword_text, weight = keyword
        else:
            # If it's just a keyword string
            keyword_text = keyword
            weight = 1.0
        
        # Get all related terms for this keyword
        search_terms = get_all_search_terms_for_extraction(keyword_text)
        
        # Add the original keyword with its weight
        enhanced_keywords.append((keyword_text, weight))
        
        # Add related terms with slightly lower weight
        for related_term in search_terms:
            if related_term.lower() != keyword_text.lower():  # Don't duplicate
                enhanced_keywords.append((related_term, weight * 0.8))  # 80% of original weight
    
    # Remove duplicates while preserving the highest weight for each keyword
    keyword_weights = {}
    for keyword, weight in enhanced_keywords:
        keyword_lower = keyword.lower()
        if keyword_lower not in keyword_weights:
            keyword_weights[keyword_lower] = weight
        else:
            # Keep the higher weight
            keyword_weights[keyword_lower] = max(keyword_weights[keyword_lower], weight)
    
    # Convert back to list of tuples
    final_enhanced_keywords = [(keyword, weight) for keyword, weight in keyword_weights.items()]
    
    return final_enhanced_keywords

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

class KeywordNode:
    """A node in the keyword linked list"""
    def __init__(self, keyword: str, weight: float, source: str, category: str):
        self.keyword = keyword
        self.weight = weight
        self.source = source  # 'user_input' or 'ai_response'
        self.category = category  # 'clothing_item', 'style', 'color', 'material', etc.
        self.timestamp = datetime.now().isoformat()
        self.mention_count = 1
        self.next = None  # Link to next related keyword

class ClothingChain:
    """A linked list representing a clothing item and its attributes"""
    def __init__(self, clothing_item: str, weight: float, source: str):
        self.head = KeywordNode(clothing_item, weight, source, 'clothing_item')
        self.clothing_category = self._get_clothing_category(clothing_item)
        self.last_updated = datetime.now().isoformat()
        self.total_nodes = 1
    
    def _get_clothing_category(self, keyword: str):
        """Determine clothing category from keyword using FashionCategories"""
        return fashion_categories.get_clothing_category(keyword) or 'unknown'
    
    def add_attribute(self, keyword: str, weight: float, source: str, category: str):
        """Add a style attribute to this clothing chain"""
        # Check if keyword already exists in chain
        current = self.head
        while current:
            if current.keyword.lower() == keyword.lower():
                # Update existing keyword
                current.weight = max(current.weight, weight)
                current.mention_count += 1
                current.timestamp = datetime.now().isoformat()
                current.source = source  # Update source to most recent
                self.last_updated = datetime.now().isoformat()
                return True
            current = current.next
        
        # Add new keyword to end of chain
        new_node = KeywordNode(keyword, weight, source, category)
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        self.total_nodes += 1
        self.last_updated = datetime.now().isoformat()
        return True
    
    def get_all_keywords(self) -> List[Tuple[str, float]]:
        """Get all keywords in this chain as (keyword, weight) tuples"""
        keywords = []
        current = self.head
        while current:
            keywords.append((current.keyword, current.weight))
            current = current.next
        return keywords
    
    def get_keywords_by_category(self, category: str) -> List[Tuple[str, float]]:
        """Get keywords of specific category from this chain"""
        keywords = []
        current = self.head
        while current:
            if current.category == category:
                keywords.append((current.keyword, current.weight))
            current = current.next
        return keywords
    
    def remove_keywords_by_category(self, category: str):
        """Remove all keywords of specific category from chain"""
        # Special case: can't remove head (clothing item)
        if self.head.category == category:
            return False
        
        current = self.head
        while current.next:
            if current.next.category == category:
                current.next = current.next.next
                self.total_nodes -= 1
            else:
                current = current.next
        self.last_updated = datetime.now().isoformat()
        return True
    
    def apply_decay(self, decay_factor: float = 0.9):
        """Apply time-based decay to all weights in chain"""
        current = self.head
        while current:
            current.weight *= decay_factor
            current = current.next
        self.last_updated = datetime.now().isoformat()

class LinkedKeywordSystem:
    """Manages multiple clothing chains using linked lists"""
    
    def __init__(self):
        self.chains: Dict[str, ClothingChain] = {}  # clothing_category -> ClothingChain
        self.last_clothing_focus = None
        
    def _categorize_keyword(self, keyword: str) -> str:
        """Categorize a keyword into type using FashionCategories"""
        keyword_lower = keyword.lower()
        
        # Check clothing items first
        if fashion_categories.is_clothing_item(keyword):
            return 'clothing_item'
        
        # Check style attributes
        if fashion_categories.is_style_term(keyword):
            return 'style'
        
        # Check colors
        if fashion_categories.is_color_term(keyword):
            return 'color'
        
        # Check sleeve/length attributes
        if any(term in keyword_lower for term in fashion_categories.SLEEVE_TERMS + fashion_categories.LENGTH_TERMS + fashion_categories.FIT_TERMS): # Added FIT_TERMS here for attribute categorization
            return 'attribute'
        
        # Check materials
        if any(term in keyword_lower for term in fashion_categories.MATERIAL_TERMS):
            return 'material'
        
        return 'other'
    
    def _get_clothing_category_from_keyword(self, keyword: str) -> str:
        """Get the clothing category for a keyword using FashionCategories"""
        return fashion_categories.get_clothing_category(keyword) or 'unknown'
    
    def detect_clothing_change(self, keywords: List[Tuple[str, float]], is_multi_item_request: bool) -> bool:
        """
        Detect if there's a major clothing category change.
        Now considers if it's a multi-item request.
        """
        current_clothing_items_in_input = set() # Store categories from current input
        
        # Find clothing items in current keywords (top 8 for better multi-item detection)
        limit = 8 # Look at more keywords to capture all items in multi-item requests
        for keyword, weight in keywords[:limit]:
            # For combined keywords like "short pants", split and check parts
            if is_multi_item_request and ' ' in keyword:
                parts = keyword.split()
                for part in parts:
                    if self._categorize_keyword(part) == 'clothing_item':
                        category = self._get_clothing_category_from_keyword(part)
                        if category and category != 'unknown':
                            current_clothing_items_in_input.add(category)
            elif self._categorize_keyword(keyword) == 'clothing_item':
                category = self._get_clothing_category_from_keyword(keyword)
                if category and category != 'unknown':
                    current_clothing_items_in_input.add(category)
        
        print(f"   👕 Detected clothing items in current input: {current_clothing_items_in_input}")
        
        if not current_clothing_items_in_input:
            print("   ⚠️ No clothing items detected in current input for change detection.")
            return False
        
        # Get existing categories in chains
        existing_chain_categories = set(self.chains.keys())
        print(f"   📚 Existing chain categories: {existing_chain_categories}")

        # If it's a multi-item request, be careful about flagging a "change" that leads to deletion.
        # We only flag a change if a *new, incompatible* primary category is introduced.
        if is_multi_item_request:
            print(f"   🤝 Multi-item request active. Checking for truly incompatible new categories.")
            for new_cat in current_clothing_items_in_input:
                is_compatible_with_any_existing = False
                for existing_cat in existing_chain_categories:
                    if self._are_compatible_categories_for_multi(new_cat, existing_cat):
                        is_compatible_with_any_existing = True
                        break
                
                # If a new category is introduced that is NOT compatible with ANY existing chain,
                # AND there are existing chains, then it's a major change requiring cleanup of old, incompatible ones.
                if not is_compatible_with_any_existing and existing_chain_categories:
                    print(f"   🔄 CLOTHING CHANGE DETECTED (Multi-item): New incompatible category '{new_cat}' introduced.")
                    return True # Means we need to clean up older, incompatible chains
            
            print("   ✅ No truly incompatible categories introduced in multi-item request. No major change detected for deletion.")
            return False # For multi-item, generally don't trigger a full reset of all chains if compatible
        
        # For single-item requests, detect a conflict if the primary clothing item is changing.
        # If the input explicitly requests a clothing item that conflicts with the highest-weighted existing chain head, then it's a change.
        if existing_chain_categories:
            primary_existing_category = self.get_primary_clothing_focus()
            if primary_existing_category:
                for new_cat in current_clothing_items_in_input:
                    if self._are_conflicting_categories(new_cat, primary_existing_category):
                        print(f"🔄 CLOTHING CHANGE DETECTED (Single-item): New '{new_cat}' conflicts with primary '{primary_existing_category}'.")
                        return True
            else: # Should ideally not happen if existing_chain_categories is not empty
                print("   ⚠️ No primary clothing focus despite existing chains.")
                return True # Treat as a change to be safe
        
        # If no existing chains, and current input has clothing items, it's a "new start" but not a "change" for deletion.
        if current_clothing_items_in_input and not existing_chain_categories:
            print("   🆕 New clothing items in input, but no existing chains. Starting new context.")
            return False
            
        print("   ✅ No major clothing change detected for deletion (single-item mode).")
        return False
    
    def _are_conflicting_categories(self, cat1: str, cat2: str) -> bool:
        """Check if two clothing categories conflict (for single-item context)"""
        separates = {'tops', 'bottoms_pants', 'bottoms_skirts', 'outerwear', 'shorts'}
        dresses = {'dresses'}
        
        # Conflict if one is separates and other is dresses
        if (cat1 in separates and cat2 in dresses) or (cat1 in dresses and cat2 in separates):
            return True
        
        # Conflict if switching between different, distinct bottoms (e.g., from pants to skirts, where only one is desired)
        bottoms = {'bottoms_pants', 'bottoms_skirts', 'shorts'}
        if cat1 in bottoms and cat2 in bottoms and cat1 != cat2:
            return True # In single-item mode, asking for "pants" then "skirts" means a change
        
        return False

    def _are_compatible_categories_for_multi(self, cat1: str, cat2: str) -> bool:
        """
        Check if two clothing categories are compatible in a multi-item context.
        E.g., tops and bottoms are compatible, pants and skirts are compatible.
        Dresses are generally not compatible with separate tops/bottoms unless explicitly layered.
        """
        if cat1 == cat2:
            return True
        
        # Define compatibility groups
        tops_group = {'tops', 'outerwear'}
        bottoms_group = {'bottoms_pants', 'bottoms_skirts', 'shorts'} # All bottoms are compatible with each other
        
        # Tops are compatible with bottoms
        if (cat1 in tops_group and cat2 in bottoms_group) or \
           (cat2 in tops_group and cat1 in bottoms_group):
            return True
        
        # Different items within the same "group" are compatible (e.g., pants and skirts, shirt and jacket)
        if (cat1 in tops_group and cat2 in tops_group) or \
           (cat1 in bottoms_group and cat2 in bottoms_group):
            return True

        # Dresses are generally NOT compatible with separate tops/bottoms in a simultaneous multi-item request
        dresses_group = {'dresses'}
        if (cat1 in dresses_group and (cat2 in tops_group or cat2 in bottoms_group)) or \
           (cat2 in dresses_group and (cat1 in tops_group or cat1 in bottoms_group)):
            return False 

        return False # Default to not compatible

    def update_keywords(self, keywords: List[Tuple[str, float]], is_user_input: bool = False, is_multi_item_request: bool = False):
        """
        Update the keyword system with new keywords.
        Now takes `is_multi_item_request` into account.
        """
        print(f"\n🔗 LINKED KEYWORD SYSTEM UPDATE")
        print("="*50)
        print(f"   🤝 Is Multi-Item Request (in LinkedKeywordSystem.update_keywords): {is_multi_item_request}")
        
        source = "user_input" if is_user_input else "ai_response"
        
        # Determine current clothing items from input, potentially grouped if multi-item
        current_input_clothing_categories = set()
        for kw, _ in keywords:
            # For combined keywords, check parts
            if is_multi_item_request and ' ' in kw:
                for part in kw.split():
                    cat = self._get_clothing_category_from_keyword(part)
                    if cat and cat != 'unknown':
                        current_input_clothing_categories.add(cat)
            else:
                cat = self._get_clothing_category_from_keyword(kw)
                if cat and cat != 'unknown':
                    current_input_clothing_categories.add(cat)
        
        print(f"   📝 Current input's clothing categories: {current_input_clothing_categories}")

        # Check for clothing changes *after* processing `keywords` to get a complete view of `current_input_clothing_categories`.
        # This function only decides if a *major deletion* should occur.
        clothing_change_detected = self.detect_clothing_change(keywords, is_multi_item_request)
        
        if clothing_change_detected:
            # This will remove truly conflicting chains for single-item requests,
            # or truly incompatible old chains for multi-item requests.
            self._handle_clothing_change(keywords, is_multi_item_request)
        
        # Process keywords to create/update chains
        # We no longer track `clothing_items_processed_in_current_update` with a set,
        # as we want to ensure *all* distinct clothing items from the input get their own chain
        # in multi-item mode, and their weights are correctly managed.
        
        for keyword, weight in keywords:
            keyword_category = self._categorize_keyword(keyword)
            
            # --- Handle combined keywords first for processing into chains ---
            if is_multi_item_request and ' ' in keyword:
                # This keyword represents a grouped item + attribute (e.g., "short pants")
                # Try to extract the main clothing item from this combined keyword
                main_clothing_item = None
                main_clothing_category = None
                attributes_in_group = []
                
                parts = keyword.split()
                for part in parts:
                    if self._categorize_keyword(part) == 'clothing_item':
                        main_clothing_item = part
                        main_clothing_category = self._get_clothing_category_from_keyword(part)
                    else: # Assuming other parts are attributes
                        attributes_in_group.append(part)
                
                if main_clothing_item and main_clothing_category and main_clothing_category != 'unknown':
                    # Create/update chain for the main clothing item
                    if main_clothing_category not in self.chains:
                        self.chains[main_clothing_category] = ClothingChain(main_clothing_item, weight, source)
                        print(f"   🆕 NEW CHAIN (Multi-item grouped): {main_clothing_category} → '{main_clothing_item}' ({weight:.1f})")
                    else:
                        # Update existing chain head with the new combined keyword's weight
                        self.chains[main_clothing_category].head.weight = max(self.chains[main_clothing_category].head.weight, weight)
                        self.chains[main_clothing_category].head.mention_count += 1
                        self.chains[main_clothing_category].last_updated = datetime.now().isoformat()
                        print(f"   🔄 UPDATED CHAIN HEAD (Multi-item grouped): {main_clothing_category} → '{main_clothing_item}' ({weight:.1f})")
                    
                    # Add attributes to this specific chain
                    for attr in attributes_in_group:
                        self.chains[main_clothing_category].add_attribute(attr, weight * 0.8, source, self._categorize_keyword(attr))
                        print(f"      ➕ ADDED ATTRIBUTE to '{main_clothing_category}': '{attr}' ({weight*0.8:.1f})")
                else:
                    # If it's a combined keyword but no clear clothing item detected, treat as generic attributes
                    # This case should ideally not happen if `extract_ranked_keywords` is working well
                    print(f"   ⚠️ Combined keyword '{keyword}' has no clear clothing item. Adding attributes to all active chains.")
                    for attr in parts:
                        self._add_attribute_to_relevant_chains(attr, weight, source, self._categorize_keyword(attr))

            elif keyword_category == 'clothing_item':
                clothing_cat = self._get_clothing_category_from_keyword(keyword)
                
                if clothing_cat and clothing_cat != 'unknown':
                    if is_multi_item_request:
                        # In multi-item mode, always create or update if it's a clothing item.
                        if clothing_cat not in self.chains:
                            self.chains[clothing_cat] = ClothingChain(keyword, weight, source)
                            print(f"   🆕 NEW CHAIN (Multi-item, single item): {clothing_cat} → '{keyword}' ({weight:.1f})")
                        else:
                            # Update existing chain head
                            self.chains[clothing_cat].head.weight = max(self.chains[clothing_cat].head.weight, weight)
                            self.chains[clothing_cat].head.mention_count += 1
                            self.chains[clothing_cat].last_updated = datetime.now().isoformat()
                            print(f"   🔄 UPDATED CHAIN HEAD (Multi-item, single item): {clothing_cat} → '{keyword}' ({weight:.1f})")
                    else: # Not a multi-item request (single item focus)
                        # In single-item mode, we maintain a single primary chain.
                        # If the input contains a clothing item that *conflicts* with the current primary,
                        # or if there's no primary, this new item becomes the primary.
                        primary_current_chain = self.get_primary_clothing_focus()
                        
                        if clothing_cat not in self.chains or \
                           (primary_current_chain and self._are_conflicting_categories(clothing_cat, primary_current_chain)):
                            # Clear all existing chains and set this new item as the primary focus
                            if self.chains:
                                print(f"   🗑️ Clearing ALL old chains for single-item focus change: {list(self.chains.keys())} -> {clothing_cat}")
                                self.chains.clear()
                            self.chains[clothing_cat] = ClothingChain(keyword, weight, source)
                            print(f"   🆕 NEW CHAIN (Single-item primary): {clothing_cat} → '{keyword}' ({weight:.1f})")
                        else:
                            # Update existing chain head (if it's already the primary or compatible primary)
                            self.chains[clothing_cat].head.weight = max(self.chains[clothing_cat].head.weight, weight)
                            self.chains[clothing_cat].head.mention_count += 1
                            self.chains[clothing_cat].last_updated = datetime.now().isoformat()
                            print(f"   🔄 UPDATED CHAIN HEAD (Single-item compatible): {clothing_cat} → '{keyword}' ({weight:.1f})")
            
            else: # Not a clothing item (e.g., style, color, material, other attribute)
                # Add as attribute to all relevant chains
                self._add_attribute_to_relevant_chains(keyword, weight, source, keyword_category)
        
        # Apply gentle decay to all chains
        self._apply_decay()
        
        # Show current state
        self._print_current_state()
    
    def _handle_clothing_change(self, keywords: List[Tuple[str, float]], is_multi_item_request: bool):
        """
        Handle major clothing category changes.
        In multi-item requests, it will be more selective about removing chains.
        """
        current_clothing_items_in_input = set()
        # Look at more keywords for new focus, especially for multi-item requests
        limit = 8 if is_multi_item_request else 3
        for keyword, weight in keywords[:limit]:
            # Handle combined keywords
            if ' ' in keyword and is_multi_item_request:
                for part in keyword.split():
                    cat = self._get_clothing_category_from_keyword(part)
                    if cat and cat != 'unknown':
                        current_clothing_items_in_input.add(cat)
            else:
                if self._categorize_keyword(keyword) == 'clothing_item':
                    category = self._get_clothing_category_from_keyword(keyword)
                    if category and category != 'unknown':
                        current_clothing_items_in_input.add(category)
        
        chains_to_remove = []
        for existing_category in list(self.chains.keys()): # Iterate over a copy as we might modify
            is_compatible_with_any_new = False
            for new_cat in current_clothing_items_in_input:
                if self._are_compatible_categories_for_multi(existing_category, new_cat):
                    is_compatible_with_any_new = True
                    break
            
            if is_multi_item_request:
                # In multi-item mode, if an existing category is NOT one of the newly requested
                # categories AND it's NOT compatible with ANY of the newly requested categories,
                # then it's an old, incompatible chain that should be removed.
                if existing_category not in current_clothing_items_in_input and not is_compatible_with_any_new:
                    chains_to_remove.append(existing_category)
                    print(f"   🗑️ REMOVING INCOMPATIBLE OLD CHAIN (Multi-item cleanup): {existing_category}")
            else: # Single-item mode
                # In single-item mode, if an existing category conflicts with any of the new input categories
                # (and is not itself a new input category, implying a direct switch)
                if existing_category not in current_clothing_items_in_input:
                    # Check if the existing category is part of the "old" context that's being replaced.
                    # This is implicitly handled by the `_are_conflicting_categories` in `detect_clothing_change`
                    # which makes `clothing_change_detected` true.
                    
                    # For single-item, if a change is detected by `detect_clothing_change`,
                    # `update_keywords` simply clears all chains and rebuilds, which is more robust.
                    # So this `_handle_clothing_change` might not be strictly necessary for single-item.
                    pass 
                
        for category in chains_to_remove:
            if category in self.chains: # Double-check before deleting
                print(f"   🗑️ ACTUALLY DELETING CONFLICTING CHAIN: {category}")
                del self.chains[category]
    
    def _add_attribute_to_relevant_chains(self, keyword: str, weight: float, source: str, category: str):
        """Add attribute to relevant clothing chains"""
        if not self.chains:
            print(f"   ⚠️ No active chains to add attribute '{keyword}' to. Skipping.")
            return # No chains to add attributes to

        # If it's a combined keyword, its attributes might need to be added to specific parts
        # This function primarily handles single attributes. If it's a combined keyword (like "short pants")
        # it should have been processed already in update_keywords.
        # This is for standalone attributes like "red" or "casual".
        
        # Add to all active chains. This assumes general attributes apply to all items in multi-item requests.
        added_count = 0
        for clothing_category, chain in self.chains.items():
            chain.add_attribute(keyword, weight, source, category)
            added_count += 1
        
        if added_count > 0:
            print(f"   ➕ ADDED ATTRIBUTE: '{keyword}' ({category}) to {added_count} chains")
    
    def _apply_decay(self, decay_factor: float = 0.95):
        """Apply gentle decay to all chains"""
        for chain in self.chains.values():
            chain.apply_decay(decay_factor)
    
    def _print_current_state(self):
        """Print current state for debugging"""
        print(f"\n🔗 CURRENT KEYWORD CHAINS:")
        
        if not self.chains:
            print("   (No active chains)")
            return

        for clothing_category, chain in self.chains.items():
            keywords = chain.get_all_keywords() # This gets all (keyword, weight) tuples
            print(f"   📦 {clothing_category.upper()}: {len(keywords)} keywords")
            
            # Now, iterate through the actual nodes in the chain to get their properties
            current_node = chain.head
            i = 0
            while current_node and i < 5: # Limit to top 5 for display
                keyword = current_node.keyword
                weight = current_node.weight
                node_category = current_node.category
                node_source = current_node.source

                if i == 0:
                    print(f"      🏷️  HEAD: '{keyword}' → {weight:.1f} (Category: {node_category}, Source: {node_source})")
                else:
                    print(f"      🔗 '{keyword}' → {weight:.1f} (Category: {node_category}, Source: {node_source})")
                
                current_node = current_node.next
                i += 1
            
            if len(keywords) > 5:
                print(f"      ... and {len(keywords) - 5} more")

    def get_flattened_keywords(self) -> List[Tuple[str, float]]:
        """Get all keywords from all chains as flat list for product search"""
        all_keywords = []
        
        for chain in self.chains.values():
            all_keywords.extend(chain.get_all_keywords())
        
        # Sort by weight
        all_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return all_keywords
    
    def get_requirements_for_category(self, clothing_category: str) -> Dict[str, List[str]]:
        """Get style requirements for a specific clothing category"""
        if clothing_category not in self.chains:
            return {'colors': [], 'fits': [], 'sleeve_lengths': [], 'clothing_lengths': []}
        
        chain = self.chains[clothing_category]
        requirements = {'colors': [], 'fits': [], 'sleeve_lengths': [], 'clothing_lengths': []}
        
        # Extract requirements from this chain only
        current = chain.head.next  # Skip head (clothing item)
        while current:
            if current.category == 'color':
                requirements['colors'].append(current.keyword)
            elif current.category == 'style' and any(fit in current.keyword.lower() for fit in fashion_categories.FIT_TERMS):
                requirements['fits'].append(current.keyword)
            elif current.category == 'attribute' and any(sleeve in current.keyword.lower() for sleeve in fashion_categories.SLEEVE_TERMS):
                requirements['sleeve_lengths'].append(current.keyword)
            elif current.category == 'attribute' and any(length in current.keyword.lower() for length in fashion_categories.LENGTH_TERMS):
                requirements['clothing_lengths'].append(current.keyword)
            
            current = current.next
        
        return requirements
    
    def get_primary_clothing_focus(self) -> Optional[str]:
        """Get the primary clothing category based on highest weight"""
        if not self.chains:
            return None
        
        max_weight = 0
        primary_category = None
        
        for category, chain in self.chains.items():
            if chain.head.weight > max_weight:
                max_weight = chain.head.weight
                primary_category = category
        
        return primary_category
    
# --- Integration functions need to pass `is_multi_item_request` ---

def convert_to_linked_system(user_context: Dict[str, Any], is_multi_item_request: bool = False):
    """Convert existing accumulated_keywords to linked system"""
    if 'linked_keyword_system' not in user_context:
        user_context['linked_keyword_system'] = LinkedKeywordSystem()
    
    if 'accumulated_keywords' in user_context:
        # Convert old system to new
        keywords = [(k, v.get('weight', 0)) for k, v in user_context['accumulated_keywords'].items()]
        # Pass is_multi_item_request to update_keywords
        user_context['linked_keyword_system'].update_keywords(keywords, is_multi_item_request=is_multi_item_request)
        
        # Keep old system as backup for now
        # del user_context['accumulated_keywords']

def update_linked_keywords(user_context: Dict[str, Any], new_keywords: List[Tuple[str, float]], is_user_input: bool = False, is_multi_item_request: bool = False):
    """
    Update keywords using linked system.
    Now takes `is_multi_item_request` into account.
    """
    if 'linked_keyword_system' not in user_context:
        user_context['linked_keyword_system'] = LinkedKeywordSystem()
    
    # Pass is_multi_item_request to the internal update_keywords method
    user_context['linked_keyword_system'].update_keywords(new_keywords, is_user_input, is_multi_item_request)
    
def get_keywords_for_product_search(user_context: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Get flattened keywords for product search"""
    if 'linked_keyword_system' not in user_context:
        return []
    
    return user_context['linked_keyword_system'].get_flattened_keywords()

def get_smart_requirements(user_context):
    """Simple version that works with existing accumulated_keywords using FashionCategories"""
    
    # Default empty requirements
    requirements = {'colors': [], 'fits': [], 'sleeve_lengths': [], 'clothing_lengths': []}
    
    if 'accumulated_keywords' not in user_context:
        return requirements
    
    # Get top 3 keywords to determine focus
    sorted_keywords = sorted(
        user_context['accumulated_keywords'].items(),
        key=lambda x: x[1].get('weight', 0),
        reverse=True
    )
    
    # Determine clothing focus from top keywords using FashionCategories
    clothing_focus = None
    for keyword, data in sorted_keywords[:3]:
        clothing_category = fashion_categories.get_clothing_category(keyword)
        if clothing_category:
            if clothing_category in ['bottoms_pants', 'bottoms_skirts']:
                clothing_focus = 'bottoms'
                break
            elif clothing_category == 'tops':
                clothing_focus = 'tops'
                break
            elif clothing_category == 'dresses':
                clothing_focus = 'dresses'
                break
    
    # Define relevant categories for each clothing type
    relevant_categories = {
        'tops': ['fits', 'sleeve_lengths', 'clothing_lengths', 'colors'],
        'bottoms': ['clothing_lengths', 'colors'],  # No fits, no sleeves
        'dresses': ['fits', 'sleeve_lengths', 'clothing_lengths', 'colors'],
        None: ['fits', 'sleeve_lengths', 'clothing_lengths', 'colors']
    }
    
    relevant = relevant_categories.get(clothing_focus, ['fits', 'sleeve_lengths', 'clothing_lengths', 'colors'])
    
    # Extract only relevant requirements using FashionCategories
    for keyword, data in sorted_keywords[:10]:
        keyword_lower = keyword.lower()
        
        # Colors
        if 'colors' in relevant and fashion_categories.is_color_term(keyword):
            requirements['colors'].append(keyword_lower)
        
        # Fits  
        if 'fits' in relevant and any(fit in keyword_lower for fit in fashion_categories.FIT_TERMS):
            requirements['fits'].append(keyword_lower)
        
        # Sleeve lengths
        if 'sleeve_lengths' in relevant and any(sleeve in keyword_lower for sleeve in fashion_categories.SLEEVE_TERMS):
            requirements['sleeve_lengths'].append(keyword_lower)
        
        # Clothing lengths
        if 'clothing_lengths' in relevant and any(length in keyword_lower for length in fashion_categories.LENGTH_TERMS):
            requirements['clothing_lengths'].append(keyword_lower)
    
    return requirements

def extract_ranked_keywords(ai_response: str = None, translated_input: str = None, accumulated_keywords=None):
    """
    Enhanced keyword extraction using FashionCategories for consistency.
    For multi-item requests, it attempts to group attributes with clothing items.
    """
    print("\n" + "="*60)
    print("🔤 ENHANCED KEYWORD EXTRACTION WITH FASHION CATEGORIES")
    print("="*60)
    
    keyword_scores = {}
    global_exclusions = set()

    # Use FashionCategories for conversation words and exclusions
    conversation_words = set(fashion_categories.BLACKLISTED_TERMS)
    
    simple_responses = {
        "yes", "ya", "iya", "oke", "ok", "okay", "sure", "tentu",
        "no", "tidak", "nope", "ga", "gak", "engga", "nah",
        "good", "bagus", "nice", "baik", "great", "mantap",
        "thanks", "terima", "kasih", "makasih", "thx"
    }
    
    current_input_categories = set() # This needs to be populated correctly for both multi/single item
    wanted_items = []
    context_items = []
    
    clothing_categories = fashion_categories.CLOTHING_CATEGORIES
    
    if translated_input:
        try:
            wanted_items, context_items = extract_specific_clothing_request(translated_input, ai_response)
        except Exception as e:
            print(f"⚠️ Error in extract_specific_clothing_request: {e}")
            wanted_items, context_items = [], []
            for category, terms in clothing_categories.items():
                for term in terms:
                    if term in translated_input.lower():
                        if any(indicator in translated_input.lower() for indicator in ['apa', 'what', 'carikan', 'tunjukkan', 'show']):
                            wanted_items.append(category)
                        else:
                            context_items.append(category)
    
    # Detect if this is a multi-item request
    is_multi_item_request = detect_multi_item_request(translated_input)
    print(f"🤝 Multi-item request (in extract_ranked_keywords): {is_multi_item_request}")
    
    # NEW: Structure to hold grouped keywords for multi-item requests
    grouped_keywords = defaultdict(lambda: {'items': set(), 'attributes': set(), 'score': 0.0})
    
    def get_keyword_score(keyword, source, frequency=1):
        """Get appropriate score using FashionCategories priority system"""
        keyword_lower = keyword.lower()
        
        if fashion_categories.is_blacklisted(keyword):
            print(f"      🚫 FILTERED blacklisted term: '{keyword}'")
            return 0, 'FILTERED'
        
        base_score = fashion_categories.get_category_priority(keyword)
        
        if source == 'user':
            base_score *= frequency * 1.2
        elif source == 'ai':
            base_score *= frequency * 1.0
        else:
            base_score *= frequency * 0.5
        
        if base_score >= 400:
            priority = 'CLOTHING'
        elif base_score >= 300:
            priority = 'STYLE'
        elif base_score >= 200:
            priority = 'ATTRIBUTE'
        else:
            priority = 'OTHER'
        
        clothing_category = fashion_categories.get_clothing_category(keyword)
        if clothing_category and clothing_category in wanted_items:
            if source == 'ai':
                boost = base_score * 2
                print(f"      🤖🚀 AI WANTED BOOST: '{keyword}' ({clothing_category}) {base_score} → {base_score + boost}")
            else:
                boost = base_score * 1.5
                print(f"      👤🚀 USER WANTED BOOST: '{keyword}' ({clothing_category}) {base_score} → {base_score + boost}")
            base_score += boost
            priority = 'SPECIFIC_REQUEST'
        
        return base_score, priority
    
    # Process user input with improved filtering
    if translated_input:
        print(f"📝 USER INPUT: '{translated_input}'")
        
        input_words = translated_input.lower().split()
        is_simple_response = (
            len(input_words) <= 2 and 
            all(word in simple_responses for word in input_words)
        )
        
        if is_simple_response:
            print(f"   ⚠️  SIMPLE RESPONSE DETECTED - Skipping")
            # Store the multi-item flag before returning, even if no keywords
            extract_ranked_keywords.is_multi_item_request = is_multi_item_request
            return []
        
        doc = nlp(translated_input)
        
        # --- REVISED: More robust check for empty or invalid doc ---
        if not doc or not doc.text.strip() or len(doc) == 0:
            print(f"   ⚠️  Processed input is empty or contains no meaningful tokens after nlp - Skipping keyword extraction.")
            # Store the multi-item flag before returning, even if no keywords
            extract_ranked_keywords.is_multi_item_request = is_multi_item_request
            return []
        # --- END REVISED ---

        user_keywords_freq = {}
        
        # --- REVISED: Unified token processing for both multi-item and single-item ---
        for token in doc:
            token_text = token.text.lower()
            
            # Skip common filters early
            if not token_text.strip() or len(token_text) <= 2 or \
               not token.is_alpha or token.is_digit or \
               token_text in simple_responses or \
               fashion_categories.is_blacklisted(token_text):
                continue
            
            # CRITICAL FIX: Populate current_input_categories for all scenarios (from user input)
            clothing_cat_from_token = fashion_categories.get_clothing_category(token_text)
            if clothing_cat_from_token:
                current_input_categories.add(clothing_cat_from_token)
            # END CRITICAL FIX

            # Prioritize multi-item grouping if active
            if is_multi_item_request:
                
                if clothing_cat_from_token and clothing_cat_from_token != 'unknown':
                    # This is a clothing item: add to user_keywords_freq and grouped_keywords
                    user_keywords_freq[token_text] = user_keywords_freq.get(token_text, 0) + 1
                    group_key = clothing_cat_from_token
                    grouped_keywords[group_key]['items'].add(token_text)
                    # Use the clothing item's score as part of the group's score
                    grouped_keywords[group_key]['score'] = max(grouped_keywords[group_key]['score'], get_keyword_score(token_text, 'user')[0])
                    
                    # Heuristic for Attribute Association: Look for preceding adjectives/nouns that might be attributes
                    if token.i > 0: # Check previous token
                        prev_token = doc[token.i - 1]
                        prev_token_text = prev_token.text.lower()
                        if prev_token.pos_ in ['ADJ', 'NOUN', 'PROPN'] and \
                           (fashion_categories.is_style_term(prev_token_text) or \
                            any(term in prev_token_text for term in fashion_categories.SLEEVE_TERMS + fashion_categories.LENGTH_TERMS + fashion_categories.MATERIAL_TERMS + fashion_categories.COLOR_TERMS + fashion_categories.FIT_TERMS)):
                            
                            # Add attribute to the same group as the clothing item
                            grouped_keywords[group_key]['attributes'].add(prev_token_text)
                            # Add a fraction of the attribute's score to the group score
                            grouped_keywords[group_key]['score'] = max(grouped_keywords[group_key]['score'], get_keyword_score(prev_token_text, 'user')[0] * 0.5)
                            user_keywords_freq[prev_token_text] = user_keywords_freq.get(prev_token_text, 0) + 1
                            print(f"      🔗 ATTRIBUTE '{prev_token_text}' associated with '{token_text}' ({group_key})")
                
                elif (token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and 
                      (fashion_categories.is_style_term(token_text) or 
                       any(term in token_text for term in fashion_categories.SLEEVE_TERMS + fashion_categories.LENGTH_TERMS + fashion_categories.MATERIAL_TERMS + fashion_categories.COLOR_TERMS + fashion_categories.FIT_TERMS))):
                    # This is an attribute (e.g., "red" or "oversized") not immediately preceding a known clothing item.
                    # Add it to user_keywords_freq for general processing. It might get grouped later or remain general.
                    user_keywords_freq[token_text] = user_keywords_freq.get(token_text, 0) + 1
                
                else: # General noun/adjective/proper noun not explicitly classified as clothing or fashion attribute
                    user_keywords_freq[token_text] = user_keywords_freq.get(token_text, 0) + 1
            
            else: # NOT a multi-item request, process as a flat list
                if token.pos_ in ['NOUN', 'ADJ', 'PROPN']:
                    user_keywords_freq[token_text] = user_keywords_freq.get(token_text, 0) + 1
                    # clothing_cat is already handled above before the if-else for multi-item
        # --- END REVISED ---
        
        # Score user keywords (for both multi-item and single-item)
        # This loop now populates `keyword_scores` from `user_keywords_freq`
        for keyword, frequency in user_keywords_freq.items():
            score, priority = get_keyword_score(keyword, 'user', frequency)
            
            if score > 0:
                keyword_scores[keyword] = score
                print(f"   📌 '{keyword}' (freq: {frequency}) → {score} ({priority})")
                
                # Get translation expansion
                try:
                    search_terms = get_search_terms_for_keyword(keyword)
                    if isinstance(search_terms, dict):
                        include_terms = search_terms.get('include', [])
                        exclude_terms = search_terms.get('exclude', [])
                        
                        for include_term in include_terms:
                            if (include_term != keyword and 
                                include_term not in keyword_scores and
                                not fashion_categories.is_blacklisted(include_term)):
                                
                                expansion_score = score * 0.7
                                
                                expansion_clothing_cat = fashion_categories.get_clothing_category(include_term)
                                if expansion_clothing_cat and expansion_clothing_cat in wanted_items:
                                    expansion_score *= 1.5
                                    print(f"      ➕ BOOSTED expansion '{keyword}' → '{include_term}' ({expansion_score:.1f})")
                                else:
                                    print(f"      ➕ Expanded '{keyword}' → '{include_term}' ({expansion_score:.1f})")
                                
                                keyword_scores[include_term] = expansion_score
                        
                        if exclude_terms:
                            global_exclusions.update(exclude_terms)
                            print(f"      🚫 Will exclude: {exclude_terms}")
                except Exception as e:
                    print(f"      ⚠️ Translation mapping error: {e}")
                    pass
    
    # Process AI response (remains largely the same, but affects global keyword_scores)
    if ai_response:
        print(f"\n🤖 AI RESPONSE processing...")
        
        bold_headings = extract_bold_headings_from_ai_response(ai_response)
        print(f"   📋 Found {len(bold_headings)} bold headings: {bold_headings}")
        
        for heading in bold_headings:
            heading_lower = heading.lower()
            cleaned_heading = re.sub(r'[^\w\s-]', '', heading_lower).strip() # Fixed escaped backslash
            
            if (cleaned_heading and 
                len(cleaned_heading) > 2 and 
                not fashion_categories.is_blacklisted(cleaned_heading)):
                
                score, priority = get_keyword_score(cleaned_heading, 'ai', 3)
                
                if score > 0:
                    if cleaned_heading not in keyword_scores:
                        keyword_scores[cleaned_heading] = score
                    else:
                        keyword_scores[cleaned_heading] = max(keyword_scores[cleaned_heading], score)
                    
                    print(f"   🔥 BOLD HEADING: '{cleaned_heading}' → {score} ({priority})")
                    
                    # Ensure current_input_categories is updated for AI response keywords too
                    clothing_cat_from_heading = fashion_categories.get_clothing_category(cleaned_heading)
                    if clothing_cat_from_heading:
                        current_input_categories.add(clothing_cat_from_heading)
    
    # Smart conflict resolution using FashionCategories
    print(f"\n⚔️ SMART CONFLICT ANALYSIS:")
    print(f"   📦 Current input categories: {current_input_categories}") # This should now be populated
    print(f"   🤝 Is multi-item request: {is_multi_item_request}")

    # Define clothing conflicts using FashionCategories structure
    separates_categories = {'tops', 'bottoms_pants', 'bottoms_skirts', 'outerwear', 'shorts'} # Added shorts
    dress_category = {'dresses'}
    
    def is_major_category_switch(current_cats, accumulated_cats, is_multi_item_flag): # Renamed param
        if not current_cats or not accumulated_cats:
            return False
        
        if not current_cats.intersection(accumulated_cats):
            if is_multi_item_flag: # Use the flag consistently
                # For multi-item, only switch between major domains
                acc_in_separates = bool(accumulated_cats.intersection(separates_categories))
                curr_in_separates = bool(current_cats.intersection(separates_categories))
                acc_in_dress = bool(accumulated_cats.intersection(dress_category))
                curr_in_dress = bool(current_cats.intersection(dress_category))
                
                # Only a major switch if transitioning between separates and dresses
                return (acc_in_separates and curr_in_dress) or (acc_in_dress and curr_in_separates)
            else:
                return True
        
        return False

    # Process accumulated keywords with conflict checking
    accumulated_categories = set()
    conflicting_keywords = []

    if accumulated_keywords:
        print(f"\n📚 ACCUMULATED keywords...")
        
        # First pass: identify categories
        for keyword, old_weight_data in accumulated_keywords[:15]: # Assuming accumulated_keywords is list of (k, data) or (k, weight)
            # Adapt to either format:
            if isinstance(old_weight_data, dict):
                old_weight = old_weight_data.get('weight', 0)
            else:
                old_weight = old_weight_data # It's just the weight
                
            if (keyword and len(keyword) > 2 and 
                keyword.lower() not in simple_responses and
                not fashion_categories.is_blacklisted(keyword) and
                not any(char.isdigit() for char in keyword)):
                
                clothing_cat = fashion_categories.get_clothing_category(keyword)
                if clothing_cat:
                    accumulated_categories.add(clothing_cat)
        
        print(f"   📦 Accumulated categories: {accumulated_categories}")
        
        # Check for major category switch
        is_switch = is_major_category_switch(current_input_categories, accumulated_categories, is_multi_item_request) # Pass the flag
        
        if is_switch:
            print(f"   🔄 MAJOR CATEGORY SWITCH DETECTED")
            
            # Apply reduction to conflicting keywords
            for keyword, old_weight_data in accumulated_keywords[:15]:
                if isinstance(old_weight_data, dict):
                    old_weight = old_weight_data.get('weight', 0)
                else:
                    old_weight = old_weight_data

                if (keyword and len(keyword) > 2 and 
                    keyword.lower() not in simple_responses and
                    not fashion_categories.is_blacklisted(keyword) and
                    not any(char.isdigit() for char in keyword)):
                    
                    clothing_cat = fashion_categories.get_clothing_category(keyword)
                    if clothing_cat and clothing_cat in accumulated_categories:
                        if clothing_cat not in current_input_categories:
                            conflicting_keywords.append(keyword)
                            print(f"   ⚔️  SWITCH CONFLICT: '{keyword}' ({clothing_cat}) being replaced by {current_input_categories}")
        
        # Add non-conflicting keywords with decay
        for keyword, old_weight_data in accumulated_keywords[:10]:
            if isinstance(old_weight_data, dict):
                old_weight = old_weight_data.get('weight', 0)
            else:
                old_weight = old_weight_data

            if (keyword and len(keyword) > 2 and 
                keyword.lower() not in simple_responses and
                not fashion_categories.is_blacklisted(keyword) and
                not any(char.isdigit() for char in keyword) and
                keyword not in conflicting_keywords):
                
                accumulated_score = old_weight * 0.4
                
                if keyword not in keyword_scores and accumulated_score > 15:
                    keyword_scores[keyword] = accumulated_score
                    print(f"   📜 '{keyword}' → {accumulated_score:.1f}")
    
    # Enhanced cleanup using FashionCategories
    cleanup_keywords = []
    for keyword in list(keyword_scores.keys()):
        if (fashion_categories.is_blacklisted(keyword) or 
            len(keyword.split()) > 3 or # Max 3 words for a keyword, e.g., "long sleeve shirt" is okay
            len(keyword) <= 2):
            cleanup_keywords.append(keyword)
    
    for keyword in cleanup_keywords:
        if keyword in keyword_scores: # Ensure it exists before deleting
            del keyword_scores[keyword]
            print(f"   🗑️ Enhanced cleanup: '{keyword}'")
    
    # --- FINAL RETURN BASED ON is_multi_item_request ---
    if is_multi_item_request and grouped_keywords:
        # Flatten grouped keywords for return
        final_output = []
        for group_key, data in grouped_keywords.items():
            # Combine items and attributes into one string for easier processing later
            # Filter out empty strings from set() if any
            combined_string_parts = [p for p in list(data['items']) + list(data['attributes']) if p.strip()]
            combined_string = " ".join(combined_string_parts)
            
            if combined_string:
                # Add group score to final output. Adjust score based on number of items/attributes.
                # A group with more specific details might be more valuable.
                adjusted_score = data['score'] * (1 + (len(combined_string_parts) * 0.1)) # Small boost for more details
                final_output.append((combined_string, adjusted_score))
        
        # Also add any general keywords that weren't part of a group
        # but are still relevant from keyword_scores
        for kw, score in keyword_scores.items():
            is_grouped = False
            for group_key, data in grouped_keywords.items():
                if kw in data['items'] or kw in data['attributes']:
                    is_grouped = True
                    break
            if not is_grouped:
                final_output.append((kw, score))
        
        # Sort the final output
        final_output.sort(key=lambda x: x[1], reverse=True)
        print(f"\n🏆 FINAL GROUPED KEYWORDS (Multi-item):")
        for i, (kw, score) in enumerate(final_output[:15]):
            print(f"   {i+1:2d}. '{kw}' → {score:.1f}")
        
        # Store results
        extract_ranked_keywords.last_exclusions = list(global_exclusions)
        extract_ranked_keywords.wanted_items = wanted_items
        extract_ranked_keywords.context_items = context_items
        extract_ranked_keywords.is_multi_item_request = is_multi_item_request # Store this
        
        return final_output[:15] # Return a list of (combined_string, score)
        
    else: # NOT a multi-item request, return flat list as before
        ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 FINAL FASHION CATEGORIES ENHANCED KEYWORDS (Single-item):")
        for i, (keyword, score) in enumerate(ranked_keywords[:15]):
            clothing_cat = fashion_categories.get_clothing_category(keyword)
            
            if clothing_cat in wanted_items:
                category_icon = "🎯"
                priority = "🚀 WANTED"
            elif clothing_cat in context_items:
                category_icon = "📝"
                priority = "📋 CONTEXT"
            elif score >= 500:
                category_icon = "⭐"
                priority = "🔥 AI-HIGH"
            elif score >= 300:
                category_icon = "👕"
                priority = "🎯 HIGH"
            elif score >= 150:
                category_icon = "📋"
                priority = "📋 MED"
            else:
                category_icon = "📝"
                priority = "📝 LOW"
            
            clothing_display = f" [{clothing_cat}]" if clothing_cat else ""
            print(f"   {i+1:2d}. {category_icon} {priority} '{keyword}'{clothing_display} → {score:.1f}")
        
        if global_exclusions:
            print(f"\n🚫 PRODUCT EXCLUSIONS:")
            for term in sorted(global_exclusions):
                print(f"   ❌ '{term}'")
        
        # Store results
        extract_ranked_keywords.last_exclusions = list(global_exclusions)
        extract_ranked_keywords.wanted_items = wanted_items
        extract_ranked_keywords.context_items = context_items
        extract_ranked_keywords.is_multi_item_request = is_multi_item_request # Store this
        
        return ranked_keywords[:15]
           
def get_search_terms_for_keyword(keyword):
    """
    Enhanced translation mapping using FashionCategories for consistency.
    Get both English and Indonesian search terms for a keyword to improve product matching.
    Returns a dictionary with 'include' and 'exclude' terms.
    """
    keyword_lower = keyword.lower().strip()
    
    # Streamlined translation mapping - only essential translations
    translation_map = {
        # Core clothing types with exclusions
        'shirt': {
            'include': ['shirt', 'kemeja', 'baju', 'atasan'],
            'exclude': ['t-shirt', 'tshirt', 'kaos', 'tank top', 'polo']
        },
        'kemeja': {
            'include': ['kemeja', 'shirt', 'baju', 'atasan'],
            'exclude': ['t-shirt', 'tshirt', 'kaos', 'tank top', 'polo']
        },
        'blouse': {
            'include': ['blouse', 'blus', 'kemeja wanita', 'atasan wanita'],
            'exclude': ['t-shirt', 'kaos', 'tank top']
        },
        'blus': {
            'include': ['blus', 'blouse', 'kemeja wanita'],
            'exclude': ['t-shirt', 'kaos', 'tank top']
        },
        't-shirt': {
            'include': ['t-shirt', 'tshirt', 'kaos', 'baju kaos'],
            'exclude': ['kemeja', 'shirt', 'blouse', 'formal shirt']
        },
        'kaos': {
            'include': ['kaos', 't-shirt', 'tshirt', 'baju kaos'],
            'exclude': ['kemeja', 'shirt', 'blouse', 'formal shirt']
        },
        
        # Bottoms
        'pants': {
            'include': ['pants', 'celana', 'bawahan'],
            'exclude': []
        },
        'celana': {
            'include': ['celana', 'pants', 'bawahan'],
            'exclude': []
        },
        'skirt': {
            'include': ['skirt', 'rok'],
            'exclude': []
        },
        'rok': {
            'include': ['rok', 'skirt'],
            'exclude': []
        },
        'jeans': {
            'include': ['jeans', 'celana jeans', 'denim'],
            'exclude': []
        },
        
        # Dresses
        'dress': {
            'include': ['dress', 'gaun', 'terusan'],
            'exclude': ['shirt', 'kemeja', 'top', 'atasan']
        },
        'gaun': {
            'include': ['gaun', 'dress', 'terusan'],
            'exclude': ['shirt', 'kemeja', 'top', 'atasan']
        },
        
        # Outerwear
        'jacket': {
            'include': ['jacket', 'jaket', 'jas'],
            'exclude': []
        },
        'jaket': {
            'include': ['jaket', 'jacket', 'jas'],
            'exclude': []
        },
        'sweater': {
            'include': ['sweater', 'baju hangat', 'jumper'],
            'exclude': []
        },
        'cardigan': {
            'include': ['cardigan', 'kardigan'],
            'exclude': []
        },
        'hoodie': {
            'include': ['hoodie', 'jaket hoodie', 'sweater hoodie'],
            'exclude': []
        },
        
        # Core colors with translations
        'white': {'include': ['white', 'putih'], 'exclude': []},
        'putih': {'include': ['putih', 'white'], 'exclude': []},
        'black': {'include': ['black', 'hitam'], 'exclude': []},
        'hitam': {'include': ['hitam', 'black'], 'exclude': []},
        'red': {'include': ['red', 'merah'], 'exclude': []},
        'merah': {'include': ['merah', 'red'], 'exclude': []},
        'blue': {'include': ['blue', 'biru'], 'exclude': []},
        'biru': {'include': ['biru', 'blue'], 'exclude': []},
        'green': {'include': ['green', 'hijau'], 'exclude': []},
        'hijau': {'include': ['hijau', 'green'], 'exclude': []},
        'yellow': {'include': ['yellow', 'kuning'], 'exclude': []},
        'kuning': {'include': ['kuning', 'yellow'], 'exclude': []},
        'navy': {'include': ['navy', 'biru tua', 'navy blue'], 'exclude': []},
        'grey': {'include': ['grey', 'gray', 'abu-abu'], 'exclude': []},
        'abu-abu': {'include': ['abu-abu', 'grey', 'gray'], 'exclude': []},
        
        # Core styles
        'casual': {'include': ['casual', 'santai', 'kasual'], 'exclude': []},
        'santai': {'include': ['santai', 'casual', 'kasual'], 'exclude': []},
        'formal': {'include': ['formal', 'resmi'], 'exclude': []},
        'resmi': {'include': ['resmi', 'formal'], 'exclude': []},
        'elegant': {'include': ['elegant', 'elegan'], 'exclude': []},
        'elegan': {'include': ['elegan', 'elegant'], 'exclude': []},
        
        # Materials
        'cotton': {'include': ['cotton', 'katun'], 'exclude': []},
        'katun': {'include': ['katun', 'cotton'], 'exclude': []},
        'silk': {'include': ['silk', 'sutra'], 'exclude': []},
        'sutra': {'include': ['sutra', 'silk'], 'exclude': []},
        'denim': {'include': ['denim', 'jeans'], 'exclude': []},
        
        # Fits with opposites excluded
        'oversized': {'include': ['oversized', 'longgar', 'loose'], 'exclude': ['slim', 'tight', 'fitted']},
        'longgar': {'include': ['longgar', 'oversized', 'loose'], 'exclude': ['slim', 'tight', 'fitted']},
        'slim': {'include': ['slim', 'ketat', 'tight', 'fitted'], 'exclude': ['oversized', 'loose', 'longgar']},
        'ketat': {'include': ['ketat', 'slim', 'tight', 'fitted'], 'exclude': ['oversized', 'loose', 'longgar']},
        'loose': {'include': ['loose', 'longgar', 'oversized'], 'exclude': ['slim', 'tight', 'fitted']},
        'tight': {'include': ['tight', 'ketat', 'slim', 'fitted'], 'exclude': ['oversized', 'loose', 'longgar']},
        
        # Sleeve lengths with opposites excluded
        'lengan panjang': {
            'include': ['lengan panjang', 'long sleeve', 'long sleeves'],
            'exclude': ['lengan pendek', 'short sleeve', 'sleeveless']
        },
        'long sleeve': {
            'include': ['long sleeve', 'long sleeves', 'lengan panjang'],
            'exclude': ['short sleeve', 'lengan pendek', 'sleeveless']
        },
        'lengan pendek': {
            'include': ['lengan pendek', 'short sleeve', 'short sleeves'],
            'exclude': ['lengan panjang', 'long sleeve', 'sleeveless']
        },
        'short sleeve': {
            'include': ['short sleeve', 'short sleeves', 'lengan pendek'],
            'exclude': ['long sleeve', 'lengan panjang', 'sleeveless']
        },
        'sleeveless': {
            'include': ['sleeveless', 'tanpa lengan'],
            'exclude': ['long sleeve', 'short sleeve', 'lengan panjang', 'lengan pendek']
        },
        'tanpa lengan': {
            'include': ['tanpa lengan', 'sleeveless'],
            'exclude': ['long sleeve', 'short sleeve', 'lengan panjang', 'lengan pendek']
        },
        
        # Lengths with opposites excluded
        'maxi': {
            'include': ['maxi', 'panjang', 'long'],
            'exclude': ['mini', 'pendek', 'short', 'crop']
        },
        'panjang': {
            'include': ['panjang', 'long', 'maxi'],
            'exclude': ['pendek', 'short', 'mini', 'crop']
        },
        'mini': {
            'include': ['mini', 'pendek', 'short'],
            'exclude': ['maxi', 'panjang', 'long']
        },
        'pendek': {
            'include': ['pendek', 'short', 'mini'],
            'exclude': ['panjang', 'long', 'maxi']
        },
        'midi': {
            'include': ['midi', 'medium length', 'sedang'],
            'exclude': []
        },
        'crop': {
            'include': ['crop', 'cropped', 'short'],
            'exclude': ['long', 'maxi', 'panjang']
        },
    }
    
    # If the keyword has a direct mapping, return it
    if keyword_lower in translation_map:
        return translation_map[keyword_lower]
    
    # Check if it's a fashion term from FashionCategories and create basic mapping
    if (fashion_categories.is_clothing_item(keyword) or 
        fashion_categories.is_style_term(keyword) or 
        fashion_categories.is_color_term(keyword)):
        return {'include': [keyword_lower], 'exclude': []}
    
    # Check if it's from any FashionCategories term lists
    all_fashion_terms = fashion_categories.get_all_fashion_terms()
    if keyword_lower in [term.lower() for term in all_fashion_terms]:
        return {'include': [keyword_lower], 'exclude': []}
    
    # Default fallback for unknown terms
    return {'include': [keyword_lower], 'exclude': []}

# Enhanced function to get all search terms for use in extract_ranked_keywords
def get_all_search_terms_for_extraction(keyword):
    """
    Helper function that uses the complete keyword mapping for extracting ranked keywords.
    This integrates with extract_ranked_keywords to improve keyword processing.
    """
    search_mapping = get_search_terms_for_keyword(keyword)
    
    # Return all include terms for broader matching in keyword extraction
    return search_mapping['include']

def preprocess_text_for_tfidf(text):
    """
    Preprocess text for better TF-IDF results
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def create_product_text_corpus(all_products):
    """
    Create a text corpus from all products for TF-IDF fitting
    """
    corpus = []
    
    for product_row in all_products:
        # Combine product name, detail, and colors into one text
        product_name = preprocess_text_for_tfidf(product_row[1])
        product_detail = preprocess_text_for_tfidf(product_row[2])
        available_colors = preprocess_text_for_tfidf(product_row[7] if len(product_row) > 7 and product_row[7] else "")
        
        # Create combined text with emphasis on product name (repeat it)
        combined_text = f"{product_name} {product_name} {product_detail} {available_colors}"
        corpus.append(combined_text)
    
    return corpus

def initialize_tfidf_model(all_products):
    """
    Initialize TF-IDF model with enhanced preprocessing for better semantic matching
    """
    global tfidf_vectorizer, TFIDF_MODEL_FITTED, product_tfidf_matrix
    
    try:
        print("🧠 Initializing enhanced TF-IDF model...")
        
        # Create enhanced text corpus
        product_texts = []
        for product_row in all_products:
            # Combine and emphasize product name
            product_name = preprocess_text_for_tfidf(product_row[1])
            product_detail = preprocess_text_for_tfidf(product_row[2])
            available_colors = preprocess_text_for_tfidf(product_row[7] if len(product_row) > 7 and product_row[7] else "")
            
            # Emphasize product name by repeating it
            combined_text = f"{product_name} {product_name} {product_detail} {available_colors}"
            product_texts.append(combined_text)
        
        if not product_texts:
            print("⚠️ No product texts found for TF-IDF")
            return False
        
        # Enhanced TF-IDF vectorizer with better parameters for fashion/clothing
        tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,  # Increased vocabulary size
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better fashion term matching
            min_df=1,  # Include all terms
            max_df=0.90,  # Allow more common fashion terms
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            sublinear_tf=True,  # Use sublinear scaling
            smooth_idf=True,   # Smooth IDF weights
            norm='l2'          # L2 normalization
        )
        
        # Fit and transform
        product_tfidf_matrix = tfidf_vectorizer.fit_transform(product_texts)
        
        TFIDF_MODEL_FITTED = True
        
        print(f"✅ Enhanced TF-IDF model fitted successfully!")
        print(f"   📊 Vocabulary size: {len(tfidf_vectorizer.get_feature_names_out())}")
        print(f"   📦 Products indexed: {len(product_texts)}")
        print(f"   🔢 TF-IDF matrix shape: {product_tfidf_matrix.shape}")
        print(f"   🧠 Semantic ranking: ENABLED")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing enhanced TF-IDF model: {str(e)}")
        TFIDF_MODEL_FITTED = False
        return False
    
def calculate_semantic_similarity(query_keywords, product_row, product_index=None):
    """
    Calculate semantic similarity using TF-IDF
    """
    global tfidf_vectorizer, product_tfidf_matrix
    
    if not TFIDF_MODEL_FITTED or tfidf_vectorizer is None:
        return 0.0
    
    try:
        # Create query string from keywords
        if isinstance(query_keywords, list):
            if query_keywords and isinstance(query_keywords[0], tuple):
                # List of (keyword, weight) tuples
                query_string = " ".join([kw for kw, weight in query_keywords[:10]])
            else:
                # List of keywords
                query_string = " ".join(query_keywords[:10])
        else:
            query_string = str(query_keywords)
        
        # Preprocess query
        query_string = preprocess_text_for_tfidf(query_string)
        
        if not query_string:
            return 0.0
        
        # Transform query to TF-IDF vector
        query_tfidf_vector = tfidf_vectorizer.transform([query_string])
        
        if query_tfidf_vector.nnz == 0:  # No common terms
            return 0.0
        
        # If we have pre-computed product matrix and index, use it
        if product_tfidf_matrix is not None and product_index is not None:
            if product_index < product_tfidf_matrix.shape[0]:
                product_vector = product_tfidf_matrix[product_index:product_index+1]
                
                if product_vector.nnz > 0:
                    similarity = sk_cosine_similarity(query_tfidf_vector, product_vector)[0][0]
                    return float(similarity)
        
        # Fallback: create product vector on the fly
        product_name = preprocess_text_for_tfidf(product_row[1])
        product_detail = preprocess_text_for_tfidf(product_row[2])
        available_colors = preprocess_text_for_tfidf(product_row[7] if len(product_row) > 7 and product_row[7] else "")
        
        product_text = f"{product_name} {product_name} {product_detail} {available_colors}"
        product_tfidf_vector = tfidf_vectorizer.transform([product_text])
        
        if product_tfidf_vector.nnz > 0:
            similarity = sk_cosine_similarity(query_tfidf_vector, product_tfidf_vector)[0][0]
            return float(similarity)
        
        return 0.0
        
    except Exception as e:
        print(f"⚠️ Error calculating semantic similarity: {str(e)}")
        return 0.0

def calculate_relevance_score(product_row, keywords, debug=False, focus_category=None, product_index=None, is_multi_item_request=False):
    """
    Enhanced relevance calculation with TF-IDF as PRIMARY ranking factor
    and STRONGER ATTRIBUTE MATCHING, emphasizing the MAIN CLOTHING ITEM,
    with special handling for multi-item requests and grouped keywords.
    """
    global tfidf_vectorizer, TFIDF_MODEL_FITTED

    product_name = product_row[1].lower()
    product_detail = product_row[2].lower()
    available_colors = product_row[7].lower() if product_row[7] else ""
    product_gender = product_row[4].lower()

    search_text = f"{product_name} {product_detail} {available_colors}"

    if debug:
        print(f"   🔍 Search text: '{search_text[:100]}...'")
        print(f"   🎯 Focus category: {focus_category}")
        print(f"   📝 Checking against {len(keywords)} keywords:")
        print(f"   🤝 Is Multi-Item Request: {is_multi_item_request}")

    # STEP 0: DETERMINE REQUESTED CLOTHING TYPES from user's keywords
    requested_clothing_types = set()
    primary_clothing_item_from_keywords = None 

    if keywords:
        for kw, weight in keywords:
            # Handle combined keywords from multi-item requests
            kw_to_check = kw
            if is_multi_item_request and ' ' in kw:
                # If it's a combined string (e.g., "short pants"), split and check components
                parts = kw.split()
                # Check for main clothing item within the parts
                for part in parts:
                    clothing_cat = fashion_categories.get_clothing_category(part)
                    if clothing_cat and clothing_cat != 'unknown':
                        requested_clothing_types.add(clothing_cat)
                        if primary_clothing_item_from_keywords is None:
                            primary_clothing_item_from_keywords = clothing_cat
                        break # Found a clothing item, move to next combined keyword
            else: # Normal single keyword
                clothing_cat = fashion_categories.get_clothing_category(kw)
                if clothing_cat and clothing_cat != 'unknown':
                    requested_clothing_types.add(clothing_cat)
                    if primary_clothing_item_from_keywords is None:
                        primary_clothing_item_from_keywords = clothing_cat
            
            # Break if a primary focus is found (especially for single-item requests)
            if primary_clothing_item_from_keywords and not is_multi_item_request:
                break 

    if debug:
        print(f"   👕 Requested Clothing Types from Keywords: {requested_clothing_types}")
        print(f"   🎯 Primary Clothing Item from Keywords (first detected): {primary_clothing_item_from_keywords}")

    # STEP 1: STRICT FILTERING (MODIFIED FOR MULTI-ITEM)
    product_category = fashion_categories.get_clothing_category(search_text)

    # Apply strict category filtering ONLY IF NOT a multi-item request AND a primary clothing type was requested
    if not is_multi_item_request and primary_clothing_item_from_keywords:
        focus_terms = fashion_categories.CLOTHING_CATEGORIES.get(primary_clothing_item_from_keywords, [])
        if focus_terms and not any(term in search_text for term in focus_terms):
            if debug:
                print(f"   🚫 PRIMARY CLOTHING MISMATCH (STRICT): Expected {primary_clothing_item_from_keywords}, Product is {product_category} ('{product_row[1]}')")
            return 0.0

    # New: Gender filtering (critical for relevance) - UNCHANGED
    user_gender_preference = None
    for kw, weight in keywords:
        if fashion_categories.is_gender_term(kw):
            if any(term in kw.lower() for term in ['perempuan', 'wanita', 'female', 'woman', 'cewek', 'cewe']):
                user_gender_preference = 'female'
                break
            elif any(term in kw.lower() for term in ['pria', 'laki-laki', 'male', 'man', 'cowok', 'cowo']):
                user_gender_preference = 'male'
                break

    if user_gender_preference and product_gender != 'unisex':
        if user_gender_preference != product_gender:
            if debug:
                print(f"   🚫 GENDER MISMATCH: User wants {user_gender_preference}, Product is {product_gender}")
            return 0.0


    # STEP 2: TF-IDF SEMANTIC SCORING (PRIMARY FACTOR) - MODIFIED FOR GROUPED KEYWORDS
    semantic_score = 0.0
    if TFIDF_MODEL_FITTED and tfidf_vectorizer is not None:
        try:
            query_parts = []
            for kw, weight in keywords:
                # For multi-item, break down combined keywords for TF-IDF
                if is_multi_item_request and ' ' in kw:
                    # Treat each part of the combined keyword as a separate term for TF-IDF
                    # Boost the clothing item, and attributes if applicable
                    for part in kw.split():
                        keyword_lower = part.lower()
                        if fashion_categories.is_clothing_item(keyword_lower):
                            query_parts.append(f"{keyword_lower} {keyword_lower} {keyword_lower} {keyword_lower}") # Quadruple boost
                        elif fashion_categories.is_color_term(keyword_lower):
                            query_parts.append(f"{keyword_lower} {keyword_lower} {keyword_lower}") # Triple boost
                        elif any(term in keyword_lower for term in fashion_categories.FIT_TERMS + fashion_categories.SLEEVE_TERMS + fashion_categories.LENGTH_TERMS):
                             query_parts.append(f"{keyword_lower} {keyword_lower}") # Double boost
                        else:
                            query_parts.append(keyword_lower)
                else: # Original single keyword logic
                    keyword_lower = kw.lower()
                    if fashion_categories.is_clothing_item(keyword_lower):
                        query_parts.append(f"{keyword_lower} {keyword_lower} {keyword_lower} {keyword_lower}")
                    elif fashion_categories.is_color_term(keyword_lower):
                        query_parts.append(f"{keyword_lower} {keyword_lower} {keyword_lower}")
                    elif any(term in keyword_lower for term in fashion_categories.FIT_TERMS + fashion_categories.SLEEVE_TERMS + fashion_categories.LENGTH_TERMS):
                         query_parts.append(f"{keyword_lower} {keyword_lower}")
                    else:
                        query_parts.append(keyword_lower)

            query_string = " ".join(query_parts)
            query_string = preprocess_text_for_tfidf(query_string)

            if query_string:
                query_tfidf_vector = tfidf_vectorizer.transform([query_string])

                if product_tfidf_matrix is not None and product_index is not None:
                    if product_index < product_tfidf_matrix.shape[0]:
                        product_vector = product_tfidf_matrix[product_index:product_index+1]

                        if query_tfidf_vector.nnz > 0 and product_vector.nnz > 0:
                            similarity = sk_cosine_similarity(query_tfidf_vector, product_vector)[0][0]
                            semantic_score = similarity * 10000

                            if debug:
                                print(f"   🧠 TF-IDF SEMANTIC SCORE (PRIMARY): {semantic_score:.2f} (similarity: {similarity:.4f})")
                else:
                    product_text = f"{product_row[1]} {product_row[2]} {product_row[7] if product_row[7] else ''}"
                    product_text_preprocessed = preprocess_text_for_tfidf(product_text)

                    if product_text_preprocessed:
                        product_tfidf_vector = tfidf_vectorizer.transform([product_text_preprocessed])

                        if query_tfidf_vector.nnz > 0 and product_tfidf_vector.nnz > 0:
                            similarity = sk_cosine_similarity(query_tfidf_vector, product_tfidf_vector)[0][0]
                            semantic_score = similarity * 10000

                            if debug:
                                print(f"   🧠 TF-IDF SEMANTIC SCORE (FALLBACK): {semantic_score:.2f}")

        except Exception as e:
            if debug:
                print(f"   ❌ TF-IDF semantic scoring failed: {e}")

    # STEP 3: KEYWORD MATCHING SCORE (SECONDARY FACTOR) - MODIFIED FOR GROUPED KEYWORDS
    keyword_score = 0.0
    matches_found = []

    for i, (keyword, weight) in enumerate(keywords[:15]):
        keyword_lower_original = keyword.lower() # Store original for logging
        position_weight = (15 - i) / 15
        # Break down combined keywords for individual matching
        keywords_to_match = [keyword_lower_original]
        if is_multi_item_request and ' ' in keyword_lower_original:
            keywords_to_match = keyword_lower_original.split()

        current_kw_match_score = 0 # Score for the current (potentially combined) keyword
        current_kw_match_type = "NO_MATCH"
        
        # Iterate over parts of the combined keyword or just the single keyword
        for single_kw_to_match in keywords_to_match:
            match_score = 0
            match_type = "NO_MATCH"

            # CLOTHING ITEM MATCH (HIGHEST PRIORITY IN KEYWORD MATCHING)
            if fashion_categories.is_clothing_item(single_kw_to_match):
                kw_clothing_cat = fashion_categories.get_clothing_category(single_kw_to_match)

                if kw_clothing_cat in requested_clothing_types:
                    if single_kw_to_match in product_name:
                        match_score = weight * position_weight * 300 # EXTREME BOOST
                        match_type = "REQUESTED_CLOTHING_EXACT_MATCH"
                    elif single_kw_to_match in product_detail:
                        match_score = weight * position_weight * 250
                        match_type = "REQUESTED_CLOTHING_DETAIL_MATCH"
                    elif any(part in search_text for part in single_kw_to_match.split()):
                        match_score = weight * position_weight * 200
                        match_type = "REQUESTED_CLOTHING_PARTIAL_MATCH"
                elif not is_multi_item_request:
                    pass # Already handled by strict filter if primary_clothing_item_from_keywords exists
                else:
                    # If it IS a multi-item request, allow OTHER clothing types to score
                    if single_kw_to_match in product_name:
                        match_score = weight * position_weight * 50
                        match_type = "OTHER_REQUESTED_CLOTHING_NAME_MATCH"
                    elif single_kw_to_match in product_detail:
                        match_score = weight * position_weight * 40
                        match_type = "OTHER_REQUESTED_CLOTHING_DESC_MATCH"

            # STYLE/COLOR/MATERIAL MATCHES (Still very important)
            elif (fashion_categories.is_style_term(single_kw_to_match) or
                  fashion_categories.is_color_term(single_kw_to_match) or
                  any(term in single_kw_to_match for term in fashion_categories.MATERIAL_TERMS) or
                  any(term in single_kw_to_match for term in fashion_categories.FIT_TERMS + fashion_categories.SLEEVE_TERMS + fashion_categories.LENGTH_TERMS)):
                if single_kw_to_match in search_text or single_kw_to_match in available_colors:
                    match_score = weight * position_weight * 100
                    match_type = "CRITICAL_ATTRIBUTE_MATCH"
                    if debug:
                        print(f"      CRITICAL ATTRIBUTE Match: '{single_kw_to_match}' in '{search_text}' (+{match_score:.2f})")
                else:
                    if i < 5:
                        if fashion_categories.is_color_term(single_kw_to_match) or any(term in single_kw_to_match for term in fashion_categories.FIT_TERMS):
                            if debug:
                                print(f"      PENALTY: Top attribute '{single_kw_to_match}' NOT found. (-{weight * position_weight * 50:.2f})")
                            current_kw_match_score -= (weight * position_weight * 50) # Apply penalty to combined score

            # OCCASION MATCHES (Medium priority)
            elif any(term in single_kw_to_match for term in fashion_categories.OCCASION_TERMS):
                if single_kw_to_match in search_text:
                    match_score = weight * position_weight * 20
                    match_type = "OCCASION_MATCH"
            
            # Aggregate scores for combined keywords
            current_kw_match_score += match_score
            if match_type != "NO_MATCH": # Only update type if a match was found
                current_kw_match_type = match_type 

        # Add the aggregated score for the original (potentially combined) keyword
        if current_kw_match_score > 0 or current_kw_match_type == "ATTRIBUTE_PENALTY": # Include penalties
            keyword_score += current_kw_match_score
            matches_found.append((keyword_lower_original, current_kw_match_type, current_kw_match_score)) # Use original for logging

            if debug:
                print(f"      ✅ '{keyword_lower_original}' → {current_kw_match_type} (+{current_kw_match_score:.2f})")
        
    # STEP 4: CATEGORY BONUS (TERTIARY FACTOR) - MODIFIED
    category_bonus = 0
    if is_multi_item_request and product_category in requested_clothing_types:
        category_bonus = 150
        if debug:
            print(f"   🎯 MULTI-ITEM CATEGORY ALIGNMENT BONUS: +{category_bonus}")
    elif not is_multi_item_request and primary_clothing_item_from_keywords and product_category == primary_clothing_item_from_keywords:
        category_bonus = 50
        if debug:
            print(f"   🎯 SINGLE-ITEM CATEGORY ALIGNMENT BONUS: +{category_bonus}")

    # STEP 5: COMBINE SCORES WITH TF-IDF AS PRIMARY FACTOR - UNCHANGED
    total_score = semantic_score + (keyword_score * 0.2) + category_bonus

    if semantic_score == 0 and TFIDF_MODEL_FITTED:
        total_score *= 0.1
        if debug:
            print(f"   ⚠️ NO SEMANTIC SIMILARITY - Applied penalty")

    if total_score < 0:
        total_score = 0.0

    if debug:
        print(f"   📊 SCORE BREAKDOWN:")
        print(f"      🧠 TF-IDF Semantic: {semantic_score:.2f} (PRIMARY)")
        print(f"      🔤 Keyword Matching: {keyword_score:.2f} (weight: 0.2)")
        print(f"      🎯 Category Bonus: {category_bonus:.2f}")
        print(f"      🔢 TOTAL: {total_score:.2f}")
        print(f"   🎯 Best keyword matches: {[f'{kw}({mt})' for kw, mt, _ in matches_found[:3]]}")

    return total_score

def get_clothing_category(keyword):
    """Get clothing category for a keyword using shared categories"""
    keyword_lower = keyword.lower()
    clothing_categories = get_shared_clothing_categories()
    
    for category, terms in clothing_categories.items():
        if any(term in keyword_lower for term in terms):
            return category
    return None

async def fetch_products_from_db(db: AsyncSession, top_keywords: list, max_results=15, gender_category=None, budget_range=None, focus_category=None, is_multi_item_request=False):
    """
    Enhanced product fetching with TF-IDF as PRIMARY ranking factor and balanced item distribution for multi-item requests.
    """
    global tfidf_vectorizer, TFIDF_MODEL_FITTED, product_tfidf_matrix
    
    print("\n" + "="*80)
    print("🧠 ENHANCED TF-IDF SEMANTIC RANKING")
    print("="*80)
    print(f"📊 Keywords received: {len(top_keywords)}")
    print(f"🎯 Top 10 keywords for TF-IDF:")
    for i, (kw, score) in enumerate(top_keywords[:10]):
        print(f"   {i+1:2d}. '{kw}' → Score: {score:.2f}")
    print(f"   🤝 Is Multi-Item Request (in fetch_products_from_db): {is_multi_item_request}")
    try:
        # Build the database query (same as your existing logic)
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

        # Apply filters
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

        if not TFIDF_MODEL_FITTED and all_products:
            print("\n🧠 Initializing Enhanced TF-IDF model...")
            success = initialize_tfidf_model(all_products)
            if success:
                print("✅ Enhanced TF-IDF model ready for semantic ranking")
            else:
                print("⚠️ TF-IDF initialization failed, using keyword-only matching")

        if not all_products:
            print("❌ No products found in database")
            return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])

        print(f"📦 Found {len(all_products)} products to rank")
        print(f"🧠 Enhanced TF-IDF ranking: {'🟢 ENABLED' if TFIDF_MODEL_FITTED else '🔴 DISABLED'}")

        print(f"\n🧮 CALCULATING ENHANCED RELEVANCE SCORES...")

        all_products_with_scores = []
        debug_count = 0
        
        # Collect all requested clothing categories from keywords for distribution logic
        requested_clothing_categories_for_distribution = set()
        for kw, _ in top_keywords:
            if is_multi_item_request and ' ' in kw: # Handle combined keywords (e.g., "short pants")
                for part in kw.split():
                    cat = fashion_categories.get_clothing_category(part)
                    if cat and cat != 'unknown':
                        requested_clothing_categories_for_distribution.add(cat)
            else: # Handle single keywords (e.g., "skirt")
                cat = fashion_categories.get_clothing_category(kw)
                if cat and cat != 'unknown':
                    requested_clothing_categories_for_distribution.add(cat)
        
        print(f"   🎯 Categories requested for distribution: {requested_clothing_categories_for_distribution}")

        for product_index, product_row in enumerate(all_products):
            debug_this_product = debug_count < 3

            if debug_this_product:
                print(f"\n🔍 DEBUGGING PRODUCT {debug_count + 1}: '{product_row[1]}'")
                print(f"   💰 Price: IDR {product_row[5]:,}")
                debug_count += 1

            # Pass the is_multi_item_request flag here
            relevance_score = calculate_relevance_score(
                product_row, top_keywords, debug_this_product, focus_category, product_index,
                is_multi_item_request=is_multi_item_request
            )

            if relevance_score <= 0:
                continue

            sizes = product_row[6].split(',') if product_row[6] else []
            colors = product_row[7].split(',') if product_row[7] else []
            
            # Get product's category for distribution
            product_category_for_distribution = fashion_categories.get_clothing_category(product_row[1].lower())

            product_data = {
                "product_id": product_row[0],
                "product": product_row[1], # Use product_name from product_row for consistency
                "description": product_row[2], # Use product_detail from product_row for consistency
                "price": product_row[5],
                "size": ", ".join(sizes) if sizes else "N/A",
                "color": ", ".join(colors) if colors else "N/A",
                "stock": product_row[8],
                "link": f"http://localhost/e-commerce-main/product-{product_row[3]}-{product_row[0]}",
                "photo": product_row[9],
                "relevance": relevance_score,
                "tfidf_enabled": TFIDF_MODEL_FITTED,
                "product_category_for_distribution": product_category_for_distribution # Store category for distribution
            }

            all_products_with_scores.append(product_data)
        
        # CRITICAL: Sort by enhanced relevance score (TF-IDF semantic score is primary)
        all_products_with_scores.sort(key=lambda x: x['relevance'], reverse=True)
        
        print(f"\n🧠 ENHANCED TF-IDF RANKING RESULTS:")
        print(f"   📊 Products ranked: {len(all_products_with_scores)}")
        if all_products_with_scores:
            print(f"   🥇 Highest score: {all_products_with_scores[0]['relevance']:.2f}")
            print(f"   🥉 Lowest score: {all_products_with_scores[-1]['relevance']:.2f}")
        else:
            print("   ❌ No products scored")
        
        final_products = []
        if is_multi_item_request and requested_clothing_categories_for_distribution:
            print(f"\nDistribution Strategy: Multi-Item Request detected. Balancing results across categories.")
            
            # Group products by their detected category
            products_by_category = defaultdict(list)
            for p_data in all_products_with_scores:
                if p_data["product_category_for_distribution"]: # Only group if a category was detected
                    products_by_category[p_data["product_category_for_distribution"]].append(p_data)
            
            print(f"   Products grouped by category: { {cat: len(prods) for cat, prods in products_by_category.items()} }")

            # --- REFINED DISTRIBUTION LOGIC ---
            # Phase 1: Ensure at least one product from each requested category
            # We want to prioritize a balanced representation in the initial view.
            added_product_ids = set()
            products_to_distribute = [] # This will hold products for balanced picking

            # Add at least one of the highest-scoring product from each requested category first
            for category in requested_clothing_categories_for_distribution:
                if category in products_by_category and products_by_category[category]:
                    top_product_in_cat = products_by_category[category][0] # Already sorted by relevance
                    if top_product_in_cat['product_id'] not in added_product_ids:
                        products_to_distribute.append(top_product_in_cat)
                        added_product_ids.add(top_product_in_cat['product_id'])
                        print(f"      Phase 1: Guaranteed 1 from '{category}' ('{top_product_in_cat['product'][:20]}...')")
            
            # Now, add remaining products from all categories, alternating to balance
            # Create iterators for all products, excluding those already added
            all_products_remaining_iter = iter([
                p for p in all_products_with_scores if p['product_id'] not in added_product_ids
            ])

            # Fill up to max_results by alternating through available categories
            # And then from the general pool if categories are exhausted
            num_categories_in_input = len(requested_clothing_categories_for_distribution)
            if num_categories_in_input > 0:
                # Create category-specific iterators for remaining products
                category_specific_iters = {
                    cat: iter([p for p in products_by_category[cat] if p['product_id'] not in added_product_ids])
                    for cat in requested_clothing_categories_for_distribution
                    if cat in products_by_category
                }
                
                current_category_idx_for_alternating = 0
                while len(products_to_distribute) < max_results:
                    found_product_in_round = False
                    
                    # Try to pick from requested categories in a round-robin fashion
                    categories_in_rotation = list(category_specific_iters.keys())
                    if not categories_in_rotation: # All specific categories exhausted
                        break
                    
                    for _ in range(len(categories_in_rotation)): # One round through remaining categories
                        cat_to_pick_from = categories_in_rotation[current_category_idx_for_alternating % len(categories_in_rotation)]
                        current_category_idx_for_alternating += 1 # Advance for next round

                        try:
                            next_prod = next(category_specific_iters[cat_to_pick_from])
                            if next_prod['product_id'] not in added_product_ids:
                                products_to_distribute.append(next_prod)
                                added_product_ids.add(next_prod['product_id'])
                                found_product_in_round = True
                                print(f"      Phase 2: Added '{next_prod['product'][:20]}...' from '{cat_to_pick_from}'. Total: {len(products_to_distribute)}/{max_results}")
                                if len(products_to_distribute) == max_results: break
                        except StopIteration:
                            del category_specific_iters[cat_to_pick_from] # Remove exhausted category
                            print(f"      Category '{cat_to_pick_from}' exhausted for alternating fill.")
                            if not category_specific_iters: # All category specific iterators are done
                                break
                    
                    if not found_product_in_round and not category_specific_iters: # No more products from specific categories
                        break
                    elif not found_product_in_round: # No unique product found in this round, but categories still have iterators
                        # This could happen if remaining products in iterators are already picked by some other logic (e.g. from Phase 1 but lower rank)
                        break

            # Phase 3: Fill any remaining slots from the overall top-ranked list
            # This ensures we always return max_results if enough products exist.
            print(f"   Phase 3: Attempting to fill remaining {max_results - len(products_to_distribute)} slots from overall best.")
            while len(products_to_distribute) < max_results:
                try:
                    next_overall_prod = next(all_products_remaining_iter)
                    if next_overall_prod['product_id'] not in added_product_ids:
                        products_to_distribute.append(next_overall_prod)
                        added_product_ids.add(next_overall_prod['product_id'])
                        print(f"      Phase 3: Added '{next_overall_prod['product'][:20]}...' from overall list. Total: {len(products_to_distribute)}/{max_results}")
                except StopIteration:
                    print("      Overall product list exhausted before reaching max_results.")
                    break
            
            final_products = products_to_distribute # Assign the distributed list
            final_products.sort(key=lambda x: x['relevance'], reverse=True) # Final sort for display
            print(f"   Final distributed products count: {len(final_products)}")

        else: # NOT a multi-item request, or no specific clothing categories requested for distribution
            print(f"\\nDistribution Strategy: Single-Item or no specific category request. Taking overall top {max_results} results.")
            final_products = all_products_with_scores[:max_results]
        
        # Convert to DataFrame
        products_df = pd.DataFrame(final_products)
        
        if not products_df.empty:
            print(f"\\n🏆 TOP {min(10, len(products_df))} PRODUCTS (ENHANCED TF-IDF RANKED & DISTRIBUTED):")
            # Drop the temporary category column before final display
            products_df = products_df.drop(columns=['product_category_for_distribution'], errors='ignore')
            for i, row in products_df.head(10).iterrows():
                semantic_indicator = "🧠🔥" if row.get('tfidf_enabled', False) else "🔤"
                print(f"   {i+1:2d}. {semantic_indicator} '{row['product'][:50]}...' → Score: {row['relevance']:.2f}, Price: IDR {row['price']:,}")
            
            print(f"\\n✅ RETURNING {len(products_df)} ENHANCED TF-IDF RANKED & DISTRIBUTED PRODUCTS")
        else:
            print(f"\\n❌ NO PRODUCTS REMAINING AFTER ENHANCED TF-IDF RANKING AND DISTRIBUTION")
        
        print("="*80)
        
        return products_df
        
    except Exception as e:
        logging.error(f"Error in enhanced TF-IDF product ranking: {str(e)}\\n{traceback.format_exc()}")
        print(f"❌ ERROR in enhanced TF-IDF product ranking: {str(e)}")
        return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
            
async def fetch_products_with_budget_awareness(db: AsyncSession, top_keywords: list, max_results=15, gender_category=None, budget_range=None, is_multi_item_request=False): # This signature is correct now
    """
    Enhanced budget-aware product fetching with TF-IDF ranking
    """
    logging.info(f"=== ENHANCED BUDGET-AWARE PRODUCT FETCH ===")
    logging.info(f"Budget range: {budget_range}")
    logging.info(f"Is Multi-Item Request: {is_multi_item_request}") # New debug line

    # Clean up empty/invalid budget ranges
    if budget_range == (None, None) or budget_range == [None, None]:
        budget_range = None

    try:
        if budget_range and any(budget_range or []):
            print(f"💰 Searching with budget constraint: {budget_range}")
            # Pass is_multi_item_request to fetch_products_from_db
            products_within_budget = await fetch_products_from_db(
                db, top_keywords, max_results, gender_category, budget_range,
                is_multi_item_request=is_multi_item_request # CORRECTED LINE
            )

            if products_within_budget is not None and not products_within_budget.empty:
                logging.info(f"Found {len(products_within_budget)} products within budget")
                return products_within_budget, "within_budget"
            else:
                logging.info("No products found within budget range")
                # Pass is_multi_item_request to fetch_products_from_db
                products_without_budget = await fetch_products_from_db(
                    db, top_keywords, max_results, gender_category, None, # Pass None for budget_range here
                    is_multi_item_request=is_multi_item_request # CORRECTED LINE
                )

                if products_without_budget is not None and not products_without_budget.empty:
                    logging.info(f"Found {len(products_without_budget)} products outside budget")
                    return products_without_budget, "no_products_in_budget"
                else:
                    logging.info("No products found even without budget constraint")
                    return pd.DataFrame(), "no_products_found"
        else:
            print(f"💰 No budget specified, searching normally")
            # Pass is_multi_item_request to fetch_products_from_db
            products = await fetch_products_from_db(
                db, top_keywords, max_results, gender_category, None,
                is_multi_item_request=is_multi_item_request # CORRECTED LINE
            )

            if products is not None and not products.empty:
                logging.info(f"Found {len(products)} products without budget constraint")
                return products, "no_budget_specified"
            else:
                logging.info("No products found")
                return pd.DataFrame(), "no_products_found"

    except Exception as e:
        logging.error(f"Error in enhanced fetch_products_with_budget_awareness: {str(e)}")
        return pd.DataFrame(), "error"
    
def generate_budget_message(budget_range, user_language, cheapest_price=None, most_expensive_price=None):
    """
    Generate appropriate budget constraint messages.
    """
    min_price, max_price = budget_range if budget_range else (None, None)
    
    # Create budget range text
    if min_price and max_price:
        budget_text = f"IDR {min_price:,} - IDR {max_price:,}"
    elif max_price:
        budget_text = f"under IDR {max_price:,}"
    elif min_price:
        budget_text = f"above IDR {min_price:,}"
    else:
        budget_text = "your specified budget"
    
    # Create price range text for alternatives
    price_info = ""
    if cheapest_price and most_expensive_price:
        price_info = f"The available products range from IDR {cheapest_price:,} to IDR {most_expensive_price:,}."
    elif cheapest_price:
        price_info = f"The cheapest available option starts from IDR {cheapest_price:,}."
    
    # English messages
    messages_en = {
        "no_products": f"I couldn't find any products matching your preferences within {budget_text}. {price_info}\n\nWould you like me to:\n1. Show you options outside your budget range?\n2. Help you adjust your search criteria?\n\nPlease let me know your preference!",
        
        "show_outside_budget": f"I found some great options for you, but they're outside your {budget_text} range. {price_info}\n\nWould you like to see these recommendations anyway? You can reply with:\n- 'Yes' to see all options\n- 'No' to adjust your search\n- 'Adjust budget' to modify your price range",
        
        "budget_adjustment": "I understand you'd prefer to stay within your budget. Would you like to:\n1. Search for different product types?\n2. Adjust your budget range?\n3. Look for similar but more affordable alternatives?\n\nJust let me know what you'd prefer!"
    }
    
    # Indonesian messages
    messages_id = {
        "no_products": f"Saya tidak dapat menemukan produk yang sesuai dengan preferensi Anda dalam rentang {budget_text}. {price_info}\n\nApakah Anda ingin saya:\n1. Tunjukkan pilihan di luar rentang anggaran Anda?\n2. Bantu menyesuaikan kriteria pencarian?\n\nSilakan beri tahu preferensi Anda!",
        
        "show_outside_budget": f"Saya menemukan beberapa pilihan bagus untuk Anda, tetapi berada di luar rentang {budget_text}. {price_info}\n\nApakah Anda ingin melihat rekomendasi ini?\n- 'Ya' untuk melihat semua pilihan\n- 'Tidak' untuk menyesuaikan pencarian\n- 'Sesuaikan anggaran' untuk mengubah rentang harga",
        
        "budget_adjustment": "Saya mengerti Anda lebih suka tetap dalam anggaran. Apakah Anda ingin:\n1. Mencari jenis produk yang berbeda?\n2. Menyesuaikan rentang anggaran?\n3. Mencari alternatif serupa yang lebih terjangkau?\n\nBeri tahu saya apa yang Anda inginkan!"
    }
    
    return messages_id if user_language != "en" else messages_en

def detect_budget_response(user_input):
    """
    Detect user response to budget constraint messages.
    Returns: "show_anyway" | "adjust_search" | "adjust_budget" | "unknown"
    """
    user_input_lower = user_input.lower().strip()
    
    # Positive responses - show products anyway
    show_anyway_patterns = [
        r'\b(yes|ya|iya|ok|okay|sure|tentu)\b',
        r'\b(show|tunjukkan|tampilkan)\s+(anyway|saja|aja)\b',
        r'\b(see|lihat)\s+(all|semua|them|mereka)\b',
        r'\b(ignore|abaikan)\s+(budget|anggaran)\b',
        r'\b(outside|di luar)\s+(budget|anggaran)\b',
    ]
    
    # Negative responses - adjust search
    adjust_search_patterns = [
        r'\b(no|tidak|nope|nah)\b',
        r'\b(adjust|sesuaikan|ubah)\s+(search|pencarian|kriteria)\b',
        r'\b(different|berbeda|lain)\s+(product|produk|type|jenis)\b',
        r'\b(stay|tetap)\s+(within|dalam)\s+(budget|anggaran)\b',
    ]
    
    # Budget adjustment responses
    adjust_budget_patterns = [
        r'\b(adjust|sesuaikan|ubah)\s+(budget|anggaran|price|harga)\b',
        r'\b(change|ganti)\s+(budget|anggaran)\b',
        r'\b(increase|naikkan|tingkatkan)\s+(budget|anggaran)\b',
        r'\b(decrease|turunkan|kurangi)\s+(budget|anggaran)\b',
        r'\b(new|baru)\s+(budget|anggaran)\b',
    ]
    
    for pattern in show_anyway_patterns:
        if re.search(pattern, user_input_lower):
            return "show_anyway"
    
    for pattern in adjust_budget_patterns:
        if re.search(pattern, user_input_lower):
            return "adjust_budget"
    
    for pattern in adjust_search_patterns:
        if re.search(pattern, user_input_lower):
            return "adjust_search"
    
    return "unknown"

def detect_budget_adjustment_request(user_input):
    """
    Extract new budget information from user input.
    Returns: (new_budget_range, confidence) or (None, 0)
    """
    # Use the existing extract_budget_from_text function
    new_budget = extract_budget_from_text(user_input)
    
    if new_budget:
        return new_budget, 1.0
    
    # Check for relative adjustments
    increase_patterns = [
        r'\b(increase|naikkan|tingkatkan)\s+(by|sebesar)?\s*(\d+)(?:rb|ribu|000|k)?\b',
        r'\b(add|tambah)\s+(\d+)(?:rb|ribu|000|k)?\b',
        r'\b(more|lebih)\s+(\d+)(?:rb|ribu|000|k)?\b',
    ]
    
    decrease_patterns = [
        r'\b(decrease|turunkan|kurangi)\s+(by|sebesar)?\s*(\d+)(?:rb|ribu|000|k)?\b',
        r'\b(reduce|kurangi)\s+(\d+)(?:rb|ribu|000|k)?\b',
        r'\b(less|kurang)\s+(\d+)(?:rb|ribu|000|k)?\b',
    ]
    
    user_input_lower = user_input.lower()
    
    for pattern in increase_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            amount_str = match.group(-1)  # Last captured group
            try:
                amount = int(amount_str)
                if 'rb' in user_input_lower or 'ribu' in user_input_lower:
                    amount *= 1000
                return ("increase", amount), 0.8
            except:
                pass
    
    for pattern in decrease_patterns:
        match = re.search(pattern, user_input_lower)
        if match:
            amount_str = match.group(-1)  # Last captured group
            try:
                amount = int(amount_str)
                if 'rb' in user_input_lower or 'ribu' in user_input_lower:
                    amount *= 1000
                return ("decrease", amount), 0.8
            except:
                pass
    
    return None, 0

def set_search_adjustment_mode(user_context, user_language, session_id):
    """
    Helper function to properly set up search adjustment mode
    KEEPS ORIGINAL FUNCTION NAME
    """
    print(f"🔧 SETTING SEARCH ADJUSTMENT MODE")
    
    # Create the "no products found" message with options
    no_products_response = "I'm sorry, I couldn't find any products matching your preferences. Would you like to try:"
    no_products_response += "\n\n1. **Different style preferences?**"
    no_products_response += "\n   (Change colors, fit, sleeve length, etc.)"
    no_products_response += "\n\n2. **Different clothing types?**" 
    no_products_response += "\n   (Switch from shirts to dresses, pants to skirts, etc.)"
    no_products_response += "\n\n3. **More general search terms?**"
    no_products_response += "\n   (Broader search with less specific requirements)"
    no_products_response += "\n\n**Just type the number (1, 2, or 3) or tell me directly what you'd like to see!**"
    
    if user_language != "en":
        no_products_response = translate_text(no_products_response, user_language, session_id)

    # Set the flags
    user_context["awaiting_search_adjustment"] = True
    user_context["awaiting_confirmation"] = False
    
    return no_products_response

def detect_search_adjustment_response(user_input):
    """
    ENHANCED: Detect user response to "no products found" options.
    KEEPS ORIGINAL FUNCTION NAME
    Returns: "different_style" | "different_type" | "general_search" | "new_clothing_request" | "unknown"
    """
    user_input_lower = user_input.lower().strip()
    
    print(f"🔍 SEARCH ADJUSTMENT DEBUG: Input = '{user_input_lower}'")
    
    # Check for new clothing request first (highest priority)
    if detect_new_clothing_request(user_input):
        print(f"   ✅ Detected new clothing request")
        return "new_clothing_request"
    
    # ENHANCED: More flexible option detection
    # Option 1: Different style preferences
    style_patterns = [
        r'^1$',  # Just "1"
        r'^\b(one|satu|pertama)\b$',  # Just the number words
        r'\b(1|one|satu|pertama)\b',  # Number in context
        r'\b(style|gaya)\b.*\b(different|berbeda|change|ubah)\b',
        r'\b(different|berbeda)\b.*\b(style|gaya)\b',
        r'\b(preference|preferensi)\b',
        r'\b(color|warna|fit|potongan|length|panjang)\b.*\b(different|berbeda)\b',
        r'\b(more\s+casual|more\s+formal|lebih\s+kasual|lebih\s+formal)\b'
    ]
    
    # Option 2: Different clothing types  
    type_patterns = [
        r'^2$',  # Just "2"
        r'^\b(two|dua|kedua)\b$',  # Just the number words
        r'\b(2|two|dua|kedua)\b',  # Number in context
        r'\b(clothing|pakaian|clothes)\b.*\b(different|berbeda|type|jenis)\b',
        r'\b(different|berbeda)\b.*\b(clothing|pakaian|type|jenis|item|barang)\b',
        r'\b(type|jenis)\b.*\b(different|berbeda)\b',
        r'\b(shirt|kemeja|dress|gaun|pants|celana|skirt|rok)\b.*\b(instead|instead of|ganti)\b',
        r'\b(show|tunjukkan|carikan)\b.*\b(different|lain|other)\b.*\b(item|barang|clothing|pakaian)\b'
    ]
    
    # Option 3: More general search
    general_patterns = [
        r'^3$',  # Just "3"
        r'^\b(three|tiga|ketiga)\b$',  # Just the number words
        r'\b(3|three|tiga|ketiga)\b',  # Number in context
        r'\b(general|umum|broader|lebih\s+luas)\b',
        r'\b(more\s+general|lebih\s+umum)\b',
        r'\b(search|pencarian|cari)\b.*\b(general|umum|broader|luas)\b',
        r'\b(expand|perluas|widen|lebarkan)\b.*\b(search|pencarian)\b',
        r'\b(less\s+specific|kurang\s+spesifik)\b'
    ]
    
    # PRIORITY CHECK: Check patterns in order
    print(f"   🔍 Checking style patterns...")
    for i, pattern in enumerate(style_patterns):
        if re.search(pattern, user_input_lower):
            print(f"   ✅ Style pattern {i+1} matched: '{pattern}'")
            return "different_style"
    
    print(f"   🔍 Checking type patterns...")
    for i, pattern in enumerate(type_patterns):
        if re.search(pattern, user_input_lower):
            print(f"   ✅ Type pattern {i+1} matched: '{pattern}'")
            return "different_type"
    
    print(f"   🔍 Checking general patterns...")
    for i, pattern in enumerate(general_patterns):
        if re.search(pattern, user_input_lower):
            print(f"   ✅ General pattern {i+1} matched: '{pattern}'")
            return "general_search"
    
    # FALLBACK: If input is very short and contains keywords, try to guess intent
    if len(user_input_lower.split()) <= 2:
        if any(word in user_input_lower for word in ['style', 'gaya', 'color', 'warna', 'fit']):
            print(f"   ✅ Short style-related input detected")
            return "different_style"
        elif any(word in user_input_lower for word in ['type', 'jenis', 'clothing', 'pakaian', 'item']):
            print(f"   ✅ Short type-related input detected")
            return "different_type"
        elif any(word in user_input_lower for word in ['general', 'umum', 'broader', 'expand']):
            print(f"   ✅ Short general-related input detected")
            return "general_search"
    
    print(f"   ❌ No patterns matched")
    return "unknown"

def detect_new_clothing_request(user_input):
    """
    ENHANCED: Detect if user is making a new clothing request
    KEEPS ORIGINAL FUNCTION NAME
    """
    user_input_lower = user_input.lower().strip()
    
    print(f"   🔍 Checking new clothing request patterns...")
    
    # BASIC FILTERS: Don't trigger on simple responses
    simple_responses = ["yes", "ya", "iya", "ok", "okay", "sure", "tentu", "no", "tidak", "nope", "ga", "engga", "1", "2", "3", "one", "two", "three", "satu", "dua", "tiga"]
    if user_input_lower in simple_responses:
        print(f"   ❌ Simple response detected: '{user_input_lower}'")
        return False
    
    # Don't trigger on very short inputs unless they're clearly clothing items
    if len(user_input_lower.split()) <= 2:
        # Check if the short input contains clothing items
        clothing_keywords = ['kemeja', 'shirt', 'dress', 'gaun', 'celana', 'pants', 'rok', 'skirt', 'jaket', 'jacket', 'kaos', 'sweater']
        if not any(clothing in user_input_lower for clothing in clothing_keywords):
            print(f"   ❌ Short input without clothing keywords: '{user_input_lower}'")
            return False
    
    # CLOTHING REQUEST PATTERNS
    clothing_request_patterns = [
        # Direct requests with action words
        r'\b(carikan|tunjukkan|show|find|cari|search)\s+\w+',
        r'\b(ada|any|have)\s+\w+\s+(yang|that)',
        r'\b(mau|want|ingin|need|butuh)\s+\w+',
        r'\b(looking\s+for|mencari)\s+\w+',
        
        # Question patterns
        r'\b(apa|what)\s+(ada|about|yang)\s+\w+',
        r'\bada\s+(tidak|ga|gak)\s+\w+',
        r'\bhow\s+about\s+\w+',
        r'\bbagaimana\s+(dengan|kalau)\s+\w+',
        
        # Specific clothing items mentioned
        r'\b(kemeja|shirt|blouse|dress|gaun|celana|pants|rok|skirt|jaket|jacket|kaos|t-shirt|sweater|cardigan|hoodie|blazer)\b',
        
        # Color + clothing combinations
        r'\b(hitam|putih|merah|biru|hijau|kuning|black|white|red|blue|green|yellow)\s+(kemeja|shirt|dress|pants|celana|gaun)\b',
        r'\b(kemeja|shirt|dress|pants|celana|gaun)\s+(hitam|putih|merah|biru|hijau|kuning|black|white|red|blue|green|yellow)\b',
        
        # Style + clothing combinations
        r'\b(oversized|slim|loose|tight|casual|formal)\s+(kemeja|shirt|dress|pants|celana|gaun)\b',
        r'\b(lengan\s+panjang|lengan\s+pendek|long\s+sleeve|short\s+sleeve)\b',
        
        # New search indicators
        r'\b(now|sekarang)\s+(show|tunjukkan|carikan)',
        r'\b(instead|sebagai\s+gantinya)\s+\w+',
        r'\b(what\s+about|bagaimana\s+dengan|gimana\s+kalau)\s+\w+',
    ]
    
    for i, pattern in enumerate(clothing_request_patterns):
        if re.search(pattern, user_input_lower):
            print(f"   ✅ New clothing request pattern {i+1} matched: '{pattern}'")
            return True
    
    # Check if input contains clothing items from shared categories
    try:
        clothing_categories = get_shared_clothing_categories()
        for category, terms in clothing_categories.items():
            if any(term in user_input_lower for term in terms):
                print(f"   ✅ Clothing item detected: {terms} in category {category}")
                return True
    except:
        # Fallback if get_shared_clothing_categories is not available
        basic_clothing_terms = ['kemeja', 'shirt', 'dress', 'gaun', 'celana', 'pants', 'rok', 'skirt', 'jaket', 'jacket', 'kaos', 't-shirt', 'sweater', 'cardigan', 'hoodie', 'blazer', 'atasan', 'bawahan']
        if any(term in user_input_lower for term in basic_clothing_terms):
            print(f"   ✅ Basic clothing term detected")
            return True
    
    print(f"   ❌ No new clothing request patterns matched")
    return False

def get_paginated_products(all_products_df, page=0, products_per_page=5):
    """
    Helper function to get a specific page of products from the full results.
    This is a SYNCHRONOUS function - no async/await needed.
    
    Args:
        all_products_df: DataFrame with all products
        page: Page number (0-based)
        products_per_page: Number of products per page
    
    Returns:
        tuple: (paginated_products_df, has_more_pages)
    """
    if all_products_df.empty:
        logging.info("No products available for pagination")
        return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"]), False
    
    start_idx = page * products_per_page
    end_idx = start_idx + products_per_page
    
    # Get the slice for this page
    paginated_products = all_products_df.iloc[start_idx:end_idx]
    
    # Check if there are more pages
    has_more = end_idx < len(all_products_df)
    
    logging.info(f"📄 Pagination: Page {page}, showing products {start_idx+1}-{min(end_idx, len(all_products_df))} of {len(all_products_df)}")
    logging.info(f"📊 Has more pages: {has_more}")
    
    return paginated_products, has_more
    
def detect_more_products_request(user_input: str) -> bool:
    """
    Detect if user is asking for more products - more precise to avoid conflicts
    """
    more_patterns = [
        # English patterns
        r'\b(more|other|another|additional|different|else)\s+(product|item|option|choice|recommendation)',
        r'\b(show|give|find|get)\s+(me\s+)?(more|other|another|additional)',
        r'\b(what|anything)\s+else',
        r'\b(more|other)\s+(suggestion|option|choice)',
        r'\belse\s+(do\s+you\s+have|available)',
        
        # Indonesian patterns  
        r'\b(lain|lainnya|yang lain|lagi)\b',
        r'\b(tunjukkan|carikan|kasih|coba)\s+(yang\s+)?(lain|lainnya)',
        r'\b(ada\s+)?(yang\s+)?(lain|lainnya)',
        r'\b(produk|barang|item)\s+(lain|lainnya)',
        r'\b(pilihan|opsi)\s+(lain|lainnya)',
        r'\b(apa\s+lagi|apalagi)',
        r'\b(selain\s+itu|besides)',
        r'\b(lebih\s+banyak|more)',
        r'\bapa\s+lagi\b',
        r'\belse\s+(do\s+you\s+have|available)',
        r'\blainnya\b.*\b(produk|barang|pilihan)',
    ]
    
    user_input_lower = user_input.lower().strip()
    
    # CRITICAL: Don't trigger on simple confirmations
    simple_responses = ["yes", "ya", "iya", "ok", "okay", "sure", "tentu", "no", "tidak", "nope", "ga", "engga"]
    if user_input_lower in simple_responses:
        return False
    
    # Don't trigger on very short responses (likely confirmations)
    if len(user_input_lower.split()) <= 2 and user_input_lower not in ["apa lagi", "yang lain", "show more"]:
        return False
    
    for pattern in more_patterns:
        if re.search(pattern, user_input_lower):
            logging.info(f"Detected specific 'more products' request: {user_input}")
            return True
    
    return False

def detect_new_product_search(user_input, current_keywords):
    """
    Detect if user is asking for a completely different product category.
    Returns: (is_new_search, confidence_level)
    """
    user_input_lower = user_input.lower()
    
    # Define product categories with more granular classification
    product_categories = {
        'shirts': ['kemeja', 'shirt', 'blouse', 'blus'],
        'tshirts': ['t-shirt', 'tshirt', 'kaos', 'tank top'],
        'tops_other': ['atasan', 'top', 'sweater', 'cardigan', 'hoodie'],
        'outerwear': ['jacket', 'jaket', 'coat', 'mantel', 'blazer'],
        'bottoms_pants': ['celana', 'pants', 'jeans', 'trousers'],
        'bottoms_skirts': ['rok', 'skirt'],
        'bottoms_general': ['bawahan', 'bottom', 'shorts'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'footwear': ['sepatu', 'shoes', 'sandal', 'boot', 'sneaker'],
        'accessories': ['tas', 'bag', 'topi', 'hat', 'belt', 'ikat pinggang', 'scarf', 'syal'],
        'undergarments': ['underwear', 'bra', 'dalam', 'celana dalam']
    }
    
    # Group related categories (less aggressive reset between related items)
    category_groups = {
        'tops': ['shirts', 'tshirts', 'tops_other'],
        'bottoms': ['bottoms_pants', 'bottoms_skirts', 'bottoms_general'],
        'full_garments': ['dresses', 'outerwear'],
        'accessories': ['footwear', 'accessories', 'undergarments']
    }
    
    # Check what category user is asking for now
    current_request_categories = []
    for category, terms in product_categories.items():
        if any(term in user_input_lower for term in terms):
            current_request_categories.append(category)
    
    # Check what categories were in previous search
    previous_categories = []
    if current_keywords:
        for category, terms in product_categories.items():
            for keyword, _ in current_keywords[:8]:  # Check more keywords for better detection
                if any(term in keyword.lower() for term in terms):
                    previous_categories.append(category)
                    break
    
    # Calculate change intensity
    change_intensity = 0
    change_type = "none"
    
    if current_request_categories and previous_categories:
        # Check if categories are in different groups
        current_groups = []
        previous_groups = []
        
        for group, categories in category_groups.items():
            if any(cat in current_request_categories for cat in categories):
                current_groups.append(group)
            if any(cat in previous_categories for cat in categories):
                previous_groups.append(group)
        
        if current_groups and previous_groups:
            if not any(group in previous_groups for group in current_groups):
                change_intensity = 3  # Major category change (tops -> bottoms)
                change_type = "major_category"
            elif not any(cat in previous_categories for cat in current_request_categories):
                change_intensity = 2  # Related category change (shirts -> t-shirts)
                change_type = "related_category"
            else:
                change_intensity = 1  # Same or very similar category
                change_type = "minor"
    
    # Check for explicit new search indicators with different weights
    explicit_patterns = {
        'strong': [
            r'\b(now|sekarang)\s+(show|tunjukkan|carikan|cari)\b',
            r'\b(instead|sebagai gantinya|ganti)\b',
            r'\b(different|berbeda|lain)\s+(type|jenis|product|item|barang)\b',
            r'\b(what about|bagaimana dengan|gimana)\s+.*\b(bottom|bawahan|pants|celana|top|atasan|dress|gaun)\b',
            r'\b(suitable|cocok|sesuai)\s+(bottom|bawahan|pants|celana|top|atasan)\b',
        ],
        'medium': [
            r'\b(also|juga)\s+(show|tunjukkan)\b',
            r'\b(recommend|rekomendasikan)\s+(some|beberapa)?\s*(different|lain|other)\b',
            r'\b(any|ada)\s+(suggestion|saran|recommendations|rekomendasi)\s+(for|untuk)\b',
        ],
        'weak': [
            r'\b(more|lebih)\s+(options|pilihan|choices)\b',
            r'\b(other|lain|lainnya)\s+(styles|gaya|options)\b',
        ]
    }
    
    pattern_intensity = 0
    for intensity, patterns in explicit_patterns.items():
        for pattern in patterns:
            if re.search(pattern, user_input_lower):
                if intensity == 'strong':
                    pattern_intensity = max(pattern_intensity, 3)
                    print(f"🔄 STRONG new search pattern: {pattern}")
                elif intensity == 'medium':
                    pattern_intensity = max(pattern_intensity, 2)
                    print(f"🔄 MEDIUM new search pattern: {pattern}")
                elif intensity == 'weak':
                    pattern_intensity = max(pattern_intensity, 1)
                    print(f"🔄 WEAK new search pattern: {pattern}")
                break
    
    # Combine intensities
    final_intensity = max(change_intensity, pattern_intensity)
    
    print(f"🔍 Change Analysis:")
    print(f"   Current categories: {current_request_categories}")
    print(f"   Previous categories: {previous_categories}")
    print(f"   Change type: {change_type}")
    print(f"   Category intensity: {change_intensity}")
    print(f"   Pattern intensity: {pattern_intensity}")
    print(f"   Final intensity: {final_intensity}")
    
    return final_intensity >= 2, final_intensity


def smart_preserve_keywords(user_context, change_intensity):
    """
    Flexible keyword preservation based on change intensity and keyword characteristics.
    """
    if "accumulated_keywords" not in user_context:
        return {}
    
    preserved_keywords = {}
    current_keywords = user_context["accumulated_keywords"]
    
    # Define keyword categories with preservation priority
    keyword_categories = {
        'user_identity': {
            'terms': ['perempuan', 'wanita', 'female', 'woman', 'pria', 'laki-laki', 'male', 'man'],
            'preserve_threshold': 1,  # Always preserve
            'weight_reduction': 0.1   # Minimal reduction
        },
        'physical_attributes': {
            'terms': ['tinggi', 'height', 'berat', 'weight', 'kulit', 'skin', 'slim', 'kurus', 
                     'gemuk', 'besar', 'kecil', 'tall', 'short', 'pendek'],
            'preserve_threshold': 1,  # Always preserve
            'weight_reduction': 0.2
        },
        'style_preferences': {
            'terms': ['formal', 'resmi', 'casual', 'santai', 'elegant', 'elegan', 'vintage', 'modern',
                     'minimalist', 'minimalis', 'bohemian', 'boho', 'streetwear'],
            'preserve_threshold': 3,  # Preserve unless major change
            'weight_reduction': 0.5
        },
        'fit_preferences': {
            'terms': ['oversized', 'longgar', 'loose', 'ketat', 'tight', 'fit'],
            'preserve_threshold': 2,  # Preserve unless major change
            'weight_reduction': 0.4
        },
        'color_preferences': {
            'terms': ['hitam', 'black', 'putih', 'white', 'merah', 'red', 'biru', 'blue', 
                     'hijau', 'green', 'kuning', 'yellow', 'coklat', 'brown', 'abu-abu', 'grey'],
            'preserve_threshold': 3,  # Only preserve for minor changes
            'weight_reduction': 0.5
        },
        'specific_features': {
            'terms': ['lengan panjang', 'lengan pendek', 'long sleeve', 'short sleeve', 'panjang', 'pendek',
                     'kerah', 'collar', 'kantong', 'pocket', 'kancing', 'button'],
            'preserve_threshold': 3,  # Only preserve for minor changes
            'weight_reduction': 0.6
        },
        'product_specific': {
            'terms': ['kemeja', 'shirt', 't-shirt', 'kaos', 'dress', 'gaun', 'celana', 'pants', 
                     'rok', 'skirt', 'jaket', 'jacket', 'atasan', 'bawahan'],
            'preserve_threshold': 4,  # Never preserve
            'weight_reduction': 1.0
        }
    }
    
    print(f"🔄 PRESERVING KEYWORDS (change intensity: {change_intensity})")
    
    for keyword, data in current_keywords.items():
        should_preserve = False
        weight_reduction = 0.6 # Default reduction
        category_matched = "uncategorized"
        
        # Check which category this keyword belongs to
        for category, config in keyword_categories.items():
            if any(term in keyword.lower() for term in config['terms']):
                category_matched = category
                if change_intensity < config['preserve_threshold']:
                    should_preserve = True
                    weight_reduction = config['weight_reduction']
                break
        
        # Additional preservation logic for high-weight keywords
        if not should_preserve and data["weight"] > 200:
            # Very high weight keywords might be important user preferences
            should_preserve = True
            weight_reduction = 0.5
            category_matched = "high_weight"
            print(f"   → Preserving high-weight keyword: '{keyword}'")
        
        # Additional preservation for recent keywords
        if not should_preserve and data.get("count", 1) >= 3:
            # Frequently mentioned keywords might be important
            should_preserve = True
            weight_reduction = 0.6
            category_matched = "frequent"
            print(f"   → Preserving frequent keyword: '{keyword}'")
        
        if should_preserve:
            new_weight = data["weight"] * (1 - weight_reduction)
            preserved_keywords[keyword] = {
                "weight": new_weight,
                "count": data["count"],
                "first_seen": data["first_seen"],
                "source": data["source"],
                "preserved_reason": category_matched
            }
            print(f"   ✅ Preserved '{keyword}' ({category_matched}): {data['weight']:.1f} → {new_weight:.1f}")
        else:
            print(f"   ❌ Removed '{keyword}' ({category_matched})")
    
    return preserved_keywords

def smart_keyword_context_update(user_input, user_context, new_keywords, is_user_input=False):
    """
    IMPROVED: Context update with nuanced conflict resolution
    """
    print(f"\n📝 IMPROVED KEYWORD CONTEXT UPDATE")
    print("="*60)
    
    # Debug: Show state before update
    if "accumulated_keywords" in user_context:
        acc_kw = user_context["accumulated_keywords"]
        print(f"📚 BEFORE - {len(acc_kw)} accumulated keywords")
        if acc_kw:
            sorted_kw = sorted(acc_kw.items(), key=lambda x: x[1].get("weight", 0), reverse=True)
            print(f"   🏆 Top 5 BEFORE:")
            for i, (kw, data) in enumerate(sorted_kw[:5]):
                source_icon = "👤" if data.get("source") == "user_input" else "🤖"
                print(f"      {i+1}. {source_icon} '{kw}' → {data.get('weight', 0):.1f}")
    
    # NEW: Retrieve multi-item flag directly from extract_ranked_keywords
    is_multi_item = getattr(extract_ranked_keywords, 'is_multi_item_request', False)
    user_context["is_multi_item_request_flag"] = is_multi_item # Store in context for other parts to read
    

    # STEP 1: Apply improved category change detection FIRST
    # Pass the is_multi_item flag here
    major_change_detected = detect_fashion_category_change(user_input, user_context, is_multi_item)
    
    # STEP 2: Apply enhanced keyword decay (unchanged)
    apply_keyword_decay(user_context)

    # --- CRITICAL FIX: Ensure convert_to_linked_system receives is_multi_item_request ---
    # This block ensures that if accumulated_keywords are being converted to linked system,
    # the multi-item flag is passed. This is usually on session start or a major context switch.
    if 'linked_keyword_system' not in user_context and "accumulated_keywords" in user_context:
        # Convert existing accumulated_keywords to linked system using the correct flag
        convert_to_linked_system(user_context, is_multi_item_request=is_multi_item) # <--- FIX APPLIED HERE
        print(f"   🔄 Initialized LinkedKeywordSystem with multi-item flag: {is_multi_item}")
    # --- END CRITICAL FIX ---
    
    # STEP 3: Apply improved scoring to new keywords
    enhanced_new_keywords = []
    
    for keyword, weight in new_keywords: # new_keywords now could be (combined_string, score)
        # If it's a combined string (from multi-item request), categorize based on its components
        if is_multi_item and ' ' in keyword: # Simple heuristic: if it contains a space and is multi-item
            # Try to get categories from individual words in the combined string
            sub_keywords = keyword.split()
            main_category = None
            for sub_kw in sub_keywords:
                cat = fashion_categories.get_clothing_category(sub_kw)
                if cat:
                    main_category = cat
                    break
            if main_category:
                category_multiplier = get_keyword_category_multiplier(main_category) # Use main clothing category for multiplier
            else:
                category_multiplier = get_keyword_category_multiplier(keyword) # Fallback to original
        else:
            category_multiplier = get_keyword_category_multiplier(keyword)
        
        # Improved boost logic - different boosts for major vs minor changes
        if is_user_input:
            if major_change_detected:
                nuclear_boost = 10.0  # 10x boost for major clothing type changes
                enhanced_weight = weight * category_multiplier * nuclear_boost
                print(f"   ☢️  MAJOR CHANGE BOOST: '{keyword}' {weight:.1f} × {category_multiplier} × {nuclear_boost} = {enhanced_weight:.1f}")
            elif is_multi_item: # Use the detected multi-item flag
                multi_boost = 7.0  # 7x boost for multi-item requests
                enhanced_weight = weight * category_multiplier * multi_boost
                print(f"   🤝 MULTI BOOST: '{keyword}' {weight:.1f} × {category_multiplier} × {multi_boost} = {enhanced_weight:.1f}")
            else:
                user_boost = 5.0  # 5x boost for regular user input
                enhanced_weight = weight * category_multiplier * user_boost
                print(f"   👤 USER BOOST: '{keyword}' {weight:.1f} × {category_multiplier} × {user_boost} = {enhanced_weight:.1f}")
        else:
            enhanced_weight = weight * category_multiplier
            
        enhanced_new_keywords.append((keyword, enhanced_weight))
    
    # STEP 4: Add new keywords with enhanced weights
    # Pass the is_multi_item flag here as well
    update_accumulated_keywords(enhanced_new_keywords, user_context, is_user_input=is_user_input, is_multi_item_request=is_multi_item)
    
    # STEP 5: Apply appropriate cleanup based on change type
    if major_change_detected:
        print(f"🧹 MAJOR CHANGE CLEANUP")
        persistence_config = {
            'clothing_items': {'decay_rate': 0.5, 'max_age_minutes': 15},  # Very fast decay for clothing items
            'style_attributes': {'decay_rate': 0.6, 'max_age_minutes': 10},
            'colors': {'decay_rate': 0.7, 'max_age_minutes': 10},
            'gender_terms': {'decay_rate': 0.05, 'max_age_minutes': 240},  # Keep gender
            'occasions': {'decay_rate': 0.8, 'max_age_minutes': 5},       # Very fast decay
            'default': {'decay_rate': 0.7, 'max_age_minutes': 10}
        }
    else:
        print(f"🧹 MINOR/NO CHANGE CLEANUP")
        persistence_config = {
            'clothing_items': {'decay_rate': 0.1, 'max_age_minutes': 120},  # Keep clothing items longer
            'style_attributes': {'decay_rate': 0.3, 'max_age_minutes': 45}, # Gentle decay for style changes
            'colors': {'decay_rate': 0.4, 'max_age_minutes': 30},          # Gentle decay for color changes
            'gender_terms': {'decay_rate': 0.05, 'max_age_minutes': 240},  # Keep gender
            'occasions': {'decay_rate': 0.6, 'max_age_minutes': 20},       # Moderate decay for occasions
            'default': {'decay_rate': 0.4, 'max_age_minutes': 25}
        }
    
    # Pass is_multi_item to category_cleanup for intelligent pruning
    # Pass the original new_keywords (which might be grouped) to cleanup for context
    category_cleanup(user_context, persistence_config, is_multi_item, new_keywords)
    
    # STEP 6: Apply rebalancing and normalization only for major changes
    if major_change_detected:
        rebalance_keywords_after_conflict(user_context, user_input)
        normalize_weights_nuclear(user_context, user_input)
        remove_irrelevant_keywords_nuclear(user_context, user_input)
    else:
        # For minor changes, just do gentle normalization
        gentle_normalization(user_context, user_input)
    
    # Debug: Show final state
    final_count = len(user_context.get("accumulated_keywords", {}))
    print(f"\n📊 AFTER IMPROVED UPDATE:")
    print(f"   📈 Total keywords: {final_count}")
    print(f"   🤝 Multi-item request: {is_multi_item}")
    print(f"   ⚔️  Major change detected: {major_change_detected}")
    
    if final_count > 0:
        sorted_final = sorted(user_context["accumulated_keywords"].items(), 
                             key=lambda x: x[1].get("weight", 0), reverse=True)
        print(f"   🏆 Top 5 AFTER:")
        for i, (kw, data) in enumerate(sorted_final[:5]):
            source_icon = "👤" if data.get("source") == "user_input" else "🤖"
            print(f"      {i+1}. {source_icon} '{kw}' → {data.get('weight', 0):.1f}")
    
    print("="*60)

def gentle_normalization(user_context, user_input):
    """
    GENTLE: Normalization for minor changes (colors, styles, materials)
    """
    print(f"🌸 GENTLE NORMALIZATION for minor changes")
    
    if "accumulated_keywords" not in user_context:
        return
    
    user_input_lower = user_input.lower()
    
    # Identify current vs old keywords
    current_keywords = []
    old_keywords = []
    
    for keyword, data in user_context["accumulated_keywords"].items():
        if keyword in user_input_lower:
            current_keywords.append(keyword)
        else:
            old_keywords.append(keyword)
    
    # Get the max weight of current keywords
    current_max = 0
    for keyword in current_keywords:
        if keyword in user_context["accumulated_keywords"]:
            current_max = max(current_max, user_context["accumulated_keywords"][keyword]["weight"])
    
    print(f"   🎯 Current keywords max weight: {current_max:.1f}")
    print(f"   📚 Current keywords: {current_keywords}")
    print(f"   🗂️  Old keywords: {len(old_keywords)}")
    
    # GENTLE: Cap old keywords to be maximum 50% of current keywords (much more gentle than nuclear 0.1%)
    weight_cap = current_max * 0.5  # 50% of current max
    
    capped_count = 0
    for keyword in old_keywords:
        if keyword in user_context["accumulated_keywords"]:
            current_weight = user_context["accumulated_keywords"][keyword]["weight"]
            if current_weight > weight_cap:
                user_context["accumulated_keywords"][keyword]["weight"] = weight_cap
                capped_count += 1
                print(f"   🌸 GENTLE CAP: '{keyword}' {current_weight:.1f} → {weight_cap:.1f}")
    
    print(f"   📊 Gently capped {capped_count} old keywords to {weight_cap:.1f}")

def remove_irrelevant_keywords_nuclear(user_context, user_input):
    """
    NUCLEAR: Remove keywords that are far too weak to be relevant
    """
    print(f"☢️  NUCLEAR IRRELEVANT KEYWORD REMOVAL")
    
    if "accumulated_keywords" not in user_context:
        return
    
    user_input_lower = user_input.lower()
    
    # Get current keyword max weight
    current_max = 0
    for keyword, data in user_context["accumulated_keywords"].items():
        if keyword in user_input_lower:
            current_max = max(current_max, data["weight"])
    
    # Remove keywords that are less than 0.01% of current max
    removal_threshold = current_max * 0.0001  # 0.01% threshold
    
    keywords_to_remove = []
    for keyword, data in user_context["accumulated_keywords"].items():
        if keyword not in user_input_lower and data["weight"] < removal_threshold:
            keywords_to_remove.append(keyword)
    
    # Remove irrelevant keywords
    removed_count = 0
    for keyword in keywords_to_remove:
        old_weight = user_context["accumulated_keywords"][keyword]["weight"]
        del user_context["accumulated_keywords"][keyword]
        removed_count += 1
        print(f"   ☢️  REMOVED: '{keyword}' (weight: {old_weight:.3f} < threshold: {removal_threshold:.3f})")
    
    print(f"   📊 Nuclear removed {removed_count} irrelevant keywords")

def normalize_weights_ultra_aggressive(user_context, user_input):
    """
    ULTRA AGGRESSIVE: Cap old keywords and boost new ones
    """
    print(f"🧯 ULTRA AGGRESSIVE WEIGHT NORMALIZATION")
    
    if "accumulated_keywords" not in user_context:
        return
    
    user_input_lower = user_input.lower()
    
    # Identify current vs old keywords
    current_keywords = []
    old_keywords = []
    
    for keyword, data in user_context["accumulated_keywords"].items():
        if keyword in user_input_lower:
            current_keywords.append(keyword)
        else:
            old_keywords.append(keyword)
    
    # Get the max weight of current keywords
    current_max = 0
    for keyword in current_keywords:
        if keyword in user_context["accumulated_keywords"]:
            current_max = max(current_max, user_context["accumulated_keywords"][keyword]["weight"])
    
    print(f"   🎯 Current keywords max weight: {current_max:.1f}")
    print(f"   📚 Current keywords: {current_keywords}")
    print(f"   🗂️  Old keywords: {len(old_keywords)}")
    
    # Cap old keywords to be maximum 25% of current keywords
    weight_cap = current_max * 0.25  # Old keywords can't exceed 25% of new max
    
    capped_count = 0
    for keyword in old_keywords:
        if keyword in user_context["accumulated_keywords"]:
            current_weight = user_context["accumulated_keywords"][keyword]["weight"]
            if current_weight > weight_cap:
                user_context["accumulated_keywords"][keyword]["weight"] = weight_cap
                capped_count += 1
                print(f"   📏 CAPPED: '{keyword}' {current_weight:.1f} → {weight_cap:.1f}")
    
    print(f"   📊 Capped {capped_count} old keywords to {weight_cap:.1f}")

def rebalance_keywords_after_conflict(user_context, user_input):
    """
    ULTRA AGGRESSIVE: Rebalance to ensure new keywords dominate
    """
    print(f"⚖️  ULTRA AGGRESSIVE REBALANCING")
    
    if "accumulated_keywords" not in user_context:
        return
    
    user_input_lower = user_input.lower()
    
    # Identify keywords from current user input
    current_input_keywords = []
    for keyword in user_context["accumulated_keywords"].keys():
        if keyword in user_input_lower:
            current_input_keywords.append(keyword)
    
    # Get current max weight
    current_max_weight = max(
        (data["weight"] for data in user_context["accumulated_keywords"].values()),
        default=0
    )
    
    print(f"   📊 Current max weight in system: {current_max_weight:.1f}")
    print(f"   🎯 Current input keywords: {current_input_keywords}")
    
    # ULTRA AGGRESSIVE BOOST: Ensure new keywords are 2x the max weight
    ultra_target_multiplier = 2.0  # Make new keywords 2x stronger than anything else
    
    for keyword in current_input_keywords:
        if keyword in user_context["accumulated_keywords"]:
            current_weight = user_context["accumulated_keywords"][keyword]["weight"]
            ultra_target_weight = current_max_weight * ultra_target_multiplier
            
            # Only boost if current weight is less than ultra target
            if current_weight < ultra_target_weight:
                user_context["accumulated_keywords"][keyword]["weight"] = ultra_target_weight
                print(f"   🚀 ULTRA BOOSTED: '{keyword}' {current_weight:.1f} → {ultra_target_weight:.1f}")
            else:
                print(f"   ✅ Already strong: '{keyword}' {current_weight:.1f}")

def extract_specific_clothing_request(user_input, ai_response):
    """
    Extract what specific clothing item the user is asking for vs. what they want to pair it with.
    Returns: (wanted_items, context_items)
    """
    user_input_lower = user_input.lower()
    ai_response_lower = ai_response.lower() if ai_response else ""
    
    print(f"   🔍 ANALYZING REQUEST: '{user_input}'")
    
    # Primary clothing items user is asking for
    clothing_requests = {
        'kemeja': ['kemeja', 'shirt', 'blouse', 'blus'],
        'celana': ['celana', 'pants', 'trousers', 'jeans'],
        'dress': ['dress', 'gaun', 'terusan'],
        'rok': ['rok', 'skirt'],
        'jaket': ['jaket', 'jacket', 'blazer', 'coat'],
        'kaos': ['kaos', 't-shirt', 'tshirt', 'tank top'],
        'sweater': ['sweater', 'cardigan', 'hoodie'],
        'atasan': ['atasan', 'top', 'blouse'],
        'bawahan': ['bawahan', 'bottom']
    }
    
    # Context items (what they want to pair WITH)
    pairing_indicators = [
        'dipadukan dengan', 'dipasangkan dengan', 'cocok dengan', 'pair with', 'match with',
        'untuk', 'dengan', 'sama', 'bareng', 'yang cocok dengan'
    ]
    
    wanted_items = []
    context_items = []
    
    # NEW: Handle "X yang cocok untuk Y" pattern specifically
    compatibility_patterns = [
        r'\b(\w+)\s+yang\s+cocok\s+untuk\s+(\w+)',  # "kemeja yang cocok untuk celana"
        r'\bcarikan\s+(\w+)\s+yang\s+cocok\s+untuk\s+(\w+)',  # "carikan kemeja yang cocok untuk celana"
        r'\btunjukkan\s+(\w+)\s+yang\s+cocok\s+untuk\s+(\w+)',  # "tunjukkan kemeja yang cocok untuk celana"
        r'\bada\s+(\w+)\s+yang\s+cocok\s+untuk\s+(\w+)',  # "ada kemeja yang cocok untuk celana"
        r'\b(\w+)\s+yang\s+sesuai\s+untuk\s+(\w+)',  # "kemeja yang sesuai untuk celana"
        r'\b(\w+)\s+yang\s+pas\s+untuk\s+(\w+)',  # "kemeja yang pas untuk celana"
    ]
    
    compatibility_found = False
    
    for pattern in compatibility_patterns:
        compatibility_match = re.search(pattern, user_input_lower)
        
        if compatibility_match:
            primary_item = compatibility_match.group(1).strip()
            context_item = compatibility_match.group(2).strip()
            
            print(f"   🎯 COMPATIBILITY PATTERN DETECTED: '{primary_item}' yang cocok untuk '{context_item}'")
            
            # Find which categories these belong to
            primary_category = None
            context_category = None
            
            for category, terms in clothing_requests.items():
                # Check if primary_item matches any terms
                if any(term in primary_item or primary_item in term for term in terms):
                    primary_category = category
                    print(f"      🎯 Primary item '{primary_item}' mapped to category: {primary_category}")
                
                # Check if context_item matches any terms  
                if any(term in context_item or context_item in term for term in terms):
                    context_category = category
                    print(f"      📝 Context item '{context_item}' mapped to category: {context_category}")
            
            if primary_category:
                wanted_items.append(primary_category)
                print(f"   🎯 WANTED: {primary_category} (from compatibility pattern)")
            
            if context_category:
                context_items.append(context_category)
                print(f"   📝 CONTEXT: {context_category} (from compatibility pattern)")
            
            compatibility_found = True
            break  # Found a pattern, no need to check others
    
    # If we found the compatibility pattern, prioritize it and return early
    if compatibility_found:
        print(f"   📊 COMPATIBILITY ANALYSIS - WANTED: {wanted_items}, CONTEXT: {context_items}")
        return wanted_items, context_items
    
    print(f"   ❌ No compatibility pattern found, trying direct request patterns...")
    
    # EXISTING LOGIC: Find what user specifically asked for (only if no compatibility pattern found)
    for category, terms in clothing_requests.items():
        for term in terms:
            # Check if it's a direct request (question words + clothing item)
            direct_request_patterns = [
                rf'\b(?:apa|what|mana|which|ada|show|tunjukkan|carikan)\s+.*?{term}\b',
                rf'\b{term}\s+(?:apa|what|yang|mana)\b',
                rf'^{term}\b',  # At start of sentence
            ]
            
            for pattern in direct_request_patterns:
                if re.search(pattern, user_input_lower):
                    if category not in wanted_items:
                        wanted_items.append(category)
                        print(f"   🎯 WANTED: {category} (from '{term}' - direct request)")
            
            # Check if it's mentioned as context/pairing
            for indicator in pairing_indicators:
                if indicator in user_input_lower:
                    # Look for clothing terms after pairing indicators
                    pairing_pattern = rf'{indicator}\s+.*?{term}\b'
                    if re.search(pairing_pattern, user_input_lower):
                        if category not in context_items:
                            context_items.append(category)
                            print(f"   📝 CONTEXT: {category} (for pairing with '{term}')")
    
    # If AI response mentions specific recommendations, add those as wanted
    if ai_response:
        try:
            bold_items = extract_bold_headings_from_ai_response(ai_response)
            for item in bold_items:
                item_lower = item.lower()
                for category, terms in clothing_requests.items():
                    if any(term in item_lower for term in terms):
                        if category not in wanted_items:
                            wanted_items.append(category)
                            print(f"   🤖 AI RECOMMENDED: {category} (from bold: '{item}')")
        except:
            pass
    
    print(f"   📊 FINAL ANALYSIS - WANTED: {wanted_items}, CONTEXT: {context_items}")
    return wanted_items, context_items

def get_shared_clothing_categories():
    """Shared clothing categories used by both fetch_products_from_db and calculate_relevance_score"""
    return {
        'tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'sweater', 'hoodie', 'cardigan', 'blazer', 'tank', 'top'],
        'bottoms_pants': ['celana', 'pants', 'jeans', 'trousers', 'leggings'],
        'bottoms_skirts': ['rok', 'skirt'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'coat', 'mantel'],
        'shorts': ['shorts', 'celana pendek']
    }

def is_clothing_item_with_priority(keyword):
    """Check if keyword is clothing and return equal priority for fairness"""
    kw_lower = keyword.lower()
    clothing_categories = get_shared_clothing_categories()
    
    # Check if it's ANY clothing item
    for category, terms in clothing_categories.items():
        if any(term in kw_lower for term in terms):
            return True, 5.0  # ALL clothing items get EQUAL priority
    
    # General clothing terms
    general_clothing = ['baju', 'pakaian', 'outfit', 'clothing', 'wear']
    if any(term in kw_lower for term in general_clothing):
        return True, 4.5  # Slightly lower for general terms
    
    return False, 0  # Not clothing

def detect_multi_item_request(user_input):
    """
    Enhanced detection for multi-item requests like "carikan kemeja dan celana" or "short pants and maxi skirt".
    KEEPS ORIGINAL FUNCTION NAME
    """
    user_input_lower = user_input.lower().strip()
    
    print(f"\n🤝 MULTI-ITEM DETECTION DEBUG START: '{user_input}'")
    
    simple_responses = {
        "yes", "ya", "iya", "ok", "okay", "sure", "tentu", "no", "tidak", "nope", "ga", "engga", "1", "2", "3", "one", "two", "three", "satu", "dua", "tiga"
    }
    if user_input_lower in simple_responses:
        print(f"   ❌ Simple response detected: '{user_input_lower}' - Returning False")
        return False
    
    # Early exit for very short inputs that are unlikely multi-item unless they contain specific terms
    if len(user_input_lower.split()) <= 2:
        clothing_keywords_for_short_input = ['kemeja', 'shirt', 'dress', 'gaun', 'celana', 'pants', 'rok', 'skirt', 'jaket', 'jacket', 'kaos', 'sweater', 'blouse', 'top', 'bottom', 'outerwear', 'shorts']
        if not any(kw in user_input_lower for kw in clothing_keywords_for_short_input):
            print(f"   ❌ Very short input without direct clothing keywords: '{user_input_lower}' - Returning False")
            return False
        # If it has a clothing keyword, it might still be a multi-item if that keyword implies multiple, e.g., "shirts and blouses"
        # but for now, let's keep it simple and assume short inputs are single-item unless connectors are very clear.
    
    # Step 1: Check for explicit multi-item connectors
    multi_connectors = [r'\b(dan|and|atau|or|with|sama|plus|\+|&)\b']
    has_explicit_connector = any(re.search(pattern, user_input_lower) for pattern in multi_connectors)
    print(f"   🔍 Has explicit connector: {has_explicit_connector}")

    # Step 2: Identify all distinct clothing item categories mentioned in the input
    clothing_categories_map = fashion_categories.CLOTHING_CATEGORIES
    
    # Store tuples of (clothing_category, term_that_matched)
    # This helps in identifying multiple categories from the input.
    found_clothing_terms_and_categories = [] 

    for category, terms in clothing_categories_map.items():
        for term in terms:
            if term in user_input_lower:
                found_clothing_terms_and_categories.append((category, term))
                # Do NOT break here, we want to find *all* categories
    
    distinct_clothing_categories_in_input = {cat for cat, _ in found_clothing_terms_and_categories}
    
    print(f"   📦 Distinct clothing categories in input: {distinct_clothing_categories_in_input} (Count: {len(distinct_clothing_categories_in_input)})")
    
    # Step 3: Check for attributes that differentiate clothing items (e.g., 'short' for pants, 'maxi' for skirt)
    # This implies multiple items if multiple categories are present.
    length_terms = fashion_categories.LENGTH_TERMS
    fit_terms = fashion_categories.FIT_TERMS
    sleeve_terms = fashion_categories.SLEEVE_TERMS

    has_differentiating_attributes = False
    for term_list in [length_terms, fit_terms, sleeve_terms]:
        for attr_term in term_list:
            if attr_term in user_input_lower:
                has_differentiating_attributes = True
                print(f"   🔍 Differentiating attribute found: '{attr_term}'")
                break
        if has_differentiating_attributes:
            break
    print(f"   Summary: Has differentiating attributes: {has_differentiating_attributes}")

    # Step 4: Decision Logic for Multi-Item Request
    
    # Strongest indicators:
    if has_explicit_connector and len(distinct_clothing_categories_in_input) >= 2:
        print(f"   ✅ DECISION: Explicit connector AND 2+ distinct clothing categories - Returning True")
        return True
    
    # Very strong indicator: multiple distinct clothing categories with differentiating attributes
    if len(distinct_clothing_categories_in_input) >= 2 and has_differentiating_attributes:
        print(f"   ✅ DECISION: 2+ distinct clothing categories AND differentiating attributes - Returning True")
        return True

    # If only one primary category, but multiple specific items mentioned (e.g., "kemeja and blouse" where both map to 'tops')
    # Or if a connector is used with 2 or more related clothing terms from the same main category.
    # We can infer multi-item if the number of *distinct clothing terms* (not categories) is >= 2 AND connector is present.
    distinct_clothing_terms_matched = {term for cat, term in found_clothing_terms_and_categories}
    if has_explicit_connector and len(distinct_clothing_terms_matched) >= 2:
        print(f"   ✅ DECISION: Explicit connector AND 2+ distinct clothing terms (same category possible) - Returning True")
        return True

    # Consider 3 or more distinct clothing categories as multi-item, even without explicit connector or attributes
    if len(distinct_clothing_categories_in_input) >= 3:
        print(f"   ✅ DECISION: 3+ distinct clothing categories - Returning True")
        return True
    
    # If a connector is present, but only one clothing category is found, it could still be multi-item
    # For example, "red shirt and blue shirt". This relies on `extract_ranked_keywords` to capture "red shirt" and "blue shirt" as combined keywords.
    if has_explicit_connector:
        print(f"   ✅ DECISION: Explicit connector detected (even if few categories) - Returning True (permissive)")
        return True 

    print(f"   ❌ DECISION: No strong multi-item patterns found - Returning False")
    return False
  
def are_compatible_categories(cat1, cat2):
    """Check if two clothing categories are compatible/related"""
    if not cat1 or not cat2:
        return False
    
    # Same category is always compatible
    if cat1 == cat2:
        return True
    
    # Define compatible groups
    bottom_categories = {'bottoms_pants', 'bottoms_skirts'}
    clothing_separates = {'tops', 'bottoms_pants', 'bottoms_skirts', 'outerwear'}
    
    # Both are bottoms
    if cat1 in bottom_categories and cat2 in bottom_categories:
        return True
    
    # Both are clothing separates (not dresses)
    if cat1 in clothing_separates and cat2 in clothing_separates:
        return True
    
    return False

def get_latest_exclusions():
    """Get exclusions from the last keyword extraction for product filtering."""
    return getattr(extract_ranked_keywords, 'last_exclusions', [])
    
def post_update_enhanced_cleanup(user_context):
    """
    Enhanced cleanup that prioritizes fashion categories over occasions
    """
    if "accumulated_keywords" not in user_context:
        return
    
    # Separate keywords by category
    fashion_keywords = {}
    occasion_keywords = {}
    other_keywords = {}
    
    fashion_terms = [
        'kemeja', 'shirt', 'dress', 'celana', 'pants', 'rok', 'jaket',
        'kaos', 'atasan', 'blouse', 'sweater', 'jeans', 'dasi', 'kerudung',
        'topi', 'sepatu', 'flat', 'sendal', 'jam', 'jam tangan', 'sabuk', 'tas',
        'hijab', 'jilbab', 'kerudung', 'tudung', 'gesper', 'belt'
    ]
    
    occasion_terms = [
        'office', 'kantor', 'party', 'pesta', 'wedding', 'beach', 'sport'
    ]
    
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        
        if any(term in keyword_lower for term in fashion_terms):
            fashion_keywords[keyword] = data
        elif any(term in keyword_lower for term in occasion_terms):
            occasion_keywords[keyword] = data
        else:
            other_keywords[keyword] = data
    
    # Keep more fashion keywords, fewer occasion keywords
    MAX_FASHION = 25
    MAX_OCCASION = 8  # Limited occasions
    MAX_OTHER = 12
    
    # Sort each category by weight
    top_fashion = dict(sorted(fashion_keywords.items(), 
                             key=lambda x: x[1]["weight"], reverse=True)[:MAX_FASHION])
    
    top_occasion = dict(sorted(occasion_keywords.items(), 
                              key=lambda x: x[1]["weight"], reverse=True)[:MAX_OCCASION])
    
    top_other = dict(sorted(other_keywords.items(), 
                           key=lambda x: x[1]["weight"], reverse=True)[:MAX_OTHER])
    
    # Combine and update
    cleaned_keywords = {**top_fashion, **top_occasion, **top_other}
    
    removed_count = len(user_context["accumulated_keywords"]) - len(cleaned_keywords)
    user_context["accumulated_keywords"] = cleaned_keywords
    
    if removed_count > 0:
        print(f"🧹 Enhanced cleanup: Fashion({len(top_fashion)}), Occasion({len(top_occasion)}), Other({len(top_other)})")
        print(f"   Removed {removed_count} lower-priority keywords")

def is_physical_description(text):
    """
    Check if text contains physical description that should not be treated as clothing requirements.
    """
    text_lower = text.lower()
    
    physical_indicators = [
        # Skin color contexts
        r'\b(?:kulit|skin)\s+\w+',
        r'\bberkulit\s+\w+',
        r'\bwarna\s+kulit',
        r'\bskin\s+tone',
        
        # Physical measurements
        r'\b\d+\s*(?:cm|kg|tahun|years?)',
        r'\b(?:tinggi|height|berat|weight|umur|age)',
        
        # Nationality/origin
        r'\b(?:dari|from)\s+(?:indonesia|malaysia|singapore|thailand)',
        
        # Gender
        r'\b(?:cowo|cowok|cewe|cewek|pria|wanita|laki-laki|perempuan)',
        
        # Activities/lifestyle
        r'\b(?:suka|like)\s+(?:olahraga|sport|gym|fitness)',
    ]
    
    for pattern in physical_indicators:
        if re.search(pattern, text_lower):
            return True
    
    return False

def detect_rapid_preference_changes(user_input, user_context):
    """
    Detect if user is rapidly changing preferences and adjust context accordingly.
    """
    rapid_change_patterns = [
        r'\b(actually|sebenarnya|wait|tunggu)\b',
        r'\b(no|tidak|bukan)\s+(that|itu)\b',
        r'\b(change|ganti|ubah)\s+(my mind|pikiran)\b',
        r'\b(different|lain)\s+(color|warna|style|gaya)\b',
        r'\b(prefer|lebih suka|mau)\s+(something|sesuatu)\s+(else|lain)\b',
    ]
    
    user_input_lower = user_input.lower()
    
    for pattern in rapid_change_patterns:
        if re.search(pattern, user_input_lower):
            print(f"🔄 RAPID CHANGE detected: {pattern}")
            
            # Reduce weights of recent keywords more aggressively
            if "accumulated_keywords" in user_context:
                for keyword, data in user_context["accumulated_keywords"].items():
                    if data.get("source") == "user_input":  # Recent user inputs
                        data["weight"] *= 0.3  # Reduce significantly
                        print(f"   → Reduced weight of recent keyword: '{keyword}'")
            
            return True
    
    return False
       
@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def render_markdown(text: str) -> str:
    extensions = [
        'tables',  # For table support
        'nl2br',   # Convert newlines to <br>
        'fenced_code',  # For code blocks
        'smarty'   # For smart quotes
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
        # Add debugging to verify the file exists and has content
        if not os.path.exists(file_location):
            logging.error(f"File not found at location: {file_location}")
            return None
            
        file_size = os.path.getsize(file_location)
        if file_size == 0:
            logging.error(f"File exists but is empty (0 bytes): {file_location}")
            return None
            
        logging.info(f"Uploading file to Cloudinary: {file_location}, Size: {file_size} bytes")
        
        # Optimisation parameters
        transformation = {
            'quality': 'auto',
            'fetch_format': 'auto',
        }

        response = cloudinary.uploader.upload(
            file_location, 
            folder="uploads/",
            transformation=transformation
        )
        logging.info(f"Cloudinary upload successful: {response['url']}")
        return response['url']
    except Exception as e:
        logging.error(f"Cloudinary upload error: {e}")
        return None  # Return None instead of raising to let the retry logic handle it

@app.post("/upload/")
async def upload(user_input: str = Form(None), file: UploadFile = None):
    if not file and not user_input:
        return JSONResponse(content={"success": False, "error": "No input or file received"})

    try:
        if file:
            # Save file with unique name in the upload directory
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Invalid file type.")

            # Read file content once
            file_content = await file.read()
            file_size = len(file_content)
            
            # Log file details for debugging
            logging.info(f"Received file: {file.filename}, Size: {file_size} bytes")
            
            if file_size == 0:
                logging.error("Uploaded file has 0 bytes")
                return JSONResponse(content={"success": False, "error": "Uploaded file is empty."})
                
            if file_size > 5 * 1024 * 1024:  # 5MB
                return JSONResponse(content={"success": False, "error": "File size exceeds 5MB limit."})
            
            # Ensure upload directory exists
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
                logging.info(f"Created upload directory: {UPLOAD_DIR}")
            
            # Generate unique file name
            unique_id = uuid.uuid4()
            sanitized_filename = slugify(file.filename.rsplit(".", 1)[0], lowercase=False)
            unique_filename = f"{unique_id}_{sanitized_filename}.{file_extension}"
            file_location = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Write the content we already read to the file
            with open(file_location, "wb") as file_object:
                file_object.write(file_content)
            
            # Verify file was written correctly
            if os.path.exists(file_location) and os.path.getsize(file_location) > 0:
                logging.info(f"File successfully saved: {file_location}, Size: {os.path.getsize(file_location)} bytes")
            else:
                logging.error(f"File not saved correctly at {file_location}")
                return JSONResponse(content={"success": False, "error": "Failed to save file."})

            # Upload to Cloudinary with retries
            image_url = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logging.info(f"Cloudinary upload attempt {attempt+1}/{max_retries}")
                    image_url = upload_to_cloudinary(file_location)
                    if image_url:
                        logging.info(f"Successfully uploaded to Cloudinary: {image_url}")
                        break
                    else:
                        logging.warning(f"Cloudinary upload returned None on attempt {attempt+1}")
                except Exception as e:
                    logging.error(f"Cloudinary upload attempt {attempt+1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to upload to Cloudinary after {max_retries} attempts: {str(e)}")
                    else:
                        logging.info(f"Sleeping for 1 second before retry {attempt+2}")
                        time.sleep(1)

            if image_url:
                # File uploaded successfully
                return JSONResponse(content={"success": True, "file_url": image_url})
            else:
                return JSONResponse(content={"success": False, "error": "Failed to upload image to Cloudinary."})

        elif user_input:
            return JSONResponse(content={"success": True})

        # If neither input nor file is present
        return JSONResponse(content={"success": False, "error": "No input or file received"})

    except Exception as e:
        logging.error(f"Error in upload endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An error occurred during file upload.")
    
chat_responses = []

async def is_small_talk(input_text):
    greetings = ["hello", "hi", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening", "selamat pagi", "pagi", "selamat siang", "siang", "malam", "selamat malam"]
    return input_text.lower() in greetings or re.match(r"^\s*(hi|hello|hey)\s*$", input_text, re.IGNORECASE)

@app.post("/chat/save")
async def save_message(message: ChatMessage, db: AsyncSession = Depends(get_db)):
    try:
        logging.info(f"Saving message: {message.session_id}, {message.message_type}, {message.content}")
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
        logging.error(f"Error saving message: {str(e)}\nTraceback: ")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        query = select(ChatHistoryDB).where(
            ChatHistoryDB.session_id == session_id
        ).order_by(ChatHistoryDB.timestamp)

        result = await db.execute(query)
        messages =  result.scalars().all()

        return ChatHistoryResponse(
            messages=[
                ChatMessage (
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

async_session = async_sessionmaker(get_db, expire_on_commit=False)

class SessionLanguageManager:
    def __init__(self):
        self.session_languages = {}  # Store language by session_id
        
    def detect_or_retrieve_language(self, text, session_id):
        # If this session already has a detected language, use it
        if session_id in self.session_languages:
            return self.session_languages[session_id]
        
        # Otherwise, detect the language for the first time
        try:
            if text and text.strip():
                lang = detect(text)
                # Store it for future messages in this session
                self.session_languages[session_id] = lang
                return lang
            return "unknown"
        except Exception as e:
            print(f"Language detection error: {e}")
            return "unknown"
            
    def reset_session(self, session_id):
        # Call this when a conversation ends
        if session_id in self.session_languages:
            del self.session_languages[session_id]

# Function to detect the language of the text
def detect_language(text):
    try:
        if not text or not text.strip():
            raise ValueError("Input text is empty or invalid.")
        return detect(text)  # Detect the language using langdetect
    except Exception as e:
        print(f"Language detection error: {e}")
        return "unknown"

# Initialize the session manager
session_manager = SessionLanguageManager()

# Function to translate text using Deep Translator
def translate_text(text, target_language, session_id=None):
    try:
        if session_id and session_id in session_manager.session_languages:
            source_language = session_manager.session_languages[session_id]
            print(f"Using stored language for translation: {source_language}")
        else:
            source_language = detect_language(text)
            print(f"Detected language for translation: {source_language}")

            if session_id:
                session_manager.session_languages[session_id] = source_language
        
        print(f"Detected source language: {source_language} ({'from session' if session_id in session_manager.session_languages else 'newly detected'})")

        # If the source and target languages are the same, no translation needed
        if source_language == target_language:
            return text

        # Use GoogleTranslator from Deep Translator to perform the translation
        translated_text = MyMemoryTranslator(source=source_language, target=target_language).translate(text)
        return translated_text

    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # Return the original text as a fallback

def extract_intent(user_input, target_language="en", session_id=None):
    """Extracts keywords and entities, translating input if necessary."""
    doc = nlp(user_input)

    try:
        if session_id:
            detected_language = session_manager.get_language(session_id, user_input)
        else:
            detected_language = detected_language(user_input)

    except:
        print(f"Language detection error: {str(e)}")
        detected_language = "en"  # Default to English if detection fails
    
    # Translate the input if it's not in the target language
    if detected_language != target_language:
        try:
            user_input = translate_text(user_input, target_language)
            doc = nlp(user_input)
        except Exception as e:
            print(f"Translation error: {str(e)}")

    # Extract keywords and entities
    keywords = [chunk.text for chunk in doc.noun_chunks]
    entities = [ent.text for ent in doc.ents]

    recommendation_phrases = [
        "recommend another product", "show me more option", "suggest something else",
        "what else do you have", "can you recommend another one", "bisakah kamu menunjukkan produk yang lain",
        "tunjukkan produk lainnya", "saran produk lainnya", "apa lagi yang kamu punya", "rekomendasikan sesuatu yang lain"
    ]

    if any(phrase in user_input.lower() for phrase in recommendation_phrases) or \
       any(keyword in recommendation_phrases for keyword in keywords):
        intent = "Product Recommendation"
    else:
        intent = "General"

    return {
        "language": detected_language,
        "translated_text": user_input,
        "keywords": keywords,
        "entities": entities, 
        "intent": intent
    }

@app.websocket("/ws")
async def chat(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    try:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        await websocket.send_text(f"{session_id}|Selamat Datang! Bagaimana saya bisa membantu Anda hari ini?\n\nWelcome! How can I help you today?")

        # Initial system prompt for fashion consultation - only style advice, no products yet
        message_objects = [{
            "role": "system",
            "content": (
                "You are a fashion consultant. Your task is to provide detailed fashion recommendations "
                "for users based on their appearance and style preferences. Respond in a friendly, natural tone "
                "and avoid using structured JSON or code format. Instead, communicate recommendations in conversational sentences.\n\n"
                
                "IMPORTANT: Always ask for their gender, weight and height, skin tone, their ethnical background and use this information as a base for your recommendations.\n\n"
                
                "IMPORTANT: When asking for style preferences, ALWAYS format them as bullet points with examples:\n\n"
                
                "**Style Preferences:**\n"
                "• **Sleeve length preference:** Please choose from sleeveless (tank tops), short sleeve (t-shirts), 3/4 sleeve (three-quarter), or long sleeve (full coverage)\n"
                "• **Clothing length preference:** \n"
                "  - For tops: crop top (above waist), regular length (at waist), tunic (below waist), or longline (hip length)\n"
                "  - For bottoms: shorts (above knee), capri (mid-calf), regular (ankle length), or long/full length (floor length)\n"
                "• **Fit preference:** Choose from oversized (loose and baggy), regular fit (standard comfort), fitted (close to body), slim fit (tailored and snug), or loose fit (relaxed but not oversized)\n"
                "• **Daily activity level and lifestyle:** Please specify sedentary/office work (mostly sitting), moderately active (walking, light exercise), very active/athletic (sports, gym, running), or mixed activities (combination of different activity levels)\n\n"
                
                "CRITICAL FORMATTING REQUIREMENTS:\n"
                "1. Always format each clothing recommendation as a bold heading with size recommendation in parentheses\n"
                "2. Use this exact format: **[Clothing Item Name] (Disarankan Ukuran [Size])**\n"
                "3. Follow each heading with a detailed paragraph explanation\n"
                "4. Always include a horizontal line (---) between different clothing recommendations\n"
                "5. Each recommendation should be in a separate paragraph block\n\n"
                
                "EXAMPLE FORMAT:\n"
                "**Kemeja Flanel Oversized (Disarankan Ukuran L)**\n"
                "Kemeja flanel oversized memberikan tampilan santai namun tetap stylish. Cocok dipadukan dengan celana jeans atau celana panjang. Dengan aktivitas yang cukup aktif, ukuran L akan memberikan kenyamanan dan gaya yang Anda inginkan.\n\n"
                "---\n\n"
                "**Kemeja Polo Slim Fit (Disarankan Ukuran M)**\n"
                "Kemeja polo slim fit cocok untuk aktivitas sehari-hari yang lebih formal. Potongan yang pas akan memberikan tampilan yang rapi dan profesional.\n\n"
                
                "IMPORTANT: When giving recommendations, mention specific clothing items and how they would suit the user's attributes, "
                "such as gender, height, weight, skin tone, and consider their daily activities when suggesting appropriate fits and styles.\n\n"
                
                "IMPORTANT: Based on the user's measurements, body type, and daily activities, provide a size recommendation (XS, S, M, L, XL, etc.) for each suggested item, explaining why that size would work best for their lifestyle and comfort needs.\n\n"
                
                "Give at least 3 items recommendation.\n\n"
                
                "IMPORTANT: If the user asks for a specific type of clothing (such as 'kemeja', 'shirt', 'dress', 'pants', etc.), "
                "make sure your recommendations focus directly on that specific clothing type.\n\n"
                
                "MANDATORY FORMATTING RULES:\n"
                "- Every clothing item must be bold with size recommendation: **[Item Name] (Disarankan Ukuran [Size])**\n"
                "- Each description must be a separate paragraph explaining why it suits the user\n"
                "- Use horizontal lines (---) to separate different recommendations\n"
                "- When asking for style preferences, ALWAYS use bullet points with clear examples\n"
                "- End with the confirmation question about showing specific products\n\n"
                
                "Do not mention any specific brand of clothing.\n"
                "After each style recommendation, always ask a yes or no question: 'Would you like to see product recommendations based on these style suggestions?' or 'Do these styles align with what you're looking for? I can show you specific products if you're interested.'\n"
                "DO NOT provide product recommendations in your initial response - only suggest styles and wait for user confirmation."
            )
        }]
        
        # Store the most recent AI response for use in confirmation handling
        last_ai_response = ""
        
        # Store user context for better recommendations
        user_context = {
            "current_image_url": None,
            "current_text_input": None,
            "pending_image_analysis": False,
            "has_shared_image": False,
            "has_shared_preferences": False,
            "last_query_type": None,
            "awaiting_confirmation": False,  # Flag to track if we're waiting for user confirmation
            "accumulated_keywords": {},  # Dictionary to store and accumulate keywords with their weights
            "preferences": {},
            "known_attributes": {},
            "user_gender": {  # New field to persistently store user gender information
                "category": None,
                "term": None,
                "confidence": 0,
                "last_updated": None
            },
            "is_multi_item_request_flag": False, # <--- ADD THIS LINE HERE
            "product_cache": {
                "all_result": pd.DataFrame(),
                "current_page": 0,
                "product_per_page": 5,
                "last_search_params": {},
                "has_more": False
            }
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
                    user_language = session_manager.detect_or_retrieve_language(session_id, user_input)
                    logging.info(f"User language '{user_language}' for session {session_id}")
                except Exception as e:
                    logging.error(f"Language detection error: {str(e)}")
                    user_language = "en"

                if user_context.get("awaiting_search_adjustment", False):
                    print(f"\n🔧 SEARCH ADJUSTMENT HANDLER")
                    print(f"   📝 User input: '{user_input}'")
                    print(f"   🏷️ Language: {user_language}")
                    
                    response_type = detect_search_adjustment_response(user_input)
                    
                    print(f"🔍 Search adjustment response: {response_type}")
                    
                    # NEW: Check if user is making a new clothing request instead of choosing options
                    if response_type == "new_clothing_request" or detect_new_clothing_request(user_input):
                        print(f"🆕 NEW CLOTHING REQUEST detected while in search adjustment mode")
                        print(f"   📝 New request: '{user_input}'")
                        
                        # Clear search adjustment flags and process as new request
                        user_context["awaiting_search_adjustment"] = False
                        user_context["awaiting_confirmation"] = False
                        
                        # Clear product cache for fresh search
                        user_context["product_cache"] = {
                            "all_results": pd.DataFrame(),
                            "current_page": 0,
                            "products_per_page": 5,
                            "has_more": False
                        }
                        
                        # Reset accumulated keywords for completely new search
                        if "accumulated_keywords" in user_context:
                            # Keep only essential user attributes (gender, basic preferences)
                            essential_keywords = {}
                            for keyword, data in user_context["accumulated_keywords"].items():
                                if any(essential in keyword.lower() for essential in ['perempuan', 'wanita', 'female', 'woman', 'pria', 'laki-laki', 'male', 'man']):
                                    essential_keywords[keyword] = {
                                        "weight": data["weight"] * 0.2,  # Reduce weight significantly
                                        "count": 1,
                                        "first_seen": data.get("first_seen", datetime.now().isoformat()),
                                        "last_seen": datetime.now().isoformat(),
                                        "source": "preserved_essential"
                                    }
                            user_context["accumulated_keywords"] = essential_keywords
                            print(f"   🧹 Reset accumulated keywords, kept {len(essential_keywords)} essential items")
                        
                        # Continue to normal text processing (don't use 'continue' here)
                        print(f"   ▶️ Processing as new clothing request...")
                        # Fall through to normal text processing below
                        
                    elif response_type == "different_style":
                        print(f"   🎨 User wants different style preferences")
                        
                        # Handle style clarification
                        style_response = "What style would you prefer instead? For example:"
                        style_response += "\n• More casual or formal?"
                        style_response += "\n• Different colors (black, white, blue, etc.)?"
                        style_response += "\n• Different fit (oversized, slim, regular, loose)?"
                        style_response += "\n• Different sleeve length (short, long, sleeveless)?"
                        style_response += "\n\nJust tell me what you'd like to change!"
                        
                        if user_language != "en":
                            style_response = translate_text(style_response, user_language, session_id)
                        
                        # Clear flags and send response
                        user_context["awaiting_search_adjustment"] = False
                        user_context["awaiting_confirmation"] = False
                        
                        # Save response to database
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=style_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        await websocket.send_text(f"{session_id}|{style_response}")
                        continue
                        
                    elif response_type == "different_type":
                        print(f"   👕 User wants different clothing types")
                        
                        # Handle type clarification
                        type_response = "What type of clothing would you like to see instead? For example:"
                        type_response += "\n• Dresses or skirts?"
                        type_response += "\n• Pants or jeans?" 
                        type_response += "\n• T-shirts or sweaters?"
                        type_response += "\n• Jackets or cardigans?"
                        type_response += "\n• Formal or casual wear?"
                        type_response += "\n\nJust let me know what type you're looking for!"
                        
                        if user_language != "en":
                            type_response = translate_text(type_response, user_language, session_id)
                        
                        # Clear flags and reset cache
                        user_context["awaiting_search_adjustment"] = False
                        user_context["awaiting_confirmation"] = False
                        user_context["product_cache"] = {
                            "all_results": pd.DataFrame(),
                            "current_page": 0,
                            "products_per_page": 5,
                            "has_more": False
                        }
                        
                        # Save response to database
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=type_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        await websocket.send_text(f"{session_id}|{type_response}")
                        continue
                        
                    elif response_type == "general_search":
                        print(f"   🔍 User wants more general search")
                        
                        # Handle general search
                        general_response = "Let me search with more general terms..."
                        
                        if user_language != "en":
                            general_response = translate_text(general_response, user_language, session_id)
                        
                        await websocket.send_text(f"{session_id}|{general_response}")
                        
                        # Simplify accumulated keywords to more general terms
                        if "accumulated_keywords" in user_context:
                            basic_keywords = {}
                            for keyword, data in user_context["accumulated_keywords"].items():
                                # Keep only basic clothing categories and high-level attributes
                                if any(basic in keyword.lower() for basic in ['kemeja', 'shirt', 'blouse', 'dress', 'celana', 'pants', 'casual', 'formal', 'elegant']):
                                    # Reduce specificity by lowering weights of very specific terms
                                    if len(keyword.split()) > 1:  # Multi-word terms are more specific
                                        data["weight"] *= 0.5
                                    basic_keywords[keyword] = data
                            
                            user_context["accumulated_keywords"] = basic_keywords
                            print(f"   🔄 Simplified to {len(basic_keywords)} general keywords")
                        
                        # Clear flags and trigger new search
                        user_context["awaiting_search_adjustment"] = False
                        user_context["awaiting_confirmation"] = True  # Trigger product search
                        continue
                        
                    else:
                        print(f"   ❓ Unknown response: '{user_input}' -> {response_type}")
                        
                        # Unknown response - ask for clarification
                        clarification = "I didn't understand your choice. Please specify:"
                        clarification += "\n1. Different style preferences (colors, fit, etc.)"
                        clarification += "\n2. Different clothing types (shirts, dresses, pants, etc.)"
                        clarification += "\n3. More general search terms"
                        clarification += "\n\nOr you can directly tell me what you're looking for!"
                        clarification += "\n\nFor example: 'show me casual dresses' or 'I want oversized sweaters'"
                        
                        if user_language != "en":
                            clarification = translate_text(clarification, user_language, session_id)
                        
                        # Save response to database
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=clarification
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        await websocket.send_text(f"{session_id}|{clarification}")
                        continue
                # Check if we're awaiting confirmation for product recommendations
                if user_context["awaiting_confirmation"]:
                    # Process confirmation response
                    is_positive = user_input.strip().lower() in ["yes", "ya", "iya", "sure", "tentu", "ok", "okay"]
                    is_negative = user_input.strip().lower() in ["no", "tidak", "nope", "nah", "tidak usah"]
                    is_more_request = detect_more_products_request(user_input)

                    print(f"\n📋 CONFIRMATION CHECK START")
                    print("="*50)
                    if "accumulated_keywords" in user_context:
                        acc_kw = user_context["accumulated_keywords"]
                        print(f"📚 Accumulated Keywords: {len(acc_kw)}")
                        if acc_kw:
                            sorted_kw = sorted(acc_kw.items(), key=lambda x: x[1].get("weight", 0), reverse=True)
                            print(f"   🏆 Top 15:")
                            for i, (kw, data) in enumerate(sorted_kw[:15]):
                                source_icon = "👤" if data.get("source") == "user_input" else "🤖"
                                print(f"      {i+1}. {source_icon} '{kw}' → {data.get('weight', 0):.1f}")
                    
                    if "user_gender" in user_context and user_context["user_gender"].get("category"):
                        gender_info = user_context["user_gender"]
                        print(f"👤 Gender: {gender_info['category']} (confidence: {gender_info.get('confidence', 0):.1f})")
                    
                    if "budget_range" in user_context and user_context["budget_range"]:
                        budget = user_context["budget_range"]
                        print(f"💰 Budget: {budget}")
                    print("="*50)

                    logging.info(f"Confirmation state - Input: '{user_input}' | Positive: {is_positive}, Negative: {is_negative}, More: {is_more_request}")
                    
                    if is_positive:
                        if "budget_range" in user_context:
                            current_budget = user_context["budget_range"]
                            print(f"🔍 TEMP DEBUG: Current budget in context: {current_budget}")
                            if current_budget == (None, None) or not any(current_budget or []):
                                print(f"🧹 TEMP FIX: Clearing phantom budget")
                                user_context["budget_range"] = None
                        # User confirmed, show product recommendations
                        try:
                            # Get accumulated keywords from context (without adding new ones)
                            accumulated_keywords = []
                            if "accumulated_keywords" in user_context:
                                # Convert from dictionary to list of tuples format for extract_ranked_keywords
                                accumulated_keywords = [(k, v["weight"]) for k, v in user_context["accumulated_keywords"].items()]
                                
                            # Rank the accumulated keywords based on importance
                            # Use last AI response and user input for better context
                            last_user_input = user_context.get("current_text_input", "")
                            ranked_keywords = get_keywords_for_product_search(user_context)
                            
                            if user_language != "en":
                                # Extract just the keywords
                                keywords_only = [kw for kw, _ in ranked_keywords]
                                
                                # Join keywords with a special delimiter
                                combined_text = " ||| ".join(keywords_only)
                                
                                # Translate the combined text
                                translated_combined = translate_text(combined_text, "en", session_id)
                                
                                # Split back into individual keywords
                                translated_keywords = translated_combined.split(" ||| ")
                                
                                # Check if we got the right number of translations
                                if len(translated_keywords) == len(ranked_keywords):
                                    # Rebuild the tuples with original scores
                                    translated_ranked_keywords = [(translated_keywords[i], score) for i, (_, score) in enumerate(ranked_keywords)]
                                else:
                                    print(f"Translation mismatch: got {len(translated_keywords)} items, expected {len(ranked_keywords)}")
                                    # Fall back to individual translation
                                    translated_ranked_keywords = [(translate_text(kw, "en", session_id), score) for kw, score in ranked_keywords]
                                
                                translated_ranked_keywords = [(translated_keywords[i], score)
                                                       for i, (_, score) in enumerate(ranked_keywords)]
                            else:
                                translated_ranked_keywords = ranked_keywords

                            logging.info(f"Using ranked keywords for product search: {translated_ranked_keywords[:15]}")
                            
                            # Get user gender and budget for filtering
                            user_gender = user_context.get("user_gender", {}).get("category", None)
                            budget_range = user_context.get("budget_range", None)
                            # Use the multi-item flag from user_context, determined during the initial query
                            is_multi_item_request = user_context.get("is_multi_item_request_flag", False)
                            print(f"   🤝 Using stored Is Multi-Item Request flag from context: {is_multi_item_request}") # Debug

                            print(f"\n💰 CONTEXT BUDGET DEBUG:")
                            print(f"   📊 user_context keys: {list(user_context.keys())}")
                            if "budget_range" in user_context:
                                print(f"   💰 budget_range in context: {user_context['budget_range']}")
                                print(f"   🔍 budget_range type: {type(user_context['budget_range'])}")
                            else:
                                print(f"   ❌ No budget_range in context")

                            if budget_range:
                                logging.info(f"Using budget filter: {budget_range}")
                                print(f"Using budget filter: {budget_range}")
                            
                            # Positive confirmation message
                            positive_response = "Great! Based on your preferences and style recommendations, here are some products that might interest you:"
                            if budget_range:
                                min_price, max_price = budget_range
                                if min_price and max_price:
                                    budget_text = f" (within your budget of IDR {min_price:,} - IDR {max_price:,})"
                                elif max_price:
                                    budget_text = f" (under {max_price:,})"
                                elif min_price:
                                    budget_text = f" (above {min_price:,})"
                                else:
                                    budget_text = ""
                                positive_response += budget_text

                            if user_language != "en":
                                positive_response = translate_text(positive_response, user_language, session_id)

                            print(f"\n🔍 PRODUCT SEARCH INPUTS:")
                            print(f"   🎯 Keywords: {len(translated_ranked_keywords)}")
                            for i, (kw, score) in enumerate(translated_ranked_keywords[:15]):
                                print(f"      {i+1}. '{kw}' → {score:.2f}")
                            print(f"   👤 Gender: {user_gender}")
                            print(f"   💰 Budget: {budget_range}")
                            print(f"   🤝 Is Multi-Item Request (passed to relevance): {is_multi_item_request}") # New debug line
                            print()
                            
                            # Fetch products using the ranked keywords
                            try:
                                recommended_products, budget_status = await fetch_products_with_budget_awareness(
                                    db=db,  # Make sure db is the AsyncSession object
                                    top_keywords=translated_ranked_keywords,  # Make sure this is a list of tuples
                                    max_results=15,
                                    gender_category=user_gender,
                                    budget_range=budget_range,
                                    is_multi_item_request=is_multi_item_request
                                )
                                
                                print(f"Successfully fetched {len(recommended_products)} products")

                                # Handle different budget scenarios
                                if budget_status == "no_products_in_budget":
                                    # No products within budget - ask user what to do
                                    cheapest_price = recommended_products['price'].min() if not recommended_products.empty else None
                                    most_expensive_price = recommended_products['price'].max() if not recommended_products.empty else None
                                    
                                    budget_messages = generate_budget_message(budget_range, user_language, cheapest_price, most_expensive_price)
                                    budget_response = budget_messages["show_outside_budget"]
                                    
                                    # Save the products for potential display and set awaiting budget decision flag
                                    user_context["pending_products"] = recommended_products
                                    user_context["awaiting_budget_decision"] = True
                                    user_context["budget_scenario"] = "show_outside_budget"
                                    
                                    # Send budget constraint message
                                    new_ai_message = ChatHistoryDB(
                                        session_id=session_id,
                                        message_type="assistant",
                                        content=budget_response
                                    )
                                    db.add(new_ai_message)
                                    await db.commit()
                                    
                                    await websocket.send_text(f"{session_id}|{budget_response}")
                                    return  # Exit early, wait for user decision
                                    
                                elif budget_status == "no_products_found":
                                    # NO PRODUCTS FOUND AT ALL - not a budget issue
                                    no_products_response = set_search_adjustment_mode(user_context, user_language, session_id)
                                    
                                    if user_language != "en":
                                        no_products_response = translate_text(no_products_response, user_language, session_id)

                                    user_context["awaiting_search_adjustment"] = True
                                    user_context["awaiting_confirmation"] = False
                                    
                                    new_ai_message = ChatHistoryDB(
                                        session_id=session_id,
                                        message_type="assistant",
                                        content=no_products_response
                                    )
                                    db.add(new_ai_message)
                                    await db.commit()
                                    
                                    await websocket.send_text(f"{session_id}|{no_products_response}")
                                    user_context["awaiting_confirmation"] = False
                                    continue  # Exit early
                                
                            except Exception as fetch_error:
                                logging.error(f"Error calling fetch_products_from_db: {str(fetch_error)}")
                                logging.error(f"Parameters passed:")
                                logging.error(f"- db: {type(db)}")
                                logging.error(f"- top_keywords: {type(translated_ranked_keywords)} - {translated_ranked_keywords[:3] if translated_ranked_keywords else 'None'}")
                                logging.error(f"- user_gender: {user_gender}")
                                logging.error(f"- budget_range: {budget_range}")
                                raise
                            
                            user_context["product_cache"]["all_results"] = recommended_products
                            user_context["product_cache"]["current_page"] = 0
                            user_context["product_cache"]["has_more"] = len(recommended_products) > 5

                            first_page_products, has_more = get_paginated_products(
                                recommended_products,
                                page=0, 
                                products_per_page=5
                            )

                            user_context["product_cache"]["has_more"] = has_more

                            # Create response with product cards
                            if not first_page_products.empty:
                                complete_response = positive_response + "\n\n"
                                
                                for _, row in first_page_products.iterrows():
                                    # Create a simpler, more robust product card structure
                                    product_card = (
                                        "<div class='product-card'>\n"
                                        f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                        f"<div class='product-info'>\n"  # Separate container for text info
                                        f"<h3>{row['product']}</h3>\n"
                                        f"<p class='price'>IDR {row['price']}</p>\n"
                                        f"<p class='description'>{row['description']}</p>\n"
                                        f"<p class='available'>Available in size: {row['size']}, Color: {row['color']}</p>\n"
                                        f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                        "</div>\n"  # Close the product-info div
                                        "</div>\n"  # Close the product-card div
                                    )
                                    complete_response += product_card
                                
                                if has_more:
                                    complete_response += "\n\nWould you like to see more options? Just ask for 'more products' or 'lainnya'!"
                            else:
                                complete_response = positive_response + "\n\nI'm sorry, but I couldn't find specific product recommendations at the moment. Would you like me to help you with something else?"
                                budget_msg = ""

                            # For translation, use a more robust method to protect HTML tags
                            if user_language != "en":
                                try:
                                    # Alternative approach to protect HTML: encode entire HTML blocks
                                    def encode_html_blocks(text):
                                        # Define a pattern that matches complete HTML elements with content
                                        pattern = r'(<div class=\'product-card\'>.*?</div>\n)'
                                        blocks = []
                                        
                                        # Replace each block with a placeholder
                                        def replace_block(match):
                                            nonlocal blocks
                                            placeholder = f"__HTML_BLOCK_{len(blocks)}__"
                                            blocks.append(match.group(0))
                                            return placeholder
                                        
                                        protected_text = re.sub(pattern, replace_block, text, flags=re.DOTALL)
                                        return protected_text, blocks
                                    
                                    # Decode HTML blocks after translation
                                    def decode_html_blocks(text, blocks):
                                        for i, block in enumerate(blocks):
                                            text = text.replace(f"__HTML_BLOCK_{i}__", block)
                                        return text
                                    
                                    # Protect complete HTML blocks
                                    protected_text, html_blocks = encode_html_blocks(complete_response)
                                    
                                    # Translate the protected text
                                    translated_protected = translate_text(protected_text, user_language, session_id)
                                    
                                    # Restore the HTML blocks
                                    translated_response = decode_html_blocks(translated_protected, html_blocks)
                                except Exception as e:
                                    logging.error(f"Error in HTML protection during translation: {str(e)}")
                                    # Fallback to original method if new method fails
                                    html_tags = {}
                                    pattern = r'<[^>]+>'
                                    
                                    for i, match in enumerate(re.finditer(pattern, complete_response)):
                                        placeholder = f"TAG_{i}"
                                        html_tags[placeholder] = match.group(0)
                                        complete_response = complete_response.replace(match.group(0), placeholder, 1)
                                    
                                    # Translate text
                                    translated_response = translate_text(complete_response, user_language, session_id)
                                    
                                    # Restore HTML tags
                                    for placeholder, tag in html_tags.items():
                                        translated_response = translated_response.replace(placeholder, tag)
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
                            
                            # Send the response
                            complete_response_html = render_markdown(translated_response)
                            print(f"{complete_response_html}")
                            await websocket.send_text(f"{session_id}|{complete_response_html}")
                            
                        except Exception as e:
                            logging.error(f"Error during product recommendation: {str(e)}")
                            error_msg = "I'm sorry, I couldn't fetch product recommendations. Is there something else you'd like to know about fashion?"
                            if user_language != "en":
                                error_msg = translate_text(error_msg, user_language, session_id)
                            await websocket.send_text(f"{session_id}|{error_msg}")
                        
                        # Reset confirmation flag
                        user_context["awaiting_confirmation"] = True

                    elif is_more_request:
                        # USER wants MORE products - handle pagination properly
                        logging.info("🔄 User requesting MORE products")
                        
                        # Check if we have cached results for pagination
                        if not user_context["product_cache"]["all_results"].empty:
                            current_page = user_context["product_cache"]["current_page"]
                            next_page = current_page + 1
                            products_per_page = user_context["product_cache"].get("products_per_page", 5)
                            
                            next_page_products, has_more = get_paginated_products(
                                user_context["product_cache"]["all_results"],
                                page=next_page,
                                products_per_page=products_per_page
                            )
                            
                            if not next_page_products.empty:
                                # Update pagination state
                                user_context["product_cache"]["current_page"] = next_page
                                user_context["product_cache"]["has_more"] = has_more
                                
                                # Create natural response for more products
                                more_responses_en = [
                                    "Here are some more options that might interest you:",
                                    "I found some additional styles you might like:",
                                    "Let me show you a few more possibilities:",
                                    "Here are some other great choices:",
                                ]
                                
                                more_responses_id = [
                                    "Berikut beberapa pilihan lain yang mungkin menarik:",
                                    "Saya menemukan beberapa gaya tambahan yang mungkin Anda suka:",
                                    "Mari saya tunjukkan beberapa kemungkinan lainnya:",
                                    "Berikut beberapa pilihan bagus lainnya:",
                                ]
                                
                                import random
                                if user_language != "en":
                                    positive_response = random.choice(more_responses_id)
                                else:
                                    positive_response = random.choice(more_responses_en)
                                
                                # Add budget context if available
                                budget_range = user_context.get("budget_range", None)
                                if budget_range:
                                    min_price, max_price = budget_range
                                    if min_price and max_price:
                                        budget_text = f" (within your budget of IDR {min_price:,} - IDR {max_price:,})"
                                    elif max_price:
                                        budget_text = f" (under IDR {max_price:,})"
                                    elif min_price:
                                        budget_text = f" (above IDR {min_price:,})"
                                    else:
                                        budget_text = ""
                                    positive_response += budget_text
                                
                                # Generate product cards for NEXT PAGE ONLY
                                complete_response = positive_response + "\n\n"
                                
                                for _, row in next_page_products.iterrows():
                                    product_card = (
                                        "<div class='product-card'>\n"
                                        f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                        f"<div class='product-info'>\n"
                                        f"<h3>{row['product']}</h3>\n"
                                        f"<p class='price'>IDR {row['price']}</p>\n"
                                        f"<p class='description'>{row['description']}</p>\n"
                                        f"<p class='available'>Available in size: {row['size']}, Color: {row['color']}</p>\n"
                                        f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                        "</div>\n"
                                        "</div>\n"
                                    )
                                    complete_response += product_card
                                
                                # Add appropriate footer based on whether more products are available
                                if has_more:
                                    if user_language != "en":
                                        more_hint = translate_text("\n\nI have even more options if you'd like to continue exploring! Just let me know if you want to see more.", user_language, session_id)
                                    else:
                                        more_hint = "\n\nI have even more options if you'd like to continue exploring! Just let me know if you want to see more."
                                    complete_response += more_hint
                                else:
                                    if user_language != "en":
                                        end_hint = translate_text("\n\nThat's all the products I found based on your preferences. Is there anything else I can help you with, or would you like to try a different search?", user_language, session_id)
                                    else:
                                        end_hint = "\n\nThat's all the products I found based on your preferences. Is there anything else I can help you with, or would you like to try a different search?"
                                    complete_response += end_hint
                                
                                # Handle translation while protecting HTML
                                if user_language != "en":
                                    try:
                                        # Protect HTML blocks during translation
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
                                        
                                        # Protect and translate
                                        protected_text, html_blocks = encode_html_blocks(complete_response)
                                        translated_protected = translate_text(protected_text, user_language, session_id)
                                        translated_response = decode_html_blocks(translated_protected, html_blocks)
                                        
                                    except Exception as e:
                                        logging.error(f"Error in HTML protection during translation: {str(e)}")
                                        # Fallback translation method
                                        html_tags = {}
                                        pattern = r'<[^>]+>'
                                        
                                        for i, match in enumerate(re.finditer(pattern, complete_response)):
                                            placeholder = f"TAG_{i}"
                                            html_tags[placeholder] = match.group(0)
                                            complete_response = complete_response.replace(match.group(0), placeholder, 1)
                                        
                                        translated_response = translate_text(complete_response, user_language, session_id)
                                        
                                        for placeholder, tag in html_tags.items():
                                            translated_response = translated_response.replace(placeholder, tag)
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
                                
                                # Send the response
                                complete_response_html = render_markdown(translated_response)
                                await websocket.send_text(f"{session_id}|{complete_response_html}")
                                
                                # Log pagination info
                                total_products = len(user_context["product_cache"]["all_results"])
                                products_shown = (next_page + 1) * products_per_page
                                logging.info(f"📄 Showed page {next_page + 1}, products {current_page * products_per_page + 1}-{min(products_shown, total_products)} of {total_products}")
                                logging.info(f"📊 Has more pages: {has_more}")
                                
                                # Keep awaiting confirmation for potential more requests
                                user_context["awaiting_confirmation"] = True
                                
                            else:
                                # No more products on next page (edge case)
                                if user_language != "en":
                                    no_more_msg = translate_text("I've shown you all the best matches I could find. Would you like to try a different style or adjust your preferences?", user_language, session_id)
                                else:
                                    no_more_msg = "I've shown you all the best matches I could find. Would you like to try a different style or adjust your preferences?"
                                
                                new_ai_message = ChatHistoryDB(
                                    session_id=session_id,
                                    message_type="assistant",
                                    content=no_more_msg
                                )
                                db.add(new_ai_message)
                                await db.commit()
                                
                                await websocket.send_text(f"{session_id}|{no_more_msg}")
                                logging.info("📭 No more products available on next page")
                                user_context["awaiting_confirmation"] = False
                        
                        else:
                            # No cached results available - this shouldn't happen if flow is correct
                            logging.warning("🚨 No cached results available for 'more' request")
                            
                            if user_language != "en":
                                no_cache_msg = translate_text("I don't have any cached product recommendations. Let me search for new recommendations for you. What style are you looking for?", user_language, session_id)
                            else:
                                no_cache_msg = "I don't have any cached product recommendations. Let me search for new recommendations for you. What style are you looking for?"
                            
                            new_ai_message = ChatHistoryDB(
                                session_id=session_id,
                                message_type="assistant",
                                content=no_cache_msg
                            )
                            db.add(new_ai_message)
                            await db.commit()
                            
                            await websocket.send_text(f"{session_id}|{no_cache_msg}")
                            user_context["awaiting_confirmation"] = False

                    elif user_context.get("awaiting_budget_decision", False):
                        # User is responding to budget constraint message
                        budget_response_type = detect_budget_response(user_input)
                        budget_scenario = user_context.get("budget_scenario", "")
                        
                        if budget_response_type == "show_anyway":
                            # User wants to see products outside budget
                            pending_products = user_context.get("pending_products", pd.DataFrame())
                            
                            if not pending_products.empty:
                                # Show the products that were outside budget
                                positive_response = "Great! Here are the product recommendations outside your budget range:"
                                if user_language != "en":
                                    positive_response = translate_text(positive_response, user_language, session_id)
                                
                                # Use existing product display logic with pending_products
                                first_page_products, has_more = get_paginated_products(
                                    pending_products, page=0, products_per_page=5
                                )
                                
                                user_context["product_cache"]["all_results"] = pending_products
                                user_context["product_cache"]["current_page"] = 0
                                user_context["product_cache"]["has_more"] = has_more
                                
                                # ... continue with existing product card generation ...
                                
                            # Clear budget decision flags
                            user_context["awaiting_budget_decision"] = False
                            user_context["pending_products"] = pd.DataFrame()
                            user_context["budget_scenario"] = ""
                            user_context["awaiting_confirmation"] = True
                            
                        elif budget_response_type == "adjust_budget":
                            # User wants to adjust budget
                            budget_adjustment, confidence = detect_budget_adjustment_request(user_input)
                            
                            if budget_adjustment and confidence > 0.6:
                                if isinstance(budget_adjustment, tuple) and budget_adjustment[0] in ["increase", "decrease"]:
                                    # Relative adjustment
                                    current_budget = user_context.get("budget_range", (None, None))
                                    action, amount = budget_adjustment
                                    
                                    if current_budget[1]:  # Has max budget
                                        if action == "increase":
                                            new_max = current_budget[1] + amount
                                            new_budget = (current_budget[0], new_max)
                                        else:  # decrease
                                            new_max = max(current_budget[1] - amount, current_budget[0] if current_budget[0] else 0)
                                            new_budget = (current_budget[0], new_max)
                                        
                                        user_context["budget_range"] = new_budget
                                        
                                        adjust_response = f"Budget adjusted to IDR {new_budget[0]:,} - IDR {new_budget[1]:,}. Let me search again with your new budget."
                                        if user_language != "en":
                                            adjust_response = translate_text(adjust_response, user_language, session_id)
                                        
                                        await websocket.send_text(f"{session_id}|{adjust_response}")
                                        
                                        # Trigger new search with adjusted budget
                                        user_context["awaiting_budget_decision"] = False
                                        user_context["awaiting_confirmation"] = False  # Trigger new search
                                    else:
                                        # Ask for complete budget range
                                        budget_help = "Please specify your complete budget range, for example: 'budget 100rb-300rb' or 'maximum 250rb'"
                                        if user_language != "en":
                                            budget_help = translate_text(budget_help, user_language, session_id)
                                        await websocket.send_text(f"{session_id}|{budget_help}")
                                else:
                                    # Absolute budget
                                    user_context["budget_range"] = budget_adjustment
                                    new_min, new_max = budget_adjustment
                                    
                                    if new_min and new_max:
                                        budget_text = f"IDR {new_min:,} - IDR {new_max:,}"
                                    elif new_max:
                                        budget_text = f"under IDR {new_max:,}"
                                    else:
                                        budget_text = f"above IDR {new_min:,}"
                                    
                                    adjust_response = f"Budget updated to {budget_text}. Let me search for products within your new budget."
                                    if user_language != "en":
                                        adjust_response = translate_text(adjust_response, user_language, session_id)
                                    
                                    await websocket.send_text(f"{session_id}|{adjust_response}")
                                    
                                    # Trigger new search
                                    user_context["awaiting_budget_decision"] = False
                                    user_context["awaiting_confirmation"] = False
                            else:
                                # Ask for clarification
                                budget_help = "Could you please specify your new budget? For example: 'budget 150rb-400rb' or 'increase budget by 100rb'"
                                if user_language != "en":
                                    budget_help = translate_text(budget_help, user_language, session_id)
                                await websocket.send_text(f"{session_id}|{budget_help}")
                        
                        elif budget_response_type == "adjust_search":
                            # User wants to adjust search criteria instead
                            budget_messages = generate_budget_message(user_context.get("budget_range"), user_language)
                            adjust_response = budget_messages["budget_adjustment"]
                            
                            user_context["awaiting_budget_decision"] = False
                            user_context["awaiting_confirmation"] = False  # Allow new search
                            
                            new_ai_message = ChatHistoryDB(
                                session_id=session_id,
                                message_type="assistant", 
                                content=adjust_response
                            )
                            db.add(new_ai_message)
                            await db.commit()
                            
                            await websocket.send_text(f"{session_id}|{adjust_response}")
                        
                        else:
                            # Unknown response, ask for clarification
                            clarification = "I didn't quite understand. Would you like to see products outside your budget (yes/no), or would you prefer to adjust your search criteria?"
                            if user_language != "en":
                                clarification = translate_text(clarification, user_language, session_id)
                            await websocket.send_text(f"{session_id}|{clarification}")
                            
                    elif is_negative:
                        # User declined product recommendations
                        negative_response = "I understand. What specific styles or fashion advice would you prefer instead? I'm here to help you find the perfect look."
                        if user_language != "en":
                            negative_response = translate_text(negative_response, user_language, session_id)
                            
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=negative_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        await websocket.send_text(f"{session_id}|{negative_response}")
                        user_context["awaiting_confirmation"] = False
                    
                    else:
                        # Extract keywords from the user's additional input
                        if user_language != "en":
                            translated_input = translate_text(user_input, "en", session_id)
                        else:
                            translated_input = user_input
                        
                        # Save current text input for later use
                        user_context["current_text_input"] = user_input
                            
                        # Extract and update keywords from this additional context
                        additional_keywords = extract_ranked_keywords("", translated_input)
                        update_accumulated_keywords(additional_keywords, user_context, is_user_input=True)
                        
                        # Continue with regular processing
                        user_context["awaiting_confirmation"] = False
                
                # Check if input contains an image URL
                url_pattern = re.compile(r'(https?://\S+\.(?:jpg|jpeg|png|gif|bmp|webp))', re.IGNORECASE)
                image_url_match = url_pattern.search(user_input)
                
                if not user_context["awaiting_confirmation"] and image_url_match:
                    # Process image input (with or without text)
                    image_url = image_url_match.group(1)
                    # Extract text content by removing the image URL
                    text_content = user_input.replace(image_url, "").strip()
                    
                    try:
                        # Update user context for image input
                        user_context["has_shared_image"] = True
                        user_context["last_query_type"] = "mixed" if text_content else "image"
                        user_context["current_image_url"] = image_url
                        user_context["current_text_input"] = text_content  # Save text content for keyword extraction

                        # Call image analysis
                        clothing_features = await analyze_uploaded_image(image_url)

                        # Error handling for image analysis
                        if clothing_features.startswith("Error:"):
                            error_message = clothing_features
                            logging.warning(f"Image analysis error: {error_message}")

                            # Save error message to database
                            new_error_message = ChatHistoryDB(
                                session_id=session_id,
                                message_type="assistant",
                                content=error_message
                            )
                            db.add(new_error_message)
                            await db.commit()

                            # Send error response
                            await websocket.send_text(f"{session_id}|{error_message}")
                            continue

                        # Direct gender detection from text content
                        if text_content:
                            detect_and_update_gender(text_content, user_context, False)

                        # Prepare prompt based on whether there's additional text or just an image
                        # Include gender information if available
                        user_gender_info = get_user_gender(user_context)
                        gender_context = ""
                        if user_gender_info["category"]:
                            gender_context = f" I am {user_gender_info['category']}."
                            
                        if text_content:
                            prompt = f"I've shared an image with the following request: '{text_content}'.{gender_context} Here's what the image shows: {clothing_features}. Please give me style recommendations based on this image and my specific request, but DO NOT offer product recommendations yet."
                        else:
                            prompt = f"I've shared an image.{gender_context} Here's what the image shows: {clothing_features}. Please give me style recommendations based on this image, but DO NOT offer product recommendations yet."
                        
                        # Generate styling recommendations based on image and text
                        message_objects.append({
                            "role": "user",
                            "content": prompt
                        })

                        # Get AI style recommendation response
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

                        # Extract and accumulate keywords from image analysis with high weight
                        image_keywords = extract_ranked_keywords(clothing_features, "")
                        update_accumulated_keywords(image_keywords, user_context, is_user_input=True)
                        
                        # Also process text content keywords if available
                        if text_content:
                            text_keywords = extract_ranked_keywords("", text_content)
                            update_accumulated_keywords(text_keywords, user_context, is_user_input=True)
                        
                        # Extract and accumulate keywords from AI style suggestions
                        style_keywords = extract_ranked_keywords(ai_response, "")
                        update_accumulated_keywords(style_keywords, user_context, is_ai_response=True)
                        
                        # Log the updated keywords and gender information
                        logging.info(f"Updated accumulated keywords after image analysis: {user_context['accumulated_keywords']}")
                        logging.info(f"Current user gender info: {user_context['user_gender']}")
                        print(f"Updated accumulated keywords after image analysis: {user_context['accumulated_keywords']}")
                        print(f"Current user gender info: {user_context['user_gender']}")

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
                        
                        # Send the response
                        ai_response_html = render_markdown(translated_ai_response)
                        await websocket.send_text(f"{session_id}|{ai_response_html}")
                        
                        # Set awaiting confirmation flag
                        user_context["awaiting_confirmation"] = True

                    except Exception as input_error:
                        logging.error(f"Error during image processing: {str(input_error)}\n{traceback.format_exc()}")
                        error_msg = "Sorry, there was an issue processing your image. Could you try again?"
                        await websocket.send_text(f"{session_id}|{error_msg}")
                
                # Handle normal text input (if no image was processed and not waiting for confirmation)
                elif not user_context["awaiting_confirmation"]:

                    print(f"\n📋 TEXT PROCESSING START")
                    print("="*50)
                    print(f"📝 User input: '{user_input}'")
                    if "budget_range" in user_context and user_context["budget_range"]:
                        print(f"💰 Current budget: {user_context['budget_range']}")
                    print("="*50)

                    # Check for small talk
                    if await is_small_talk(user_input):
                        ai_response = "Hello! How can I assist you with fashion recommendations today? Feel free to share information about your style preferences or upload an image for personalized suggestions."
                        if user_language != "en":
                            ai_response = translate_text(ai_response, user_language, session_id)
                            
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=ai_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        await websocket.send_text(f"{session_id}|{ai_response}")
                        continue
                    
                    # Process text input for style recommendations
                    user_context["last_query_type"] = "text"
                    user_context["current_text_input"] = user_input

                    # Translate if needed
                    if user_language != "en":
                        translated_input = translate_text(user_input, "en", session_id)
                    else:
                        translated_input = user_input
                    
                    # Direct gender detection from text
                    detect_and_update_gender(translated_input, user_context, False)
                        
                    # Extract and accumulate keywords from user input with high weight
                    input_keywords = extract_ranked_keywords("", translated_input)
                    is_multi_item_request_flag = getattr(extract_ranked_keywords, 'is_multi_item_request', False)
                    user_context["is_multi_item_request_flag"] = is_multi_item_request_flag
                    update_linked_keywords(user_context, input_keywords, is_user_input=True, is_multi_item_request=is_multi_item_request_flag)

                    # Add to message history - keep it simple
                    message_objects.append({
                        "role": "user",
                        "content": translated_input,
                    })
                    
                    # Get AI response with style recommendations
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
                    
                    # Extract and accumulate keywords from AI response
                    response_keywords = extract_ranked_keywords(ai_response, "")

                    update_linked_keywords(user_context, response_keywords, is_user_input=False, is_multi_item_request=user_context["is_multi_item_request_flag"])

                    # Log the updated keywords and gender information
                    logging.info(f"Updated accumulated keywords after text conversation: {user_context['accumulated_keywords']}")
                    logging.info(f"Current user gender info: {user_context['user_gender']}")
                    print(f"Updated accumulated keywords after text conversation: {user_context['accumulated_keywords']}")
                    print(f"Current user gender info: {user_context['user_gender']}")
                    
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
                    
                    # Render and send
                    ai_response_html = render_markdown(translated_response)
                    await websocket.send_text(f"{session_id}|{ai_response_html}")

                    # Check for rapid preference changes
                    detect_rapid_preference_changes(user_input, user_context)
                    
                    # Save current text input
                    user_context["current_text_input"] = user_input
                  
                    # Set awaiting confirmation flag
                    user_context["awaiting_confirmation"] = True
                
            except WebSocketDisconnect:
                logging.info(f"Websocket disconnected for session {session_id}")
                session_manager.reset_session(session_id)
                break
                
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")
                error_message = "I'm sorry, I encountered an error while processing your request. Please try again."
                if user_language != "en":
                    try:
                        error_message = translate_text(error_message, user_language, session_id)
                    except:
                        pass
                await websocket.send_text(f"{session_id}|{error_message}")
                
    except Exception as e:
        logging.error(f"Websocket error: {str(e)}\n{traceback.format_exc()}")
        try:
            await websocket.close()
        except:
            pass

def extract_budget_from_text(text):
    """
    Extract budget information with smart physical context detection.
    """
    if not text:
        return None
    
    print(f"\n💰 BUDGET EXTRACTION DEBUG")
    print(f"   📝 Input text: '{text}'")
    
    text_lower = text.lower()
    
    # STEP 1: Identify and exclude physical measurements
    physical_measurement_patterns = [
        r'\b(\d+)\s*(?:kg|kilogram|gram|gr)\b',  # Weight: 50kg, 60 kilogram
        r'\b(\d+)\s*(?:cm|centimeter|meter|m)\b',  # Height: 150cm, 1.7m
        r'\b(?:tinggi|height)\s*(\d+)(?:\s*cm)?\b',  # Height context: tinggi 150
        r'\b(?:berat|weight)\s*(\d+)(?:\s*kg)?\b',  # Weight context: berat 50
        r'\b(?:umur|age)\s*(\d+)(?:\s*tahun|years?)?\b',  # Age: umur 25
        r'\b(\d+)\s*(?:tahun|years?)\b',  # Age: 25 tahun
        r'\b(\d+)\s*(?:inch|ft|feet)\b',  # Imperial measurements
    ]
    
    # Extract all physical measurements to exclude them
    physical_numbers = set()
    for pattern in physical_measurement_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                physical_numbers.update(match)
            else:
                physical_numbers.add(match)
    
    print(f"   🏃 Physical measurements detected: {physical_numbers}")
    
    # STEP 2: Check for budget indicators with PRECISE currency detection
    explicit_budget_keywords = ['budget', 'anggaran', 'harga', 'price', 'biaya', 'cost']
    constraint_indicators = [
        'dibawah', 'under', 'maksimal', 'max', 'kurang dari', 'less than',
        'diatas', 'over', 'minimal', 'min', 'lebih dari', 'more than',
        'sekitar', 'around', 'kisaran', 'range'
    ]
    
    # FIXED: More precise currency detection using word boundaries
    currency_patterns = [
        r'\b(\d+)rb\b',           # 300rb
        r'\b(\d+)ribu\b',         # 300ribu  
        r'\b(\d+)k\b',            # 300k (but not 50kg)
        r'\b(\d+)jt\b',           # 1jt
        r'\b(\d+)juta\b',         # 1juta
        r'\brupiah\b',            # rupiah
        r'\brp\.?\s*\d+\b',       # rp 300000
        r'\bidr\b'                # idr
    ]
    
    has_explicit_budget = any(keyword in text_lower for keyword in explicit_budget_keywords)
    has_constraint = any(indicator in text_lower for indicator in constraint_indicators)
    
    # Check for currency with precise patterns (avoid kg, cm, etc.)
    has_currency = False
    found_currency_matches = []
    for pattern in currency_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            has_currency = True
            found_currency_matches.extend(matches)
    
    print(f"   💰 Explicit budget: {has_explicit_budget}")
    print(f"   🎯 Constraint indicators: {has_constraint}")
    print(f"   💵 Currency found: {has_currency} - {found_currency_matches}")
    
    # STEP 3: Decision logic with physical exclusion
    should_process_budget = False
    
    if has_explicit_budget:
        should_process_budget = True
        print(f"   ✅ Explicit budget keyword found")
    elif has_constraint and has_currency:
        should_process_budget = True
        print(f"   ✅ Budget constraint + currency found")
    elif has_currency:
        # Check for standalone currency amounts, but exclude physical measurements
        currency_number_patterns = [
            r'\b(\d+)rb\b',
            r'\b(\d+)ribu\b',
            r'\b(\d+)k\b',
            r'\b(\d+)jt\b',
            r'\b(\d+)juta\b',
            r'\brp\.?\s*(\d+)\b'
        ]
        
        found_currency_numbers = []
        for pattern in currency_number_patterns:
            matches = re.findall(pattern, text_lower)
            found_currency_numbers.extend(matches)
        
        # Only proceed if we found currency numbers that aren't physical measurements
        valid_currency_numbers = [num for num in found_currency_numbers if num not in physical_numbers]
        
        if valid_currency_numbers:
            should_process_budget = True
            print(f"   ✅ Valid currency amounts found: {valid_currency_numbers}")
        else:
            print(f"   ❌ Currency amounts are physical measurements: {found_currency_numbers}")
    
    if not should_process_budget:
        print(f"   ❌ No valid budget indicators found")
        return None
    
    # STEP 4: Extract budget with physical exclusion
    def convert_to_rupiah(amount_str, unit):
        try:
            amount = int(amount_str)
            if unit in ['rb', 'ribu', 'k']:
                return amount * 1000
            elif unit in ['jt', 'juta']:
                return amount * 1000000
            elif unit == '000':
                return amount * 1000
            else:
                return amount
        except:
            return None
    
    budget_patterns = [
        # Range patterns
        (r'(?:budget|anggaran|harga)?\s*(?:antara|between)?\s*(\d+)(?:rb|ribu|k|jt|juta)?\s*(?:-|sampai|hingga|to)\s*(\d+)(?:rb|ribu|k|jt|juta)?', "RANGE"),
        
        # Constraint patterns
        (r'(?:dibawah|under|maksimal|max|kurang\s+dari|less\s+than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta|000)?', "MAX"),
        (r'(?:diatas|over|minimal|min|lebih\s+dari|more\s+than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta|000)?', "MIN"),
        
        # Exact/around patterns
        (r'(?:budget|anggaran|sekitar|around|kisaran)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta|000)?', "EXACT"),
        
        # Standalone currency (only if passed validation above)
        (r'\b(\d+)(?:rb|ribu|k|jt|juta)\b', "STANDALONE"),
    ]
    
    for pattern_idx, (pattern, pattern_type) in enumerate(budget_patterns):
        matches = list(re.finditer(pattern, text_lower))
        
        for match in matches:
            groups = match.groups()
            match_text = match.group(0)
            
            # CRITICAL: Exclude matches that contain physical measurements
            matched_numbers = [g for g in groups if g and g.isdigit()]
            if any(num in physical_numbers for num in matched_numbers):
                print(f"   ⚠️ Skipping pattern match with physical number: {match_text}")
                continue
            
            # Process valid matches
            if pattern_type == "RANGE" and len(groups) >= 2 and groups[0] and groups[1]:
                unit = 'rb' if any(x in match_text for x in ['rb', 'ribu', 'k']) else 'jt' if 'jt' in match_text else None
                
                min_price = convert_to_rupiah(groups[0], unit)
                max_price = convert_to_rupiah(groups[1], unit)
                
                if min_price and max_price:
                    result = (min(min_price, max_price), max(min_price, max_price))
                    print(f"   🎯 RANGE BUDGET: {result}")
                    return result
            
            elif len(groups) >= 1 and groups[0]:
                unit = None
                if any(x in match_text for x in ['rb', 'ribu', 'k']):
                    unit = 'rb'
                elif any(x in match_text for x in ['jt', 'juta']):
                    unit = 'jt'
                elif '000' in match_text:
                    unit = '000'
                
                amount = convert_to_rupiah(groups[0], unit)
                
                if amount:
                    if pattern_type == "MAX":
                        result = (None, amount)
                        print(f"   🎯 MAX BUDGET: {result}")
                        return result
                    elif pattern_type == "MIN":
                        result = (amount, None)
                        print(f"   🎯 MIN BUDGET: {result}")
                        return result
                    elif pattern_type in ["EXACT", "STANDALONE"]:
                        min_range = int(amount * 0.8)
                        max_range = int(amount * 1.2)
                        result = (min_range, max_range)
                        print(f"   🎯 EXACT BUDGET: {result}")
                        return result
    
    print(f"   ❌ No valid budget patterns after physical exclusion")
    return None
    
def extract_bold_headings_from_ai_response(ai_response):
    """
    Extract bold headings from AI response text to use as keywords.
    These are typically the main clothing items or style recommendations.
    """
    if not ai_response:
        return []
    
    bold_headings = []
    
    # Pattern 1: **Bold Text** (Markdown bold)
    markdown_bold_pattern = r'\*\*(.*?)\*\*'
    markdown_matches = re.findall(markdown_bold_pattern, ai_response)
    
    # Pattern 2: **Bold Text** at the beginning of lines (heading style)
    heading_pattern = r'^\s*\*\*(.*?)\*\*\s*$'
    heading_matches = re.findall(heading_pattern, ai_response, re.MULTILINE)
    
    # Pattern 3: **Bold Text** followed by description (clothing item pattern)
    clothing_pattern = r'\*\*(.*?)\*\*\s*[–-]\s*'
    clothing_matches = re.findall(clothing_pattern, ai_response)
    
    # Combine all matches
    all_matches = markdown_matches + heading_matches + clothing_matches
    
    # Clean and filter the matches
    for match in all_matches:
        cleaned = match.strip()
        
        # Skip very short or generic terms
        if len(cleaned) < 3:
            continue
            
        # Skip generic phrases
        generic_phrases = [
            'for you', 'untuk anda', 'recommendation', 'rekomendasi',
            'style', 'gaya', 'fashion', 'outfit', 'look', 'perfect',
            'great', 'amazing', 'beautiful', 'cantik', 'bagus'
        ]
        
        if any(generic in cleaned.lower() for generic in generic_phrases):
            continue
        
        # Focus on clothing items and specific style terms
        clothing_indicators = [
            'kemeja', 'shirt', 'blouse', 'dress', 'gaun', 'celana', 'pants',
            'rok', 'skirt', 'jaket', 'jacket', 'sweater', 'cardigan',
            'kaos', 't-shirt', 'atasan', 'top', 'bawahan', 'bottom',
            'hoodie', 'blazer', 'coat', 'mantel', 'jeans', 'denim'
        ]
        
        style_indicators = [
            'casual', 'formal', 'elegant', 'vintage', 'modern', 'minimalist',
            'bohemian', 'oversized', 'slim', 'cropped', 'long sleeve',
            'short sleeve', 'off shoulder', 'button up', 'graphic'
        ]
        
        # Check if the heading contains clothing or style terms
        cleaned_lower = cleaned.lower()
        is_relevant = (
            any(indicator in cleaned_lower for indicator in clothing_indicators) or
            any(indicator in cleaned_lower for indicator in style_indicators) or
            len(cleaned.split()) <= 4  # Short phrases are likely item names
        )
        
        if is_relevant:
            bold_headings.append(cleaned)
    
    # Remove duplicates while preserving order
    unique_headings = []
    seen = set()
    for heading in bold_headings:
        if heading.lower() not in seen:
            unique_headings.append(heading)
            seen.add(heading.lower())
    
    return unique_headings

def update_accumulated_keywords(keywords, user_context, is_user_input=False, is_ai_response=False, is_multi_item_request=False): # ADD THIS PARAMETER
    """
    Enhanced keyword update using FashionCategories for better categorization
    """
    from datetime import datetime
    
    print(f"\n📝 ENHANCED KEYWORD UPDATE WITH FASHION CATEGORIES")
    print("="*40)
    print(f"   🤝 Is Multi-Item Request (in update_accumulated_keywords): {is_multi_item_request}") # Debug
    
    if "accumulated_keywords" not in user_context:
        user_context["accumulated_keywords"] = {}
    
    # Extract budget separately (unchanged)
    if is_user_input and user_context.get("current_text_input"):
        text_input = user_context["current_text_input"].lower()
        
        budget_context_keywords = [
            'budget', 'anggaran', 'harga', 'price', 'biaya', 'cost',
            'maksimal', 'minimal', 'dibawah', 'diatas', 'under', 'over',
            'sekitar', 'around', 'kisaran', 'range', 'rupiah', 'rp', 'idr'
        ]
        
        has_budget_context = any(keyword in text_input for keyword in budget_context_keywords)
        
        if has_budget_context:
            budget_info = extract_budget_from_text(user_context["current_text_input"])
            if budget_info:
                user_context["budget_range"] = budget_info
                print(f"💰 Budget detected and set: {budget_info}")
            else:
                print(f"💰 Budget context found but no valid budget extracted")
        
        # Validate existing budget
        if "budget_range" in user_context and user_context["budget_range"]:
            budget = user_context["budget_range"]
            if isinstance(budget, tuple) and len(budget) == 2:
                min_price, max_price = budget
                
                if (min_price and min_price < 10000) or (max_price and max_price < 10000):
                    print(f"💰 CLEARING suspicious budget: {budget} (too low)")
                    user_context["budget_range"] = None

    # Enhanced persistence using FashionCategories
    persistence_config = {
        'clothing_items': {'decay_rate': 0.1, 'max_age_minutes': 120},
        'style_attributes': {'decay_rate': 0.15, 'max_age_minutes': 90},
        'colors': {'decay_rate': 0.2, 'max_age_minutes': 60},
        'gender_terms': {'decay_rate': 0.05, 'max_age_minutes': 240},
        'occasions': {'decay_rate': 0.4, 'max_age_minutes': 30},
        'default': {'decay_rate': 0.25, 'max_age_minutes': 45}
    }
    
    def get_keyword_category(keyword):
        """Determine keyword category using FashionCategories"""
        keyword_lower = keyword.lower()
        
        if fashion_categories.is_clothing_item(keyword):
            return 'clothing_items'
        elif fashion_categories.is_style_term(keyword) or any(term in keyword_lower for term in fashion_categories.SLEEVE_TERMS + fashion_categories.FIT_TERMS + fashion_categories.LENGTH_TERMS):
            return 'style_attributes'
        elif fashion_categories.is_color_term(keyword):
            return 'colors'
        elif fashion_categories.is_gender_term(keyword):
            return 'gender_terms'
        elif any(term in keyword_lower for term in fashion_categories.OCCASION_TERMS):
            return 'occasions'
        else:
            return 'default'
    
    updates_made = 0
    new_keywords_added = 0
    
    for keyword, score in keywords:
        if not keyword or len(keyword) < 2:
            continue
        
        keyword_lower = keyword.lower()
        category = get_keyword_category(keyword)
        
        # Convert score to frequency estimate
        if is_user_input:
            frequency_boost = 2.0
            estimated_frequency = max(1, score / 100)
        else:
            frequency_boost = 1.0
            estimated_frequency = max(1, score / 50)
        
        if keyword_lower in user_context["accumulated_keywords"]:
            # Update existing keyword
            data = user_context["accumulated_keywords"][keyword_lower]
            
            old_frequency = data.get("total_frequency", 1)
            new_frequency = old_frequency + (estimated_frequency * frequency_boost)
            
            # Category-aware weight calculation
            config = persistence_config.get(category, persistence_config['default'])
            base_weight = new_frequency * 30
            
            # Apply category multiplier using FashionCategories priorities
            if category == 'gender_terms':
                base_weight *= 0.5
            elif category == 'clothing_items':
                base_weight *= 1.5
            elif category == 'occasions':
                base_weight *= 0.7
            
            data["weight"] = base_weight
            data["total_frequency"] = new_frequency
            data["category"] = category
            data["mention_count"] = data.get("mention_count", 0) + 1
            data["last_seen"] = datetime.now().isoformat()
            
            updates_made += 1
            print(f"   📈 '{keyword}' ({category}) freq: {old_frequency:.1f} → {new_frequency:.1f}")
            
        else:
            # Add new keyword
            config = persistence_config.get(category, persistence_config['default'])
            initial_frequency = estimated_frequency * frequency_boost
            base_weight = initial_frequency * 30
            
            # Apply category multiplier for new keywords
            if category == 'gender_terms':
                base_weight *= 0.5
            elif category == 'clothing_items':
                base_weight *= 1.5
            elif category == 'occasions':
                base_weight *= 0.7
            
            user_context["accumulated_keywords"][keyword_lower] = {
                "weight": base_weight,
                "total_frequency": initial_frequency,
                "category": category,
                "mention_count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source": "user_input" if is_user_input else "ai_response"
            }
            new_keywords_added += 1
            print(f"   🆕 '{keyword}' ({category}) initial freq: {initial_frequency:.1f}")
    
    # Enhanced cleanup with category awareness, pass the multi-item flag
    category_cleanup(user_context, persistence_config, is_multi_item_request, keywords)
    
    print(f"📊 Updates: {updates_made}, New: {new_keywords_added}")
    print(f"📚 Total: {len(user_context['accumulated_keywords'])}")
    print("="*40)

def category_cleanup(user_context, persistence_config, is_multi_item_request=False, new_keywords_from_current_update=None): # ADD THIS NEW PARAMETER
    """
    Enhanced cleanup using FashionCategories for fashion category change detection compatibility
    Now takes `is_multi_item_request` into account to preserve more.
    """
    if "accumulated_keywords" not in user_context:
        return
    
    from datetime import datetime, timedelta
    current_time = datetime.now()
    
    # Use FashionCategories for change detection
    fashion_change_categories = fashion_categories.CLOTHING_CATEGORIES
    
    def is_change_detection_keyword(keyword):
        """Check if keyword is important for fashion category change detection"""
        return fashion_categories.get_clothing_category(keyword) is not None
    
    keywords_to_remove = []
    change_detection_keywords = {} # Keywords that are clothing items and should be preserved
    
    # NEW: Determine current requested clothing categories from the `new_keywords_from_current_update`
    current_requested_clothing_cats = set()
    if new_keywords_from_current_update:
        current_requested_clothing_cats = set(
            fashion_categories.get_clothing_category(kw)
            for kw, _ in new_keywords_from_current_update # CORRECTED: Iterate over the passed list of tuples
            if fashion_categories.get_clothing_category(kw) is not None
        )
    print(f"   👕 Current requested clothing categories for cleanup: {current_requested_clothing_cats}") # Debug

    # First pass: Identify which keywords should be considered "change detection keywords"
    # and apply initial decay/removal based on general rules.
    for keyword, data in user_context["accumulated_keywords"].items():
        category = data.get("category", "default")
        config = persistence_config.get(category, persistence_config["default"])
        
        is_clothing_item_keyword = is_change_detection_keyword(keyword) # Rename for clarity
        
        try:
            last_seen = datetime.fromisoformat(data.get("last_seen", data.get("first_seen", "")))
            minutes_since_last_seen = (current_time - last_seen).total_seconds() / 60
        except:
            minutes_since_last_seen = 999
        
        # Apply time-based decay based on category
        decay_factor = 1.0 # No decay initially
        if minutes_since_last_seen > 10: # Start decay after 10 minutes
            decay_factor = max(config['decay_rate'], 1 - (minutes_since_last_seen / (config['max_age_minutes'] * 1.5))) # Slower decay
        
        data["total_frequency"] *= decay_factor
        data["weight"] = data["total_frequency"] * 30
        
        # Apply category weight adjustment after decay
        if category == 'gender_terms':
            data["weight"] *= 0.5
        elif category == 'clothing_items':
            data["weight"] *= 1.5
        elif category == 'occasions':
            data["weight"] *= 0.7
        
        min_weight_threshold = 5 # Default threshold
        if category == 'gender_terms':
            min_weight_threshold = 2
        elif is_clothing_item_keyword:
            min_weight_threshold = 10 # Keep clothing items more persistently
        
        if is_clothing_item_keyword:
            # If multi-item request is active, preserve all clothing item keywords (don't remove yet)
            if is_multi_item_request:
                # Do NOT add to keywords_to_remove here for multi-item clothing items
                # We will handle their removal more carefully later based on overall context.
                if data["weight"] > min_weight_threshold: # Still apply a minimal threshold
                    change_detection_keywords[keyword] = {
                        'data': data,
                        'minutes_old': minutes_since_last_seen
                    }
                    print(f"   🔒 PRESERVING for multi-item: '{keyword}' (weight: {data['weight']:.1f})")
                else:
                    keywords_to_remove.append(keyword) # Remove very weak clothing items even in multi-item
            else: # Single-item request
                if minutes_since_last_seen > config["max_age_minutes"] * 1.5 or data["weight"] < min_weight_threshold:
                    keywords_to_remove.append(keyword)
                else:
                    change_detection_keywords[keyword] = {
                        'data': data,
                        'minutes_old': minutes_since_last_seen
                    }
        else: # Not a clothing item keyword
            if minutes_since_last_seen > config["max_age_minutes"] or data["weight"] < min_weight_threshold:
                keywords_to_remove.append(keyword)

    # Apply removals from the first pass
    for keyword in keywords_to_remove:
        if keyword in user_context["accumulated_keywords"] and keyword not in change_detection_keywords:
            del user_context["accumulated_keywords"][keyword]
            print(f"   🗑️ Removed by first pass: '{keyword}'")

    # Second pass: Apply category limits more intelligently, especially for multi-item
    category_limits = {
        'clothing_items': 15,  # Allow more clothing items for multi-item requests
        'style_attributes': 10,
        'colors': 8,
        'gender_terms': 2,
        'occasions': 5,
        'default': 8
    }
    
    by_category = defaultdict(list)
    for keyword, data in user_context["accumulated_keywords"].items():
        by_category[data.get("category", "default")].append((keyword, data))
    
    final_keywords = {}
    
    for category, items in by_category.items():
        limit = category_limits.get(category, 8)
        sorted_items = sorted(items, key=lambda x: x[1]["weight"], reverse=True)
        
        if category == 'clothing_items' and is_multi_item_request:
            # For multi-item, try to keep *all* relevant clothing items unless they are very weak
            # Prioritize the ones explicitly requested now.
            
            # Ensure newly requested clothing items are kept
            for keyword, data in sorted_items:
                kw_cat = fashion_categories.get_clothing_category(keyword)
                if kw_cat and kw_cat in current_requested_clothing_cats: # Check if it's one of the *currently requested* cats
                    final_keywords[keyword] = data
                    print(f"   ✅ Keeping newly requested clothing item: '{keyword}' ({kw_cat})")
            
            # Then add other high-ranking clothing items up to the limit
            added_count = len([kw for kw in final_keywords if fashion_categories.is_clothing_item(kw)])
            for keyword, data in sorted_items:
                if keyword not in final_keywords and added_count < limit:
                    final_keywords[keyword] = data
                    added_count += 1
                    print(f"   ➕ Keeping other high-rank clothing item: '{keyword}'")
        else:
            # For other categories, or for single-item clothing requests, apply normal limits
            for i, (keyword, data) in enumerate(sorted_items):
                if i < limit:
                    final_keywords[keyword] = data
    
    removed_count = len(user_context["accumulated_keywords"]) - len(final_keywords)
    user_context["accumulated_keywords"] = final_keywords
    
    if removed_count > 0:
        print(f"📉 Category limits applied: removed {removed_count} keywords")
    print(f"📚 Final keyword count after cleanup: {len(user_context['accumulated_keywords'])}")

def frequency_cleanup(user_context):
    """
    Clean up keywords based on frequency and usage patterns.
    """
    if "accumulated_keywords" not in user_context:
        return
    
    from datetime import datetime, timedelta
    current_time = datetime.now()
    
    keywords_to_remove = []
    
    for keyword, data in user_context["accumulated_keywords"].items():
        # Remove low-frequency keywords that haven't been mentioned recently
        total_freq = data.get("total_frequency", 0)
        mention_count = data.get("mention_count", 0)
        
        # Check last seen time
        try:
            last_seen = datetime.fromisoformat(data.get("last_seen", data.get("first_seen", "")))
            minutes_since_last_seen = (current_time - last_seen).total_seconds() / 60
        except:
            minutes_since_last_seen = 60  # Default to old if can't parse
        
        # Remove criteria based on frequency and recency
        should_remove = False
        
        # Remove very low frequency keywords that are old
        if total_freq < 2 and minutes_since_last_seen > 30:
            should_remove = True
            
        # Remove keywords mentioned only once and are old
        elif mention_count <= 1 and minutes_since_last_seen > 45:
            should_remove = True
            
        # Remove excluded terms that might have slipped through
        elif any(excluded in keyword for excluded in ['rb', 'ribu', 'jt', 'budget', 'kulit', 'skin']):
            should_remove = True
            
        # Remove compound phrases
        elif len(keyword.split()) > 2:
            should_remove = True
        
        if should_remove:
            keywords_to_remove.append(keyword)
    
    # Apply natural decay to remaining keywords over time
    for keyword, data in user_context["accumulated_keywords"].items():
        if keyword not in keywords_to_remove:
            try:
                last_seen = datetime.fromisoformat(data.get("last_seen", data.get("first_seen", "")))
                minutes_since_last_seen = (current_time - last_seen).total_seconds() / 60
                
                # Apply gentle decay over time
                if minutes_since_last_seen > 10:
                    decay_factor = max(0.5, 1 - (minutes_since_last_seen - 10) / 120)  # Decay over 2 hours
                    data["total_frequency"] *= decay_factor
                    data["weight"] = data["total_frequency"] * 30
            except:
                pass
    
    # Remove identified keywords
    for keyword in keywords_to_remove:
        del user_context["accumulated_keywords"][keyword]
    
    # Keep only top 25 most frequent keywords
    if len(user_context["accumulated_keywords"]) > 25:
        sorted_keywords = sorted(
            user_context["accumulated_keywords"].items(),
            key=lambda x: x[1].get("total_frequency", 0),
            reverse=True
        )
        user_context["accumulated_keywords"] = dict(sorted_keywords[:25])
    
    if keywords_to_remove:
        print(f"🧹 Removed {len(keywords_to_remove)} low-frequency/old keywords")

def clean_accumulated_keywords(user_context):
    """
    Clean up accumulated keywords to prevent bloat and maintain quality.
    """
    if "accumulated_keywords" not in user_context:
        return
    
    keywords_to_remove = []
    
    # Remove problematic keywords
    for keyword, data in user_context["accumulated_keywords"].items():
        # Remove budget-related terms
        if any(budget in keyword for budget in ['rb', 'ribu', 'jt', 'juta', '000', 'budget', 'anggaran']):
            keywords_to_remove.append(keyword)
        # Remove compound phrases that are too specific
        elif len(keyword.split()) > 2:
            keywords_to_remove.append(keyword)
        # Remove very low weight keywords
        elif data["weight"] < 5:
            keywords_to_remove.append(keyword)
    
    # Remove identified keywords
    for keyword in keywords_to_remove:
        del user_context["accumulated_keywords"][keyword]
    
    # Keep only top 20 keywords by weight
    if len(user_context["accumulated_keywords"]) > 20:
        sorted_keywords = sorted(
            user_context["accumulated_keywords"].items(),
            key=lambda x: x[1]["weight"],
            reverse=True
        )
        user_context["accumulated_keywords"] = dict(sorted_keywords[:20])
    
    if keywords_to_remove:
        print(f"🧹 Cleaned {len(keywords_to_remove)} problematic keywords")

def get_keyword_category_multiplier(keyword):
    """Return multiplier based on keyword category using FashionCategories to prioritize fashion over occasions"""
    keyword_lower = keyword.lower()
    
    # Use FashionCategories for consistent categorization
    if fashion_categories.is_clothing_item(keyword):
        return 4.0  # HIGHEST priority for clothing
    elif fashion_categories.is_style_term(keyword) or any(term in keyword_lower for term in fashion_categories.SLEEVE_TERMS + fashion_categories.FIT_TERMS + fashion_categories.LENGTH_TERMS):
        return 3.0  # HIGH priority for styles and attributes
    elif fashion_categories.is_color_term(keyword) or any(term in keyword_lower for term in fashion_categories.MATERIAL_TERMS):
        return 2.0  # MEDIUM priority for colors/materials
    elif any(term in keyword_lower for term in fashion_categories.OCCASION_TERMS):
        return 0.5  # LOW priority for occasions
    else:
        return 1.0  # Default

def detect_fashion_category_change(user_input, user_context, is_multi_item_request): # ADD THIS PARAMETER
    """
    Enhanced conflict resolution using FashionCategories structure
    """
    print(f"\n🔍 FASHION CATEGORY CHANGE DETECTION")
    print("="*50)
    
    user_input_lower = user_input.lower()

    # Use FashionCategories clothing categories
    clothing_types = fashion_categories.CLOTHING_CATEGORIES
    
    # Define conflicts using FashionCategories structure
    separates_categories = {'tops', 'bottoms_pants', 'bottoms_skirts', 'outerwear', 'shorts'} # Added shorts
    dress_category = {'dresses'}
    
    def get_clothing_type_from_keyword(keyword):
        return fashion_categories.get_clothing_category(keyword)
    
    # No longer needed due to _are_compatible_categories_for_multi
    # def is_true_type_conflict(cat1, cat2):
    #     if cat1 in separates_categories and cat2 in dress_category:
    #         return True
    #     if cat1 in dress_category and cat2 in separates_categories:
    #         return True
    #     return False
    
    # Analyze current input
    current_clothing_types = set()
    
    # Find clothing types in current input
    for category, terms in clothing_types.items():
        for term in terms:
            if term in user_input_lower:
                current_clothing_types.add(category)
                break
    
    # Analyze accumulated keywords
    accumulated_clothing_types = {}
    
    if "accumulated_keywords" in user_context:
        for keyword, data in user_context["accumulated_keywords"].items():
            clothing_type = get_clothing_type_from_keyword(keyword)
            if clothing_type:
                if clothing_type not in accumulated_clothing_types:
                    accumulated_clothing_types[clothing_type] = []
                accumulated_clothing_types[clothing_type].append((keyword, data["weight"]))
    
    print(f"📝 Current clothing types: {current_clothing_types}")
    print(f"📚 Accumulated clothing types: {list(accumulated_clothing_types.keys())}")
    print(f"🤝 Is multi-item request: {is_multi_item_request}") # Use the parameter here
    
    # Enhanced decision logic
    major_change_detected = False
    
    if current_clothing_types and accumulated_clothing_types:
        print(f"🔍 Checking for category conflicts...")
        
        acc_has_separates = bool(accumulated_clothing_types.keys() & separates_categories)
        acc_has_dresses = bool(accumulated_clothing_types.keys() & dress_category)
        curr_has_separates = bool(current_clothing_types & separates_categories)
        curr_has_dresses = bool(current_clothing_types & dress_category)
        
        # True conflict: separates ↔ dresses
        if (acc_has_separates and curr_has_dresses) or (acc_has_dresses and curr_has_separates):
            major_change_detected = True
            print(f"   ⚔️  TRUE CONFLICT: separates ↔ dresses")
        
        # Major switch: complete category abandonment (single item requests only)
        # If it's NOT a multi-item request AND the current input has no intersection
        # with the accumulated categories, then it's a major change.
        elif not is_multi_item_request and not current_clothing_types.intersection(accumulated_clothing_types.keys()):
            major_change_detected = True
            print(f"   🔄 CATEGORY ABANDONMENT (Single-item): {list(accumulated_clothing_types.keys())} → {list(current_clothing_types)}")
        
        else:
            print(f"   ✅ NO CONFLICT: Categories are compatible (or multi-item is active)")

    if major_change_detected:
        print(f"☢️ APPLYING NUCLEAR REDUCTION for major category conflict")
        
        nuclear_reduction_applied = 0
        for acc_type, keywords_in_type in accumulated_clothing_types.items():
            should_reduce = False
            
            # Reduce conflicting categories based on the current context (multi-item or single)
            if is_multi_item_request:
                # In multi-item, only reduce if the accumulated type is NOT compatible with ANY of the newly requested types
                if not any(fashion_categories._are_compatible_categories_for_multi(acc_type, new_cat) for new_cat in current_clothing_types):
                    should_reduce = True
            else: # Single-item mode
                # In single-item mode, if the accumulated type conflicts with any of the new types (and is not one of the new types)
                if any(fashion_categories._are_conflicting_categories(acc_type, new_cat) for new_cat in current_clothing_types) \
                   and acc_type not in current_clothing_types:
                    should_reduce = True
            
            if should_reduce:
                for keyword, weight in keywords_in_type:
                    if keyword in user_context["accumulated_keywords"]:
                        old_weight = user_context["accumulated_keywords"][keyword]["weight"]
                        new_weight = old_weight * 0.001  # 99.9% reduction
                        user_context["accumulated_keywords"][keyword]["weight"] = new_weight
                        nuclear_reduction_applied += 1
                        print(f"   ☢️  NUCLEAR: '{keyword}' {old_weight:.1f} → {new_weight:.1f}")
        
        print(f"   📊 Applied nuclear reduction to {nuclear_reduction_applied} conflicting keywords")
        return True

    print(f"✅ No major conflicts detected")
    return False

def normalize_weights_nuclear(user_context, user_input):
    """
    NUCLEAR: Extreme weight normalization
    """
    print(f"☢️  NUCLEAR WEIGHT NORMALIZATION")
    
    if "accumulated_keywords" not in user_context:
        return
    
    user_input_lower = user_input.lower()
    
    # Identify current vs old keywords
    current_keywords = []
    old_keywords = []
    
    for keyword, data in user_context["accumulated_keywords"].items():
        if keyword in user_input_lower:
            current_keywords.append(keyword)
        else:
            old_keywords.append(keyword)
    
    # Get the max weight of current keywords
    current_max = 0
    for keyword in current_keywords:
        if keyword in user_context["accumulated_keywords"]:
            current_max = max(current_max, user_context["accumulated_keywords"][keyword]["weight"])
    
    print(f"   🎯 Current keywords max weight: {current_max:.1f}")
    print(f"   📚 Current keywords: {current_keywords}")
    print(f"   🗂️  Old keywords: {len(old_keywords)}")
    
    # NUCLEAR: Cap old keywords to be maximum 0.1% of current keywords
    weight_cap = current_max * 0.001  # 0.1% of current max!
    
    capped_count = 0
    for keyword in old_keywords:
        if keyword in user_context["accumulated_keywords"]:
            current_weight = user_context["accumulated_keywords"][keyword]["weight"]
            if current_weight > weight_cap:
                user_context["accumulated_keywords"][keyword]["weight"] = weight_cap
                capped_count += 1
                print(f"   ☢️  NUCLEAR CAP: '{keyword}' {current_weight:.1f} → {weight_cap:.1f}")
    
    print(f"   📊 Nuclear capped {capped_count} old keywords to {weight_cap:.1f}")

def detect_occasion_change(user_input, accumulated_keywords):
    """
    Separate function to detect occasion-specific changes
    """
    user_input_lower = user_input.lower()
    
    occasion_terms = [
        'office', 'kantor', 'party', 'pesta', 'wedding', 'pernikahan', 
        'beach', 'pantai', 'sport', 'olahraga', 'work', 'kerja',
        'casual', 'formal', 'elegant'
    ]
    
    # Check if user input contains occasion terms
    current_occasion = None
    for term in occasion_terms:
        if term in user_input_lower:
            current_occasion = term
            break
    
    # Check if previous keywords had different occasions
    previous_occasion = None
    if accumulated_keywords:
        for keyword, _ in accumulated_keywords[:5]:
            keyword_lower = keyword.lower()
            for term in occasion_terms:
                if term in keyword_lower:
                    previous_occasion = term
                    break
            if previous_occasion:
                break
    
    # Only return True for significant occasion changes
    if current_occasion and previous_occasion and current_occasion != previous_occasion:
        print(f"🎪 OCCASION CHANGE: {previous_occasion} → {current_occasion}")
        return True
    
    return False

def apply_keyword_decay(user_context):
    """Apply decay to persistent keywords, especially occasions"""
    if "accumulated_keywords" not in user_context:
        return
    
    occasion_keywords = ['office', 'kantor', 'party', 'pesta', 'wedding', 'pernikahan', 'beach', 'pantai']
    
    decay_applied = 0
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        
        # Strong decay for occasion terms
        if any(occasion in keyword_lower for occasion in occasion_keywords):
            old_weight = data["weight"]
            data["weight"] *= 0.6  # 40% decay for occasions
            decay_applied += 1
            print(f"   🎪 Occasion decay: '{keyword}' {old_weight:.1f} → {data['weight']:.1f}")
        
        # Normal decay for other terms  
        elif data.get("source") != "user_input":  # Don't decay recent user input
            data["weight"] *= 0.85  # 15% decay for non-user terms
    
    if decay_applied > 0:
        print(f"⏰ Applied decay to {decay_applied} occasion keywords")

def detect_major_context_switch_in_update(user_input, user_context):
    """
    Detect major context switches during keyword updates.
    """
    if not user_input or "accumulated_keywords" not in user_context:
        return False
    
    user_input_lower = user_input.lower()
    
    # Fashion item categories for detection
    fashion_categories = {
        'tops': ['kemeja', 'shirt', 'kaos', 't-shirt', 'blouse', 'atasan', 'top'],
        'bottoms': ['celana', 'pants', 'rok', 'skirt', 'jeans'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'sweater', 'cardigan', 'hoodie'],
        'bags': ['tas', 'bag', 'handbag', 'backpack', 'clutch'],
        'shoes': ['sepatu', 'shoes', 'sneaker', 'heels', 'boots'],
        'accessories': ['hijab', 'scarf', 'belt', 'topi', 'hat']
    }
    
    # Find current categories in accumulated keywords
    current_categories = set()
    for keyword in user_context["accumulated_keywords"].keys():
        for category, terms in fashion_categories.items():
            if any(term in keyword for term in terms):
                current_categories.add(category)
                break
    
    # Find new categories in user input
    new_categories = set()
    for category, terms in fashion_categories.items():
        if any(term in user_input_lower for term in terms):
            new_categories.add(category)
    
    # Check for complete category switch
    if new_categories and current_categories:
        if not new_categories.intersection(current_categories):
            print(f"   🔄 Fashion category switch: {current_categories} → {new_categories}")
            return True
    
    # Check for explicit reset phrases
    reset_phrases = [
        'instead', 'not that', 'forget about', 'different',
        'ganti', 'bukan itu', 'lupakan', 'berbeda',
        'now show', 'sekarang', 'now i want', 'i want different'
    ]
    
    for phrase in reset_phrases:
        if phrase in user_input_lower:
            print(f"   🔄 Reset phrase detected: '{phrase}'")
            return True
    
    return False

def reset_accumulated_keywords_in_update(user_context, reason):
    """
    Reset accumulated keywords while preserving essential info.
    """
    print(f"🔄 RESETTING KEYWORDS (Reason: {reason})")
    
    # Preserve essential user attributes
    essential_keywords = {}
    if "accumulated_keywords" in user_context:
        essential_terms = [
            'perempuan', 'wanita', 'female', 'woman',
            'pria', 'laki-laki', 'male', 'man'
        ]
        
        for keyword, data in user_context["accumulated_keywords"].items():
            if any(essential in keyword.lower() for essential in essential_terms):
                essential_keywords[keyword] = {
                    "weight": data["weight"] * 0.1,  # Drastically reduce weight
                    "count": 1,
                    "first_seen": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "source": "preserved_essential"
                }
                print(f"   ✅ Preserved: '{keyword}' (reduced weight)")
    
    user_context["accumulated_keywords"] = essential_keywords
    
    # Clear product cache
    user_context["product_cache"] = {
        "all_results": pd.DataFrame(),
        "current_page": 0,
        "products_per_page": 5,
        "has_more": False
    }

def clean_old_keywords_in_update(user_context):
    """
    Clean old and low-weight keywords from accumulated context.
    """
    if "accumulated_keywords" not in user_context:
        return
    
    from datetime import datetime, timedelta
    current_time = datetime.now()
    
    keywords_to_remove = []
    WEIGHT_THRESHOLD = 3.0  # Remove keywords below this weight
    MAX_AGE_MINUTES = 45    # Remove keywords older than 45 minutes
    
    for keyword, data in user_context["accumulated_keywords"].items():
        # Check age
        try:
            if "last_seen" in data:
                last_seen = datetime.fromisoformat(data["last_seen"])
            else:
                last_seen = datetime.fromisoformat(data["first_seen"])
            
            age_minutes = (current_time - last_seen).total_seconds() / 60
            
            # Apply time decay
            if age_minutes > 20:  # Start decay after 20 minutes
                decay_factor = max(0.3, 1 - (age_minutes - 20) / 60)  # Gradual decay
                data["weight"] *= decay_factor
            
            # Remove very old keywords
            if age_minutes > MAX_AGE_MINUTES:
                keywords_to_remove.append(keyword)
                continue
                
        except:
            # Invalid timestamp, apply decay
            data["weight"] *= 0.5
        
        # Remove low weight keywords
        if data["weight"] < WEIGHT_THRESHOLD:
            keywords_to_remove.append(keyword)
    
    # Remove identified keywords
    removed_count = 0
    for keyword in keywords_to_remove:
        del user_context["accumulated_keywords"][keyword]
        removed_count += 1
    
    if removed_count > 0:
        print(f"🧹 Cleaned {removed_count} old/low-weight keywords")

def post_update_cleanup(user_context):
    """
    Final cleanup after keyword updates.
    """
    if "accumulated_keywords" not in user_context:
        return
    
    # Keep only top 40 keywords by weight
    MAX_KEYWORDS = 40
    
    if len(user_context["accumulated_keywords"]) > MAX_KEYWORDS:
        sorted_keywords = sorted(
            user_context["accumulated_keywords"].items(),
            key=lambda x: x[1]["weight"],
            reverse=True
        )
        
        top_keywords = dict(sorted_keywords[:MAX_KEYWORDS])
        removed_count = len(user_context["accumulated_keywords"]) - MAX_KEYWORDS
        
        user_context["accumulated_keywords"] = top_keywords
        print(f"📉 Kept top {MAX_KEYWORDS} keywords, removed {removed_count} lowest-weight")

def print_keyword_summary(user_context):
    """
    Print a summary of current keywords for debugging.
    """
    if "accumulated_keywords" not in user_context:
        return
    
    # Get top 8 keywords
    top_keywords = sorted(
        user_context["accumulated_keywords"].items(),
        key=lambda x: x[1]["weight"],
        reverse=True
    )[:8]
    
    print(f"\n📊 CURRENT TOP KEYWORDS:")
    for i, (keyword, data) in enumerate(top_keywords):
        source_icon = "🗣️" if data["source"] == "user_input" else "🤖" if data["source"] == "ai_response" else "✨"
        print(f"   {i+1}. {source_icon} '{keyword}' → {data['weight']:.1f} (count: {data['count']})")
    print()
    
def detect_and_update_gender(user_input, user_context, force_update=False):
    """
    Detect gender from user input and update context using FashionCategories.
    """
    current_gender = user_context.get("user_gender", {})
    has_existing_gender = current_gender.get("category") is not None
    
    if has_existing_gender and not force_update:
        print(f"👤 Using existing gender: {current_gender['category']} (confidence: {current_gender.get('confidence', 0):.1f})")
        return current_gender["category"]
    
    # Use FashionCategories gender terms
    gender_patterns = {
        'male': [
            r'\b(' + '|'.join([term for term in fashion_categories.GENDER_TERMS if term in ['pria', 'laki-laki', 'male', 'man', 'cowok', 'cowo']]) + r')\b',
        ],
        'female': [
            r'\b(' + '|'.join([term for term in fashion_categories.GENDER_TERMS if term in ['perempuan', 'wanita', 'female', 'woman', 'cewek', 'cewe']]) + r')\b',
        ]
    }
    
    user_input_lower = user_input.lower()
    detected_gender = None
    detected_term = None
    confidence = 0
    
    # Check for gender patterns
    for gender, patterns in gender_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                detected_gender = gender
                detected_term = match.group(1) if match.lastindex else match.group(0)
                confidence = 10.0
                break
        if detected_gender:
            break
    
    # Update gender if detected
    if detected_gender:
        user_context["user_gender"] = {
            "category": detected_gender,
            "term": detected_term,
            "confidence": confidence,
            "last_updated": datetime.now().isoformat()
        }
        print(f"👤 Gender detected and saved: {detected_gender} (term: {detected_term}, confidence: {confidence})")
        return detected_gender
    
    if has_existing_gender:
        print(f"👤 No new gender detected, using existing: {current_gender['category']}")
        return current_gender["category"]
    
    print("👤 No gender detected")
    return None

#Caching image analysis
image_analysis_cache = {}  

async def analyze_uploaded_image(image_url: str):

    try:
        if not image_url:
            return "Error: No image URL provided."
        
        logging.info(f"Analyzing image at URL: {image_url}")
        print(f"Analyzing image URL at: {image_url}")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create (
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
                #Cached the result 
                image_analysis_cache[image_url] = analysis

                return analysis
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.info(f"Failed to analyze image at URL: {image_url}. Error: {str(e)}\n{traceback.format_exc()}")
                    print(f"Failed to analyze image at URL: {image_url}. Error: {str(e)}\n{traceback.format_exc()}")
                    return f"Error: Unable to analyse image. Please try again or use text description instead."
                else:
                    logging.info(f"Retrying image analysis after error: {str(e)}")
                    await asyncio.sleep(2) #wait before retry

    except Exception as e:
        print(f"Error during image analysis: {e}")
        return f"Error: {str(e)}"