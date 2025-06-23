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

stop_words = set([
    "a", "an", "and", "are", "as", "at", "be", "but", "by", 
    "for", "if", "in", "into", "is", "it", "its", "of", "on", 
    "or", "so", "such", "that", "the", "their", "then", 
    "there", "these", "they", "this", "to", "too", "was", 
    "will", "with", "you", "your", "do", "does", "did", 
    "have", "has", "having", "we", "us", "our", "ours", 
    "I", "me", "my", "mine", "he", "him", "his", "she", 
    "her", "hers", "itself", "themselves", "yourself", 
    "yourselves", "those", "from", "which", "or", "any", 
    "all", "some", "each", "every", "one", "once", "while", 
    "when", "where", "how", "what", "why", "about", "like", 
    "over", "under", "more", "less", "up", "down", "out", 
    "around", "just", "only", "even", "always", "never", 
    "not", "can", "could", "should", "would", "might", 
    "must", "shall", "may", "perhaps", "often", "sometimes",
    "always", "usually", "great", "very", "really", "sure", "color", "colour",
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan", 
    "sebagai", "adalah", "atau", "itu", "ini", "tidak", "bukan", 
    "sudah", "belum", "akan", "masih", "juga", "hanya", "namun", 
    "sangat", "lebih", "kurang", "ada", "dalam", "oleh", "karena", 
    "tersebut", "kemudian", "jadi", "sehingga", "agar", "supaya", 
    "bahwa", "kalau", "apa", "siapa", "bagaimana", "kapan", "dimana", 
    "mengapa", "dengan", "tetapi", "meskipun", "sebab", "hingga", 
    "untuk", "akan", "terhadap", "antara", "sesuatu", "kita", "kami", 
    "mereka", "dia", "itu", "saya", "aku", "anda", "kau", "kamu", 
    "nya", "lah", "pun", "lagi", "pernah", "sedang", "begitu", "seperti",
    "saja", "hingga", "harus", "bisa", "dapat", "mungkin", "sering", 
    "selalu", "jarang", "tampilan", "memberikan", "warna", "potongan", 
    "sebuah", "pilih", "menarik", "pilihlah", "carikan", "gaya", "menjadi"
])

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

def extract_ranked_keywords(ai_response: str = None, translated_input: str = None, accumulated_keywords=None):
    """
    ENHANCED: Context-aware keyword extraction with physical description filtering.
    """
    print("\n" + "="*60)
    print("🔤 ENHANCED KEYWORD EXTRACTION WITH CONTEXT AWARENESS")
    print("="*60)
    
    keyword_scores = {}
    global_exclusions = set()

    # Simple responses filter
    simple_responses = {
        "yes", "ya", "iya", "oke", "ok", "okay", "sure", "tentu",
        "no", "tidak", "nope", "ga", "gak", "engga", "nah",
        "good", "bagus", "nice", "baik", "great", "mantap",
        "thanks", "terima", "kasih", "makasih", "thx"
    }
    
    # Initialize variables
    current_input_categories = set()
    wanted_items = []
    context_items = []
    
    # Define clothing categories
    clothing_categories = {
        'kemeja': ['kemeja', 'shirt', 'blouse', 'blus'],
        'celana': ['celana', 'pants', 'jeans', 'trousers', 'ankle pants'],
        'dress': ['dress', 'gaun', 'terusan'],
        'rok': ['rok', 'skirt'],
        'jaket': ['jaket', 'jacket', 'blazer', 'coat'],
        'kaos': ['kaos', 't-shirt', 'tshirt', 'tank'],
        'atasan': ['atasan', 'top', 'blouse'],
        'sweater': ['sweater', 'cardigan', 'hoodie']
    }
    
    def get_clothing_category(keyword):
        """Get clothing category for a keyword"""
        keyword_lower = keyword.lower()
        for category, terms in clothing_categories.items():
            if any(term in keyword_lower for term in terms):
                return category
        return None
    
    # Extract specific clothing request if possible
    if translated_input:
        try:
            wanted_items, context_items = extract_specific_clothing_request(translated_input, ai_response)
        except:
            # If function doesn't exist, try simple detection
            wanted_items, context_items = [], []
            for category, terms in clothing_categories.items():
                for term in terms:
                    if term in translated_input.lower():
                        if any(indicator in translated_input.lower() for indicator in ['apa', 'what', 'carikan', 'tunjukkan', 'show']):
                            wanted_items.append(category)
                        else:
                            context_items.append(category)
    
    # REVISED: Rebalanced scoring - Clothing type first, then equal features
    scoring_categories = {
        'clothing_items': {
            'terms': ['kemeja', 'shirt', 'blouse', 'blus', 'dress', 'gaun', 'rok', 'skirt',
                    'celana', 'pants', 'jeans', 'jacket', 'jaket', 'sweater', 'cardigan',
                    'atasan', 'top', 'kaos', 't-shirt', 'hoodie', 'blazer', 'coat', 'ankle pants'],
            'user_score': 400,      # HIGHEST PRIORITY - Clothing types
            'ai_score': 500,        
            'priority': 'HIGHEST'
        },
        'style_attributes': {
            'terms': ['lengan panjang', 'lengan pendek', 'long sleeve', 'short sleeve',
                    'panjang', 'long', 'pendek', 'short', 'oversized', 'slim', 'regular', 
                    'loose', 'ketat', 'longgar', 'tight', 'fitted', 'relaxed'],
            'user_score': 200,      # EQUAL with colors and materials
            'ai_score': 220,        
            'priority': 'HIGH'
        },
        'colors': {
            'terms': ['white', 'putih', 'black', 'hitam', 'red', 'merah', 'blue', 'biru',
                    'green', 'hijau', 'yellow', 'kuning', 'brown', 'coklat', 'pink',
                    'purple', 'ungu', 'orange', 'oranye', 'grey', 'abu-abu', 'navy', 'beige'],
            'user_score': 200,      # EQUAL with style attributes - REDUCED from 150
            'ai_score': 220,        
            'priority': 'HIGH'
        },
        'materials_fit': {
            'terms': ['cotton', 'katun', 'silk', 'sutra', 'denim', 'wool', 'wol', 
                    'polyester', 'linen', 'leather', 'kulit', 'casual', 'formal', 'elegant'],
            'user_score': 200,      # EQUAL with colors and style attributes
            'ai_score': 220,
            'priority': 'HIGH'
        },
        'gender_terms': {
            'terms': ['perempuan', 'wanita', 'female', 'woman', 'pria', 'laki-laki', 'male', 'man'],
            'user_score': 50,       # LOW - Gender is just a filter
            'ai_score': 30,
            'priority': 'FILTER'
        },
        'occasions': {
            'terms': ['office', 'kantor', 'party', 'pesta', 'wedding', 'pernikahan',
                    'beach', 'pantai', 'sport', 'olahraga', 'work', 'kerja'],
            'user_score': 150,      # LOWER than features
            'ai_score': 170,
            'priority': 'MEDIUM'
        }
    }
    
    def get_keyword_score(keyword, source, frequency=1):
        """Get appropriate score with BALANCED WEIGHTING"""
        keyword_lower = keyword.lower()
        
        base_score = 0
        priority = 'DEFAULT'
        
        for category, config in scoring_categories.items():
            if any(term in keyword_lower for term in config['terms']):
                if source == 'user':
                    base_score = config['user_score'] * frequency
                elif source == 'ai':
                    base_score = config['ai_score'] * frequency
                else:
                    base_score = config['user_score'] * frequency * 0.5
                priority = config['priority']
                break
        
        if base_score == 0:
            if source == 'user':
                base_score = 100 * frequency
            elif source == 'ai':
                base_score = 120 * frequency
            else:
                base_score = 50 * frequency
            priority = 'DEFAULT'
        
        # Moderate specific request boost
        clothing_category = get_clothing_category(keyword)
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
    
    def is_physical_description(text):
        """Check if text contains physical description"""
        text_lower = text.lower()
        
        physical_indicators = [
            r'\b(?:kulit|skin)\s+\w+',
            r'\bberkulit\s+\w+',
            r'\bwarna\s+kulit',
            r'\bskin\s+tone',
            r'\b\d+\s*(?:cm|kg|tahun|years?)',
            r'\b(?:tinggi|height|berat|weight|umur|age)',
            r'\b(?:dari|from)\s+(?:indonesia|malaysia|singapore|thailand)',
            r'\b(?:cowo|cowok|cewe|cewek|pria|wanita|laki-laki|perempuan)',
            r'\b(?:suka|like)\s+(?:olahraga|sport|gym|fitness)',
        ]
        
        for pattern in physical_indicators:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def filter_skin_colors(keyword):
        """Filter out skin color mentions"""
        keyword_lower = keyword.lower()
        
        # Check for skin color contexts
        skin_color_patterns = [
            r'\b(?:kulit|skin)\s+([a-zA-Z]+)\b',
            r'\b([a-zA-Z]+)\s+(?:kulit|skin)\b',
            r'\bberkulit\s+([a-zA-Z]+)\b',
            r'\b(?:skin\s+tone|warna\s+kulit)\s+([a-zA-Z]+)\b',
        ]
        
        for pattern in skin_color_patterns:
            if re.search(pattern, keyword_lower):
                print(f"      🚫 SKIN COLOR FILTERED: '{keyword}' - not a clothing color")
                return True
        
        # Check if "tan" appears in physical context
        if keyword_lower == 'tan' and any(phys in keyword_lower for phys in ['kulit', 'skin']):
            print(f"      🚫 SKIN COLOR FILTERED: '{keyword}' - tan in skin context")
            return True
        
        return False
    
    # Process user input with physical description filtering
    if translated_input:
        print(f"📝 USER INPUT: '{translated_input}'")
        
        # Check for simple response
        input_words = translated_input.lower().split()
        is_simple_response = (
            len(input_words) <= 2 and 
            all(word in simple_responses for word in input_words)
        )
        
        if is_simple_response:
            print(f"   ⚠️  SIMPLE RESPONSE DETECTED - Skipping")
            return []
        
        # NEW: Check if input is primarily physical description
        if is_physical_description(translated_input):
            print(f"   🚫 PHYSICAL DESCRIPTION DETECTED - Filtering out non-clothing terms")
            
            # Split input into clothing vs physical parts
            clothing_terms = []
            physical_terms = []
            
            for word in translated_input.split():
                if any(clothing in word.lower() for clothing in ['kemeja', 'shirt', 'celana', 'pants', 'lengan', 'sleeve', 'oversized', 'loose', 'tight', 'kaos', 'dress', 'gaun', 'rok', 'jaket', 'pendek', 'panjang']):
                    clothing_terms.append(word)
                elif any(physical in word.lower() for physical in ['kulit', 'skin', 'cm', 'kg', 'tinggi', 'berat', 'dari', 'indonesia', 'cowo', 'cewe', 'cowok', 'cewek', 'pria', 'wanita', 'tahun', 'umur']):
                    physical_terms.append(word)
                else:
                    clothing_terms.append(word)  # Default to clothing context
            
            print(f"   👕 CLOTHING TERMS: {clothing_terms}")
            print(f"   🚫 PHYSICAL TERMS: {physical_terms}")
            
            # Only process clothing terms
            filtered_input = ' '.join(clothing_terms)
            doc = nlp(filtered_input) if filtered_input.strip() else nlp("")
        else:
            doc = nlp(translated_input)
        
        # Extract keywords using spaCy
        user_keywords = {}
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and 
                len(token.text) > 2 and 
                not token.text.isdigit() and
                token.is_alpha and
                token.text.lower() not in simple_responses):
                
                keyword = token.text.lower()
                
                # Filter out skin colors
                if filter_skin_colors(keyword):
                    continue
                
                user_keywords[keyword] = user_keywords.get(keyword, 0) + 1
                
                # Track clothing categories
                clothing_cat = get_clothing_category(keyword)
                if clothing_cat:
                    current_input_categories.add(clothing_cat)
        
        # Score user keywords
        for keyword, frequency in user_keywords.items():
            score, priority = get_keyword_score(keyword, 'user', frequency)
            keyword_scores[keyword] = score
            
            print(f"   📌 '{keyword}' (freq: {frequency}) → {score} ({priority})")
            
            # Get translation expansion and exclusions
            try:
                search_terms = get_search_terms_for_keyword(keyword)
                if isinstance(search_terms, dict):
                    include_terms = search_terms.get('include', [])
                    exclude_terms = search_terms.get('exclude', [])
                    
                    for include_term in include_terms:
                        if include_term != keyword and include_term not in keyword_scores:
                            expansion_score = score * 0.7
                            
                            # Apply wanted item boost to expansions
                            expansion_clothing_cat = get_clothing_category(include_term)
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
    
    # Process AI response (HIGH PRIORITY)
    if ai_response:
        print(f"\n🤖 AI RESPONSE processing (HIGH PRIORITY)...")
        
        bold_headings = extract_bold_headings_from_ai_response(ai_response)
        print(f"   📋 Found {len(bold_headings)} bold headings: {bold_headings}")
        
        for heading in bold_headings:
            heading_lower = heading.lower()
            cleaned_heading = re.sub(r'[^\w\s-]', '', heading_lower).strip()
            
            if cleaned_heading and len(cleaned_heading) > 2:
                score, priority = get_keyword_score(cleaned_heading, 'ai', 3)
                
                # Always add AI headings
                if cleaned_heading not in keyword_scores:
                    keyword_scores[cleaned_heading] = score
                else:
                    keyword_scores[cleaned_heading] = max(keyword_scores[cleaned_heading], score)
                
                print(f"   🔥 BOLD HEADING: '{cleaned_heading}' → {score} ({priority})")
                
                # Track clothing categories
                clothing_cat = get_clothing_category(cleaned_heading)
                if clothing_cat:
                    current_input_categories.add(clothing_cat)
                
                # Expand bold headings
                try:
                    search_terms = get_search_terms_for_keyword(cleaned_heading)
                    if isinstance(search_terms, dict):
                        include_terms = search_terms.get('include', [])
                        exclude_terms = search_terms.get('exclude', [])
                        
                        for include_term in include_terms:
                            if include_term not in keyword_scores:
                                expansion_score = score * 0.8
                                
                                expansion_clothing_cat = get_clothing_category(include_term)
                                if expansion_clothing_cat and expansion_clothing_cat in wanted_items:
                                    expansion_score *= 1.5
                                    print(f"      ➕ BOOSTED AI expansion: '{cleaned_heading}' → '{include_term}' ({expansion_score:.1f})")
                                else:
                                    print(f"      ➕ AI expansion: '{cleaned_heading}' → '{include_term}' ({expansion_score:.1f})")
                                
                                keyword_scores[include_term] = expansion_score
                        
                        if exclude_terms:
                            global_exclusions.update(exclude_terms)
                except:
                    pass
    
    # Process accumulated keywords
    if accumulated_keywords:
        print(f"\n📚 ACCUMULATED keywords...")
        
        for keyword, old_weight in accumulated_keywords[:10]:
            if (keyword and len(keyword) > 2 and 
                keyword.lower() not in simple_responses and
                not any(char.isdigit() for char in keyword)):
                
                # Filter out skin colors from accumulated keywords too
                if filter_skin_colors(keyword):
                    continue
                
                accumulated_score = old_weight * 0.4
                
                if keyword not in keyword_scores and accumulated_score > 15:
                    keyword_scores[keyword] = accumulated_score
                    print(f"   📜 '{keyword}' → {accumulated_score:.1f}")
    
    # Clean up unwanted terms
    excluded_terms = [
        # Budget and numbers
        "rb", "ribu", "rupiah", "budget", "anggaran", "harga", "price",
        
        # Generic conversation words
        "yang", "dan", "atau", "dengan", "untuk", "dari", "pada", "akan",
        "dapat", "ada", "adalah", "ini", "itu", "saya", "anda", "kamu",
        "mereka", "dia", "sangat", "lebih", "kurang", "bagus", "baik",
        "cocok", "sesuai", "tepat", "bisa", "juga", "hanya", "sudah",
        
        # AI response fillers
        "recommendation", "rekomendasi", "suggestion", "saran", "option",
        "pilihan", "choice", "style", "gaya", "tampilan", "fit",
        
        # Physical descriptors
        "kulit", "skin", "tubuh", "body", "tinggi", "height", "berat", "weight",
        "cowo", "cewe", "pria", "wanita", "indonesia", "dari", "cm", "kg"
    ]
    
    cleanup_keywords = []
    for keyword in list(keyword_scores.keys()):
        if (keyword in excluded_terms or 
            len(keyword.split()) > 3 or
            len(keyword) <= 2):
            cleanup_keywords.append(keyword)
    
    for keyword in cleanup_keywords:
        del keyword_scores[keyword]
        print(f"   🗑️ Cleaned: '{keyword}'")
    
    # Sort and return
    ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 FINAL CONTEXT-AWARE KEYWORDS:")
    for i, (keyword, score) in enumerate(ranked_keywords[:15]):
        clothing_cat = get_clothing_category(keyword)
        
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
    
    print(f"\n📊 CONTEXT-AWARE SUMMARY:")
    print(f"   🎯 Wanted items: {wanted_items}")
    print(f"   📝 Context items: {context_items}")
    ai_high = len([k for k, s in ranked_keywords if s >= 500])
    user_high = len([k for k, s in ranked_keywords if 300 <= s < 500])
    medium = len([k for k, s in ranked_keywords if 150 <= s < 300])
    
    print(f"   🤖 AI high priority (500+): {ai_high}")
    print(f"   👤 User high priority (300-499): {user_high}")
    print(f"   📋 Medium priority (150-299): {medium}")
    print(f"   📝 Total keywords: {len(ranked_keywords)}")
    print("="*60)
    
    # Store results
    extract_ranked_keywords.last_exclusions = list(global_exclusions)
    extract_ranked_keywords.wanted_items = wanted_items
    extract_ranked_keywords.context_items = context_items
    
    return ranked_keywords[:15]

def get_search_terms_for_keyword(keyword):
    """
    Complete version with ALL keywords from the original function, enhanced with exclusions.
    Get both English and Indonesian search terms for a keyword to improve product matching.
    Returns a dictionary with 'include' and 'exclude' terms.
    """
    keyword_lower = keyword.lower().strip()
    
    # Complete translation mapping with ALL original keywords plus exclusions
    translation_map = {
        # Clothing types - WITH PROPER EXCLUSIONS
        'shirt': {
            'include': ['shirt', 'kemeja', 'baju', 'atasan'],
            'exclude': ['t-shirt', 'tshirt', 'kaos', 'baju kaos', 'tank top', 'polo']
        },
        'kemeja': {
            'include': ['kemeja', 'shirt', 'baju', 'atasan'],
            'exclude': ['t-shirt', 'tshirt', 'kaos', 'baju kaos', 'tank top', 'polo']
        },
        'blouse': {
            'include': ['blouse', 'blus', 'kemeja wanita', 'atasan wanita'],
            'exclude': ['t-shirt', 'kaos', 'tank top']
        },
        'blus': {
            'include': ['blus', 'blouse', 'kemeja wanita'],
            'exclude': ['t-shirt', 'kaos', 'tank top']
        },
        'dress': {
            'include': ['dress', 'gaun', 'terusan'],
            'exclude': ['shirt', 'kemeja', 'top', 'atasan']
        },
        'gaun': {
            'include': ['gaun', 'dress', 'terusan'],
            'exclude': ['shirt', 'kemeja', 'top', 'atasan']
        },
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
        'kardigan': {
            'include': ['kardigan', 'cardigan'],
            'exclude': []
        },
        'jeans': {
            'include': ['jeans', 'jins', 'celana jeans', 'denim'],
            'exclude': []
        },
        'hoodie': {
            'include': ['hoodie', 'jaket hoodie', 'sweater hoodie'],
            'exclude': []
        },
        'coat': {
            'include': ['coat', 'mantel', 'jaket panjang'],
            'exclude': []
        },
        'mantel': {
            'include': ['mantel', 'coat', 'jaket panjang'],
            'exclude': []
        },
        'blazer': {
            'include': ['blazer', 'jas blazer'],
            'exclude': []
        },
        'top': {
            'include': ['top', 'atasan', 'baju atas'],
            'exclude': []
        },
        'atasan': {
            'include': ['atasan', 'top', 'baju atas'],
            'exclude': []
        },
        # T-SHIRTS - SEPARATE CATEGORY (this was missing proper exclusions)
        't-shirt': {
            'include': ['t-shirt', 'tshirt', 'kaos', 'baju kaos'],
            'exclude': ['kemeja', 'shirt', 'dress shirt', 'button shirt', 'formal shirt']
        },
        'kaos': {
            'include': ['kaos', 't-shirt', 'tshirt', 'baju kaos'],
            'exclude': ['kemeja', 'shirt', 'dress shirt', 'button shirt', 'formal shirt']
        },
        
        # Colors - ALL ORIGINAL COLORS
        'white': {
            'include': ['white', 'putih'],
            'exclude': ['black', 'hitam']
        },
        'putih': {
            'include': ['putih', 'white'],
            'exclude': ['hitam', 'black']
        },
        'black': {
            'include': ['black', 'hitam'],
            'exclude': ['white', 'putih']
        },
        'hitam': {
            'include': ['hitam', 'black'],
            'exclude': ['putih', 'white']
        },
        'red': {
            'include': ['red', 'merah'],
            'exclude': []
        },
        'merah': {
            'include': ['merah', 'red'],
            'exclude': []
        },
        'blue': {
            'include': ['blue', 'biru'],
            'exclude': []
        },
        'biru': {
            'include': ['biru', 'blue'],
            'exclude': []
        },
        'green': {
            'include': ['green', 'hijau'],
            'exclude': []
        },
        'hijau': {
            'include': ['hijau', 'green'],
            'exclude': []
        },
        'yellow': {
            'include': ['yellow', 'kuning'],
            'exclude': []
        },
        'kuning': {
            'include': ['kuning', 'yellow'],
            'exclude': []
        },
        'pink': {
            'include': ['pink', 'merah muda', 'rosa'],
            'exclude': []
        },
        'purple': {
            'include': ['purple', 'ungu', 'violet'],
            'exclude': []
        },
        'ungu': {
            'include': ['ungu', 'purple', 'violet'],
            'exclude': []
        },
        'orange': {
            'include': ['orange', 'oranye', 'jingga'],
            'exclude': []
        },
        'oranye': {
            'include': ['oranye', 'orange', 'jingga'],
            'exclude': []
        },
        'brown': {
            'include': ['brown', 'coklat', 'cokelat'],
            'exclude': []
        },
        'coklat': {
            'include': ['coklat', 'brown', 'cokelat'],
            'exclude': []
        },
        'grey': {
            'include': ['grey', 'gray', 'abu-abu'],
            'exclude': []
        },
        'gray': {
            'include': ['gray', 'grey', 'abu-abu'],
            'exclude': []
        },
        'navy': {
            'include': ['navy', 'biru tua', 'biru dongker'],
            'exclude': []
        },
        'beige': {
            'include': ['beige', 'krem', 'cream'],
            'exclude': []
        },
        'krem': {
            'include': ['krem', 'beige', 'cream'],
            'exclude': []
        },
        
        # Styles - ALL ORIGINAL STYLES
        'casual': {
            'include': ['casual', 'santai', 'kasual'],
            'exclude': []
        },
        'santai': {
            'include': ['santai', 'casual', 'kasual'],
            'exclude': []
        },
        'kasual': {
            'include': ['kasual', 'casual', 'santai'],
            'exclude': []
        },
        'formal': {
            'include': ['formal', 'resmi'],
            'exclude': []
        },
        'resmi': {
            'include': ['resmi', 'formal'],
            'exclude': []
        },
        'elegant': {
            'include': ['elegant', 'elegan'],
            'exclude': []
        },
        'elegan': {
            'include': ['elegan', 'elegant'],
            'exclude': []
        },
        'modern': {
            'include': ['modern', 'kontemporer'],
            'exclude': []
        },
        'vintage': {
            'include': ['vintage', 'klasik', 'retro'],
            'exclude': []
        },
        'klasik': {
            'include': ['klasik', 'vintage', 'retro'],
            'exclude': []
        },
        'bohemian': {
            'include': ['bohemian', 'boho'],
            'exclude': []
        },
        'boho': {
            'include': ['boho', 'bohemian'],
            'exclude': []
        },
        'minimalist': {
            'include': ['minimalist', 'minimalis', 'simple'],
            'exclude': []
        },
        'minimalis': {
            'include': ['minimalis', 'minimalist', 'simple'],
            'exclude': []
        },
        'feminine': {
            'include': ['feminine', 'feminin'],
            'exclude': []
        },
        'feminin': {
            'include': ['feminin', 'feminine'],
            'exclude': []
        },
        'masculine': {
            'include': ['masculine', 'maskulin'],
            'exclude': []
        },
        'maskulin': {
            'include': ['maskulin', 'masculine'],
            'exclude': []
        },
        'ethnic': {
            'include': ['ethnic', 'etnik', 'tradisional'],
            'exclude': []
        },
        'etnik': {
            'include': ['etnik', 'ethnic', 'tradisional'],
            'exclude': []
        },
        'streetwear': {
            'include': ['streetwear', 'jalanan'],
            'exclude': []
        },
        'oversized': {
            'include': ['oversized', 'longgar', 'besar'],
            'exclude': []
        },
        'longgar': {
            'include': ['longgar', 'oversized', 'loose'],
            'exclude': []
        },
        'slim': {
            'include': ['slim', 'ketat', 'fit'],
            'exclude': []
        },
        'ketat': {
            'include': ['ketat', 'slim', 'tight'],
            'exclude': []
        },
        
        # Materials - ALL ORIGINAL MATERIALS
        'cotton': {
            'include': ['cotton', 'katun'],
            'exclude': []
        },
        'katun': {
            'include': ['katun', 'cotton'],
            'exclude': []
        },
        'silk': {
            'include': ['silk', 'sutra'],
            'exclude': []
        },
        'sutra': {
            'include': ['sutra', 'silk'],
            'exclude': []
        },
        'wool': {
            'include': ['wool', 'wol'],
            'exclude': []
        },
        'wol': {
            'include': ['wol', 'wool'],
            'exclude': []
        },
        'linen': {
            'include': ['linen', 'linen'],
            'exclude': []
        },
        'polyester': {
            'include': ['polyester', 'poliester'],
            'exclude': []
        },
        'leather': {
            'include': ['leather', 'kulit'],
            'exclude': []
        },
        'kulit': {
            'include': ['kulit', 'leather'],
            'exclude': []
        },
        'denim': {
            'include': ['denim', 'jeans'],
            'exclude': []
        },
        'knit': {
            'include': ['knit', 'rajut'],
            'exclude': []
        },
        'rajut': {
            'include': ['rajut', 'knit'],
            'exclude': []
        },
        'satin': {
            'include': ['satin'],
            'exclude': []
        },
        'velvet': {
            'include': ['velvet', 'beludru'],
            'exclude': []
        },
        'beludru': {
            'include': ['beludru', 'velvet'],
            'exclude': []
        },
        
        # Features - ALL ORIGINAL FEATURES
        'sleeve': {
            'include': ['sleeve', 'lengan'],
            'exclude': []
        },
        'lengan': {
            'include': ['lengan', 'sleeve'],
            'exclude': []
        },
        'collar': {
            'include': ['collar', 'kerah'],
            'exclude': []
        },
        'kerah': {
            'include': ['kerah', 'collar'],
            'exclude': []
        },
        'pocket': {
            'include': ['pocket', 'kantong', 'saku'],
            'exclude': []
        },
        'kantong': {
            'include': ['kantong', 'pocket', 'saku'],
            'exclude': []
        },
        'button': {
            'include': ['button', 'kancing'],
            'exclude': []
        },
        'kancing': {
            'include': ['kancing', 'button'],
            'exclude': []
        },
        'zipper': {
            'include': ['zipper', 'resleting'],
            'exclude': []
        },
        'resleting': {
            'include': ['resleting', 'zipper'],
            'exclude': []
        },
        'embroidery': {
            'include': ['embroidery', 'bordir'],
            'exclude': []
        },
        'bordir': {
            'include': ['bordir', 'embroidery'],
            'exclude': []
        },
        'pattern': {
            'include': ['pattern', 'motif', 'pola'],
            'exclude': []
        },
        'motif': {
            'include': ['motif', 'pattern', 'pola'],
            'exclude': []
        },
        'print': {
            'include': ['print', 'cetak'],
            'exclude': []
        },
        'colorful': {
            'include': ['colorful', 'berwarna', 'warni'],
            'exclude': []
        },
        'berwarna': {
            'include': ['berwarna', 'colorful', 'warni'],
            'exclude': []
        },
        'plain': {
            'include': ['plain', 'polos'],
            'exclude': []
        },
        'polos': {
            'include': ['polos', 'plain'],
            'exclude': []
        },
        'lace': {
            'include': ['lace', 'renda'],
            'exclude': []
        },
        'renda': {
            'include': ['renda', 'lace'],
            'exclude': []
        },
        'panjang': {
            'include': ['panjang', 'long', 'maxi'],
            'exclude': ['pendek', 'short', 'mini']
        },
        'long': {
            'include': ['long', 'panjang', 'maxi'],
            'exclude': ['pendek', 'short', 'mini']
        },
        'maxi': {
            'include': ['maxi', 'panjang', 'long'],
            'exclude': ['mini', 'pendek', 'short']
        },
        'short': {
            'include': ['short', 'pendek', 'mini'],
            'exclude': ['panjang', 'long', 'maxi']
        },
        'pendek': {
            'include': ['pendek', 'short', 'mini'],
            'exclude': ['panjang', 'long', 'maxi']
        },
        'mini': {
            'include': ['mini', 'pendek', 'short'],
            'exclude': ['maxi', 'panjang', 'long']
        },
        'midi': {
            'include': ['midi', 'medium', 'sedang'],
            'exclude': ['mini', 'maxi', 'pendek', 'panjang', 'short', 'long']
        },
        'medium': {
            'include': ['medium', 'midi', 'sedang'],
            'exclude': ['mini', 'maxi', 'pendek', 'panjang', 'short', 'long']
        },
        'sedang': {
            'include': ['sedang', 'medium', 'midi'],
            'exclude': ['mini', 'maxi', 'pendek', 'panjang', 'short', 'long']
        },
        'lengan panjang': {
            'include': ['lengan panjang', 'panjang lengan', 'long sleeve', 'long-sleeve'],
            'exclude': ['lengan pendek', 'pendek lengan', 'short sleeve', 'short-sleeve', 'pendek']
        },
        'panjang lengan': {
            'include': ['panjang lengan', 'lengan panjang', 'long sleeve', 'long-sleeve'],
            'exclude': ['lengan pendek', 'pendek lengan', 'short sleeve', 'short-sleeve', 'pendek']
        },
        'long sleeve': {
            'include': ['long sleeve', 'long-sleeve', 'lengan panjang', 'panjang lengan'],
            'exclude': ['short sleeve', 'short-sleeve', 'lengan pendek', 'pendek lengan']
        },
        'lengan pendek': {
            'include': ['lengan pendek', 'pendek lengan', 'short sleeve', 'short-sleeve'],
            'exclude': ['lengan panjang', 'panjang lengan', 'long sleeve', 'long-sleeve', 'panjang']
        },
        'pendek lengan': {
            'include': ['pendek lengan', 'lengan pendek', 'short sleeve', 'short-sleeve'],
            'exclude': ['lengan panjang', 'panjang lengan', 'long sleeve', 'long-sleeve', 'panjang']
        },
        'short sleeve': {
            'include': ['short sleeve', 'short-sleeve', 'lengan pendek', 'pendek lengan'],
            'exclude': ['long sleeve', 'long-sleeve', 'lengan panjang', 'panjang lengan']
        },
        
        # Sizes/Fits - ALL ORIGINAL SIZES
        'small': {
            'include': ['small', 'kecil', 's'],
            'exclude': []
        },
        'kecil': {
            'include': ['kecil', 'small', 's'],
            'exclude': []
        },
        'medium': {
            'include': ['medium', 'sedang', 'm'],
            'exclude': []
        },
        'sedang': {
            'include': ['sedang', 'medium', 'm'],
            'exclude': []
        },
        'large': {
            'include': ['large', 'besar', 'l'],
            'exclude': []
        },
        'besar': {
            'include': ['besar', 'large', 'l'],
            'exclude': []
        },
        'extra large': {
            'include': ['extra large', 'xl', 'sangat besar'],
            'exclude': []
        },
        'tight': {
            'include': ['tight', 'ketat'],
            'exclude': []
        },
        'loose': {
            'include': ['loose', 'longgar'],
            'exclude': []
        },
        
        # Occasions - ALL ORIGINAL OCCASIONS
        'office': {
            'include': ['office', 'kantor', 'kerja'],
            'exclude': []
        },
        'kantor': {
            'include': ['kantor', 'office', 'kerja'],
            'exclude': []
        },
        'party': {
            'include': ['party', 'pesta'],
            'exclude': []
        },
        'pesta': {
            'include': ['pesta', 'party'],
            'exclude': []
        },
        'wedding': {
            'include': ['wedding', 'pernikahan'],
            'exclude': []
        },
        'pernikahan': {
            'include': ['pernikahan', 'wedding'],
            'exclude': []
        },
        'beach': {
            'include': ['beach', 'pantai'],
            'exclude': []
        },
        'pantai': {
            'include': ['pantai', 'beach'],
            'exclude': []
        },
        'sport': {
            'include': ['sport', 'olahraga'],
            'exclude': []
        },
        'olahraga': {
            'include': ['olahraga', 'sport'],
            'exclude': []
        },
    }
    
    # If the keyword has a direct mapping, return it
    if keyword_lower in translation_map:
        return translation_map[keyword_lower]
    
    # If no direct mapping, try to find partial matches
    search_terms = [keyword_lower]
    exclude_terms = []
    
    # Check if the keyword contains any mapped terms
    for mapped_term, mapping in translation_map.items():
        if mapped_term in keyword_lower or keyword_lower in mapped_term:
            search_terms.extend(mapping['include'])
            exclude_terms.extend(mapping['exclude'])
            break
    
    # Remove duplicates and return
    return {
        'include': list(set(search_terms)),
        'exclude': list(set(exclude_terms))
    }

# Enhanced function to get all search terms for use in extract_ranked_keywords
def get_all_search_terms_for_extraction(keyword):
    """
    Helper function that uses the complete keyword mapping for extracting ranked keywords.
    This integrates with extract_ranked_keywords to improve keyword processing.
    """
    search_mapping = get_search_terms_for_keyword(keyword)
    
    # Return all include terms for broader matching in keyword extraction
    return search_mapping['include']

async def fetch_products_from_db(db: AsyncSession, top_keywords: list, max_results=15, gender_category=None, budget_range=None, focus_category=None):
    """
    Enhanced product fetching with strict filtering and balanced distribution.
    ALWAYS returns a DataFrame (empty if no results).
    """
    print("\n" + "="*80)
    print("🔍 PRODUCT SEARCH WITH STRICT FILTERING & BALANCED DISTRIBUTION")
    print("="*80)
    print(f"📊 Total keywords received: {len(top_keywords)}")
    print(f"🎯 Top 15 keywords being used:")
    for i, (kw, score) in enumerate(top_keywords[:15]):
        print(f"   {i+1:2d}. '{kw}' → Score: {score:.2f}")
    print(f"🎯 Focus category: {focus_category}")
    print(f"👤 Gender filter: {gender_category}")
    print(f"💰 Budget filter: {budget_range}")
    
    # Define clothing categories for balanced distribution
    clothing_categories = {
        'tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'sweater', 'hoodie', 'cardigan', 'blazer', 'tank', 'top'],
        'bottoms_pants': ['celana', 'pants', 'jeans', 'trousers', 'leggings'],
        'bottoms_skirts': ['rok', 'skirt'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'coat', 'mantel'],
        'shorts': ['shorts', 'celana pendek']
    }
    
    def get_clothing_category(keyword):
        """Get clothing category for a keyword"""
        keyword_lower = keyword.lower()
        for category, terms in clothing_categories.items():
            if any(term in keyword_lower for term in terms):
                return category
        return None
    
    # Analyze keywords to identify requested clothing categories
    requested_categories = {}
    category_keywords = {}
    
    for keyword, score in top_keywords[:15]:
        category = get_clothing_category(keyword)
        if category:
            if category not in requested_categories:
                requested_categories[category] = 0
                category_keywords[category] = []
            requested_categories[category] += score
            category_keywords[category].append((keyword, score))
    
    print(f"\n📦 CLOTHING CATEGORIES DETECTED:")
    for category, total_score in requested_categories.items():
        keywords_in_cat = [kw for kw, _ in category_keywords[category]]
        print(f"   🏷️  {category}: {total_score:.1f} (keywords: {keywords_in_cat})")
    
    # ADD: Get exclusions from keyword extraction
    exclusions = get_latest_exclusions()
    if exclusions:
        print(f"🚫 Product exclusions: {exclusions}")
    else:
        print("🚫 No product exclusions")
    
    print("="*80)
    
    logging.info(f"=== STRICT FILTERED PRODUCT SEARCH ===")
    logging.info(f"Keywords: {[(kw, score) for kw, score in top_keywords[:10]]}")
    logging.info(f"Categories: {requested_categories}")
    logging.info(f"Gender: {gender_category}, Budget: {budget_range}")
    logging.info(f"Exclusions: {exclusions}")
    
    try:
        # Get products with variants
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
        
        # Main query - NO ORDER BY, let Python handle sorting
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

        if focus_category:
            print(f"🎯 APPLYING CATEGORY FOCUS: {focus_category}")
            
            category_terms = {
                'kemeja': ['kemeja', 'shirt', 'blouse', 'blus'],
                'celana': ['celana', 'pants', 'trousers', 'jeans'],
                'dress': ['dress', 'gaun', 'terusan'],
                'jaket': ['jaket', 'jacket', 'blazer', 'coat'],
                'kaos': ['kaos', 't-shirt', 'tshirt', 'tank'],
                'atasan': ['atasan', 'top'],
                'rok': ['rok', 'skirt']
            }
            
            if focus_category in category_terms:
                focus_terms = category_terms[focus_category]
                
                # Create OR condition for category terms
                category_conditions = []
                for term in focus_terms:
                    category_conditions.append(Product.product_name.contains(term))
                    category_conditions.append(Product.product_detail.contains(term))
                
                if category_conditions:
                    base_query = base_query.where(or_(*category_conditions))
                    print(f"   📝 Added category filter for: {focus_terms}")
            
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
                print(f"💰 Budget filter: IDR {min_price:,} - IDR {max_price:,}")
            elif max_price:
                base_query = base_query.where(variant_subquery.c.min_price <= max_price)
                print(f"💰 Max budget: IDR {max_price:,}")
            elif min_price:
                base_query = base_query.where(variant_subquery.c.min_price >= min_price)
                print(f"💰 Min budget: IDR {min_price:,}")
        
        # Execute query
        result = await db.execute(base_query)
        all_products = result.fetchall()
        
        if not all_products:
            print("❌ No products found in database")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
        
        print(f"📦 Found {len(all_products)} products to analyze")
        
        # Calculate relevance scores with strict filtering
        print(f"\n🧮 CALCULATING RELEVANCE SCORES WITH STRICT FILTERING...")
        
        categorized_products = {}
        for category in requested_categories.keys():
            categorized_products[category] = []
        categorized_products['other'] = []  # For products that don't match specific categories
        
        debug_count = 0
        strict_filter_failures = 0
        
        for product_row in all_products:
            # Debug first 3 products in detail
            debug_this_product = debug_count < 3
            
            if debug_this_product:
                print(f"\n🔍 DEBUGGING PRODUCT {debug_count + 1}: '{product_row[1]}'")
                print(f"   💰 Price: IDR {product_row[5]:,}")
            
            relevance_score = calculate_relevance_score(product_row, top_keywords, debug_this_product)
            
            # Skip products that failed strict filtering (score = 0)
            if relevance_score == 0:
                strict_filter_failures += 1
                if debug_this_product:
                    print(f"   🚫 PRODUCT EXCLUDED BY STRICT FILTERING")
                continue
            
            # Determine which category this product belongs to
            product_name_lower = product_row[1].lower()
            product_detail_lower = product_row[2].lower()
            product_text = f"{product_name_lower} {product_detail_lower}"
            
            product_category = None
            best_category_match = 0
            
            # Find the best matching category for this product
            for category, terms in clothing_categories.items():
                category_match_score = 0
                for term in terms:
                    if term in product_text:
                        category_match_score += len(term)  # Longer matches get higher scores
                
                if category_match_score > best_category_match:
                    best_category_match = category_match_score
                    product_category = category
            
            # Only assign to category if it's one of the requested categories
            if product_category and product_category in requested_categories:
                target_category = product_category
            else:
                target_category = 'other'
            
            if debug_this_product:
                print(f"   📂 Categorized as: {target_category} (match score: {best_category_match})")
                print(f"   📊 Final Relevance Score: {relevance_score:.2f}")
                debug_count += 1
            
            # Format data
            sizes = product_row[6].split(',') if product_row[6] else []
            colors = product_row[7].split(',') if product_row[7] else []
            
            product_data = {
                "product_id": product_row[0],
                "product": product_row[1],
                "description": product_row[2],
                "price": product_row[5],
                "size": ", ".join(sizes) if sizes else "N/A",
                "color": ", ".join(colors) if colors else "N/A", 
                "stock": product_row[8],
                "link": f"http://localhost/e-commerce-main/product-{product_row[3]}-{product_row[0]}",
                "photo": product_row[9],
                "relevance": relevance_score,
                "category": target_category
            }
            
            categorized_products[target_category].append(product_data)
        
        print(f"\n🚫 STRICT FILTERING RESULTS:")
        print(f"   ❌ Products excluded: {strict_filter_failures}")
        print(f"   ✅ Products passed: {sum(len(products) for products in categorized_products.values())}")
        
        # Sort products within each category by relevance
        for category in categorized_products:
            categorized_products[category].sort(key=lambda x: x['relevance'], reverse=True)
        
        print(f"\n📂 PRODUCTS BY CATEGORY AFTER STRICT FILTERING:")
        for category, products in categorized_products.items():
            if products:
                print(f"   {category}: {len(products)} products")
        
        # BALANCED DISTRIBUTION LOGIC (same as before)
        final_products = []
        
        if len(requested_categories) > 1:
            print(f"\n⚖️ APPLYING BALANCED DISTRIBUTION for {len(requested_categories)} categories")
            
            # Calculate products per category
            products_per_category = max_results // len(requested_categories)
            remaining_slots = max_results % len(requested_categories)
            
            print(f"   📊 Base allocation: {products_per_category} per category")
            print(f"   ➕ Extra slots to distribute: {remaining_slots}")
            
            # Distribute products evenly across categories
            category_list = list(requested_categories.keys())
            for i, category in enumerate(category_list):
                # Calculate allocation for this category
                allocation = products_per_category
                if i < remaining_slots:  # Distribute extra slots to first categories
                    allocation += 1
                
                available_products = categorized_products.get(category, [])
                selected_products = available_products[:allocation]
                
                print(f"   📦 {category}: taking {len(selected_products)}/{len(available_products)} (allocated: {allocation})")
                
                final_products.extend(selected_products)
            
            # Fill remaining slots with best products from any category if we're short
            if len(final_products) < max_results:
                remaining_needed = max_results - len(final_products)
                print(f"   🔄 Need {remaining_needed} more products, checking other categories...")
                
                # Collect remaining products from all categories
                all_remaining = []
                for category, products in categorized_products.items():
                    if category in requested_categories:
                        # Take products beyond what we already selected
                        allocation = products_per_category + (1 if category_list.index(category) < remaining_slots else 0)
                        remaining_from_category = products[allocation:]
                        all_remaining.extend(remaining_from_category)
                    elif category == 'other':
                        all_remaining.extend(products)
                
                # Sort remaining by relevance and take the best ones
                all_remaining.sort(key=lambda x: x['relevance'], reverse=True)
                final_products.extend(all_remaining[:remaining_needed])
                
                print(f"   ➕ Added {min(len(all_remaining), remaining_needed)} additional products")
        
        else:
            print(f"\n📦 SINGLE CATEGORY REQUEST - Using standard relevance sorting")
            # Single category - use all products sorted by relevance
            all_products_list = []
            for products in categorized_products.values():
                all_products_list.extend(products)
            
            all_products_list.sort(key=lambda x: x['relevance'], reverse=True)
            final_products = all_products_list[:max_results]
        
        # Convert to DataFrame
        products_df = pd.DataFrame(final_products)
        
        # APPLY EXCLUSION FILTERING
        if exclusions and not products_df.empty:
            print(f"\n🚫 APPLYING EXCLUSION FILTERING...")
            original_count = len(products_df)
            
            for exclusion in exclusions:
                # Remove products whose name or description contains excluded terms
                mask = ~(
                    products_df['product'].str.lower().str.contains(exclusion, na=False) |
                    products_df['description'].str.lower().str.contains(exclusion, na=False)
                )
                products_df = products_df[mask]
                
                current_count = len(products_df)
                removed_this_round = original_count - current_count
                if removed_this_round > 0:
                    print(f"   ❌ Excluded '{exclusion}': removed {removed_this_round} products")
                    original_count = current_count
            
            total_removed = len(final_products) - len(products_df)
            if total_removed > 0:
                print(f"   🗑️ Total filtered out: {total_removed} products with excluded terms")
                print(f"   ✅ Remaining products: {len(products_df)}")
            else:
                print(f"   ✅ No products were filtered out")
        
        # Final sorting by relevance within the balanced selection
        if not products_df.empty:
            products_df = products_df.sort_values(by=['relevance'], ascending=False).reset_index(drop=True)
            
            print(f"\n🏆 FINAL STRICT FILTERED & BALANCED PRODUCT DISTRIBUTION:")
            
            # Show distribution by category
            if 'category' in products_df.columns:
                category_counts = products_df['category'].value_counts()
                for category, count in category_counts.items():
                    print(f"   📦 {category}: {count} products")
            
            print(f"\n🏆 TOP {min(10, len(products_df))} PRODUCTS AFTER STRICT FILTERING:")
            for i, row in products_df.head(10).iterrows():
                category_display = f" [{row.get('category', 'unknown')}]" if 'category' in row else ""
                print(f"   {i+1:2d}. '{row['product'][:40]}...'${category_display} → Relevance: {row['relevance']:.2f}, Price: IDR {row['price']:,}")
            
            print(f"\n✅ RETURNING {len(products_df)} STRICTLY FILTERED PRODUCTS")
        else:
            print(f"\n❌ NO PRODUCTS REMAINING AFTER STRICT FILTERING")
            products_df = pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
        
        print("="*80)
        
        return products_df
        
    except Exception as e:
        logging.error(f"Error in fetch_products_from_db: {str(e)}")
        print(f"❌ ERROR in fetch_products_from_db: {str(e)}")
        # Always return empty DataFrame with correct columns instead of None
        return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
            
def calculate_relevance_score(product_row, keywords, debug=False, focus_category=None):
    """
    Enhanced relevance calculation with fair clothing detection and balanced scoring.
    Returns 0 if product fails strict filtering criteria.
    """
    product_name = product_row[1].lower()
    product_detail = product_row[2].lower()
    available_colors = product_row[7].lower() if product_row[7] else ""
    
    search_text = f"{product_name} {product_detail} {available_colors}"

    if debug:
        print(f"   🔍 Search text: '{search_text[:100]}...'")
        print(f"   🎯 Focus category: {focus_category}")
        print(f"   📝 Checking against {len(keywords)} keywords:")
    
    # Define color synonyms for common colors (flexible approach)
    color_synonyms = {
        'black': ['black', 'hitam', 'gelap'],
        'white': ['white', 'putih', 'ivory', 'cream', 'off-white'],
        'red': ['red', 'merah', 'crimson', 'burgundy', 'maroon'],
        'blue': ['blue', 'biru', 'navy', 'royal', 'cobalt', 'azure'],
        'green': ['green', 'hijau', 'lime', 'olive', 'emerald', 'forest'],
        'yellow': ['yellow', 'kuning', 'golden', 'amber', 'mustard'],
        'brown': ['brown', 'coklat', 'cokelat', 'tan', 'khaki', 'chocolate'],
        'pink': ['pink', 'merah muda', 'rose', 'blush', 'coral'],
        'purple': ['purple', 'ungu', 'violet', 'lavender', 'plum'],
        'orange': ['orange', 'oranye', 'peach', 'tangerine'],
        'grey': ['grey', 'gray', 'abu-abu', 'silver', 'charcoal'],
        'gold': ['gold', 'emas', 'golden'],
        'beige': ['beige', 'krem', 'nude', 'sand', 'taupe']
    }
    
    def detect_color_from_keyword(keyword):
        """
        Enhanced flexible color detection that PROPERLY excludes skin color contexts.
        Returns the color term if found, None otherwise.
        """
        keyword_lower = keyword.lower().strip()
        
        if debug:
            print(f"      🔍 Checking color detection for: '{keyword}'")
        
        # FIRST: Check for skin color contexts and exclude them - ENHANCED PATTERNS
        skin_color_patterns = [
            r'\b(?:kulit|skin)\s+([a-zA-Z]+)\b',  # "kulit putih", "skin white"
            r'\b([a-zA-Z]+)\s+(?:kulit|skin)\b',  # "putih kulit", "white skin"  
            r'\b(?:berkulit|has\s+skin)\s+([a-zA-Z]+)\b',  # "berkulit putih"
            r'\b(?:skin\s+tone|warna\s+kulit)\s+([a-zA-Z]+)\b',  # "skin tone white"
            r'\b(?:complexion|warna\s+kulit)\s+([a-zA-Z]+)\b',  # "complexion fair"
        ]
        
        # Check the ORIGINAL keyword input, not just the processed version
        original_context = keyword  # This should be the full original input
        
        for pattern in skin_color_patterns:
            match = re.search(pattern, original_context.lower())
            if match:
                if debug:
                    print(f"      🚫 SKIN COLOR CONTEXT DETECTED: '{keyword}' in '{original_context}' - NOT a clothing color requirement")
                return None
        
        # SECOND: Check if this specific keyword appears near skin-related words
        # This is important for when individual color words are extracted from "kulit putih"
        if keyword_lower in ['putih', 'white', 'hitam', 'black', 'coklat', 'brown', 'tan', 'fair', 'dark']:
            # For common skin colors, be extra careful about context
            skin_indicators = ['kulit', 'skin', 'berkulit', 'complexion', 'tone']
            
            # Check if any skin indicators appear near this color word
            # This would catch cases where "putih" is extracted from "kulit putih"
            if any(indicator in original_context.lower() for indicator in skin_indicators):
                if debug:
                    print(f"      🚫 SKIN COLOR FILTERED: '{keyword}' appears with skin indicators in '{original_context}'")
                return None
        
        # THIRD: Check for physical/body contexts and exclude them
        physical_contexts = [
            r'\b(?:tinggi|height|berat|weight|umur|age)\b',
            r'\b(?:cm|kg|tahun|years?)\b',
            r'\b(?:dari|from)\s+(?:indonesia|malaysia|singapore)\b',
            r'\b(?:cowo|cowok|cewe|cewek|pria|wanita|laki-laki|perempuan)\b',
        ]
        
        for pattern in physical_contexts:
            if re.search(pattern, original_context.lower()):
                if debug:
                    print(f"      🚫 PHYSICAL CONTEXT: '{keyword}' appears in physical description - skipping color detection")
                return None
        
        # FOURTH: Only proceed with color detection if it's clearly clothing-related
        # Check for clothing context indicators
        clothing_indicators = [
            'kemeja', 'shirt', 'dress', 'gaun', 'celana', 'pants', 'rok', 'skirt',
            'jaket', 'jacket', 'kaos', 't-shirt', 'atasan', 'bawahan', 'pakai',
            'warna', 'color', 'baju'
        ]
        
        has_clothing_context = any(indicator in original_context.lower() for indicator in clothing_indicators)
        
        # If no clothing context and it's a common skin color, skip
        if not has_clothing_context and keyword_lower in ['putih', 'white', 'hitam', 'black', 'coklat', 'brown', 'tan']:
            if debug:
                print(f"      🚫 NO CLOTHING CONTEXT: '{keyword}' has no clothing indicators, likely skin color")
            return None
        
        # FIFTH: Proceed with normal color detection only if passed all filters above
        # Check known color synonyms
        for base_color, synonyms in color_synonyms.items():
            if any(synonym in keyword_lower for synonym in synonyms):
                if debug:
                    print(f"      ✅ CLOTHING COLOR DETECTED: '{keyword_lower}' (passed all skin color filters)")
                return keyword_lower
        
        # Enhanced color detection patterns (only if no physical context)
        color_patterns = [
            r'\bwarna\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b',
            r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+(?:colored?|colou?red?)\b',
            r'\b(light|dark|bright|deep|pale|soft|muted|vivid|intense|pastel|neon|electric|terang|gelap)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\b',
            r'\b(navy|maroon|burgundy|teal|coral|salmon|olive|mint|rose|cherry|lemon|peach|cream|vanilla|chocolate|coffee|wine)\b',
        ]
        
        for i, pattern in enumerate(color_patterns):
            match = re.search(pattern, keyword_lower)
            if match:
                if i == 0:  # "warna X" pattern
                    color_part = match.group(1).strip()
                    if debug:
                        print(f"      ✅ CLOTHING COLOR DETECTED: 'warna {color_part}' (explicit color pattern)")
                    return f"warna {color_part}"
                elif i == 1:  # "X colored" pattern  
                    color_part = match.group(1).strip()
                    if debug:
                        print(f"      ✅ CLOTHING COLOR DETECTED: '{color_part}' (colored pattern)")
                    return color_part
                elif i == 2:  # "modifier + color" pattern
                    modifier = match.group(1).strip()
                    color_part = match.group(2).strip()
                    if debug:
                        print(f"      ✅ CLOTHING COLOR DETECTED: '{modifier} {color_part}' (modifier pattern)")
                    return f"{modifier} {color_part}"
                else:
                    if debug:
                        print(f"      ✅ CLOTHING COLOR DETECTED: '{match.group(0).strip()}' (special color)")
                    return match.group(0).strip()
        
        if debug:
            print(f"      ❌ NO COLOR DETECTED: '{keyword}' is not a recognizable color")
        return None
    
    detect_color_from_keyword.original_input = ' '.join([kw for kw, _ in keywords])

    def detect_color_in_product(product_text, color_term):
        """
        Enhanced check if a color term or its variations exist in product text.
        Handles "warna X" patterns and modifiers better.
        """
        if not color_term:
            return False
            
        # Handle "warna X" pattern specifically
        warna_match = re.search(r'warna\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)', color_term.lower())
        if warna_match:
            actual_color = warna_match.group(1).strip()
            
            if actual_color in product_text or f"warna {actual_color}" in product_text:
                return True
            
            # Also check synonyms for the extracted color
            for base_color, synonyms in color_synonyms.items():
                if actual_color in synonyms or any(synonym in actual_color for synonym in synonyms):
                    if any(synonym in product_text for synonym in synonyms):
                        return True
        
        # Handle modifier + color patterns
        modifier_match = re.search(r'(light|dark|bright|deep|pale|soft|muted|vivid|intense|pastel|neon|electric|terang|gelap)\s+([a-zA-Z]+)', color_term.lower())
        if modifier_match:
            modifier = modifier_match.group(1)
            main_color = modifier_match.group(2)
            
            if color_term.lower() in product_text:
                return True
            
            if main_color in product_text:
                return True
            
            if modifier in product_text and main_color in product_text:
                return True
                
            # Check synonyms for main color
            for base_color, synonyms in color_synonyms.items():
                if main_color in synonyms or any(synonym in main_color for synonym in synonyms):
                    if any(synonym in product_text for synonym in synonyms):
                        return True
        
        # Direct match
        if color_term.lower() in product_text:
            return True
        
        # Check if it's a known color with synonyms
        for base_color, synonyms in color_synonyms.items():
            if color_term in synonyms or any(synonym in color_term for synonym in synonyms):
                if any(synonym in product_text for synonym in synonyms):
                    return True
        
        # Partial matching for compound colors
        color_words = color_term.split()
        if len(color_words) > 1:
            for word in color_words:
                if len(word) > 3:
                    if word not in ['light', 'dark', 'bright', 'deep', 'pale', 'soft', 'warna', 'color', 'colored', 'terang', 'gelap']:
                        if word in product_text:
                            return True
                        
                        for base_color, synonyms in color_synonyms.items():
                            if word in synonyms:
                                if any(synonym in product_text for synonym in synonyms):
                                    return True
        
        return False
    
    # Define strict filtering categories with flexible color handling
    strict_filters = {
        'fits': {
            'oversized': ['oversized', 'oversize', 'loose fit', 'longgar', 'besar'],
            'slim': ['slim', 'slim fit', 'ketat', 'skinny'],
            'regular': ['regular', 'normal', 'standard'],
            'fitted': ['fitted', 'fit', 'pas badan'],
            'tight': ['tight', 'ketat sekali', 'sangat ketat'],
            'loose': ['loose', 'longgar', 'rileks']
        },
        'sleeve_lengths': {
            'sleeveless': ['sleeveless', 'tanpa lengan', 'tank', 'singlet'],
            'short_sleeve': ['short sleeve', 'lengan pendek', 'pendek lengan', 'short-sleeve'],
            'three_quarter': ['3/4 sleeve', 'three quarter', 'lengan 3/4'],
            'long_sleeve': ['long sleeve', 'lengan panjang', 'panjang lengan', 'long-sleeve'],
            'cap_sleeve': ['cap sleeve', 'lengan kap']
        },
        'clothing_lengths': {
            'crop': ['crop', 'cropped', 'pendek', 'short'],
            'regular_length': ['regular length', 'normal', 'standard'],
            'tunic': ['tunic', 'panjang', 'longline'],
            'maxi': ['maxi', 'maksimal', 'sangat panjang'],
            'mini': ['mini', 'sangat pendek'],
            'midi': ['midi', 'sedang', 'medium length'],
            'knee_length': ['knee length', 'sebatas lutut'],
            'ankle_length': ['ankle length', 'sebatas mata kaki']
        }
    }
    
    # Extract user requirements from keywords with flexible color detection
    user_requirements = {
        'colors': [],
        'fits': [],
        'sleeve_lengths': [],
        'clothing_lengths': []
    }
    
    total_score = 0
    matches_found = []
    strict_violations = []
    
    # First pass: identify user requirements with flexible color detection
    for i, (keyword, weight) in enumerate(keywords[:15]):
        keyword_lower = keyword.lower()
        
        # FLEXIBLE COLOR DETECTION with context awareness
        detected_color = detect_color_from_keyword(keyword_lower)
        if detected_color:
            # QUICK FIX: Check if any keyword contains skin context with this color
            has_skin_context = any(
                re.search(rf'\b(?:kulit|skin)\s+{re.escape(detected_color)}\b', kw.lower()) 
                for kw, _ in keywords
            )
            
            if has_skin_context:
                if debug:
                    print(f"      🚫 SKIN COLOR FILTERED: '{detected_color}' found with 'kulit/skin' context")
            elif detected_color not in user_requirements['colors']:
                user_requirements['colors'].append(detected_color)
                if debug:
                    print(f"      🎨 CLOTHING COLOR REQUIREMENT: {detected_color} (from '{keyword}')")
        # Check other strict filter categories (non-color)
        for filter_category, filter_options in strict_filters.items():
            for option_name, option_terms in filter_options.items():
                if any(term in keyword_lower for term in option_terms):
                    if option_name not in user_requirements[filter_category]:
                        user_requirements[filter_category].append(option_name)
                        if debug:
                            print(f"      🎯 REQUIREMENT: {filter_category} = {option_name} (from '{keyword}')")
    
    if debug:
        print(f"   📋 User Requirements Summary:")
        for category, requirements in user_requirements.items():
            if requirements:
                print(f"      {category}: {requirements}")
    
    # Second pass: strict filtering check with flexible color matching
    strict_filter_passed = True
    
    for filter_category, required_options in user_requirements.items():
        if required_options:
            category_matched = False
            
            if filter_category == 'colors':
                # FLEXIBLE COLOR MATCHING
                for required_color in required_options:
                    if detect_color_in_product(search_text, required_color):
                        category_matched = True
                        if debug:
                            print(f"      ✅ FLEXIBLE COLOR MATCH: found '{required_color}' in product")
                        break
                
                if not category_matched:
                    strict_filter_passed = False
                    strict_violations.append(f"colors: required {required_options}")
                    if debug:
                        print(f"      ❌ COLOR FILTER FAILED: required {required_options}")
            
            else:
                # Standard matching for non-color categories
                for required_option in required_options:
                    option_terms = strict_filters[filter_category][required_option]
                    
                    if any(term in search_text for term in option_terms):
                        category_matched = True
                        if debug:
                            print(f"      ✅ STRICT FILTER PASSED: {filter_category} = {required_option}")
                        break
                
                if not category_matched:
                    strict_filter_passed = False
                    strict_violations.append(f"{filter_category}: required {required_options}")
                    if debug:
                        print(f"      ❌ STRICT FILTER FAILED: {filter_category} required {required_options}")
    
    # If strict filtering fails, return 0 score (exclude product)
    if not strict_filter_passed:
        if debug:
            print(f"   🚫 PRODUCT EXCLUDED due to strict filter violations: {strict_violations}")
        return 0.0
    
    # NEW: Category focus boost/penalty
    if focus_category:
        category_terms = {
            'kemeja': ['kemeja', 'shirt', 'blouse', 'blus'],
            'celana': ['celana', 'pants', 'trousers', 'jeans'],
            'dress': ['dress', 'gaun', 'terusan'],
            'jaket': ['jaket', 'jacket', 'blazer', 'coat'],
            'kaos': ['kaos', 't-shirt', 'tshirt', 'tank'],
            'atasan': ['atasan', 'top'],
            'rok': ['rok', 'skirt']
        }
        
        if focus_category in category_terms:
            focus_terms = category_terms[focus_category]
            category_match = any(term in search_text for term in focus_terms)
            
            if category_match:
                category_bonus = 2000  # Huge bonus for matching focus category
                total_score += category_bonus
                if debug:
                    print(f"   🎯 CATEGORY FOCUS MATCH: +{category_bonus}")
            else:
                # Heavy penalty for not matching focus category
                if debug:
                    print(f"   ❌ CATEGORY FOCUS MISMATCH: Heavy penalty applied")
                return 0  # Return 0 to exclude non-matching products
    
    # Third pass: calculate relevance score with FAIR clothing detection
    for i, (keyword, weight) in enumerate(keywords[:15]):
        keyword_lower = keyword.lower()
        
        # Position importance (earlier keywords are more important)
        position_weight = (15 - i) / 15
        
        match_score = 0
        match_type = "NO_MATCH"
        
        # Check if keyword is a clothing item (all get equal treatment)
        is_clothing, _ = is_clothing_item_with_priority(keyword_lower)
        
        # PRIORITY 1: All clothing items - EQUAL TREATMENT
        if is_clothing:
            clothing_base_score = 5.0  # Same for all clothing types
            
            if keyword_lower in product_name:
                match_score = weight * position_weight * clothing_base_score
                match_type = "CLOTHING_NAME_MATCH"
            elif keyword_lower in product_detail:
                match_score = weight * position_weight * (clothing_base_score * 0.8)
                match_type = "CLOTHING_DESC_MATCH"
            elif any(part in search_text for part in keyword_lower.split()):
                match_score = weight * position_weight * (clothing_base_score * 0.6)
                match_type = "CLOTHING_PARTIAL_MATCH"
            else:
                match_score = weight * position_weight * (clothing_base_score * 0.4)
                match_type = "CLOTHING_WEAK_MATCH"
        
        # PRIORITY 2: Style features - EQUAL with each other
        elif any(feature in keyword_lower for feature in ['lengan', 'sleeve', 'oversized', 'slim', 'loose', 'tight', 'panjang', 'pendek', 'fitted', 'regular']):
            if keyword_lower in search_text:
                match_score = weight * position_weight * 2.5
                match_type = "STYLE_MATCH"
        
        # PRIORITY 2: Color matches - EQUAL with style features  
        elif detect_color_in_product(search_text, keyword_lower):
            match_score = weight * position_weight * 2.5
            match_type = "COLOR_MATCH"
        
        # PRIORITY 2: Material/fabric matches - EQUAL with style/color
        elif any(material in keyword_lower for material in ['cotton', 'katun', 'silk', 'sutra', 'denim', 'wool', 'polyester', 'linen']):
            if keyword_lower in search_text:
                match_score = weight * position_weight * 2.5
                match_type = "MATERIAL_MATCH"
        
        # PRIORITY 3: Exact keyword match (non-clothing)
        elif keyword_lower in search_text:
            if keyword_lower in product_name:
                match_score = weight * position_weight * 2.0
                match_type = "NAME_MATCH"
            elif keyword_lower in product_detail:
                match_score = weight * position_weight * 1.5
                match_type = "DESC_MATCH"
            else:
                match_score = weight * position_weight * 1.0
                match_type = "OTHER_MATCH"
        
        # PRIORITY 4: Partial match (lowest)
        elif any(word in search_text for word in keyword_lower.split()):
            match_score = weight * position_weight * 0.5
            match_type = "PARTIAL"
        
        if match_score > 0:
            total_score += match_score
            matches_found.append((keyword, match_type, match_score))
            
            if debug:
                print(f"      ✅ '{keyword}' → {match_type} (+{match_score:.2f})")
        elif debug and i < 8:
            print(f"      ❌ '{keyword}' → NO_MATCH")
    
    # Bonus for products that match strict requirements
    if strict_filter_passed and any(user_requirements.values()):
        strict_bonus = 50
        total_score += strict_bonus
        if debug:
            print(f"   🎁 STRICT REQUIREMENTS BONUS: +{strict_bonus}")
    
    if debug:
        print(f"   📊 Total matches found: {len(matches_found)}")
        print(f"   🎯 Best matches: {[f'{kw}({mt})' for kw, mt, _ in matches_found[:3]]}")
        print(f"   ✅ Strict filtering: {'PASSED' if strict_filter_passed else 'FAILED'}")
        print(f"   🔢 Final score: {total_score:.2f}")
    
    return total_score

async def fetch_products_with_budget_awareness(db: AsyncSession, top_keywords: list, max_results=15, gender_category=None, budget_range=None):
    """
    Enhanced product fetching that checks budget constraints and returns appropriate data.
    """
    logging.info(f"=== BUDGET-AWARE PRODUCT FETCH ===")
    logging.info(f"Budget range: {budget_range}")
    
    # Clean up empty/invalid budget ranges
    if budget_range == (None, None) or budget_range == [None, None]:
        budget_range = None
    
    try:
        if budget_range and any(budget_range):  # Only if budget has actual values
            print(f"💰 Searching with budget constraint: {budget_range}")
            products_within_budget = await fetch_products_from_db(db, top_keywords, max_results, gender_category, budget_range)
            
            if products_within_budget is not None and not products_within_budget.empty:
                logging.info(f"Found {len(products_within_budget)} products within budget")
                return products_within_budget, "within_budget"
            else:
                logging.info("No products found within budget range")
                # Try without budget constraint to see if products exist at all
                products_without_budget = await fetch_products_from_db(db, top_keywords, max_results, gender_category, None)
                
                if products_without_budget is not None and not products_without_budget.empty:
                    logging.info(f"Found {len(products_without_budget)} products outside budget")
                    return products_without_budget, "no_products_in_budget"
                else:
                    logging.info("No products found even without budget constraint")
                    return pd.DataFrame(), "no_products_found"
        else:
            print(f"💰 No budget specified, searching normally")
            # No budget specified, search normally
            products = await fetch_products_from_db(db, top_keywords, max_results, gender_category, None)
            
            if products is not None and not products.empty:
                logging.info(f"Found {len(products)} products without budget constraint")
                return products, "no_budget_specified"
            else:
                logging.info("No products found")
                return pd.DataFrame(), "no_products_found"
                
    except Exception as e:
        logging.error(f"Error in fetch_products_with_budget_awareness: {str(e)}")
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
    
    # STEP 1: Apply improved category change detection FIRST
    major_change_detected = detect_fashion_category_change(user_input, user_context)
    
    # STEP 2: Apply enhanced keyword decay (unchanged)
    apply_keyword_decay(user_context)
    
    # STEP 3: Apply improved scoring to new keywords
    enhanced_new_keywords = []
    is_multi_item = detect_multi_item_request(user_input)
    
    for keyword, weight in new_keywords:
        category_multiplier = get_keyword_category_multiplier(keyword)
        
        # Improved boost logic - different boosts for major vs minor changes
        if is_user_input:
            if major_change_detected:
                nuclear_boost = 10.0  # 10x boost for major clothing type changes
                enhanced_weight = weight * category_multiplier * nuclear_boost
                print(f"   ☢️  MAJOR CHANGE BOOST: '{keyword}' {weight:.1f} × {category_multiplier} × {nuclear_boost} = {enhanced_weight:.1f}")
            elif is_multi_item:
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
    update_accumulated_keywords(enhanced_new_keywords, user_context, is_user_input=is_user_input)
    
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
    
    category_cleanup(user_context, persistence_config)
    
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
        Enhanced detection for multi-item requests like "carikan kemeja dan celana"
        KEEPS ORIGINAL FUNCTION NAME
        """
        user_input_lower = user_input.lower().strip()
        
        print(f"🤝 MULTI-ITEM DETECTION: '{user_input}'")
        
        # BASIC FILTERS: Don't trigger on simple responses
        simple_responses = ["yes", "ya", "iya", "ok", "okay", "sure", "tentu", "no", "tidak", "nope", "ga", "engga", "1", "2", "3", "one", "two", "three", "satu", "dua", "tiga"]
        if user_input_lower in simple_responses:
            print(f"   ❌ Simple response detected: '{user_input_lower}'")
            return False
        
        # Don't trigger on very short inputs unless they're clearly clothing items
        if len(user_input_lower.split()) <= 2:
            clothing_keywords = ['kemeja', 'shirt', 'dress', 'gaun', 'celana', 'pants', 'rok', 'skirt', 'jaket', 'jacket', 'kaos', 'sweater']
            if not any(clothing in user_input_lower for clothing in clothing_keywords):
                print(f"   ❌ Short input without clothing keywords: '{user_input_lower}'")
                return False
        
        # Multi-item indicators (connectors)
        multi_indicators = [
            r'\b(dan|and|atau|or|with|sama|plus|\+|&)\b',  # Connectors
            r'\b(both|keduanya|semua|all)\b',  # Both/all indicators
            r'\b(recommendation|rekomendasi).*(dan|and|atau|or)',  # "recommendation for X and Y"
            r'\b(outfit|set|setelan|lengkap|complete)\b',  # Complete outfit requests
            r'\b(carikan|tunjukkan).*(dan|and|atau|or)',  # "carikan X dan Y"
        ]
        
        has_multi_indicator = any(re.search(pattern, user_input_lower) for pattern in multi_indicators)
        
        # Count distinct clothing categories mentioned
        clothing_categories = get_shared_clothing_categories()
        mentioned_categories = set()
        
        for category, terms in clothing_categories.items():
            if any(term in user_input_lower for term in terms):
                mentioned_categories.add(category)
        
        print(f"   🔍 Multi indicators found: {has_multi_indicator}")
        print(f"   📦 Categories mentioned: {mentioned_categories}")
        
        # Decision logic
        if has_multi_indicator and len(mentioned_categories) >= 2:
            print(f"   ✅ MULTI-ITEM REQUEST: Connectors + Multiple categories")
            return True
        elif len(mentioned_categories) >= 3:  # 3+ categories = likely multi-item even without connectors
            print(f"   ✅ MULTI-ITEM REQUEST: 3+ categories mentioned")
            return True
        elif has_multi_indicator:
            print(f"   🤔 POSSIBLE MULTI-ITEM: Has connectors but limited categories")
            return True  # Be permissive with connectors
        else:
            print(f"   ❌ SINGLE-ITEM REQUEST")
            return False

def extract_ranked_keywords(ai_response: str = None, translated_input: str = None, accumulated_keywords=None):
    """
    CORRECTED: Context-aware keyword extraction with PRESERVED conflict resolution for single-type changes
    but ALLOWS multi-item requests when explicitly requested.
    ENHANCED: Now includes improved exclusion filtering from the enhanced version.
    """
    print("\n" + "="*60)
    print("🔤 ENHANCED KEYWORD EXTRACTION WITH SMART CONFLICT RESOLUTION")
    print("="*60)
    
    keyword_scores = {}
    global_exclusions = set()

    # Enhanced exclusion lists (integrated from enhanced version)
    conversation_words = {
        # Indonesian conversation words
        "jadi", "ganti", "oke", "ya", "iya", "tidak", "bisa", "ada", "yang", "dan", "atau", 
        "dengan", "untuk", "dari", "pada", "akan", "dapat", "adalah", "ini", "itu", "saya", 
        "anda", "kamu", "mereka", "dia", "sangat", "lebih", "kurang", "bagus", "baik",
        "cocok", "sesuai", "tepat", "juga", "hanya", "sudah", "mau", "ingin", "buat",
        "gimana", "kayak", "kalau", "kalo", "terus", "lalu", "abis", "udah", "belum",
        
        # English conversation words  
        "yes", "no", "okay", "sure", "can", "could", "would", "should", "might", "must",
        "will", "shall", "may", "do", "does", "did", "have", "has", "having", "am", "is", 
        "are", "was", "were", "been", "being", "get", "got", "make", "made", "take", "took",
        "go", "went", "come", "came", "see", "saw", "know", "knew", "think", "thought",
        "want", "like", "need", "use", "used", "way", "time", "day", "year", "good", "new",
        "first", "last", "long", "great", "little", "own", "other", "old", "right", "big",
        "high", "different", "small", "large", "next", "early", "young", "important", "few",
        "public", "bad", "same", "able", "thanks", "thank", "please"
    }

    # Simple responses filter
    simple_responses = {
        "yes", "ya", "iya", "oke", "ok", "okay", "sure", "tentu",
        "no", "tidak", "nope", "ga", "gak", "engga", "nah",
        "good", "bagus", "nice", "baik", "great", "mantap",
        "thanks", "terima", "kasih", "makasih", "thx"
    }
    
    # Initialize variables
    current_input_categories = set()
    wanted_items = []
    context_items = []
    
    # Define clothing categories
    clothing_categories = get_shared_clothing_categories()
    
    # RESTORED: Define clothing conflicts (but make them smarter)
    clothing_conflicts = {
        'kemeja': {
            'keywords': ['kemeja', 'shirt', 'blouse', 'blus'],
            'conflicts_with': ['celana', 'dress', 'rok']  # Shirts conflict with bottoms/dresses
        },
        'celana': {
            'keywords': ['celana', 'pants', 'jeans', 'trousers'],
            'conflicts_with': ['kemeja', 'dress', 'atasan', 'kaos']  # Pants conflict with tops/dresses
        },
        'dress': {
            'keywords': ['dress', 'gaun', 'terusan'],
            'conflicts_with': ['kemeja', 'celana', 'atasan', 'kaos', 'rok']  # Dresses conflict with separates
        },
        'rok': {
            'keywords': ['rok', 'skirt'],
            'conflicts_with': ['kemeja', 'dress', 'atasan', 'kaos']  # Skirts conflict with tops/dresses
        },
        'jaket': {
            'keywords': ['jaket', 'jacket', 'blazer', 'coat'],
            'conflicts_with': []  # Outerwear doesn't conflict (can be worn with anything)
        },
        'kaos': {
            'keywords': ['kaos', 't-shirt', 'tshirt', 'tank'],
            'conflicts_with': ['celana', 'dress', 'rok']  # T-shirts conflict with bottoms/dresses
        },
        'atasan': {
            'keywords': ['atasan', 'top', 'blouse'],
            'conflicts_with': ['celana', 'dress', 'rok']  # Tops conflict with bottoms/dresses
        }
    }
    
    def get_clothing_category(keyword):
        """Get clothing category for a keyword"""
        keyword_lower = keyword.lower()
        for category, terms in clothing_categories.items():
            if any(term in keyword_lower for term in terms):
                return category
        return None
    

    # Extract specific clothing request if possible
    if translated_input:
        try:
            wanted_items, context_items = extract_specific_clothing_request(translated_input, ai_response)
        except:
            wanted_items, context_items = [], []
            for category, terms in clothing_categories.items():
                for term in terms:
                    if term in translated_input.lower():
                        if any(indicator in translated_input.lower() for indicator in ['apa', 'what', 'carikan', 'tunjukkan', 'show']):
                            wanted_items.append(category)
                        else:
                            context_items.append(category)
    
    # NEW: Detect if this is a multi-item request
    is_multi_item_request = False
    if translated_input:
        is_multi_item_request = detect_multi_item_request(translated_input)
    
    # REVISED: Balanced scoring that prioritizes AI recommendations
    scoring_categories = {
        'clothing_items': {
            'terms': ['kemeja', 'shirt', 'blouse', 'blus', 'dress', 'gaun', 'rok', 'skirt',
                     'celana', 'pants', 'jeans', 'jacket', 'jaket', 'sweater', 'cardigan',
                     'atasan', 'top', 'kaos', 't-shirt', 'hoodie', 'blazer', 'coat', 'ankle pants'],
            'user_score': 300,
            'ai_score': 400,
            'priority': 'HIGHEST'
        },
        'style_attributes': {
            'terms': ['lengan panjang', 'lengan pendek', 'long sleeve', 'short sleeve',
                     'panjang', 'long', 'pendek', 'short', 'slim', 'regular', 'loose', 'ketat',
                     'longgar', 'tight', 'oversized', 'casual', 'formal', 'elegant'],
            'user_score': 200,
            'ai_score': 250,
            'priority': 'HIGH'
        },
        'colors': {
            'terms': ['white', 'putih', 'black', 'hitam', 'red', 'merah', 'blue', 'biru',
                     'green', 'hijau', 'yellow', 'kuning', 'brown', 'coklat', 'pink',
                     'purple', 'ungu', 'orange', 'oranye', 'grey', 'abu-abu', 'navy', 'beige'],
            'user_score': 200,  # Equal with style attributes
            'ai_score': 220,
            'priority': 'HIGH'
        },
        'materials_fit': {
            'terms': ['cotton', 'katun', 'silk', 'sutra', 'denim', 'wool', 'wol', 
                     'polyester', 'linen', 'leather', 'kulit'],
            'user_score': 200,
            'ai_score': 220,
            'priority': 'HIGH'
        },
        'gender_terms': {
            'terms': ['perempuan', 'wanita', 'female', 'woman', 'pria', 'laki-laki', 'male', 'man'],
            'user_score': 50,
            'ai_score': 20,
            'priority': 'FILTER'
        },
        'occasions': {
            'terms': ['office', 'kantor', 'party', 'pesta', 'wedding', 'pernikahan',
                     'beach', 'pantai', 'sport', 'olahraga', 'work', 'kerja'],
            'user_score': 150,
            'ai_score': 170,
            'priority': 'MEDIUM'
        }
    }
    
    def get_keyword_score(keyword, source, frequency=1):
        """Get appropriate score with BALANCED WEIGHTING and enhanced filtering"""
        keyword_lower = keyword.lower()
        
        # ENHANCED: Filter out conversation words immediately
        if keyword_lower in conversation_words:
            print(f"      🚫 FILTERED conversation word: '{keyword}'")
            return 0, 'FILTERED'
        
        base_score = 0
        priority = 'DEFAULT'
        
        for category, config in scoring_categories.items():
            if any(term in keyword_lower for term in config['terms']):
                if source == 'user':
                    base_score = config['user_score'] * frequency
                elif source == 'ai':
                    base_score = config['ai_score'] * frequency
                else:
                    base_score = config['user_score'] * frequency * 0.5
                priority = config['priority']
                break
        
        if base_score == 0:
            if source == 'user':
                base_score = 100 * frequency
            elif source == 'ai':
                base_score = 120 * frequency
            else:
                base_score = 50 * frequency
            priority = 'DEFAULT'
        
        # Moderate specific request boost
        clothing_category = get_clothing_category(keyword)
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
    
    def is_physical_description(text):
        """Check if text contains physical description"""
        text_lower = text.lower()
        
        physical_indicators = [
            r'\b(?:kulit|skin)\s+\w+',
            r'\bberkulit\s+\w+',
            r'\bwarna\s+kulit',
            r'\bskin\s+tone',
            r'\b\d+\s*(?:cm|kg|tahun|years?)',
            r'\b(?:tinggi|height|berat|weight|umur|age)',
            r'\b(?:dari|from)\s+(?:indonesia|malaysia|singapore|thailand)',
            r'\b(?:cowo|cowok|cewe|cewek|pria|wanita|laki-laki|perempuan)',
            r'\b(?:suka|like)\s+(?:olahraga|sport|gym|fitness)',
        ]
        
        for pattern in physical_indicators:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def filter_skin_colors(keyword):
        """Filter out skin color mentions"""
        keyword_lower = keyword.lower()
        
        skin_color_patterns = [
            r'\b(?:kulit|skin)\s+([a-zA-Z]+)\b',
            r'\b([a-zA-Z]+)\s+(?:kulit|skin)\b',
            r'\bberkulit\s+([a-zA-Z]+)\b',
            r'\b(?:skin\s+tone|warna\s+kulit)\s+([a-zA-Z]+)\b',
        ]
        
        for pattern in skin_color_patterns:
            if re.search(pattern, keyword_lower):
                print(f"      🚫 SKIN COLOR FILTERED: '{keyword}' - not a clothing color")
                return True
        
        if keyword_lower == 'tan' and any(phys in keyword_lower for phys in ['kulit', 'skin']):
            print(f"      🚫 SKIN COLOR FILTERED: '{keyword}' - tan in skin context")
            return True
        
        return False
    
    # Process user input with physical description filtering
    if translated_input:
        print(f"📝 USER INPUT: '{translated_input}'")
        print(f"🤝 Multi-item request: {is_multi_item_request}")
        
        # Check for simple response
        input_words = translated_input.lower().split()
        is_simple_response = (
            len(input_words) <= 2 and 
            all(word in simple_responses for word in input_words)
        )
        
        if is_simple_response:
            print(f"   ⚠️  SIMPLE RESPONSE DETECTED - Skipping")
            return []
        
        # Physical description filtering with shared clothing categories
        if is_physical_description(translated_input):
            print(f"   🚫 PHYSICAL DESCRIPTION DETECTED - Filtering out non-clothing terms")
            
            # Get all clothing terms from shared categories
            shared_categories = get_shared_clothing_categories()
            all_clothing_terms = []
            for category_terms in shared_categories.values():
                all_clothing_terms.extend(category_terms)
            
            # Add style attributes to clothing terms
            style_terms = ['lengan', 'sleeve', 'oversized', 'loose', 'tight', 'pendek', 'panjang', 'slim', 'fitted', 'regular']
            all_clothing_terms.extend(style_terms)
            
            print(f"   👕 Using {len(all_clothing_terms)} clothing terms from shared categories")
            
            clothing_terms = []
            physical_terms = []
            
            for word in translated_input.split():
                # Check against shared clothing categories
                if any(clothing in word.lower() for clothing in all_clothing_terms):
                    clothing_terms.append(word)
                elif any(physical in word.lower() for physical in ['kulit', 'skin', 'cm', 'kg', 'tinggi', 'berat', 'dari', 'indonesia', 'cowo', 'cewe', 'cowok', 'cewek', 'pria', 'wanita', 'tahun', 'umur']):
                    physical_terms.append(word)
                else:
                    clothing_terms.append(word)  # Default to clothing context
            
            print(f"   👕 CLOTHING TERMS: {clothing_terms}")
            print(f"   🚫 PHYSICAL TERMS: {physical_terms}")
            
            filtered_input = ' '.join(clothing_terms)
            doc = nlp(filtered_input) if filtered_input.strip() else nlp("")
        else:
            doc = nlp(translated_input)    
        # Extract keywords using spaCy with enhanced filtering
        user_keywords = {}
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and 
                len(token.text) > 2 and 
                not token.text.isdigit() and
                token.is_alpha and
                token.text.lower() not in simple_responses and
                token.text.lower() not in conversation_words):  # Enhanced filtering
                
                keyword = token.text.lower()
                
                if filter_skin_colors(keyword):
                    continue
                
                user_keywords[keyword] = user_keywords.get(keyword, 0) + 1
                
                clothing_cat = get_clothing_category(keyword)
                if clothing_cat:
                    current_input_categories.add(clothing_cat)
        
        # Score user keywords with enhanced scoring
        for keyword, frequency in user_keywords.items():
            score, priority = get_keyword_score(keyword, 'user', frequency)
            
            if score > 0:  # Only add non-filtered keywords
                keyword_scores[keyword] = score
                print(f"   📌 '{keyword}' (freq: {frequency}) → {score} ({priority})")
                
                # Get translation expansion and exclusions
                try:
                    search_terms = get_search_terms_for_keyword(keyword)
                    if isinstance(search_terms, dict):
                        include_terms = search_terms.get('include', [])
                        exclude_terms = search_terms.get('exclude', [])
                        
                        for include_term in include_terms:
                            if (include_term != keyword and 
                                include_term not in keyword_scores and
                                include_term not in conversation_words):  # Filter expansions too
                                
                                expansion_score = score * 0.7
                                
                                expansion_clothing_cat = get_clothing_category(include_term)
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
    
    # Process AI response with enhanced filtering
    if ai_response:
        print(f"\n🤖 AI RESPONSE processing (HIGH PRIORITY)...")
        
        bold_headings = extract_bold_headings_from_ai_response(ai_response)
        print(f"   📋 Found {len(bold_headings)} bold headings: {bold_headings}")
        
        for heading in bold_headings:
            heading_lower = heading.lower()
            cleaned_heading = re.sub(r'[^\w\s-]', '', heading_lower).strip()
            
            if (cleaned_heading and 
                len(cleaned_heading) > 2 and 
                cleaned_heading not in conversation_words):  # Filter AI headings too
                
                score, priority = get_keyword_score(cleaned_heading, 'ai', 3)
                
                if score > 0:  # Only add non-filtered keywords
                    if cleaned_heading not in keyword_scores:
                        keyword_scores[cleaned_heading] = score
                    else:
                        keyword_scores[cleaned_heading] = max(keyword_scores[cleaned_heading], score)
                    
                    print(f"   🔥 BOLD HEADING: '{cleaned_heading}' → {score} ({priority})")
                    
                    clothing_cat = get_clothing_category(cleaned_heading)
                    if clothing_cat:
                        current_input_categories.add(clothing_cat)
                    
                    try:
                        search_terms = get_search_terms_for_keyword(cleaned_heading)
                        if isinstance(search_terms, dict):
                            include_terms = search_terms.get('include', [])
                            exclude_terms = search_terms.get('exclude', [])
                            
                            for include_term in include_terms:
                                if (include_term not in keyword_scores and
                                    include_term not in conversation_words):  # Filter AI expansions too
                                    
                                    expansion_score = score * 0.8
                                    
                                    expansion_clothing_cat = get_clothing_category(include_term)
                                    if expansion_clothing_cat and expansion_clothing_cat in wanted_items:
                                        expansion_score *= 1.5
                                        print(f"      ➕ BOOSTED AI expansion: '{cleaned_heading}' → '{include_term}' ({expansion_score:.1f})")
                                    else:
                                        print(f"      ➕ AI expansion: '{cleaned_heading}' → '{include_term}' ({expansion_score:.1f})")
                                    
                                    keyword_scores[include_term] = expansion_score
                            
                            if exclude_terms:
                                global_exclusions.update(exclude_terms)
                    except:
                        pass
    
    # SMART CONFLICT RESOLUTION (RESTORED but improved)
    print(f"\n⚔️ SMART CONFLICT ANALYSIS:")
    print(f"   📦 Current input categories: {current_input_categories}")
    print(f"   🤝 Is multi-item request: {is_multi_item_request}")
    
    # Process accumulated keywords with SMART conflict checking and enhanced filtering
    accumulated_categories = set()
    conflicting_keywords = []
    
    if accumulated_keywords:
        print(f"\n📚 ACCUMULATED keywords (with SMART conflict detection and enhanced filtering)...")
        
        # First pass: identify categories and conflicts
        for keyword, old_weight in accumulated_keywords[:15]:
            if (keyword and len(keyword) > 2 and 
                keyword.lower() not in simple_responses and
                keyword.lower() not in conversation_words and  # Filter accumulated too
                not any(char.isdigit() for char in keyword)):
                
                clothing_cat = get_clothing_category(keyword)
                if clothing_cat:
                    accumulated_categories.add(clothing_cat)
                    
                    # SMART CONFLICT LOGIC: Only apply conflicts for single-item requests
                    if not is_multi_item_request and current_input_categories:
                        for current_cat in current_input_categories:
                            if current_cat in clothing_conflicts:
                                conflicts_with = clothing_conflicts[current_cat]['conflicts_with']
                                if clothing_cat in conflicts_with:
                                    conflicting_keywords.append(keyword)
                                    print(f"   ⚔️  CONFLICT: '{keyword}' ({clothing_cat}) conflicts with current {current_input_categories}")
                                    break
        
        print(f"   📦 Accumulated categories: {accumulated_categories}")
        print(f"   🗑️  Conflicting keywords to suppress: {len(conflicting_keywords)}")
        
        # Second pass: add non-conflicting keywords with decay
        for keyword, old_weight in accumulated_keywords[:10]:
            if (keyword and len(keyword) > 2 and 
                keyword.lower() not in simple_responses and
                keyword.lower() not in conversation_words and  # Filter accumulated too
                not any(char.isdigit() for char in keyword) and
                keyword not in conflicting_keywords):  # SKIP CONFLICTING KEYWORDS (only for single-item requests)
                
                if filter_skin_colors(keyword):
                    continue
                
                accumulated_score = old_weight * 0.4
                
                if keyword not in keyword_scores and accumulated_score > 15:
                    keyword_scores[keyword] = accumulated_score
                    print(f"   📜 '{keyword}' → {accumulated_score:.1f}")
        
        # Show what was filtered out
        if conflicting_keywords:
            print(f"   ❌ SUPPRESSED conflicting keywords:")
            for conflicting_kw in conflicting_keywords:
                print(f"      ❌ '{conflicting_kw}' (conflicts with {current_input_categories})")
        else:
            print(f"   ✅ No conflicts detected (multi-item request or no conflicts)")
    
    # Enhanced cleanup - more comprehensive exclusions
    enhanced_excluded_terms = [
        # Budget terms
        "rb", "ribu", "rupiah", "budget", "anggaran", "harga", "price", "juta", "jt",
        
        # Conversation words (expanded) - duplicated from conversation_words for safety
        "yang", "dan", "atau", "dengan", "untuk", "dari", "pada", "akan", "dapat", "ada", 
        "adalah", "ini", "itu", "saya", "anda", "kamu", "mereka", "dia", "sangat", "lebih", 
        "kurang", "bagus", "baik", "cocok", "sesuai", "tepat", "bisa", "juga", "hanya", "sudah",
        "jadi", "ganti", "oke", "ya", "iya", "tidak", "mau", "ingin", "buat", "gimana", "kayak",
        
        # Generic terms  
        "recommendation", "rekomendasi", "suggestion", "saran", "option", "pilihan", "choice", 
        "style", "gaya", "tampilan", "fit",
        
        # Physical descriptors
        "kulit", "skin", "tubuh", "body", "tinggi", "height", "berat", "weight",
        "cowo", "cewe", "pria", "wanita", "indonesia", "dari", "cm", "kg"
    ]
    
    cleanup_keywords = []
    for keyword in list(keyword_scores.keys()):
        if (keyword in enhanced_excluded_terms or 
            keyword in conversation_words or  # Double check
            len(keyword.split()) > 3 or
            len(keyword) <= 2):
            cleanup_keywords.append(keyword)
    
    for keyword in cleanup_keywords:
        del keyword_scores[keyword]
        print(f"   🗑️ Enhanced cleanup: '{keyword}'")
    
    # Sort and return
    ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 FINAL SMART CONFLICT-AWARE KEYWORDS (with enhanced filtering):")
    for i, (keyword, score) in enumerate(ranked_keywords[:15]):
        clothing_cat = get_clothing_category(keyword)
        
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
    
    print(f"\n📊 SMART CONFLICT RESOLUTION SUMMARY:")
    print(f"   🎯 Wanted items: {wanted_items}")
    print(f"   📝 Context items: {context_items}")
    print(f"   🤝 Multi-item request: {is_multi_item_request}")
    print(f"   ⚔️  Conflicts suppressed: {len(conflicting_keywords) if not is_multi_item_request else 0}")
    
    ai_high = len([k for k, s in ranked_keywords if s >= 500])
    user_high = len([k for k, s in ranked_keywords if 300 <= s < 500])
    medium = len([k for k, s in ranked_keywords if 150 <= s < 300])
    
    print(f"   🤖 AI high priority (500+): {ai_high}")
    print(f"   👤 User high priority (300-499): {user_high}")
    print(f"   📋 Medium priority (150-299): {medium}")
    print(f"   📝 Total keywords: {len(ranked_keywords)}")
    print("="*60)
    
    # Store results
    extract_ranked_keywords.last_exclusions = list(global_exclusions)
    extract_ranked_keywords.wanted_items = wanted_items
    extract_ranked_keywords.context_items = context_items
    
    return ranked_keywords[:15]

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
            "product_cache": {
                "all_result": pd.DataFrame(),
                "current_page": 0,
                "product_per_page": 5,
                "last_search_params": {},
                "has_more": False
            }
        }

        # Define the GENDER_BOOST constant
        GENDER_BOOST = 2.0  # High confidence score for direct gender matches

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
                            ranked_keywords = extract_ranked_keywords(
                                last_ai_response,  # Use the last AI response 
                                last_user_input,   # Use the user's input 
                                accumulated_keywords
                            )
                            
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
                            print()
                            
                            # Fetch products using the ranked keywords
                            try:
                                recommended_products, budget_status = await fetch_products_with_budget_awareness(
                                    db=db,  # Make sure db is the AsyncSession object
                                    top_keywords=translated_ranked_keywords,  # Make sure this is a list of tuples
                                    max_results=15,
                                    gender_category=user_gender,
                                    budget_range=budget_range
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
                            force_update = detect_gender_change_request(text_content)
                            detect_and_update_gender(text_content, user_context, force_update)

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
                    force_update = detect_gender_change_request(user_input)
                    detect_and_update_gender(translated_input, user_context, force_update)
                        
                    # Extract and accumulate keywords from user input with high weight
                    input_keywords = extract_ranked_keywords("", translated_input)
                    update_accumulated_keywords(input_keywords, user_context, is_user_input=True)
                    
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
                    update_accumulated_keywords(response_keywords, user_context, is_ai_response=True)
                    
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
                    
                    # SMART KEYWORD EXTRACTION with flexible context
                    input_keywords = extract_ranked_keywords("", translated_input, 
                                                                accumulated_keywords=[(k, v["weight"]) for k, v in user_context.get("accumulated_keywords", {}).items()])                                                       
                    # FLEXIBLE CONTEXT UPDATE
                    smart_keyword_context_update(user_input, user_context, input_keywords, is_user_input=True)
                                        
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


# Define a comprehensive list of gender terms in multiple languages
ALL_GENDER_TERMS = [
    # English
    "man", "woman", "male", "female", "boy", "girl",
    # Indonesian
    "pria", "laki-laki", "perempuan", "wanita", "lelaki", "cewek", "cowok", "cewe", "cowo"
]

# Gender categories for mapping specific terms to broader gender categories
GENDER_CATEGORIES = {
    "male": ["man", "male", "boy", "pria", "laki-laki", "lelaki", "cowok", "cowo"],
    "female": ["woman", "female", "girl", "perempuan", "wanita", "cewek", "cewe"]
}

# Create gender term mapping
GENDER_TERM_MAP = {}
for gender, terms in GENDER_CATEGORIES.items():
    for term in terms:
        GENDER_TERM_MAP[term] = gender

# Function to identify gender from keywords
def identify_gender_from_keywords(keywords):
    """
    Extract gender information from keywords.
    Returns a tuple of (gender_category, gender_term, confidence_score)
    """
    gender_hits = {}
    best_term = None
    highest_score = 0
    
    for keyword, score in keywords:
        # Check if the keyword is a gender term
        keyword_lower = keyword.lower()
        
        # Direct match with a gender term
        if keyword_lower in GENDER_TERM_MAP:
            gender = GENDER_TERM_MAP[keyword_lower]
            if gender not in gender_hits:
                gender_hits[gender] = 0
            gender_hits[gender] += score
            
            if score > highest_score:
                highest_score = score
                best_term = keyword_lower
        
        # Check if any word in a multi-word keyword is a gender term
        elif ' ' in keyword_lower:
            words = keyword_lower.split()
            for word in words:
                if word in GENDER_TERM_MAP:
                    gender = GENDER_TERM_MAP[word]
                    if gender not in gender_hits:
                        gender_hits[gender] = 0
                    gender_hits[gender] += score / len(words)  # Dilute the score for multi-word matches
                    
                    if score / len(words) > highest_score:
                        highest_score = score / len(words)
                        best_term = word
    
    # If we found gender hits, return the highest scoring one
    if gender_hits:
        best_gender = max(gender_hits.items(), key=lambda x: x[1])
        return (best_gender[0], best_term, best_gender[1])
    
    return (None, None, 0)

# Add a direct text check for gender inside update_accumulated_keywords
def detect_gender_directly_from_text(text, current_confidence):
    """Directly check text for gender terms without keyword extraction"""
    gender_category = None
    gender_term = None
    confidence = 0
    
    text_lower = text.lower()
    
    # Check each gender term directly
    for term in ALL_GENDER_TERMS:
        if term in text_lower:
            # Find category
            for category, terms in GENDER_CATEGORIES.items():
                if term in terms:
                    # Assign high confidence for direct matches
                    confidence = 10.0  # GENDER_BOOST value
                    if confidence > current_confidence:
                        gender_category = category
                        gender_term = term
                        return gender_category, gender_term, confidence
    
    return gender_category, gender_term, confidence

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

def update_accumulated_keywords(keywords, user_context, is_user_input=False, is_ai_response=False):
    """
    ENHANCED: Better frequency tracking with category-aware persistence.
    """
    from datetime import datetime
    
    print(f"\n📝 ENHANCED KEYWORD UPDATE")
    print("="*40)
    
    if "accumulated_keywords" not in user_context:
        user_context["accumulated_keywords"] = {}
    
    # Extract budget separately
    if is_user_input and user_context.get("current_text_input"):
        text_input = user_context["current_text_input"].lower()
        
        # Only extract budget if budget-related context exists
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
        else:
            print(f"💰 No budget context in input: '{text_input[:50]}...'")
        
        # Validate existing budget for sanity
        if "budget_range" in user_context and user_context["budget_range"]:
            budget = user_context["budget_range"]
            if isinstance(budget, tuple) and len(budget) == 2:
                min_price, max_price = budget
                
                # Clear unreasonably low budgets (likely from physical measurements)
                if (min_price and min_price < 10000) or (max_price and max_price < 10000):
                    print(f"💰 CLEARING suspicious budget: {budget} (too low, likely physical measurement)")
                    user_context["budget_range"] = None

    # Category-based persistence settings
    persistence_config = {
        'clothing_items': {'decay_rate': 0.1, 'max_age_minutes': 120},  # Long persistence
        'style_attributes': {'decay_rate': 0.15, 'max_age_minutes': 90},
        'colors': {'decay_rate': 0.2, 'max_age_minutes': 60},
        'gender_terms': {'decay_rate': 0.05, 'max_age_minutes': 240},  # Very long persistence but low weight
        'occasions': {'decay_rate': 0.4, 'max_age_minutes': 30},       # Short persistence
        'default': {'decay_rate': 0.25, 'max_age_minutes': 45}
    }
    
    def get_keyword_category(keyword):
        """Determine keyword category for persistence settings"""
        keyword_lower = keyword.lower()
        
        categories = {
            'clothing_items': ['kemeja', 'shirt', 'blouse', 'dress', 'gaun', 'celana', 'pants', 'jacket'],
            'style_attributes': ['casual', 'formal', 'elegant', 'panjang', 'pendek', 'slim', 'oversized'],
            'colors': ['white', 'black', 'red', 'blue', 'putih', 'hitam', 'merah', 'biru'],
            'gender_terms': ['perempuan', 'wanita', 'female', 'woman', 'pria', 'laki-laki', 'male', 'man'],
            'occasions': ['office', 'kantor', 'party', 'pesta', 'wedding', 'beach']
        }
        
        for category, terms in categories.items():
            if any(term in keyword_lower for term in terms):
                return category
        
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
            estimated_frequency = max(1, score / 100)  # User scores are higher
        else:
            frequency_boost = 1.0
            estimated_frequency = max(1, score / 50)   # AI scores are lower
        
        if keyword_lower in user_context["accumulated_keywords"]:
            # Update existing keyword
            data = user_context["accumulated_keywords"][keyword_lower]
            
            old_frequency = data.get("total_frequency", 1)
            new_frequency = old_frequency + (estimated_frequency * frequency_boost)
            
            # Category-aware weight calculation
            config = persistence_config.get(category, persistence_config['default'])
            base_weight = new_frequency * 30
            
            # Apply category multiplier
            if category == 'gender_terms':
                base_weight *= 0.5  # Reduce gender weight
            elif category == 'clothing_items':
                base_weight *= 1.5  # Boost clothing items
            elif category == 'occasions':
                base_weight *= 0.7  # Reduce occasion weight
            
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
            
            # Apply category multiplier for new keywords too
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
    
    # Enhanced cleanup with category awareness
    category_cleanup(user_context, persistence_config)
    
    print(f"📊 Updates: {updates_made}, New: {new_keywords_added}")
    print(f"📚 Total: {len(user_context['accumulated_keywords'])}")
    print("="*40)

def category_cleanup(user_context, persistence_config):
    """
    Enhanced cleanup that respects category-based persistence rules while preserving 
    keywords needed for fashion category change detection.
    """
    if "accumulated_keywords" not in user_context:
        return
    
    from datetime import datetime, timedelta
    current_time = datetime.now()
    
    # Define fashion categories for change detection compatibility
    fashion_change_categories = {
        'tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'sweater', 'hoodie'],
        'bottoms': ['celana', 'pants', 'rok', 'skirt', 'jeans'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'blazer', 'coat', 'mantel'],
        'shoes': ['sepatu', 'shoes', 'heels', 'sneaker', 'boots'],
        'bags': ['tas', 'bag', 'handbag', 'backpack'],
        'occasions': ['office', 'kantor', 'party', 'pesta', 'wedding', 'pernikahan', 'beach', 'pantai', 'sport', 'olahraga'],
        'styles': ['casual', 'formal', 'elegant', 'vintage', 'modern', 'minimalist', 'bohemian']
    }
    
    def is_change_detection_keyword(keyword):
        """Check if keyword is important for fashion category change detection"""
        keyword_lower = keyword.lower()
        for category, terms in fashion_change_categories.items():
            if any(term in keyword_lower for term in terms):
                return True, category
        return False, None
    
    keywords_to_remove = []
    change_detection_keywords = {}  # Track keywords important for change detection
    
    for keyword, data in user_context["accumulated_keywords"].items():
        category = data.get("category", "default")
        config = persistence_config.get(category, persistence_config["default"])
        
        # Check if this keyword is important for change detection
        is_change_keyword, change_category = is_change_detection_keyword(keyword)
        
        # Check age
        try:
            last_seen = datetime.fromisoformat(data.get("last_seen", data.get("first_seen", "")))
            minutes_since_last_seen = (current_time - last_seen).total_seconds() / 60
        except:
            minutes_since_last_seen = 999  # Remove if can't parse
        
        # PRESERVE keywords important for change detection with extended lifetime
        if is_change_keyword:
            # Extended max age for change detection keywords
            extended_max_age = config["max_age_minutes"] * 2  # Double the lifetime
            
            if minutes_since_last_seen > extended_max_age:
                # Only remove if VERY old
                keywords_to_remove.append(keyword)
                continue
            else:
                # Track this as a change detection keyword
                change_detection_keywords[keyword] = {
                    'data': data,
                    'change_category': change_category,
                    'minutes_old': minutes_since_last_seen
                }
                print(f"   🔄 PRESERVED for change detection: '{keyword}' ({change_category}, {minutes_since_last_seen:.1f}min old)")
        else:
            # Apply normal aging rules for non-change-detection keywords
            if minutes_since_last_seen > config["max_age_minutes"]:
                keywords_to_remove.append(keyword)
                continue
        
        # Apply time-based decay (more gentle for change detection keywords)
        if minutes_since_last_seen > 10:
            if is_change_keyword:
                # Gentler decay for change detection keywords
                decay_factor = max(0.6, 1 - (minutes_since_last_seen / (config["max_age_minutes"] * 2)))
            else:
                # Normal decay for regular keywords
                decay_factor = max(0.3, 1 - (minutes_since_last_seen / config["max_age_minutes"]))
            
            data["total_frequency"] *= decay_factor
            data["weight"] = data["total_frequency"] * 30
            
            # Apply category weight adjustment after decay
            if category == 'gender_terms':
                data["weight"] *= 0.5
            elif category == 'clothing_items':
                data["weight"] *= 1.5
            elif category == 'occasions':
                data["weight"] *= 0.7
        
        # More lenient weight threshold for change detection keywords
        min_weight_threshold = 2 if is_change_keyword else 5
        if data["weight"] < min_weight_threshold:
            if not is_change_keyword:  # Don't remove change detection keywords based on weight alone
                keywords_to_remove.append(keyword)
    
    # Remove aged-out keywords (excluding change detection keywords)
    actual_removals = []
    for keyword in keywords_to_remove:
        if keyword not in change_detection_keywords:  # Double-check
            category = user_context["accumulated_keywords"][keyword].get("category", "unknown")
            del user_context["accumulated_keywords"][keyword]
            actual_removals.append(keyword)
    
    if actual_removals:
        print(f"🧹 Removed {len(actual_removals)} aged-out keywords (preserved {len(change_detection_keywords)} for change detection)")
    
    # Enhanced category limits that preserve change detection diversity
    category_limits = {
        'clothing_items': 15,  # Keep more clothing items
        'style_attributes': 10,
        'colors': 8,
        'gender_terms': 2,     # Keep only a few gender terms
        'occasions': 5,        # Limit occasions
        'default': 8
    }
    
    # Group by category
    by_category = {}
    for keyword, data in user_context["accumulated_keywords"].items():
        category = data.get("category", "default")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((keyword, data))
    
    # Limit each category while ensuring change detection keyword diversity
    final_keywords = {}
    
    # First, ensure we keep at least one keyword from each fashion change category
    change_category_representation = {}
    for keyword, change_info in change_detection_keywords.items():
        change_cat = change_info['change_category']
        if change_cat not in change_category_representation:
            change_category_representation[change_cat] = []
        change_category_representation[change_cat].append((keyword, change_info['data']))
    
    # Ensure representation from each change category (keep top keyword from each)
    guaranteed_keywords = set()
    for change_cat, keywords_in_cat in change_category_representation.items():
        if keywords_in_cat:
            # Keep the highest weight keyword from each change category
            top_keyword, top_data = max(keywords_in_cat, key=lambda x: x[1]["weight"])
            final_keywords[top_keyword] = top_data
            guaranteed_keywords.add(top_keyword)
            print(f"   🔒 GUARANTEED for '{change_cat}': '{top_keyword}' (weight: {top_data['weight']:.1f})")
    
    # Now apply normal category limits for remaining keywords
    for category, items in by_category.items():
        limit = category_limits.get(category, 8)
        sorted_items = sorted(items, key=lambda x: x[1]["weight"], reverse=True)
        
        added_from_category = 0
        for keyword, data in sorted_items:
            # Skip if already guaranteed
            if keyword in guaranteed_keywords:
                added_from_category += 1
                continue
                
            # Add up to limit
            if added_from_category < limit:
                final_keywords[keyword] = data
                added_from_category += 1
    
    removed_count = len(user_context["accumulated_keywords"]) - len(final_keywords)
    user_context["accumulated_keywords"] = final_keywords
    
    if removed_count > 0:
        print(f"📉 Category limits applied: removed {removed_count} keywords")
        print(f"🔄 Change detection diversity: {len(change_category_representation)} categories represented")

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
    """Return multiplier based on keyword category to prioritize fashion over occasions"""
    keyword_lower = keyword.lower()
    
    # HIGHEST PRIORITY - Core clothing items
    clothing_items = [
        "kemeja", "shirt", "blouse", "blus", "dress", "gaun", "celana", "pants", 
        "rok", "skirt", "jeans", "jaket", "jacket", "sweater", "cardigan", 
        "hoodie", "blazer", "coat", "mantel", "atasan", "kaos", "t-shirt"
    ]
    
    # HIGH PRIORITY - Style attributes  
    style_attributes = [
        "casual", "formal", "elegant", "vintage", "modern", "minimalist",
        "bohemian", "oversized", "slim", "ketat", "longgar"
    ]
    
    # MEDIUM PRIORITY - Colors and materials
    colors_materials = [
        "white", "black", "red", "blue", "green", "putih", "hitam", "merah",
        "cotton", "silk", "denim", "katun", "sutra"
    ]
    
    # LOW PRIORITY - Occasions (this is the fix!)
    occasions = [
        "office", "kantor", "party", "pesta", "wedding", "pernikahan", 
        "beach", "pantai", "sport", "olahraga", "work", "kerja"
    ]
    
    # Check category and return appropriate multiplier
    if any(item in keyword_lower for item in clothing_items):
        return 4.0  # HIGHEST priority for clothing
    elif any(style in keyword_lower for style in style_attributes):
        return 3.0  # HIGH priority for styles
    elif any(color in keyword_lower for color in colors_materials):
        return 2.0  # MEDIUM priority for colors/materials
    elif any(occasion in keyword_lower for occasion in occasions):
        return 0.5  # LOW priority for occasions (KEY FIX!)
    else:
        return 1.0  # Default

def detect_fashion_category_change(user_input, user_context):
    """
    IMPROVED: Nuanced conflict resolution - nuclear for clothing type changes, gentle for style changes
    """
    print(f"\n🔍 IMPROVED CATEGORY CHANGE DETECTION")
    print("="*50)
    
    user_input_lower = user_input.lower()
    
    # STEP 1: Check if this is a multi-item request
    is_multi_item = detect_multi_item_request(user_input)
    
    # Define clothing TYPES (major categories) vs ATTRIBUTES (minor categories)
    clothing_types = {
        'tops': {
            'keywords': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'sweater', 'hoodie', 'cardigan', 'blazer', 'tank', 'top'],
            'conflicts_with': ['bottoms_pants', 'bottoms_skirts', 'dresses']
        },
        'bottoms_pants': {
            'keywords': ['celana', 'pants', 'jeans', 'trousers', 'leggings'],
            'conflicts_with': ['tops', 'dresses', 'bottoms_skirts']
        },
        'bottoms_skirts': {
            'keywords': ['rok', 'skirt'],
            'conflicts_with': ['tops', 'dresses', 'bottoms_pants']
        },
        'dresses': {
            'keywords': ['dress', 'gaun', 'terusan'],
            'conflicts_with': ['tops', 'bottoms_pants', 'bottoms_skirts']
        },
        'outerwear': {
            'keywords': ['jaket', 'jacket', 'coat', 'mantel'],
            'conflicts_with': []
        }
    }
    
    # Define ATTRIBUTE categories (these should NOT trigger nuclear reduction)
    attribute_categories = {
        'colors': ['white', 'putih', 'black', 'hitam', 'red', 'merah', 'blue', 'biru', 
                  'green', 'hijau', 'yellow', 'kuning', 'brown', 'coklat', 'pink', 
                  'purple', 'ungu', 'orange', 'oranye', 'grey', 'abu-abu', 'navy', 'beige'],
        'styles': ['casual', 'formal', 'elegant', 'vintage', 'modern', 'minimalist', 
                  'bohemian', 'oversized', 'slim', 'ketat', 'longgar', 'loose', 'tight', 
                  'fitted', 'relaxed'],
        'sleeves': ['lengan panjang', 'lengan pendek', 'long sleeve', 'short sleeve', 
                   'sleeveless', 'tanpa lengan', 'panjang', 'pendek'],
        'materials': ['cotton', 'katun', 'silk', 'sutra', 'denim', 'wool', 'wol', 
                     'polyester', 'linen', 'leather', 'kulit'],
        'lengths': ['maxi', 'mini', 'midi', 'crop', 'cropped', 'long', 'short'],
        'patterns': ['striped', 'polka', 'floral', 'geometric', 'plain', 'polos']
    }
    
    def get_clothing_type_from_keyword(keyword):
        keyword_lower = keyword.lower()
        for category, data in clothing_types.items():
            if any(term in keyword_lower for term in data['keywords']):
                return category
        return None
    
    def get_attribute_type_from_keyword(keyword):
        keyword_lower = keyword.lower()
        for category, terms in attribute_categories.items():
            if any(term in keyword_lower for term in terms):
                return category
        return None
    
    def is_true_type_conflict(cat1, cat2):
        if cat1 in clothing_types and cat2 in clothing_types[cat1]['conflicts_with']:
            return True
        return False
    
    # STEP 2: Analyze current input
    current_clothing_types = set()
    current_attributes = {}  # category -> [keywords]
    
    # Find clothing types in current input
    for category, data in clothing_types.items():
        for term in data['keywords']:
            if term in user_input_lower:
                current_clothing_types.add(category)
                break
    
    # Find attributes in current input
    for attr_category, terms in attribute_categories.items():
        current_attrs_in_category = []
        for term in terms:
            if term in user_input_lower:
                current_attrs_in_category.append(term)
        if current_attrs_in_category:
            current_attributes[attr_category] = current_attrs_in_category
    
    # STEP 3: Analyze accumulated keywords
    accumulated_clothing_types = {}
    accumulated_attributes = {}
    
    if "accumulated_keywords" in user_context:
        for keyword, data in user_context["accumulated_keywords"].items():
            # Check clothing types
            clothing_type = get_clothing_type_from_keyword(keyword)
            if clothing_type:
                if clothing_type not in accumulated_clothing_types:
                    accumulated_clothing_types[clothing_type] = []
                accumulated_clothing_types[clothing_type].append((keyword, data["weight"]))
            
            # Check attributes
            attr_type = get_attribute_type_from_keyword(keyword)
            if attr_type:
                if attr_type not in accumulated_attributes:
                    accumulated_attributes[attr_type] = []
                accumulated_attributes[attr_type].append((keyword, data["weight"]))
    
    print(f"📝 Current clothing types: {current_clothing_types}")
    print(f"🎨 Current attributes: {current_attributes}")
    print(f"📚 Accumulated clothing types: {list(accumulated_clothing_types.keys())}")
    print(f"🎨 Accumulated attributes: {list(accumulated_attributes.keys())}")
    print(f"🤝 Is multi-item request: {is_multi_item}")
    
    # STEP 4: DECISION LOGIC - Major vs Minor Changes
    
    # Check for MAJOR CHANGE (clothing type conflicts)
    major_change_detected = False
    minor_change_detected = False
    
    if current_clothing_types and accumulated_clothing_types:
        if is_multi_item:
            print(f"🤝 MULTI-ITEM REQUEST - checking for type conflicts")
            
            for current_type in current_clothing_types:
                for acc_type in accumulated_clothing_types.keys():
                    if is_true_type_conflict(current_type, acc_type):
                        major_change_detected = True
                        print(f"   ⚔️  MAJOR CONFLICT: '{current_type}' conflicts with '{acc_type}'")
                        break
                if major_change_detected:
                    break
        else:
            print(f"👕 SINGLE-ITEM REQUEST - checking for type conflicts")
            
            for current_type in current_clothing_types:
                if current_type in clothing_types:
                    conflicts_with = clothing_types[current_type]['conflicts_with']
                    for acc_type in accumulated_clothing_types.keys():
                        if acc_type in conflicts_with:
                            major_change_detected = True
                            print(f"   ⚔️  MAJOR CONFLICT: '{current_type}' conflicts with '{acc_type}'")
                            break
                if major_change_detected:
                    break
    
    # Check for MINOR CHANGE (attribute changes)
    if current_attributes and accumulated_attributes:
        for attr_category, current_attrs in current_attributes.items():
            if attr_category in accumulated_attributes:
                # Check if we're changing to different values in same category
                acc_keywords_in_category = [kw for kw, _ in accumulated_attributes[attr_category]]
                current_terms_lower = [term.lower() for term in current_attrs]
                
                # If none of the current attributes match accumulated ones
                overlap = any(any(term in acc_kw.lower() for term in current_terms_lower) 
                             for acc_kw in acc_keywords_in_category)
                
                if not overlap:
                    minor_change_detected = True
                    print(f"   🎨 MINOR CHANGE in {attr_category}: {acc_keywords_in_category} → {current_attrs}")
    
    # STEP 5: APPLY APPROPRIATE RESOLUTION
    
    if major_change_detected:
        print(f"   ☢️  APPLYING NUCLEAR REDUCTION for clothing type conflicts")
        
        nuclear_reduction_applied = 0
        for acc_type, keywords_in_type in accumulated_clothing_types.items():
            should_reduce = False
            
            if is_multi_item:
                # For multi-item, reduce conflicting types
                for current_type in current_clothing_types:
                    if is_true_type_conflict(current_type, acc_type):
                        should_reduce = True
                        break
            else:
                # For single-item, reduce types that conflict with current type
                for current_type in current_clothing_types:
                    if current_type in clothing_types:
                        conflicts_with = clothing_types[current_type]['conflicts_with']
                        if acc_type in conflicts_with:
                            should_reduce = True
                            break
            
            if should_reduce:
                for keyword, weight in keywords_in_type:
                    if keyword in user_context["accumulated_keywords"]:
                        old_weight = user_context["accumulated_keywords"][keyword]["weight"]
                        new_weight = old_weight * 0.001  # 99.9% reduction
                        user_context["accumulated_keywords"][keyword]["weight"] = new_weight
                        nuclear_reduction_applied += 1
                        print(f"   ☢️  NUCLEAR: '{keyword}' {old_weight:.1f} → {new_weight:.1f}")
        
        print(f"   📊 Applied nuclear reduction to {nuclear_reduction_applied} conflicting clothing type keywords")
        return True
        
    elif minor_change_detected:
        print(f"   🎨 APPLYING GENTLE REDUCTION for attribute changes")
        
        gentle_reduction_applied = 0
        for attr_category, current_attrs in current_attributes.items():
            if attr_category in accumulated_attributes:
                acc_keywords_in_category = accumulated_attributes[attr_category]
                current_terms_lower = [term.lower() for term in current_attrs]
                
                for keyword, weight in acc_keywords_in_category:
                    # Check if this keyword conflicts with current attributes
                    conflicts_with_current = not any(term in keyword.lower() for term in current_terms_lower)
                    
                    if conflicts_with_current and keyword in user_context["accumulated_keywords"]:
                        old_weight = user_context["accumulated_keywords"][keyword]["weight"]
                        new_weight = old_weight * 0.3  # 70% reduction (gentle)
                        user_context["accumulated_keywords"][keyword]["weight"] = new_weight
                        gentle_reduction_applied += 1
                        print(f"   🎨 GENTLE: '{keyword}' {old_weight:.1f} → {new_weight:.1f}")
        
        print(f"   📊 Applied gentle reduction to {gentle_reduction_applied} conflicting attribute keywords")
        return False  # Return False since this is not a major change
    
    print(f"✅ No conflicts detected")
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
    Detect gender from user input and update context.
    Only updates if no gender exists or if force_update=True.
    """
    current_gender = user_context.get("user_gender", {})
    has_existing_gender = current_gender.get("category") is not None
    
    # Don't detect if we already have gender (unless forced)
    if has_existing_gender and not force_update:
        print(f"👤 Using existing gender: {current_gender['category']} (confidence: {current_gender.get('confidence', 0):.1f})")
        return current_gender["category"]
    
    # Gender detection patterns
    gender_patterns = {
        'male': [
            r'\b(pria|laki-laki|male|man|cowok|cowo)\b',
            r'\buntuk\s+(pria|laki-laki|male|man|cowok)\b',
            r'\b(saya|i am|i\'m)\s+(pria|laki-laki|male|man)\b'
        ],
        'female': [
            r'\b(perempuan|wanita|female|woman|cewek|cewe)\b',
            r'\buntuk\s+(perempuan|wanita|female|woman|cewek)\b',
            r'\b(saya|i am|i\'m)\s+(perempuan|wanita|female|woman)\b'
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
                confidence = 10.0  # High confidence for direct detection
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
    
    # Return existing gender if available
    if has_existing_gender:
        print(f"👤 No new gender detected, using existing: {current_gender['category']}")
        return current_gender["category"]
    
    print("👤 No gender detected")
    return None

def get_user_gender(user_context):
    """
    Safely get user gender information from context.
    """
    gender_info = user_context.get("user_gender", {})
    
    return {
        "category": gender_info.get("category"),
        "term": gender_info.get("term"),
        "confidence": gender_info.get("confidence", 0),
        "last_updated": gender_info.get("last_updated")
    }

def detect_gender_change_request(user_input):
    """
    Detect if user is explicitly trying to change their gender.
    """
    change_patterns = [
        r'\b(actually|sebenarnya)\s+(i am|saya)\s+(male|female|pria|wanita)',
        r'\b(change|ganti|ubah)\s+(to|ke|menjadi)\s+(male|female|pria|wanita)',
        r'\b(i\'m|saya)\s+(not|bukan)\s+(male|female|pria|wanita)',
        r'\b(correction|koreksi|ralat)',
    ]
    
    user_input_lower = user_input.lower()
    
    for pattern in change_patterns:
        if re.search(pattern, user_input_lower):
            return True
    
    return False

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