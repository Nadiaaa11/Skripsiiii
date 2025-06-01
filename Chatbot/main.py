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

def extract_ranked_keywords(ai_response: str = None, translated_input: str = None, accumulated_keywords=None):
    """
    Extract and rank keywords primarily from the current conversation (latest input and response),
    using accumulated keywords only as fallback or supplementary information.
    
    Parameters:
    - ai_response: The latest AI response text
    - translated_input: The latest user input text
    - accumulated_keywords: Historical keywords (used only for fallback)
    
    Returns:
    - List of (keyword, score) tuples sorted by score, prioritizing current conversation terms
    """
    # Constants for weighting
    FREQUENCY_WEIGHT = 2.0
    POSITION_WEIGHT = 1.0
    POS_WEIGHT = 1.5
    USER_KEYWORD_BOOST = 5.0  # Increased to prioritize current user input
    MULTI_WORD_USER_BOOST = 6.0  # Increased to prioritize multi-word phrases
    MULTI_WORD_AI_BOOST = 3.0
    FASHION_TERM_BOOST = 4.0
    ATTRIBUTE_BOOST = 3.5
    GENDER_BOOST = 2.0  # Very high boost for gender terms
    CURRENT_CONVERSATION_BOOST = 2.0  # Boost for keywords from current conversation
    
    # Initialize dictionary to store keyword scores
    keyword_scores = {}
    
    # Flag to track if we have content in current conversation
    has_current_content = bool(translated_input or ai_response)
    
    # Check if we have any content to process from current conversation
    if not has_current_content:
        # No current content, use accumulated keywords as fallback
        if accumulated_keywords and isinstance(accumulated_keywords, list):
            return sorted([(kw, score) for kw, score in accumulated_keywords 
                          if kw and isinstance(kw, str) and len(kw) > 2], 
                         key=lambda x: x[1], reverse=True)
        else:
            return []

    # Fashion-related terms in English and Indonesian to boost
    fashion_terms = {
        # Clothing types
        "kemeja": 1.0, "shirt": 1.0, "blouse": 1.0, "blus": 1.0, 
        "dress": 1.0, "gaun": 1.0, "rok": 1.0, "skirt": 1.0,
        "celana": 1.0, "pants": 1.0, "jeans": 1.0, "denim": 1.0,
        "jacket": 1.0, "jaket": 1.0, "sweater": 1.0, "cardigan": 1.0,
        "atasan": 1.0, "top": 0.8, "t-shirt": 0.9, "kaos": 0.9,
        "hoodie": 0.9, "coat": 0.9, "mantel": 0.9, "blazer": 1.0,
        
        # Styles
        "formal": 0.9, "casual": 0.9, "kasual": 0.9, "santai": 0.9,
        "vintage": 0.9, "modern": 0.8, "elegant": 0.9, "elegan": 0.9,
        "bohemian": 0.9, "boho": 0.9, "minimalist": 0.9, "minimalis": 0.9,
        "feminine": 0.9, "feminin": 0.9, "masculine": 0.9, "maskulin": 0.9,
        "etnik": 0.9, "ethnic": 0.9, "preppy": 0.8, "streetwear": 0.9,
        
        # Materials
        "cotton": 0.8, "katun": 0.8, "silk": 0.8, "sutra": 0.8,
        "wool": 0.8, "wol": 0.8, "linen": 0.8, "polyester": 0.8,
        "leather": 0.8, "kulit": 0.8, "denim": 0.8, "knit": 0.8,
        "rajut": 0.8, "satin": 0.8, "velvet": 0.8, "beludru": 0.8,
        
        # Features
        "sleeve": 0.7, "lengan": 0.7, "collar": 0.7, "kerah": 0.7,
        "pocket": 0.7, "kantong": 0.7, "button": 0.7, "kancing": 0.7,
        "zipper": 0.7, "resleting": 0.7, "embroidery": 0.8, "bordir": 0.8,
        "pattern": 0.8, "motif": 0.8, "print": 0.8, "colorful": 0.7,
        "berwarna": 0.7, "plain": 0.7, "polos": 0.7, "renda": 0.8, "lace": 0.8,
        
        # Body attributes (boosted higher)
        "height": 1.2, "tinggi": 1.2, "weight": 1.2, "berat": 1.2,
        "skinny": 1.1, "kurus": 1.1, "slim": 1.1, "langsing": 1.1,
        "plus size": 1.2, "ukuran plus": 1.2, "tall": 1.1, "tinggi": 1.1,
        "short": 1.1, "pendek": 1.1, "petite": 1.1, "mungil": 1.1,
        "skin tone": 1.2, "kulit": 1.2, "warna kulit": 1.2,
        
        # Colors
        "white": 0.7, "putih": 0.7, "black": 0.7, "hitam": 0.7,
        "red": 0.7, "merah": 0.7, "blue": 0.7, "biru": 0.7,
        "green": 0.7, "hijau": 0.7, "yellow": 0.7, "kuning": 0.7,
        "brown": 0.7, "coklat": 0.7, "pink": 0.7, "merah muda": 0.7,
        "purple": 0.7, "ungu": 0.7, "orange": 0.7, "oranye": 0.7,
        "grey": 0.7, "abu-abu": 0.7, "navy": 0.7, "biru tua": 0.7,
        "beige": 0.7, "krem": 0.7,
        
        # Gender terms (very high boost)
        "perempuan": 2.0, "wanita": 2.0, "female": 2.0, "woman": 2.0,
        "pria": 2.0, "laki-laki": 2.0, "male": 2.0, "man": 2.0
    }

    # First, prioritize processing user input (most important)
    if translated_input and isinstance(translated_input, str) and not translated_input.startswith(("http://", "https://")):
        doc_user = nlp(translated_input)
        user_noun_chunks = [
            chunk.text.lower() for chunk in doc_user.noun_chunks 
            if len(chunk.text) > 2 and not any(token.lower_ in stop_words or token.pos_ == 'PRON' for token in chunk)
        ]
        user_entities = [
            ent.text.lower() for ent in doc_user.ents 
            if len(ent.text) > 2 and not any(token.lower_ in stop_words or token.pos_ == 'PRON' for token in ent)
        ]
        user_pos_keywords = [
            token.text.lower() for token in doc_user 
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] 
            and len(token.text) > 2 
            and token.text.lower() not in stop_words
        ]
        all_user_keywords = user_noun_chunks + user_entities + user_pos_keywords
        user_keyword_freq = Counter(all_user_keywords)

        # Direct gender and attribute detection from raw input (highest priority)
        text_lower = translated_input.lower()
        
        # Directly check for gender terms
        for gender_term in ["perempuan", "wanita", "female", "woman", "pria", "laki-laki", "male", "man"]:
            if gender_term in text_lower:
                # Give extremely high weight to gender terms
                keyword_scores[gender_term] = keyword_scores.get(gender_term, 0) + GENDER_BOOST
                
                # Also add gender to multi-word context
                for chunk in user_noun_chunks:
                    if gender_term in chunk:
                        keyword_scores[chunk] = keyword_scores.get(chunk, 0) + GENDER_BOOST * 0.8
        
        # Directly check for body attributes that are critical for fashion
        body_attributes = ["tinggi", "height", "berat", "weight", "kulit", "skin", "ukuran", "size"]
        for attr in body_attributes:
            if attr in text_lower:
                # Give high weight to body attributes
                keyword_scores[attr] = keyword_scores.get(attr, 0) + ATTRIBUTE_BOOST * CURRENT_CONVERSATION_BOOST
                
                # Also check for attribute phrases
                for chunk in user_noun_chunks:
                    if attr in chunk:
                        keyword_scores[chunk] = keyword_scores.get(chunk, 0) + ATTRIBUTE_BOOST * CURRENT_CONVERSATION_BOOST

        # Update keyword scores with user keywords and apply boost
        for word, count in user_keyword_freq.items():
            # Add fashion term boost
            fashion_boost = 0
            for fashion_term, boost_factor in fashion_terms.items():
                if fashion_term in word or word in fashion_term:
                    fashion_boost = FASHION_TERM_BOOST * boost_factor * CURRENT_CONVERSATION_BOOST
                    break
                    
            keyword_scores[word] = keyword_scores.get(word, 0) + (count * USER_KEYWORD_BOOST) + fashion_boost
        
        # Boost multi-word user phrases
        for chunk in user_noun_chunks:
            if ' ' in chunk and len(chunk.split()) <= 3:
                # Check if phrase contains fashion terms
                fashion_boost = 0
                for fashion_term, boost_factor in fashion_terms.items():
                    if fashion_term in chunk:
                        fashion_boost = FASHION_TERM_BOOST * boost_factor * CURRENT_CONVERSATION_BOOST
                        break
                        
                keyword_scores[chunk] = keyword_scores.get(chunk, 0) + MULTI_WORD_USER_BOOST + fashion_boost

        # Special boost for phrases that combine multiple important factors
        important_combos = []
        for chunk in all_user_keywords:
            # Find combinations like "kemeja untuk perempuan" or "baju kulit putih"
            if any(gender in chunk for gender in ["perempuan", "wanita", "female", "pria", "laki-laki", "male"]) and \
               any(clothing in chunk for clothing in ["kemeja", "baju", "dress", "gaun", "rok", "celana", "atasan"]):
                important_combos.append(chunk)
                
        for combo in important_combos:
            keyword_scores[combo] = keyword_scores.get(combo, 0) + GENDER_BOOST + FASHION_TERM_BOOST * CURRENT_CONVERSATION_BOOST

        # Check for numeric expressions that may indicate height or weight
        # Find height/weight patterns like "150 cm" or "50 kg"
        height_weight_pattern = re.compile(r'(\d+)\s*(cm|kg)')
        for match in height_weight_pattern.finditer(translated_input.lower()):
            full_match = match.group(0)
            value = match.group(1)
            unit = match.group(2)
            
            if unit == 'cm':
                keyword_scores["tinggi"] = keyword_scores.get("tinggi", 0) + ATTRIBUTE_BOOST * CURRENT_CONVERSATION_BOOST
                keyword_scores[f"tinggi {value} cm"] = keyword_scores.get(f"tinggi {value} cm", 0) + ATTRIBUTE_BOOST * 1.5 * CURRENT_CONVERSATION_BOOST
            elif unit == 'kg':
                keyword_scores["berat"] = keyword_scores.get("berat", 0) + ATTRIBUTE_BOOST * CURRENT_CONVERSATION_BOOST
                keyword_scores[f"berat {value} kg"] = keyword_scores.get(f"berat {value} kg", 0) + ATTRIBUTE_BOOST * 1.5 * CURRENT_CONVERSATION_BOOST

    # Then process AI response (second priority)
    if ai_response:
        doc_ai = nlp(ai_response)

        # Extract keywords from AI response
        ai_noun_chunks = [
            chunk.text.lower() for chunk in doc_ai.noun_chunks 
            if len(chunk.text) > 2 and not any(token.lower_ in stop_words or token.pos_ == 'PRON' for token in chunk)
        ]
        ai_entities = [
            ent.text.lower() for ent in doc_ai.ents 
            if len(ent.text) > 2 and not any(token.lower_ in stop_words or token.pos_ == 'PRON' for token in ent)
        ]
        ai_pos_keywords = [
            token.text.lower() for token in doc_ai 
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] 
            and len(token.text) > 2 
            and token.text.lower() not in stop_words
        ]
        all_ai_keywords = ai_noun_chunks + ai_entities + ai_pos_keywords
        ai_keyword_freq = Counter(all_ai_keywords)

        # Update keyword scores with AI keywords
        for i, token in enumerate(doc_ai):
            token_text = token.text.lower()
            if len(token_text) < 3 or token_text in stop_words:
                continue

            # Frequency score
            freq_score = ai_keyword_freq.get(token_text, 0)
            # Position score (less important now)
            position_score = 1.0 - (i / len(doc_ai))
            # POS score
            pos_score = {'PROPN': 3, 'NOUN': 2, 'ADJ': 1}.get(token.pos_, 0)
            # Fashion term boost
            fashion_boost = 0
            for fashion_term, boost_factor in fashion_terms.items():
                if fashion_term in token_text or token_text in fashion_term:
                    fashion_boost = FASHION_TERM_BOOST * boost_factor
                    break
                    
            # Final score with fashion boost
            ai_score = (freq_score * FREQUENCY_WEIGHT) + (position_score * POSITION_WEIGHT) + \
                       (pos_score * POS_WEIGHT) + fashion_boost
                       
            # Update keyword scores
            keyword_scores[token_text] = keyword_scores.get(token_text, 0) + ai_score

        # Boost multi-word AI phrases
        for chunk in ai_noun_chunks:
            if ' ' in chunk and len(chunk.split()) <= 3:
                freq_score = ai_keyword_freq.get(chunk, 0)
                
                # Check if phrase contains fashion terms
                fashion_boost = 0
                for fashion_term, boost_factor in fashion_terms.items():
                    if fashion_term in chunk:
                        fashion_boost = FASHION_TERM_BOOST * boost_factor
                        break
                        
                chunk_score = freq_score * FREQUENCY_WEIGHT + MULTI_WORD_AI_BOOST + fashion_boost
                keyword_scores[chunk] = keyword_scores.get(chunk, 0) + chunk_score

    # Only use accumulated keywords as supplementary if we have current conversation content
    # and with reduced weight to prioritize current conversation
    if has_current_content and accumulated_keywords and isinstance(accumulated_keywords, list):
        for keyword, weight in accumulated_keywords:
            if keyword and isinstance(keyword, str) and len(keyword) > 2:
                # Only use accumulated keywords that didn't appear in current conversation
                # or boost those that did appear (reinforcement)
                if keyword.lower() in keyword_scores:
                    # Keyword already in current conversation - small reinforcement
                    keyword_scores[keyword.lower()] += weight * 0.3  # Reduced weight for historical keywords
                else:
                    # Check if this is an important fashion or attribute term
                    is_important_term = False
                    for term in list(fashion_terms.keys()) + ["height", "weight", "tinggi", "berat", "kulit"]:
                        if term in keyword.lower():
                            is_important_term = True
                            break
                    
                    # Only add historical keywords if they are important fashion terms
                    if is_important_term:
                        keyword_scores[keyword.lower()] = weight * 0.25  # Even more reduced weight for new terms

    # Clean the keyword dictionary by removing very generic terms
    generic_terms = ["saya", "anda", "dia", "yang", "dan", "atau", "juga", "dengan", "ini", "itu", 
                    "pada", "bisa", "untuk", "dari", "akan", "dalam", "dah", "lah", "ada", "tersebut",
                    "sangat", "lebih", "paling", "semua", "setiap", "beberapa", "tentu", "apakah", "bolehkah"]
    for term in generic_terms:
        if term in keyword_scores:
            del keyword_scores[term]

    # Sort keywords by score
    ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    # Log the top keywords being used
    logging.info(f"Latest conversation keywords: {ranked_keywords[:10]}")
    print(f"Latest conversation keywords: {ranked_keywords[:10]}")

    # Return the most meaningful keywords 
    return ranked_keywords[:30] 

# Add this comprehensive translation mapping function to your code
def get_search_terms_for_keyword(keyword):
    """
    Get both English and Indonesian search terms for a keyword to improve product matching.
    Returns a list of terms to search for in the database.
    """
    keyword_lower = keyword.lower().strip()
    
    # Comprehensive translation mapping
    translation_map = {
        # Clothing types
        'shirt': ['shirt', 'kemeja', 'baju', 'atasan'],
        'kemeja': ['kemeja', 'shirt', 'baju', 'atasan'],
        'blouse': ['blouse', 'blus', 'kemeja wanita', 'atasan wanita'],
        'blus': ['blus', 'blouse', 'kemeja wanita'],
        'dress': ['dress', 'gaun', 'terusan'],
        'gaun': ['gaun', 'dress', 'terusan'],
        'pants': ['pants', 'celana', 'bawahan'],
        'celana': ['celana', 'pants', 'bawahan'],
        'skirt': ['skirt', 'rok'],
        'rok': ['rok', 'skirt'],
        'jacket': ['jacket', 'jaket', 'jas'],
        'jaket': ['jaket', 'jacket', 'jas'],
        'sweater': ['sweater', 'baju hangat', 'jumper'],
        'cardigan': ['cardigan', 'kardigan'],
        'kardigan': ['kardigan', 'cardigan'],
        'jeans': ['jeans', 'jins', 'celana jeans', 'denim'],
        'hoodie': ['hoodie', 'jaket hoodie', 'sweater hoodie'],
        'coat': ['coat', 'mantel', 'jaket panjang'],
        'mantel': ['mantel', 'coat', 'jaket panjang'],
        'blazer': ['blazer', 'jas blazer'],
        'top': ['top', 'atasan', 'baju atas'],
        'atasan': ['atasan', 'top', 'baju atas'],
        't-shirt': ['t-shirt', 'tshirt', 'kaos', 'baju kaos'],
        'kaos': ['kaos', 't-shirt', 'tshirt', 'baju kaos'],
        
        # Colors
        'white': ['white', 'putih'],
        'putih': ['putih', 'white'],
        'black': ['black', 'hitam'],
        'hitam': ['hitam', 'black'],
        'red': ['red', 'merah'],
        'merah': ['merah', 'red'],
        'blue': ['blue', 'biru'],
        'biru': ['biru', 'blue'],
        'green': ['green', 'hijau'],
        'hijau': ['hijau', 'green'],
        'yellow': ['yellow', 'kuning'],
        'kuning': ['kuning', 'yellow'],
        'pink': ['pink', 'merah muda', 'rosa'],
        'purple': ['purple', 'ungu', 'violet'],
        'ungu': ['ungu', 'purple', 'violet'],
        'orange': ['orange', 'oranye', 'jingga'],
        'oranye': ['oranye', 'orange', 'jingga'],
        'brown': ['brown', 'coklat', 'cokelat'],
        'coklat': ['coklat', 'brown', 'cokelat'],
        'grey': ['grey', 'gray', 'abu-abu'],
        'gray': ['gray', 'grey', 'abu-abu'],
        'navy': ['navy', 'biru tua', 'biru dongker'],
        'beige': ['beige', 'krem', 'cream'],
        'krem': ['krem', 'beige', 'cream'],
        
        # Styles
        'casual': ['casual', 'santai', 'kasual'],
        'santai': ['santai', 'casual', 'kasual'],
        'kasual': ['kasual', 'casual', 'santai'],
        'formal': ['formal', 'resmi'],
        'resmi': ['resmi', 'formal'],
        'elegant': ['elegant', 'elegan'],
        'elegan': ['elegan', 'elegant'],
        'modern': ['modern', 'kontemporer'],
        'vintage': ['vintage', 'klasik', 'retro'],
        'klasik': ['klasik', 'vintage', 'retro'],
        'bohemian': ['bohemian', 'boho'],
        'boho': ['boho', 'bohemian'],
        'minimalist': ['minimalist', 'minimalis', 'simple'],
        'minimalis': ['minimalis', 'minimalist', 'simple'],
        'feminine': ['feminine', 'feminin'],
        'feminin': ['feminin', 'feminine'],
        'masculine': ['masculine', 'maskulin'],
        'maskulin': ['maskulin', 'masculine'],
        'ethnic': ['ethnic', 'etnik', 'tradisional'],
        'etnik': ['etnik', 'ethnic', 'tradisional'],
        'streetwear': ['streetwear', 'jalanan'],
        'oversized': ['oversized', 'longgar', 'besar'],
        'longgar': ['longgar', 'oversized', 'loose'],
        'slim': ['slim', 'ketat', 'fit'],
        'ketat': ['ketat', 'slim', 'tight'],
        
        # Materials
        'cotton': ['cotton', 'katun'],
        'katun': ['katun', 'cotton'],
        'silk': ['silk', 'sutra'],
        'sutra': ['sutra', 'silk'],
        'wool': ['wool', 'wol'],
        'wol': ['wol', 'wool'],
        'linen': ['linen', 'linen'],
        'polyester': ['polyester', 'poliester'],
        'leather': ['leather', 'kulit'],
        'kulit': ['kulit', 'leather'],
        'denim': ['denim', 'jeans'],
        'knit': ['knit', 'rajut'],
        'rajut': ['rajut', 'knit'],
        'satin': ['satin'],
        'velvet': ['velvet', 'beludru'],
        'beludru': ['beludru', 'velvet'],
        
        # Features
        'sleeve': ['sleeve', 'lengan'],
        'lengan': ['lengan', 'sleeve'],
        'collar': ['collar', 'kerah'],
        'kerah': ['kerah', 'collar'],
        'pocket': ['pocket', 'kantong', 'saku'],
        'kantong': ['kantong', 'pocket', 'saku'],
        'button': ['button', 'kancing'],
        'kancing': ['kancing', 'button'],
        'zipper': ['zipper', 'resleting'],
        'resleting': ['resleting', 'zipper'],
        'embroidery': ['embroidery', 'bordir'],
        'bordir': ['bordir', 'embroidery'],
        'pattern': ['pattern', 'motif', 'pola'],
        'motif': ['motif', 'pattern', 'pola'],
        'print': ['print', 'cetak'],
        'colorful': ['colorful', 'berwarna', 'warni'],
        'berwarna': ['berwarna', 'colorful', 'warni'],
        'plain': ['plain', 'polos'],
        'polos': ['polos', 'plain'],
        'lace': ['lace', 'renda'],
        'renda': ['renda', 'lace'],
        
        # Sizes/Fits
        'small': ['small', 'kecil', 's'],
        'kecil': ['kecil', 'small', 's'],
        'medium': ['medium', 'sedang', 'm'],
        'sedang': ['sedang', 'medium', 'm'],
        'large': ['large', 'besar', 'l'],
        'besar': ['besar', 'large', 'l'],
        'extra large': ['extra large', 'xl', 'sangat besar'],
        'tight': ['tight', 'ketat'],
        'loose': ['loose', 'longgar'],
        
        # Occasions
        'office': ['office', 'kantor', 'kerja'],
        'kantor': ['kantor', 'office', 'kerja'],
        'party': ['party', 'pesta'],
        'pesta': ['pesta', 'party'],
        'wedding': ['wedding', 'pernikahan'],
        'pernikahan': ['pernikahan', 'wedding'],
        'beach': ['beach', 'pantai'],
        'pantai': ['pantai', 'beach'],
        'sport': ['sport', 'olahraga'],
        'olahraga': ['olahraga', 'sport'],
    }
    
    # If the keyword has a direct mapping, return all variations
    if keyword_lower in translation_map:
        return translation_map[keyword_lower]
    
    # If no direct mapping, try to find partial matches
    search_terms = [keyword_lower]
    
    # Check if the keyword contains any mapped terms
    for mapped_term, variations in translation_map.items():
        if mapped_term in keyword_lower or keyword_lower in mapped_term:
            search_terms.extend(variations)
            break
    
    # Remove duplicates and return
    return list(set(search_terms))

async def fetch_products_from_db(db: AsyncSession, top_keywords: list, max_results=5, gender_category=None, budget_range=None):
    """
    Completely rewritten product fetching function that actually works.
    """
    logging.info(f"=== PRODUCT FETCH DEBUG ===")
    logging.info(f"Top keywords received: {[(kw, score) for kw, score in top_keywords[:10]]}")
    logging.info(f"Gender category: {gender_category}")
    logging.info(f"Budget range: {budget_range}")
    logging.info(f"Max results requested: {max_results}")
    
    try:
        # Step 1: Build the base query to get ALL available products first
        base_query = (
            select(Product.product_id, Product.product_name, Product.product_detail, 
                   Product.product_seourl, Product.product_gender,
                   ProductVariant.product_price, ProductVariant.size, ProductVariant.color, ProductVariant.stock,
                   ProductPhoto.productphoto_path)
            .select_from(Product)
            .join(ProductVariant, Product.product_id == ProductVariant.product_id)
            .join(ProductPhoto, Product.product_id == ProductPhoto.product_id)
            .where(ProductVariant.stock > 0)
            .distinct(Product.product_id)  # Ensure unique products
        )
        
        # Step 2: Apply gender filter
        if gender_category:
            if gender_category.lower() in ['female', 'woman', 'perempuan', 'wanita', 'cewek']:
                base_query = base_query.where(Product.product_gender == 'female')
                logging.info("Applied female gender filter")
            elif gender_category.lower() in ['male', 'man', 'pria', 'laki-laki', 'cowok']:
                base_query = base_query.where(Product.product_gender == 'male')
                logging.info("Applied male gender filter")
        
        # Step 3: Apply budget filter
        if budget_range and isinstance(budget_range, (tuple, list)) and len(budget_range) == 2:
            min_price, max_price = budget_range
            if min_price is not None and max_price is not None:
                base_query = base_query.where(ProductVariant.product_price.between(min_price, max_price))
                logging.info(f"Applied budget filter: {min_price} - {max_price}")
            elif min_price is not None:
                base_query = base_query.where(ProductVariant.product_price >= min_price)
                logging.info(f"Applied minimum price filter: {min_price}")
            elif max_price is not None:
                base_query = base_query.where(ProductVariant.product_price <= max_price)
                logging.info(f"Applied maximum price filter: {max_price}")
        
        # Step 4: Get all products that match basic criteria
        all_products_result = await db.execute(base_query)
        all_products = all_products_result.fetchall()
        logging.info(f"Found {len(all_products)} products matching basic criteria")
        
        if not all_products:
            logging.warning("No products found matching basic criteria")
            return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
        
        # Step 5: Create comprehensive keyword mapping for better matching
        def get_all_search_terms(keyword):
            """Get all possible search terms for a keyword"""
            keyword_lower = keyword.lower().strip()
            
            # Comprehensive mapping
            keyword_map = {
                # Clothing items
                'shirt': ['shirt', 'kemeja', 'baju kemeja', 'kemeja pria', 'kemeja wanita'],
                'kemeja': ['kemeja', 'shirt', 'baju kemeja', 'kemeja pria', 'kemeja wanita'],
                'blouse': ['blouse', 'blus', 'kemeja wanita', 'atasan wanita'],
                'blus': ['blus', 'blouse', 'kemeja wanita', 'atasan wanita'],
                'dress': ['dress', 'gaun', 'dres', 'terusan'],
                'gaun': ['gaun', 'dress', 'dres', 'terusan'],
                'pants': ['pants', 'celana', 'celana panjang'],
                'celana': ['celana', 'pants', 'celana panjang'],
                'skirt': ['skirt', 'rok', 'rok mini', 'rok panjang'],
                'rok': ['rok', 'skirt', 'rok mini', 'rok panjang'],
                'jacket': ['jacket', 'jaket', 'jas'],
                'jaket': ['jaket', 'jacket', 'jas'],
                'cardigan': ['cardigan', 'kardigan'],
                'sweater': ['sweater', 'baju hangat'],
                'hoodie': ['hoodie', 'jaket hoodie'],
                'jeans': ['jeans', 'celana jeans', 'denim'],
                'top': ['top', 'atasan', 'baju atasan'],
                'atasan': ['atasan', 'top', 'baju atasan'],
                't-shirt': ['t-shirt', 'tshirt', 'kaos', 'baju kaos'],
                'kaos': ['kaos', 't-shirt', 'tshirt', 'baju kaos'],
                'blazer': ['blazer', 'jas blazer'],
                
                # Colors
                'white': ['white', 'putih'],
                'putih': ['putih', 'white'],
                'black': ['black', 'hitam'],
                'hitam': ['hitam', 'black'],
                'red': ['red', 'merah'],
                'merah': ['merah', 'red'],
                'blue': ['blue', 'biru'],
                'biru': ['biru', 'blue'],
                'green': ['green', 'hijau'],
                'hijau': ['hijau', 'green'],
                'pink': ['pink', 'merah muda'],
                'yellow': ['yellow', 'kuning'],
                'kuning': ['kuning', 'yellow'],
                'brown': ['brown', 'coklat'],
                'coklat': ['coklat', 'brown'],
                'navy': ['navy', 'biru tua'],
                'grey': ['grey', 'gray', 'abu-abu'],
                'purple': ['purple', 'ungu'],
                'ungu': ['ungu', 'purple'],
                
                # Styles
                'casual': ['casual', 'santai', 'kasual'],
                'formal': ['formal', 'resmi'],
                'elegant': ['elegant', 'elegan'],
                'vintage': ['vintage', 'klasik'],
                'modern': ['modern'],
                'oversized': ['oversized', 'longgar'],
                'slim': ['slim', 'ketat'],
                'cropped': ['cropped', 'crop'],
                'long sleeve': ['long sleeve', 'lengan panjang'],
                'short sleeve': ['short sleeve', 'lengan pendek'],
                
                # Materials
                'cotton': ['cotton', 'katun'],
                'denim': ['denim', 'jeans'],
                'silk': ['silk', 'sutra'],
                'leather': ['leather', 'kulit'],
            }
            
            # Return mapped terms or just the original keyword
            return keyword_map.get(keyword_lower, [keyword_lower])
        
        # Step 6: Score each product based on keyword matches
        scored_products = []
        
        for product_row in all_products:
            # Extract product info from the row
            product_id = product_row[0]
            product_name = product_row[1]
            product_detail = product_row[2]
            product_seourl = product_row[3]
            product_gender = product_row[4]
            product_price = product_row[5]
            size = product_row[6]
            color = product_row[7]
            stock = product_row[8]
            photo_path = product_row[9]
            
            # Create searchable text
            search_text = f"{product_name} {product_detail} {color}".lower()
            
            relevance_score = 0
            matched_keywords = []
            
            # Score based on keyword matches
            for i, (keyword, keyword_weight) in enumerate(top_keywords[:10]):  # Check top 10 keywords
                search_terms = get_all_search_terms(keyword)
                
                # Check if any search term appears in the product
                for search_term in search_terms:
                    if search_term.lower() in search_text:
                        # Higher score for more important keywords (earlier in list)
                        importance_multiplier = (10 - i) / 10  # 1.0 for first keyword, 0.1 for 10th
                        match_score = keyword_weight * importance_multiplier
                        relevance_score += match_score
                        matched_keywords.append(f"{keyword}->{search_term}")
                        break  # Only count once per keyword
            
            # Add the product with its relevance score
            product_data = {
                "product_id": product_id,
                "product": product_name,
                "description": product_detail,
                "price": product_price,
                "size": size,
                "color": color,
                "stock": stock,
                "link": f"http://localhost/e-commerce-main/product-{product_seourl}-{product_id}",
                "photo": photo_path,
                "relevance": relevance_score,
                "matched_keywords": matched_keywords
            }
            
            scored_products.append(product_data)
        
        # Step 7: Sort by relevance score (highest first)
        scored_products.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Log top scoring products
        logging.info("=== TOP SCORING PRODUCTS ===")
        for i, product in enumerate(scored_products[:10]):
            logging.info(f"{i+1}. {product['product']} - Score: {product['relevance']:.2f} - Matched: {product['matched_keywords']}")
        
        # Step 8: Select the best products
        selected_products = []
        seen_product_ids = set()
        
        # First, take products with high relevance scores
        for product in scored_products:
            if len(selected_products) >= max_results:
                break
                
            if product["product_id"] not in seen_product_ids and product["relevance"] > 0:
                selected_products.append(product)
                seen_product_ids.add(product["product_id"])
        
        # If we don't have enough products with keyword matches, add some without matches
        if len(selected_products) < max_results:
            logging.info(f"Only found {len(selected_products)} products with keyword matches, adding random products")
            for product in scored_products:
                if len(selected_products) >= max_results:
                    break
                    
                if product["product_id"] not in seen_product_ids:
                    selected_products.append(product)
                    seen_product_ids.add(product["product_id"])
        
        # Clean up the products (remove debug info)
        final_products = []
        for product in selected_products:
            clean_product = {
                "product_id": product["product_id"],
                "product": product["product"],
                "description": product["description"],
                "price": product["price"],
                "size": product["size"],
                "color": product["color"],
                "stock": product["stock"],
                "link": product["link"],
                "photo": product["photo"],
                "relevance": product["relevance"]
            }
            final_products.append(clean_product)
        
        logging.info(f"=== FINAL RESULT ===")
        logging.info(f"Returning {len(final_products)} products")
        for i, product in enumerate(final_products):
            logging.info(f"{i+1}. {product['product']} - Price: {product['price']} - Relevance: {product['relevance']:.2f}")
        
        return pd.DataFrame(final_products)
        
    except Exception as e:
        logging.error(f"Error in fetch_products_from_db: {str(e)}")
        logging.error(f"Full traceback: ", exc_info=True)
        return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
        
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
                "IMPORTANT: Always ask for their gender, weight and height, skin tone, their ethnical background and use this information as a base for your recommendations."
                "IMPORTANT: When giving recommendations, mention specific clothing items and how they would suit the user's attributes, "
                "IMPORTANT: such as gender, height, weight, and skin tone. Use descriptive phrases and consider mentioning outfit ideas for casual occasions, "
                "unless the user specifies a different occasion.\n"
                "give at least 3 items recommendation.\n\n"
                "IMPORTANT: If the user asks for a specific type of clothing (such as 'kemeja', 'shirt', 'dress', 'pants', etc.), "
                "make sure your recommendations focus directly on that specific clothing type.\n\n"
                "make each clothing item as a bold text and for the explanation make it as a paragraph and in different line from the title, for new or different item make it in different line."
                "Example response:\n"
                "For casual wear, here are some styles that would look great on you:  \n"
                "- Cropped Tees or Tank Tops  \n– Perfect for a laid-back look, and they can help accentuate your waist and balance out your height.  \n"
                "- Oversized T-shirts  \n– A relaxed fit in earthy or jewel tones would be flattering. You could tuck them into high-waisted jeans or shorts to add some shape.  \n"
                "- Off-Shoulder Tops  \n– These are cute and can show off some skin while keeping things casual. They also come in various styles, like long-sleeved or with ruffles, which add a nice detail.  \n"
                "- Button-Up Shirts (Linen or Cotton)  \n– A lightweight button-up, especially in tan, beige, or pastel shades, could be a staple. You can wear it loose or tied at the waist.  \n"
                "- Graphic Tees  \n– These are always in style and add personality to any casual outfit. You could go for retro or minimalist designs in colors that complement your skin tone.  \n\n"
                "Do not mention any specific brand of clothing."
                "After each style recommendation, always ask a yes or no question: 'Would you like to see product recommendations based on these style suggestions?' or 'Do these styles align with what you're looking for? I can show you specific products if you're interested.'"
                "DO NOT provide product recommendations in your initial response - only suggest styles and wait for user confirmation.")
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
            }
        }

        # Define the GENDER_BOOST constant
        GENDER_BOOST = 10.0  # High confidence score for direct gender matches

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
                
                # Check if we're awaiting confirmation for product recommendations
                if user_context["awaiting_confirmation"]:
                    # Process confirmation response
                    is_positive = user_input.strip().lower() in ["yes", "ya", "iya", "sure", "tentu", "ok", "okay"]
                    is_negative = user_input.strip().lower() in ["no", "tidak", "nope", "nah", "tidak usah"]
                    
                    if is_positive:
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

                            logging.info(f"Using ranked keywords for product search: {translated_ranked_keywords[:10]}")
                            
                            # Get user gender and budget for filtering
                            user_gender = user_context.get("user_gender", {}).get("category", None)
                            budget_range = user_context.get("budget_range", None)

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
                            
                            # Fetch products using the ranked keywords
                            try:
                                recommended_products = await fetch_products_from_db(
                                    db=db,  # Make sure db is the AsyncSession object
                                    top_keywords=translated_ranked_keywords,  # Make sure this is a list of tuples
                                    max_results=5,
                                    gender_category=user_gender,
                                    budget_range=budget_range
                                )
                                
                                print(f"Successfully fetched {len(recommended_products)} products")
                                
                            except Exception as fetch_error:
                                logging.error(f"Error calling fetch_products_from_db: {str(fetch_error)}")
                                logging.error(f"Parameters passed:")
                                logging.error(f"- db: {type(db)}")
                                logging.error(f"- top_keywords: {type(translated_ranked_keywords)} - {translated_ranked_keywords[:3] if translated_ranked_keywords else 'None'}")
                                logging.error(f"- user_gender: {user_gender}")
                                logging.error(f"- budget_range: {budget_range}")
                                raise

                            # Create response with product cards
                            if not recommended_products.empty:
                                complete_response = positive_response + "\n\n"
                                
                                for _, row in recommended_products.iterrows():
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
                                
                                complete_response += "\n\nIs there anything else you'd like to know about these items?"
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
                        user_context["awaiting_confirmation"] = False
                        
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
                            text_lower = text_content.lower()
                            # Check for gender terms directly
                            for gender_term in ["perempuan", "wanita", "female", "woman", "pria", "laki-laki", "male", "man"]:
                                if gender_term in text_lower:
                                    gender_cat = "female" if gender_term in ["perempuan", "wanita", "female", "woman"] else "male"
                                    user_context["user_gender"] = {
                                        "category": gender_cat,
                                        "term": gender_term,
                                        "confidence": GENDER_BOOST,
                                        "last_updated": datetime.now().isoformat()
                                    }
                                    logging.info(f"Direct gender detection updated: {gender_cat} (term: {gender_term})")
                                    print(f"Direct gender detection updated: {gender_cat} (term: {gender_term})")
                                    break

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
                    text_lower = translated_input.lower()
                    # Check for gender terms directly
                    for gender_term in ["perempuan", "wanita", "female", "woman", "pria", "laki-laki", "male", "man"]:
                        if gender_term in text_lower:
                            gender_cat = "female" if gender_term in ["perempuan", "wanita", "female", "woman"] else "male"
                            user_context["user_gender"] = {
                                "category": gender_cat,
                                "term": gender_term,
                                "confidence": GENDER_BOOST,
                                "last_updated": datetime.now().isoformat()
                            }
                            logging.info(f"Direct gender detection updated: {gender_cat} (term: {gender_term})")
                            print(f"Direct gender detection updated: {gender_cat} (term: {gender_term})")
                            break
                        
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
    "pria", "laki-laki", "perempuan", "wanita", "lelaki", "cewek", "cowok"
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
    Extract budget information from user input text.
    Returns tuple (min_price, max_price) or None if no budget mentioned.
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Pattern for Indonesian Rupiah (IDR, Rp, rupiah)
    # Matches patterns like: "budget 100000", "maksimal 500rb", "dibawah 1jt", "antara 50rb-200rb"
    
    # Common Indonesian budget phrases
    budget_patterns = [
        # Range patterns: "50rb-200rb", "100000-500000", "50 ribu sampai 200 ribu"
        r'(?:budget|anggaran|harga|harganya)?\s*(?:antara|between)?\s*(\d+)(?:rb|ribu|000|k)?\s*(?:-|sampai|hingga|to)\s*(\d+)(?:rb|ribu|000|k)?',
        r'(\d+)(?:rb|ribu|000)?\s*(?:-|sampai|hingga|to)\s*(\d+)(?:rb|ribu|000)?',
        
        # Maximum patterns: "maksimal 200rb", "dibawah 500000", "under 1jt"
        r'(?:maksimal|max|dibawah|under|kurang dari|less than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|jt|juta|000|k)?',
        
        # Minimum patterns: "minimal 100rb", "diatas 50000", "over 1jt" 
        r'(?:minimal|min|diatas|over|lebih dari|more than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|jt|juta|000|k)?',
        
        # Exact budget: "budget 150rb", "anggaran 200000"
        r'(?:budget|anggaran|harga)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|jt|juta|000|k)?',
        
        # Simple number with currency indicators
        r'(?:rp\.?\s*)?(\d+)(?:rb|ribu|jt|juta|k)',
    ]
    
    def convert_to_rupiah(amount_str, unit):
        """Convert amount string with unit to actual rupiah value"""
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
    
    # Try each pattern
    for pattern in budget_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            groups = match.groups()
            
            # Range pattern (two amounts)
            if len(groups) >= 2 and groups[0] and groups[1]:
                # Extract units from the original match
                match_text = match.group(0)
                
                # Determine units for first amount
                if 'rb' in match_text or 'ribu' in match_text:
                    unit1 = 'rb'
                elif 'k' in match_text:
                    unit1 = 'rb'
                elif 'jt' in match_text or 'juta' in match_text:
                    unit1 = 'jt'
                elif '000' in groups[0]:
                    unit1 = '000'
                else:
                    unit1 = None
                
                # Determine units for second amount  
                if 'rb' in match_text.split('-')[-1] or 'ribu' in match_text.split('-')[-1]:
                    unit2 = 'rb'
                elif 'k' in match_text.split('-')[-1]:
                    unit2 = 'rb'
                elif 'jt' in match_text.split('-')[-1] or 'juta' in match_text.split('-')[-1]:
                    unit2 = 'jt'
                elif '000' in groups[1]:
                    unit2 = '000'
                else:
                    unit2 = unit1  # Use same unit as first amount
                
                min_price = convert_to_rupiah(groups[0], unit1)
                max_price = convert_to_rupiah(groups[1], unit2)
                
                if min_price and max_price:
                    return (min(min_price, max_price), max(min_price, max_price))
            
            # Single amount pattern
            elif len(groups) >= 1 and groups[0]:
                match_text = match.group(0)
                
                # Determine unit
                if 'rb' in match_text or 'ribu' in match_text:
                    unit = 'rb'
                elif 'k' in match_text:
                    unit = 'rb'
                elif 'jt' in match_text or 'juta' in match_text:
                    unit = 'jt'
                elif '000' in groups[0]:
                    unit = '000'
                else:
                    unit = None
                
                amount = convert_to_rupiah(groups[0], unit)
                
                if amount:
                    # Determine if it's min, max, or exact based on context
                    if any(word in match_text for word in ['maksimal', 'max', 'dibawah', 'under', 'kurang dari', 'less than']):
                        return (None, amount)  # Maximum budget
                    elif any(word in match_text for word in ['minimal', 'min', 'diatas', 'over', 'lebih dari', 'more than']):
                        return (amount, None)  # Minimum budget
                    else:
                        # For exact budget, create a range (±20%)
                        min_range = int(amount * 0.8)
                        max_range = int(amount * 1.2)
                        return (min_range, max_range)
    
    # Also check for simple number patterns that might indicate budget
    simple_number_pattern = r'\b(\d{5,7})\b'  # 5-7 digits (typical Indonesian prices)
    matches = re.finditer(simple_number_pattern, text_lower)
    for match in matches:
        amount = int(match.group(1))
        # Only consider if it's a reasonable price range (10k - 10M IDR)
        if 10000 <= amount <= 10000000:
            # Create a range around this amount
            min_range = int(amount * 0.8)
            max_range = int(amount * 1.2)
            return (min_range, max_range)
    
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
    # Initialize if not present
    if "accumulated_keywords" not in user_context:
        user_context["accumulated_keywords"] = {}
    
    # Extract budget information if this is user input
    if is_user_input and user_context.get("current_text_input"):
        budget_info = extract_budget_from_text(user_context["current_text_input"])
        if budget_info:
            user_context["budget_range"] = budget_info
            logging.info(f"Budget extracted from user input: {budget_info}")
            print(f"Budget extracted from user input: {budget_info}")
    
    # Extract bold headings if this is an AI response
    if is_ai_response and "last_ai_response" in user_context:
        bold_headings = extract_bold_headings_from_ai_response(user_context["last_ai_response"])
        if bold_headings:
            logging.info(f"Bold headings extracted from AI response: {bold_headings}")
            print(f"Bold headings extracted from AI response: {bold_headings}")
            
            # Add bold headings as high-priority keywords
            for heading in bold_headings:
                # Give bold headings very high weight since they're the main recommendations
                heading_weight = 200.0  # Higher than normal keyword weights
                
                if heading.lower() in user_context["accumulated_keywords"]:
                    current_weight = user_context["accumulated_keywords"][heading.lower()]["weight"]
                    user_context["accumulated_keywords"][heading.lower()]["weight"] = max(current_weight, heading_weight)
                    user_context["accumulated_keywords"][heading.lower()]["count"] += 1
                else:
                    user_context["accumulated_keywords"][heading.lower()] = {
                        "weight": heading_weight,
                        "count": 1,
                        "first_seen": datetime.now().isoformat(),
                        "source": "ai_bold_heading"
                    }
    
    # Direct gender detection from raw text input
    if is_user_input and user_context.get("current_text_input"):
        raw_text = user_context["current_text_input"]
        current_confidence = user_context.get("user_gender", {}).get("confidence", 0)
        
        # Directly check for gender terms in raw text
        gender_cat, gender_term, confidence = detect_gender_directly_from_text(raw_text, current_confidence)
        
        # Update user gender if we found a match with higher confidence
        if gender_cat and confidence > current_confidence:
            user_context["user_gender"] = {
                "category": gender_cat,
                "term": gender_term,
                "confidence": confidence,
                "last_updated": datetime.now().isoformat()
            }
            logging.info(f"Direct gender detection updated: {gender_cat} (term: {gender_term}, confidence: {confidence})")
            print(f"Direct gender detection updated: {gender_cat} (term: {gender_term}, confidence: {confidence})")
    
    # Weight multipliers based on source
    user_multiplier = 3.0 if is_user_input else 1.0  # Increased user input importance
    ai_multiplier = 2.0 if is_ai_response else 1.0    # Increased AI response importance
    
    # Update the accumulated keywords
    for keyword, weight in keywords:
        if not keyword or len(keyword) < 3:
            continue
            
        # Calculate final weight with multipliers
        final_weight = weight * user_multiplier * ai_multiplier
        
        # Update in accumulator
        keyword_lower = keyword.lower()
        if keyword_lower in user_context["accumulated_keywords"]:
            # Increase weight for repeated mentions
            current_weight = user_context["accumulated_keywords"][keyword_lower]["weight"]
            user_context["accumulated_keywords"][keyword_lower]["weight"] = max(current_weight, final_weight)
            user_context["accumulated_keywords"][keyword_lower]["count"] += 1
        else:
            # Add new keyword
            user_context["accumulated_keywords"][keyword_lower] = {
                "weight": final_weight,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "source": "user_input" if is_user_input else "ai_response"
            }
    
    # Check keywords for gender information
    gender_cat, gender_term, confidence = identify_gender_from_keywords(keywords)
    current_confidence = user_context.get("user_gender", {}).get("confidence", 0)
    
    # Update user gender if we found a match with higher confidence
    if gender_cat and confidence > current_confidence:
        user_context["user_gender"] = {
            "category": gender_cat,
            "term": gender_term,
            "confidence": confidence,
            "last_updated": datetime.now().isoformat()
        }
        logging.info(f"Gender detection from keywords updated: {gender_cat} (term: {gender_term}, confidence: {confidence})")
        print(f"Gender detection from keywords updated: {gender_cat} (term: {gender_term}, confidence: {confidence})")

def get_user_gender(user_context):
    if "user_gender" in user_context and user_context["user_gender"]["category"]:
        return user_context

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