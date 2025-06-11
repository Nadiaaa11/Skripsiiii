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

app = FastAPI()

app.mount("/static", StaticFiles(directory="Chatbot/static"), name="static")

templates = Jinja2Templates(directory="Chatbot/templates")

UPLOAD_DIR = "static/uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "jfif"}

Base = declarative_base()

DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/ecommerce"

engine = create_async_engine(DATABASE_URL, echo=True)

database = Database(DATABASE_URL)

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

    def matches_main_category(query, product):
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

    def semantic_search(self, query: str, top_k: int = 10, threshold: float = 0.3, 
                   gender_filter: str = None, budget_range: tuple = None) -> list:
        """
        CLEANER semantic search with better product filtering
        """
        if self.products_df is None or self.products_df.empty:
            print("⚠️ No products available for semantic search")
            return []
        
        print(f"\n🔍 CLEAN SEMANTIC SEARCH")
        print(f"   Query: '{query}'")
        print(f"   Gender filter: {gender_filter}")
        print(f"   Budget: {budget_range}")
        
        # Clean the query - remove numbers and irrelevant words
        cleaned_query = self.clean_semantic_query(query)
        print(f"   🧹 Cleaned query: '{cleaned_query}'")
        
        # Preprocess query with minimal expansion
        enhanced_query = self.semantic_system.preprocess_fashion_text(cleaned_query)
        print(f"   ✨ Enhanced query: '{enhanced_query}'")
        
        # Get query embedding
        query_embedding = self.semantic_system.get_semantic_embedding(enhanced_query)
        
        # Calculate similarities with product pre-filtering
        similarities = []
        for _, product in self.products_df.iterrows():
            product_id = product['product_id']
            
            if product_id in self.product_embeddings:
                # PRE-FILTER: Skip products that don't match the main category
                if not self.matches_main_category(cleaned_query, product):
                    continue
                    
                product_embedding = self.product_embeddings[product_id]
                
                # Calculate semantic similarity
                if self.semantic_system.model and SENTENCE_TRANSFORMERS_AVAILABLE:
                    similarity = np.dot(query_embedding, product_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding)
                    )
                else:
                    similarity = self.semantic_system.calculate_semantic_similarity(
                        enhanced_query, product['enhanced_description']
                    )
                
                similarities.append({
                    'product_id': product_id,
                    'similarity': float(similarity),
                    'product_data': product
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)    
        # Apply filters
        filtered_results = []
        for result in similarities:
            product_data = result['product_data']
            
            # Similarity threshold
            if result['similarity'] < threshold:
                continue
            
            # Gender filter
            if gender_filter:
                if gender_filter.lower() in ['female', 'woman', 'perempuan', 'wanita']:
                    if product_data['gender'] != 'female':
                        continue
                elif gender_filter.lower() in ['male', 'man', 'pria', 'laki-laki']:
                    if product_data['gender'] != 'male':
                        continue
            
            # Budget filter
            if budget_range:
                min_price, max_price = budget_range
                product_price = product_data['price']
                
                if min_price and product_price < min_price:
                    continue
                if max_price and product_price > max_price:
                    continue
            
            # Format result to match your existing format
            sizes = product_data['sizes'].split(',') if product_data['sizes'] else []
            colors = product_data['colors'].split(',') if product_data['colors'] else []
            
            filtered_results.append({
                'product_id': product_data['product_id'],
                'product': product_data['product_name'],
                'description': product_data['product_detail'],
                'price': product_data['price'],
                'size': ", ".join(sizes) if sizes else "N/A",
                'color': ", ".join(colors) if colors else "N/A",
                'stock': product_data['stock'],
                'link': f"http://localhost/e-commerce-main/product-{product_data['seourl']}-{product_data['product_id']}",
                'photo': product_data['photo'],
                'relevance': result['similarity'],
                'semantic_match': True
            })
            
            if len(filtered_results) >= top_k:
                break
        
        print(f"   Found {len(filtered_results)} semantic matches above threshold {threshold}")
        if filtered_results:
            print(f"   Top match: '{filtered_results[0]['product']}' (similarity: {filtered_results[0]['relevance']:.3f})")
        
        return filtered_results
    
    def _get_products_hash(self, products_data):
        """Create a hash of product data to detect changes"""
        # Create a string representation of key product info
        product_info = []
        for product in products_data:
            info = f"{product[0]}_{product[1]}_{product[2]}_{product[5]}"  # id, name, detail, price
            product_info.append(info)
        
        combined_info = "|".join(sorted(product_info))
        return hashlib.md5(combined_info.encode()).hexdigest()
    
    def _save_embeddings_to_cache(self, products_hash):
        """Save embeddings and product data to disk"""
        try:
            print("💾 Saving embeddings to cache...")
            
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.product_embeddings, f)
            
            # Save products DataFrame
            with open(self.products_file, 'wb') as f:
                pickle.dump(self.products_df, f)
            
            # Save metadata
            metadata = {
                'products_hash': products_hash,
                'created_at': datetime.now().isoformat(),
                'model_type': 'sentence-transformers' if self.semantic_system.model else 'tfidf',
                'total_products': len(self.products_df) if self.products_df is not None else 0
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"✅ Embeddings cached successfully ({metadata['total_products']} products)")
            
        except Exception as e:
            print(f"❌ Error saving embeddings to cache: {str(e)}")
    
    def _load_embeddings_from_cache(self, current_products_hash):
        """Load embeddings from disk if they're still valid"""
        try:
            # Check if all cache files exist
            if not all(os.path.exists(f) for f in [self.embeddings_file, self.products_file, self.metadata_file]):
                print("📂 Cache files not found")
                return False
            
            # Load metadata first
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Check if cache is still valid
            cached_hash = metadata.get('products_hash')
            cache_age_hours = 0
            
            try:
                created_at = datetime.fromisoformat(metadata.get('created_at', ''))
                cache_age_hours = (datetime.now() - created_at).total_seconds() / 3600
            except:
                cache_age_hours = 999  # Force refresh if we can't parse date
            
            # Cache is invalid if:
            # 1. Product data has changed (different hash)
            # 2. Cache is older than 24 hours
            # 3. Model type has changed
            current_model_type = 'sentence-transformers' if self.semantic_system.model else 'tfidf'
            cached_model_type = metadata.get('model_type', 'unknown')
            
            if (cached_hash != current_products_hash or 
                cache_age_hours > 24 or 
                cached_model_type != current_model_type):
                
                print(f"🔄 Cache invalid:")
                print(f"   Products changed: {cached_hash != current_products_hash}")
                print(f"   Cache age: {cache_age_hours:.1f}h (max: 24h)")
                print(f"   Model changed: {cached_model_type} → {current_model_type}")
                return False
            
            # Load embeddings and products
            print("📂 Loading embeddings from cache...")
            
            with open(self.embeddings_file, 'rb') as f:
                self.product_embeddings = pickle.load(f)
            
            with open(self.products_file, 'rb') as f:
                self.products_df = pickle.load(f)
            
            print(f"✅ Loaded {len(self.product_embeddings)} embeddings from cache")
            print(f"   Cache created: {metadata.get('created_at', 'unknown')}")
            print(f"   Products: {metadata.get('total_products', 0)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading embeddings from cache: {str(e)}")
            return False
    
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
            
            # Create hash of current products
            current_products_hash = self._get_products_hash(all_products)
            print(f"🔢 Products hash: {current_products_hash[:8]}...")
            
            # Try to load from cache first (unless forced refresh)
            if not force_refresh and self._load_embeddings_from_cache(current_products_hash):
                print("🚀 Using cached embeddings - FAST STARTUP!")
                print("="*50)
                return
            
            # Cache miss or forced refresh - create new embeddings
            print("🔄 Creating new embeddings...")
            print("   This will take 2-5 minutes for the first time...")
            
            # Create enhanced product descriptions
            enhanced_descriptions = []
            product_data = []
            
            for i, product_row in enumerate(all_products):
                if i % 100 == 0 and i > 0:
                    print(f"   📝 Processing product descriptions: {i}/{len(all_products)}")
                
                # Enhanced description combining all available information
                desc = f"{product_row[1]} {product_row[2]}"
                
                # Add attributes from variants
                if product_row[7]:  # colors
                    colors = product_row[7].split(',')
                    desc += f" colors: {' '.join(colors)}"
                
                if product_row[6]:  # sizes
                    sizes = product_row[6].split(',') 
                    desc += f" sizes: {' '.join(sizes)}"
                
                # Add gender
                desc += f" gender: {product_row[4]}"
                
                # Preprocess with cultural context
                enhanced_desc = self.semantic_system.preprocess_fashion_text(desc)
                enhanced_descriptions.append(enhanced_desc)
                
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
                    "seourl": product_row[3],
                    "enhanced_description": enhanced_desc
                })
            
            self.products_df = pd.DataFrame(product_data)
            print(f"📋 Created product DataFrame with {len(self.products_df)} products")
            
            # Generate embeddings for all products
            print(f"🧠 Generating embeddings ({len(enhanced_descriptions)} products)...")
            start_time = datetime.now()
            
            self.product_embeddings = {}
            
            for i, desc in enumerate(enhanced_descriptions):
                if i % 50 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if i > 0:
                        rate = i / elapsed
                        remaining = (len(enhanced_descriptions) - i) / rate
                        print(f"   Progress: {i}/{len(enhanced_descriptions)} ({i/len(enhanced_descriptions)*100:.1f}%) - ETA: {remaining:.0f}s")
                
                embedding = self.semantic_system.get_semantic_embedding(desc, use_cache=True)
                self.product_embeddings[self.products_df.iloc[i]['product_id']] = embedding
            
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Generated {len(self.product_embeddings)} embeddings in {total_time:.1f}s")
            
            # Save to cache for next time
            self._save_embeddings_to_cache(current_products_hash)
            
            print("="*50)
            
        except Exception as e:
            logging.error(f"Error in enhanced preprocess_products: {str(e)}")
            print(f"❌ Error preprocessing products: {str(e)}")

        
    
    def clear_cache(self):
        """Clear the embeddings cache"""
        try:
            for file_path in [self.embeddings_file, self.products_file, self.metadata_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            print("🗑️ Embeddings cache cleared")
        except Exception as e:
            print(f"❌ Error clearing cache: {str(e)}")

# Initialize enhanced product matcher
enhanced_matcher = EnhancedSemanticProductMatcher(semantic_system)

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
    FIXED: Keyword extraction that properly uses translation mapping for expansion.
    """
    print("\n" + "="*60)
    print("🔤 KEYWORD EXTRACTION WITH TRANSLATION EXPANSION")
    print("="*60)
    
    keyword_scores = {}
    global_exclusions = set()

    # Simple responses filter
    simple_responses = {
        "yes", "ya", "iya", "ok", "okay", "sure", "tentu",
        "no", "tidak", "nope", "ga", "gak", "engga", "nah"
    }
    
    # Core product terms (clothing items get highest priority)
    core_clothing_terms = [
        # CORE CLOTHING TYPES (HIGHEST PRIORITY) - Increased weights
        "kemeja", "shirt", "blouse", "blus", 
        "dress", "gaun", "rok", "skirt",
        "celana", "pants", "jeans", "denim",
        "jacket", "jaket", "sweater", "cardigan",
        "atasan", "top", "kaos", "t-shirt",
        "hoodie", "blazer", "coat", "mantel",

        # STYLE ATTRIBUTES (HIGH PRIORITY) - Increased weights
        "lengan panjang", "panjang lengan", "long sleeve",
        "lengan pendek", "pendek lengan", "short sleeve",
        "panjang", "long", "pendek", "short",
        "slim", "regular", "loose", "ketat",
        "longgar", "tight", "oversized",
        
        # STYLE CATEGORIES (MEDIUM-HIGH PRIORITY) - Increased weights  
        "formal", "casual", "santai", "elegant", "elegan",
        "vintage", "modern", "minimalist", "minimalis",
        
        # COLORS (MEDIUM PRIORITY) - Slightly increased
        "white", "putih", "black", "hitam",
        "red", "merah", "blue", "biru",
        "green", "hijau", "yellow", "kuning",
        "brown", "coklat", "pink", "merah muda",
        "purple", "ungu", "orange", "oranye",
        "grey", "abu-abu", "navy", "biru tua",
        "beige", "krem",

        # OCCASIONS (LOW PRIORITY) - SIGNIFICANTLY REDUCED
        "office", "kantor", "party", "pesta",
        "wedding", "pernikahan", "beach", "pantai",
        "sport", "olahraga", "work", "kerja"
    ]
    
    # Process user input
    if translated_input:
        print(f"📝 USER INPUT: '{translated_input}'")
        
        # Check for simple response
        input_words = translated_input.lower().split()
        is_simple_response = (
            len(input_words) == 1 and input_words[0] in simple_responses
        )
        
        if is_simple_response:
            print(f"   ⚠️  SIMPLE RESPONSE - Skipping")
            return []
        
        # Extract base keywords from user input
        doc = nlp(translated_input)
        base_keywords = {}
        
        for token in doc:
            if (len(token.text) > 1 and
                not token.text.isdigit() and
                token.is_alpha):
                keyword = token.text.lower()
                base_keywords[keyword] = base_keywords.get(keyword, 0) + 1
        
        print(f"   🔍 Base keywords extracted: {list(base_keywords.keys())}")
        
        # EXPAND keywords using translation mapping
        expanded_keywords = {}
        
        for keyword, frequency in base_keywords.items():
            # Add the original keyword
            expanded_keywords[keyword] = frequency
            
            # Get translation expansion
            try:
                search_terms = get_search_terms_for_keyword(keyword)
                if isinstance(search_terms, dict) and 'include' in search_terms:
                    include_terms = search_terms.get('include', [])
                    exclude_terms = search_terms.get('exclude', [])
                    
                    # Add include terms with slightly lower frequency
                    for include_term in include_terms:
                        if include_term != keyword:  # Don't duplicate
                            expanded_keywords[include_term] = expanded_keywords.get(include_term, 0) + (frequency * 0.8)
                            print(f"      ➕ Expanded '{keyword}' → '{include_term}'")
                    
                    # Store exclusions
                    if exclude_terms:
                        global_exclusions.update(exclude_terms)
                        print(f"      🚫 Will exclude: {exclude_terms}")
                        
            except Exception as e:
                print(f"      ⚠️ Translation mapping error for '{keyword}': {e}")
                pass
        
        print(f"   📈 After expansion: {len(expanded_keywords)} keywords")
        
        # Score expanded keywords with CLOTHING PRIORITY
        for keyword, frequency in expanded_keywords.items():
            base_score = frequency * 100  # Base score
            
            # HIGHEST PRIORITY: Core clothing items
            if any(clothing in keyword.lower() for clothing in core_clothing_terms):
                clothing_bonus = 200
                print(f"      👕 CLOTHING ITEM: '{keyword}' gets +{clothing_bonus}")
            # High priority: Fashion-related terms
            elif any(fashion in keyword.lower() for fashion in ['neck', 'sleeve', 'shoulder', 'style', 'casual', 'formal']):
                clothing_bonus = 100
                print(f"      👗 FASHION TERM: '{keyword}' gets +{clothing_bonus}")
            # Medium priority: Colors and materials
            elif any(attr in keyword.lower() for attr in ['white', 'black', 'red', 'blue', 'putih', 'hitam', 'cotton', 'silk']):
                clothing_bonus = 50
                print(f"      🎨 COLOR/MATERIAL: '{keyword}' gets +{clothing_bonus}")
            # Low priority: Context terms
            else:
                clothing_bonus = 0
                if keyword.lower() in ['carikan', 'cocok', 'cerah', 'indonesia', 'tinggi', 'cm', 'kg']:
                    print(f"      📝 CONTEXT ONLY: '{keyword}' (low priority)")
            
            final_score = base_score + clothing_bonus
            keyword_scores[keyword] = final_score
            
            print(f"   📌 '{keyword}' (freq: {frequency:.1f}) → {final_score}")
    
    # Process AI response (extract clothing items mentioned)
    if ai_response:
        print(f"\n🤖 AI RESPONSE processing...")
        
        # Extract bold headings (clothing items)
        bold_headings = extract_bold_headings_from_ai_response(ai_response)
        for heading in bold_headings:
            heading_lower = heading.lower()
            cleaned_heading = re.sub(r'[^\w\s-]', '', heading_lower).strip()
            
            if cleaned_heading and len(cleaned_heading) > 2:
                # Expand AI-extracted headings too
                try:
                    search_terms = get_search_terms_for_keyword(cleaned_heading)
                    if isinstance(search_terms, dict) and 'include' in search_terms:
                        include_terms = search_terms.get('include', [])
                        
                        for include_term in include_terms:
                            ai_score = 80  # Good score for AI-extracted clothing items
                            if include_term not in keyword_scores or keyword_scores[include_term] < ai_score:
                                keyword_scores[include_term] = ai_score
                                print(f"   🤖 AI clothing: '{include_term}' → {ai_score}")
                except:
                    pass
                
                # Also add the original heading
                ai_score = 80
                if cleaned_heading not in keyword_scores or keyword_scores[cleaned_heading] < ai_score:
                    keyword_scores[cleaned_heading] = ai_score
                    print(f"   🤖 AI heading: '{cleaned_heading}' → {ai_score}")
    
    # Process accumulated keywords with decay
    if accumulated_keywords:
        print(f"\n📚 ACCUMULATED keywords...")
        
        for keyword, old_weight in accumulated_keywords[:10]:
            if (keyword and len(keyword) > 1 and
                not any(char.isdigit() for char in keyword)):
                
                # Apply expansion to accumulated keywords too
                try:
                    search_terms = get_search_terms_for_keyword(keyword)
                    if isinstance(search_terms, dict) and 'include' in search_terms:
                        include_terms = search_terms.get('include', [])
                        
                        for include_term in include_terms:
                            accumulated_score = old_weight * 0.4  # Decay factor
                            if include_term not in keyword_scores:
                                keyword_scores[include_term] = accumulated_score
                                print(f"   📜 Accumulated expansion: '{keyword}' → '{include_term}' ({accumulated_score:.1f})")
                except:
                    pass
                
                # Add original accumulated keyword
                accumulated_score = old_weight * 0.4
                if keyword not in keyword_scores:
                    keyword_scores[keyword] = accumulated_score
                    print(f"   📜 '{keyword}' → {accumulated_score:.1f}")
    
    # Clean up obviously irrelevant terms
    cleanup_keywords = []
    irrelevant_terms = ['carikan', 'cocok', 'bisa', 'yang', 'dari', 'untuk', 'dengan']
    
    for keyword in list(keyword_scores.keys()):
        if keyword in irrelevant_terms or len(keyword.split()) > 3:
            cleanup_keywords.append(keyword)
    
    for keyword in cleanup_keywords:
        del keyword_scores[keyword]
        print(f"   🗑️ Cleaned: '{keyword}'")
    
    # Sort and return
    ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 FINAL KEYWORDS WITH TRANSLATION EXPANSION:")
    for i, (keyword, score) in enumerate(ranked_keywords[:15]):
        priority = "🎯 HIGH" if score >= 200 else "📋 MED" if score >= 50 else "📝 LOW"
        print(f"   {i+1:2d}. {priority} '{keyword}' → {score:.1f}")
    
    if global_exclusions:
        print(f"\n🚫 PRODUCT EXCLUSIONS:")
        for term in sorted(global_exclusions):
            print(f"   ❌ '{term}'")
    
    print(f"\n📊 SUMMARY:")
    print(f"   🎯 High priority (200+): {len([k for k, s in ranked_keywords if s >= 200])}")
    print(f"   📋 Medium priority (50+): {len([k for k, s in ranked_keywords if s >= 50])}")
    print(f"   📝 Total keywords: {len(ranked_keywords)}")
    print("="*60)
    
    # Store exclusions
    extract_ranked_keywords.last_exclusions = list(global_exclusions)
    
    return ranked_keywords[:15]

def enhanced_extract_ranked_keywords(ai_response: str = None, translated_input: str = None, 
                                   accumulated_keywords=None, use_semantic: bool = True):
    """
    Enhanced version of your extract_ranked_keywords that adds semantic understanding
    """
    print("\n" + "="*70)
    print("🔤 ENHANCED SEMANTIC KEYWORD EXTRACTION")
    print("="*70)
    
    # Use your existing extraction logic as base
    base_keywords = extract_ranked_keywords(ai_response, translated_input, accumulated_keywords)
    
    if not use_semantic or not base_keywords or not semantic_system.model:
        return base_keywords
    
    # Enhance with semantic expansion
    enhanced_keywords = []
    semantic_expansions = 0
    
    for keyword, score in base_keywords:
        # Add original keyword
        enhanced_keywords.append((keyword, score))
        
        # Semantic expansion for fashion terms
        if any(fashion_term in keyword.lower() for fashion_term in 
               ['kemeja', 'celana', 'dress', 'kaos', 'jaket', 'formal', 'casual']):
            
            # Find semantically similar terms
            fashion_terms = [
                'shirt', 'blouse', 'top', 'dress', 'pants', 'skirt', 'jacket', 
                'casual', 'formal', 'elegant', 'modern', 'traditional'
            ]
            
            for term in fashion_terms:
                if term != keyword.lower():
                    similarity = semantic_system.calculate_semantic_similarity(keyword, term)
                    if similarity > 0.6:  # High similarity threshold
                        expansion_score = score * similarity * 0.7  # Reduced score for expansions
                        enhanced_keywords.append((term, expansion_score))
                        semantic_expansions += 1
                        print(f"   🧠 Semantic expansion: '{keyword}' → '{term}' (sim: {similarity:.2f})")
    
    # Remove duplicates and sort
    keyword_dict = {}
    for keyword, score in enhanced_keywords:
        if keyword in keyword_dict:
            keyword_dict[keyword] = max(keyword_dict[keyword], score)
        else:
            keyword_dict[keyword] = score
    
    final_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   🧠 Added {semantic_expansions} semantic expansions")
    print(f"   📊 Total enhanced keywords: {len(final_keywords)}")
    print("="*70)
    
    return final_keywords[:20]  # Return top 20

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

async def fetch_products_from_db(db: AsyncSession, top_keywords: list, max_results=15, gender_category=None, budget_range=None):
    """
    Simplified product fetching with relevance-based sorting and exclusion filtering.
    ALWAYS returns a DataFrame (empty if no results).
    """
    print("\n" + "="*80)
    print("🔍 PRODUCT SEARCH DEBUG")
    print("="*80)
    print(f"📊 Total keywords received: {len(top_keywords)}")
    print(f"🎯 Top 15 keywords being used:")
    for i, (kw, score) in enumerate(top_keywords[:15]):
        print(f"   {i+1:2d}. '{kw}' → Score: {score:.2f}")
    print(f"👤 Gender filter: {gender_category}")
    print(f"💰 Budget filter: {budget_range}")
    
    # ADD: Get exclusions from keyword extraction
    exclusions = get_latest_exclusions()
    if exclusions:
        print(f"🚫 Product exclusions: {exclusions}")
    else:
        print("🚫 No product exclusions")
    
    print("="*80)
    
    logging.info(f"=== PRODUCT SEARCH ===")
    logging.info(f"Keywords: {[(kw, score) for kw, score in top_keywords[:10]]}")
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
        
        # Calculate relevance scores
        print(f"\n🧮 CALCULATING RELEVANCE SCORES...")
        print(f"📝 Using top {min(15, len(top_keywords))} keywords for scoring")
        
        scored_products = []
        debug_count = 0
        
        for product_row in all_products:
            # Debug first 3 products in detail
            debug_this_product = debug_count < 3
            
            if debug_this_product:
                print(f"\n🔍 DEBUGGING PRODUCT {debug_count + 1}: '{product_row[1]}'")
                print(f"   💰 Price: IDR {product_row[5]:,}")
            
            relevance_score = calculate_relevance_score(product_row, top_keywords, debug_this_product)
            
            if debug_this_product:
                print(f"   📊 Final Relevance Score: {relevance_score:.2f}")
                debug_count += 1
            
            # Format data
            sizes = product_row[6].split(',') if product_row[6] else []
            colors = product_row[7].split(',') if product_row[7] else []
            
            scored_products.append({
                "product_id": product_row[0],
                "product": product_row[1],
                "description": product_row[2],
                "price": product_row[5],
                "size": ", ".join(sizes) if sizes else "N/A",
                "color": ", ".join(colors) if colors else "N/A", 
                "stock": product_row[8],
                "link": f"http://localhost/e-commerce-main/product-{product_row[3]}-{product_row[0]}",
                "photo": product_row[9],
                "relevance": relevance_score
            })
        
        # CREATE DataFrame
        products_df = pd.DataFrame(scored_products)
        
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
            
            total_removed = len(scored_products) - len(products_df)
            if total_removed > 0:
                print(f"   🗑️ Total filtered out: {total_removed} products with excluded terms")
                print(f"   ✅ Remaining products: {len(products_df)}")
            else:
                print(f"   ✅ No products were filtered out")
        
        # Sort by relevance (highest first)
        if not products_df.empty:
            print(f"\n📈 SORTING {len(products_df)} PRODUCTS BY RELEVANCE...")
            products_df = products_df.sort_values(by=['relevance'], ascending=False).reset_index(drop=True)
            
            # Take top results
            final_products = products_df[:max_results]
            
            print(f"\n🏆 TOP {min(10, len(final_products))} PRODUCTS AFTER SORTING AND FILTERING:")
            for i, row in final_products.head(10).iterrows():
                print(f"   {i+1:2d}. '{row['product'][:40]}...' → Relevance: {row['relevance']:.2f}, Price: IDR {row['price']:,}")
            
            print(f"\n✅ RETURNING {len(final_products)} FILTERED PRODUCTS")
        else:
            print(f"\n❌ NO PRODUCTS REMAINING AFTER FILTERING")
            final_products = pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
        
        print("="*80)
        
        return final_products
        
    except Exception as e:
        logging.error(f"Error in fetch_products_from_db: {str(e)}")
        print(f"❌ ERROR in fetch_products_from_db: {str(e)}")
        # Always return empty DataFrame with correct columns instead of None
        return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"])
    
def calculate_relevance_score(product_row, keywords, debug=False):
    """
    Simplified relevance calculation with debug output.
    """
    product_name = product_row[1].lower()
    product_detail = product_row[2].lower()
    available_colors = product_row[7].lower() if product_row[7] else ""
    
    search_text = f"{product_name} {product_detail} {available_colors}"
    
    if debug:
        print(f"   🔍 Search text: '{search_text[:100]}...'")
        print(f"   📝 Checking against {len(keywords)} keywords:")
    
    total_score = 0
    matches_found = []
    
    for i, (keyword, weight) in enumerate(keywords[:15]):
        keyword_lower = keyword.lower()
        
        # Position importance (earlier keywords are more important)
        position_weight = (15 - i) / 15
        
        match_score = 0
        match_type = "NO_MATCH"
        
        # Exact match bonus
        if keyword_lower in search_text:
            if keyword_lower in product_name:
                # Product name match gets highest score
                match_score = weight * position_weight * 3.0
                match_type = "NAME_MATCH"
            elif keyword_lower in product_detail:
                # Description match gets medium score
                match_score = weight * position_weight * 2.0
                match_type = "DESC_MATCH"
            else:
                # Color/other match gets base score
                match_score = weight * position_weight * 1.0
                match_type = "COLOR_MATCH"
            
            total_score += match_score
            matches_found.append((keyword, match_type, match_score))
            
            if debug:
                print(f"      ✅ '{keyword}' → {match_type} (+{match_score:.2f})")
        
        # Partial match (less points)
        elif any(word in search_text for word in keyword_lower.split()):
            partial_score = weight * position_weight * 0.5
            total_score += partial_score
            matches_found.append((keyword, "PARTIAL", partial_score))
            
            if debug:
                print(f"      ⚡ '{keyword}' → PARTIAL (+{partial_score:.2f})")
        else:
            if debug and i < 8:  # Only show first 8 for readability
                print(f"      ❌ '{keyword}' → NO_MATCH")
    
    if debug:
        print(f"   📊 Total matches found: {len(matches_found)}")
        print(f"   🎯 Best matches: {[f'{kw}({mt})' for kw, mt, _ in matches_found[:3]]}")
    
    return total_score
        
async def fetch_products_with_budget_awareness(db: AsyncSession, top_keywords: list, max_results=15, gender_category=None, budget_range=None):
    """
    Enhanced product fetching that checks budget constraints and returns appropriate data.
    Products are ALWAYS sorted by HIGHEST RELEVANCE SCORE first, not price.
    Returns: (products_df, budget_status)
    budget_status: "within_budget" | "no_products_in_budget" | "no_budget_specified" | "no_products_found"
    """
    logging.info(f"=== BUDGET-AWARE PRODUCT FETCH (RELEVANCE-FIRST SORTING) ===")
    logging.info(f"Budget range: {budget_range}")
    
    try:
        if budget_range:
            products_within_budget = await fetch_products_from_db(db, top_keywords, max_results, gender_category, budget_range)
            
            if products_within_budget is not None and not products_within_budget.empty:
                logging.info(f"Found {len(products_within_budget)} products within budget (sorted by relevance)")
                return products_within_budget, "within_budget"
            else:
                logging.info("No products found within budget range")
                # Try without budget constraint
                products_without_budget = await fetch_products_from_db(db, top_keywords, max_results, gender_category, None)
                
                if products_without_budget is not None and not products_without_budget.empty:
                    # Verify sorting
                    if 'relevance' in products_without_budget.columns:
                        products_without_budget = products_without_budget.sort_values(
                            by=['relevance'], 
                            ascending=False
                        ).reset_index(drop=True)
                        print(f"🔄 Verified relevance sorting for outside-budget products")
                    
                    logging.info(f"Found {len(products_without_budget)} products outside budget (sorted by relevance)")
                    return products_without_budget, "no_products_in_budget"
                else:
                    logging.info("No products found even without budget constraint")
                    return pd.DataFrame(), "no_products_found"
        else:
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
        # Always return a tuple, even on error
        return pd.DataFrame(), "error"
    
async def enhanced_fetch_products_with_semantic(db: AsyncSession, top_keywords: list, 
                                              max_results=15, gender_category=None, 
                                              budget_range=None, use_semantic=True):
    """
    SAFE VERSION: Falls back gracefully if semantic search fails
    """
    print("\n" + "="*80)
    print("🔍 SAFE ENHANCED PRODUCT SEARCH")
    print("="*80)
    
    # Try semantic search first if available
    if use_semantic and semantic_system.model:
        try:
            # Check if semantic_search method exists
            if hasattr(enhanced_matcher, 'semantic_search'):
                # Ensure products are preprocessed
                if enhanced_matcher.products_df is None:
                    print("🔄 First-time product preprocessing...")
                    await enhanced_matcher.preprocess_products(db)
                
                # Create semantic query from keywords
                if top_keywords:
                    top_terms = [kw for kw, _ in top_keywords[:10]]
                    semantic_query = " ".join(top_terms)
                    
                    print(f"🧠 Semantic query: '{semantic_query}'")
                    
                    # Perform semantic search
                    semantic_results = enhanced_matcher.semantic_search(
                        query=semantic_query,
                        top_k=max_results,
                        threshold=0.2,
                        gender_filter=gender_category,
                        budget_range=budget_range
                    )
                    
                    if semantic_results:
                        print(f"✅ Semantic search found {len(semantic_results)} results")
                        results_df = pd.DataFrame(semantic_results)
                        budget_status = "within_budget" if budget_range else "no_budget_specified"
                        return results_df, budget_status
                    else:
                        print("⚠️ Semantic search found no results, falling back to keyword search")
            else:
                print("⚠️ semantic_search method not available, falling back to keyword search")
            
        except Exception as e:
            print(f"❌ Semantic search failed: {str(e)}")
            logging.error(f"Semantic search error: {str(e)}")
    
    # Fallback to your existing keyword-based search
    print("🔄 Using existing keyword-based search as fallback")
    try:
        fallback_results = await fetch_products_with_budget_awareness(
            db, top_keywords, max_results, gender_category, budget_range
        )
        
        print(f"✅ Fallback search completed")
        return fallback_results
        
    except Exception as e:
        print(f"❌ Fallback search also failed: {str(e)}")
        logging.error(f"Fallback search error: {str(e)}")
        return pd.DataFrame(), "error"
    
def rebalance_keywords_for_current_request(user_context, current_user_input):
    """
    CRITICAL FIX: Rebalance keywords so current request gets priority
    """
    print(f"\n⚖️ REBALANCING KEYWORDS FOR CURRENT REQUEST")
    print("="*60)
    print(f"📝 Current input: '{current_user_input}'")
    
    if "accumulated_keywords" not in user_context or not current_user_input:
        return
    
    # Define keyword categories with importance levels
    keyword_categories = {
        'current_clothing_request': {
            'terms': [],  # Will be filled dynamically
            'target_weight_range': (200000, 500000),  # MUCH HIGHER target weight for current request
            'priority': 1  # Highest priority
        },
        'clothing_items': {
            'terms': ['kemeja', 'shirt', 'blouse', 'celana', 'pants', 'rok', 'skirt', 'dress', 'gaun', 'atasan', 'kaos'],
            'target_weight_range': (5000, 20000),  # Medium weight for other clothing
            'priority': 2
        },
        'style_attributes': {
            'terms': ['casual', 'formal', 'elegant', 'vintage', 'modern', 'minimalist', 'santai'],
            'target_weight_range': (1000, 5000),  # Lower weight for styles
            'priority': 3
        },
        'colors': {
            'terms': ['black', 'white', 'red', 'blue', 'hitam', 'putih', 'merah', 'biru'],
            'target_weight_range': (500, 2000),  # Even lower for colors
            'priority': 4
        },
        'user_attributes': {
            'terms': ['female', 'male', 'woman', 'man', 'perempuan', 'cewe', 'indonesia'],
            'target_weight_range': (100, 1000),  # Low weight for user attributes
            'priority': 5
        },
        'context_terms': {
            'terms': ['carikan', 'tone', 'baju', 'tinggi', 'cm', 'kg'],
            'target_weight_range': (50, 500),  # Lowest weight for context
            'priority': 6
        }
    }
    
    # Extract current clothing request terms from user input
    user_input_lower = current_user_input.lower()
    current_clothing_terms = []
    
    # Check for clothing items in current input
    all_clothing_terms = [
        'kemeja', 'shirt', 'blouse', 'celana', 'pants', 'rok', 'skirt', 'dress', 'gaun', 
        'atasan', 'kaos', 't-shirt', 'jaket', 'jacket', 'sweater', 'bawahan'
    ]
    
    for term in all_clothing_terms:
        if term in user_input_lower:
            current_clothing_terms.append(term)
    
    # Update current request category
    keyword_categories['current_clothing_request']['terms'] = current_clothing_terms
    
    print(f"🎯 Current clothing request terms: {current_clothing_terms}")
    
    # Categorize and rebalance keywords
    rebalanced_count = 0
    max_weights_by_category = {}
    
    # First pass: find current max weights per category
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        current_weight = get_weight_compatible(data)
        
        # Find which category this keyword belongs to
        category_found = None
        for category_name, category_info in keyword_categories.items():
            if any(term in keyword_lower for term in category_info['terms']):
                category_found = category_name
                break
        
        if category_found:
            if category_found not in max_weights_by_category:
                max_weights_by_category[category_found] = current_weight
            else:
                max_weights_by_category[category_found] = max(max_weights_by_category[category_found], current_weight)
    
    print(f"📊 Current max weights by category: {max_weights_by_category}")
    
    # Second pass: rebalance weights
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        current_weight = get_weight_compatible(data)
        
        # Find which category this keyword belongs to
        category_found = None
        for category_name, category_info in keyword_categories.items():
            if any(term in keyword_lower for term in category_info['terms']):
                category_found = category_name
                break
        
        if not category_found:
            category_found = 'context_terms'  # Default category
        
        category_info = keyword_categories[category_found]
        target_min, target_max = category_info['target_weight_range']
        
        # Calculate new weight
        new_weight = current_weight
        
        # If weight is way too high for this category, bring it down
        if current_weight > target_max * 2:
            new_weight = target_max
            print(f"   📉 CAPPING '{keyword}': {current_weight:.1f} → {new_weight:.1f} ({category_found})")
            rebalanced_count += 1
        
        # If this is a current clothing request and weight is too low, boost it
        elif category_found == 'current_clothing_request' and current_weight < target_min:
            new_weight = max(target_max, current_weight * 3.0)  # AGGRESSIVE boost
            print(f"   📈 AGGRESSIVELY BOOSTING '{keyword}': {current_weight:.1f} → {new_weight:.1f} (CURRENT REQUEST)")
            rebalanced_count += 1
        
        # If weight is significantly out of range, adjust it
        elif current_weight > target_max * 5:  # Way too high
            new_weight = target_max * 2  # Bring down but not too aggressively
            print(f"   📉 REDUCING '{keyword}': {current_weight:.1f} → {new_weight:.1f} ({category_found})")
            rebalanced_count += 1
        
        elif current_weight < target_min / 5 and category_found in ['clothing_items', 'current_clothing_request']:  # Too low for important items
            new_weight = target_min
            print(f"   📈 RAISING '{keyword}': {current_weight:.1f} → {new_weight:.1f} ({category_found})")
            rebalanced_count += 1
        
        # Update the weight
        if isinstance(data, dict):
            data["weight"] = new_weight
        else:
            user_context["accumulated_keywords"][keyword] = {
                "weight": new_weight,
                "total_frequency": 1,
                "mention_count": 1,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source": "rebalanced",
                "category": category_found
            }
    
    print(f"⚖️ Rebalanced {rebalanced_count} keywords")
    
    # Show top keywords after rebalancing
    if user_context["accumulated_keywords"]:
        sorted_kw = sorted(user_context["accumulated_keywords"].items(), 
                          key=lambda x: get_weight_compatible(x[1]), reverse=True)
        print(f"\n🏆 TOP 10 AFTER REBALANCING:")
        for i, (kw, data) in enumerate(sorted_kw[:10]):
            weight = get_weight_compatible(data)
            source_icon = "👤" if get_source_compatible(data) == "user_input" else "🤖"
            print(f"      {i+1}. {source_icon} '{kw}' → {weight:.1f}")
    
    print("="*60)

def apply_current_request_boost(user_context, current_user_input):
    """
    IMPROVED: More targeted boost for current request terms
    """
    if not current_user_input or "accumulated_keywords" not in user_context:
        return
    
    print(f"\n🚀 IMPROVED CURRENT REQUEST BOOST")
    print("="*50)
    
    user_input_lower = current_user_input.lower()
    input_words = set(user_input_lower.split())
    
    # Identify key terms in current request
    key_terms = []
    specificity_terms = ['maxi', 'mini', 'midi', 'cropped', 'oversized', 'only', 'just', 'saja']
    clothing_terms = ['skirt', 'rok', 'dress', 'gaun', 'pants', 'celana', 'kemeja', 'shirt']
    
    for word in input_words:
        if word in specificity_terms or word in clothing_terms:
            key_terms.append(word)
    
    print(f"📝 Key terms in current request: {key_terms}")
    
    boosted_count = 0
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        
        # Check if keyword is relevant to current request
        relevance_score = 0
        
        # Exact match with key terms
        if any(term in keyword_lower for term in key_terms):
            relevance_score = 3.0
        # Partial match with input words
        elif any(word in keyword_lower for word in input_words if len(word) > 2):
            relevance_score = 1.5
        
        if relevance_score > 0:
            current_weight = get_weight_compatible(data)
            
            # Determine boost factor
            if any(spec in keyword_lower for spec in specificity_terms):
                boost_factor = 10.0  # HUGE boost for specificity
            elif any(clothing in keyword_lower for clothing in clothing_terms):
                boost_factor = 5.0   # Large boost for clothing
            else:
                boost_factor = relevance_score
            
            new_weight = current_weight * boost_factor
            
            # Update weight
            if isinstance(data, dict):
                data["weight"] = new_weight
                data["last_seen"] = datetime.now().isoformat()
            
            print(f"   🚀 BOOSTED '{keyword}': {current_weight:.1f} → {new_weight:.1f} (×{boost_factor})")
            boosted_count += 1
    
    print(f"🚀 Applied improved boost to {boosted_count} relevant keywords")
    print("="*50)

def cap_extreme_weights(user_context, max_weight_cap=100000):
    """
    Cap extremely high weights that dominate everything else
    """
    if "accumulated_keywords" not in user_context:
        return
    
    print(f"\n🧢 APPLYING WEIGHT CAP (max: {max_weight_cap:,})")
    print("="*50)
    
    capped_count = 0
    
    for keyword, data in user_context["accumulated_keywords"].items():
        current_weight = get_weight_compatible(data)
        
        if current_weight > max_weight_cap:
            if isinstance(data, dict):
                data["weight"] = max_weight_cap
            
            print(f"   🧢 CAPPED '{keyword}': {current_weight:.1f} → {max_weight_cap}")
            capped_count += 1
    
    print(f"🧢 Capped {capped_count} extreme weights")
    print("="*50)

async def enhanced_product_search_with_rebalancing(websocket, user_context, session_id, db, user_language, semantic_enabled):
    """
    Enhanced product search that rebalances keywords first
    """
    print(f"\n🎯 ENHANCED PRODUCT SEARCH WITH REBALANCING")
    print("="*60)
    
    try:
        # STEP 1: Cap extreme weights first
        cap_extreme_weights(user_context, max_weight_cap=50000)
        
        # STEP 2: Rebalance keywords for current request
        current_user_input = user_context.get("current_text_input", "")
        if current_user_input:
            rebalance_keywords_for_current_request(user_context, current_user_input)
            apply_current_request_boost(user_context, current_user_input)
        
        # STEP 3: Get keywords safely
        accumulated_keywords = []
        if "accumulated_keywords" in user_context:
            accumulated_keywords = [(k, get_weight_compatible(v)) 
                                  for k, v in user_context["accumulated_keywords"].items()]
        
        # STEP 4: Sort by weight (after rebalancing)
        accumulated_keywords.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 FINAL KEYWORD RANKING FOR SEARCH:")
        for i, (kw, weight) in enumerate(accumulated_keywords[:15]):
            print(f"   {i+1:2d}. '{kw}' → {weight:.1f}")
        
        # STEP 5: Enhanced keyword extraction with FILTERING
        last_user_input = user_context.get("current_text_input", "")
        last_ai_response = user_context.get("last_ai_response", "")

        if semantic_enabled:
            ranked_keywords = enhanced_extract_ranked_keywords(
                ai_response=last_ai_response,
                translated_input=last_user_input,
                accumulated_keywords=accumulated_keywords,
                use_semantic=True
            )
        else:
            ranked_keywords = extract_ranked_keywords(
                last_ai_response, last_user_input, accumulated_keywords
            )

        # CRITICAL FIX: Filter out conflicting clothing categories from semantic query
        if current_user_input:
            ranked_keywords = filter_conflicting_categories_from_query(ranked_keywords, current_user_input)
        
        # STEP 6: Prioritize current clothing request in final ranking
        final_keywords = prioritize_current_clothing_request(ranked_keywords, current_user_input)
        
        print(f"\n🎯 FINAL SEARCH KEYWORDS:")
        for i, (kw, weight) in enumerate(final_keywords[:15]):
            print(f"   {i+1:2d}. '{kw}' → {weight:.1f}")

        # STEP 7: Create CLEAN semantic query - FIXED VERSION
        if user_context["semantic_enabled"]:
            current_input = user_context.get("current_text_input", "")
            
            if current_input:
                # Extract ONLY the main clothing item from current input
                main_clothing_item = extract_main_clothing_item(current_input)
                
                if main_clothing_item:
                    # Use ONLY the main item for semantic search
                    semantic_query = main_clothing_item
                    print(f"🎯 CLEAN SEMANTIC QUERY: '{semantic_query}'")
                else:
                    # Fallback: use top 3 keywords only
                    top_terms = [kw for kw, _ in final_keywords[:3]]
                    semantic_query = " ".join(top_terms)
                    print(f"📝 FALLBACK QUERY: '{semantic_query}'")
            else:
                # No current input, use top 3 accumulated keywords
                top_terms = [kw for kw, _ in final_keywords[:3]]
                semantic_query = " ".join(top_terms)
                print(f"📚 ACCUMULATED QUERY: '{semantic_query}'")
        else:
            # Non-semantic: still use clean approach
            if current_input:
                main_item = extract_main_clothing_item(current_input)
                if main_item:
                    # Find keywords that match the main item
                    relevant_keywords = [kw for kw, _ in final_keywords[:5] 
                                    if main_item.lower() in kw.lower()]
                    semantic_query = " ".join([main_item] + relevant_keywords)
                else:
                    semantic_query = " ".join([kw for kw, _ in final_keywords[:5]])
            else:
                semantic_query = " ".join([kw for kw, _ in final_keywords[:5]])

        # STEP 8: Translate keywords if needed
        if user_language != "en":
            translated_keywords = []
            for kw, score in final_keywords:
                try:
                    translated_kw = translate_text(kw, "en", session_id)
                    translated_keywords.append((translated_kw, score))
                except:
                    translated_keywords.append((kw, score))
        else:
            translated_keywords = final_keywords

        # STEP 9: Get user context
        user_gender = user_context.get("user_gender", {}).get("category", None)
        budget_range = user_context.get("budget_range", None)
        
        # STEP 10: Search with rebalanced keywords
        if semantic_enabled:
            recommended_products, budget_status = await enhanced_fetch_products_with_semantic(
                db=db,
                top_keywords=translated_keywords,
                max_results=15,
                gender_category=user_gender,
                budget_range=budget_range,
                use_semantic=True
            )
        else:
            recommended_products, budget_status = await fetch_products_with_budget_awareness(
                db=db,
                top_keywords=translated_keywords,
                max_results=15,
                gender_category=user_gender,
                budget_range=budget_range
            )
        
        print(f"✅ Search completed: {len(recommended_products)} products found")
        print("="*60)
        
        return recommended_products, budget_status
        
    except Exception as e:
        logging.error(f"Error in enhanced_product_search_with_rebalancing: {str(e)}")
        raise

def extract_main_clothing_item(user_input):
    """
    Extract the PRIMARY clothing item the user is asking for
    """
    if not user_input:
        return None
    
    user_input_lower = user_input.lower()
    
    # Define clothing items in priority order (most specific first)
    clothing_items = [
        # Specific blazer types
        'cropped blazer', 'oversized blazer', 'fitted blazer', 'long blazer',
        # General blazer
        'blazer', 'jas',
        # Specific shirt types  
        'long sleeve shirt', 'short sleeve shirt', 'button up shirt', 'dress shirt',
        'kemeja lengan panjang', 'kemeja lengan pendek',
        # General shirts
        'kemeja', 'shirt', 'blouse', 'blus',
        # Specific pants
        'wide leg pants', 'skinny pants', 'high waist pants', 'palazzo pants',
        # General pants
        'celana', 'pants', 'trousers',
        # Specific skirts
        'maxi skirt', 'mini skirt', 'a-line skirt', 'pencil skirt',
        'rok maxi', 'rok mini', 'rok panjang', 'rok pendek',
        # General skirts
        'rok', 'skirt',
        # Dresses
        'maxi dress', 'mini dress', 'midi dress', 'dress', 'gaun',
        # Other items
        'sweater', 'cardigan', 'hoodie', 'jacket', 'jaket',
        'atasan', 'kaos', 't-shirt'
    ]
    
    # Find the FIRST (most specific) match
    for item in clothing_items:
        if item in user_input_lower:
            print(f"   🎯 MAIN ITEM DETECTED: '{item}'")
            return item
    
    print(f"   ⚠️ NO MAIN ITEM FOUND in: '{user_input}'")
    return None

def analyze_query_complexity(user_input, accumulated_keywords):
    """
    Analyze query to determine complexity and intent
    """
    if not user_input:
        return {'type': 'normal', 'primary_terms': []}
    
    user_input_lower = user_input.lower()
    words = user_input_lower.split()
    
    # Extract primary clothing terms from input
    clothing_terms = [
        'kemeja', 'shirt', 'blouse', 'blus', 'dress', 'gaun', 'celana', 'pants', 
        'rok', 'skirt', 'jaket', 'jacket', 'sweater', 'atasan', 'kaos', 'sepatu'
    ]
    
    primary_terms = []
    for word in words:
        if any(clothing in word for clothing in clothing_terms):
            primary_terms.append(word)
    
    # Analyze intent indicators
    very_specific_indicators = ['saja', 'only', 'just', 'hanya', 'cuma']
    specific_indicators = ['tunjukkan', 'carikan', 'show me', 'want', 'mau']
    broad_indicators = ['also', 'juga', 'or', 'atau', 'options', 'pilihan', 'suggestions']
    
    if any(indicator in user_input_lower for indicator in very_specific_indicators):
        query_type = 'very_specific'
    elif any(indicator in user_input_lower for indicator in specific_indicators):
        query_type = 'specific'
    elif any(indicator in user_input_lower for indicator in broad_indicators):
        query_type = 'broad'
    elif len(primary_terms) >= 2:  # Multiple clothing items mentioned
        query_type = 'specific'
    elif len(words) >= 6:  # Long query suggests broad exploration
        query_type = 'broad'
    else:
        query_type = 'normal'
    
    return {
        'type': query_type,
        'primary_terms': primary_terms,
        'word_count': len(words),
        'clothing_terms_count': len(primary_terms)
    }

def filter_conflicting_categories_from_query(ranked_keywords, current_user_input):
    """
    FLEXIBLE: Dynamic conflict filtering based on user intent intensity
    """
    print(f"\n🔍 FLEXIBLE CONFLICT FILTERING")
    print("="*60)
    print(f"📝 Current input: '{current_user_input}'")
    
    user_input_lower = current_user_input.lower()
    
    # Analyze user intent intensity
    intent_indicators = {
        'very_specific': ['saja', 'only', 'just', 'hanya', 'cuma'],
        'specific': ['show me', 'tunjukkan', 'carikan', 'want', 'mau', 'need', 'butuh'],
        'change_focus': ['now', 'sekarang', 'instead', 'ganti', 'different', 'lain'],
        'broad_exploration': ['also', 'juga', 'or', 'atau', 'maybe', 'mungkin']
    }
    
    # Determine intent level
    intent_level = 'normal'
    for level, indicators in intent_indicators.items():
        if any(indicator in user_input_lower for indicator in indicators):
            intent_level = level
            break
    
    print(f"   🎯 Intent level: {intent_level}")
    
    # Define clothing categories dynamically
    clothing_categories = {
        'tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'sweater', 'hoodie'],
        'bottoms': ['celana', 'pants', 'rok', 'skirt', 'bawahan', 'jeans', 'shorts'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'coat', 'blazer'],
        'footwear': ['sepatu', 'shoes', 'sneaker', 'heels', 'boots'],
        'accessories': ['tas', 'bag', 'topi', 'hat', 'scarf']
    }
    
    # Find current categories mentioned
    current_categories = set()
    mentioned_terms = []
    
    for category_name, terms in clothing_categories.items():
        found_terms = [term for term in terms if term in user_input_lower]
        if found_terms:
            current_categories.add(category_name)
            mentioned_terms.extend(found_terms)
    
    print(f"   📝 Current categories: {current_categories}")
    print(f"   🔤 Mentioned terms: {mentioned_terms}")
    
    # Apply filtering based on intent level
    if intent_level == 'very_specific':
        # Very strict filtering - keep only exact matches
        filter_strength = 0.9
        keep_related = False
    elif intent_level == 'specific':
        # Moderate filtering - keep main category + styles
        filter_strength = 0.7
        keep_related = True
    elif intent_level == 'change_focus':
        # Strong filtering for category changes
        filter_strength = 0.8
        keep_related = True
    elif intent_level == 'broad_exploration':
        # Light filtering - keep most things
        filter_strength = 0.3
        keep_related = True
    else:
        # Normal filtering
        filter_strength = 0.5
        keep_related = True
    
    # Dynamic keyword filtering
    filtered_keywords = []
    
    for keyword, weight in ranked_keywords:
        keyword_lower = keyword.lower()
        should_keep = True
        keep_reason = "default"
        weight_multiplier = 1.0
        
        # Check if keyword matches current categories
        matches_current = False
        for category in current_categories:
            if any(term in keyword_lower for term in clothing_categories[category]):
                matches_current = True
                keep_reason = f"matches_{category}"
                weight_multiplier = 2.0  # Boost matching categories
                break
        
        # Check if keyword matches mentioned terms specifically
        matches_mentioned = any(term in keyword_lower for term in mentioned_terms)
        if matches_mentioned:
            weight_multiplier = max(weight_multiplier, 3.0)  # Extra boost for exact terms
            keep_reason = "exact_match"
        
        # Check for conflicting categories
        conflicts_with_current = False
        if current_categories and not matches_current:
            for category_name, terms in clothing_categories.items():
                if category_name not in current_categories:
                    if any(term in keyword_lower for term in terms):
                        conflicts_with_current = True
                        break
        
        # Apply filtering logic based on intent
        if matches_current or matches_mentioned:
            # Always keep matching keywords
            should_keep = True
        elif conflicts_with_current:
            # Apply filter strength to conflicting items
            if intent_level == 'very_specific':
                should_keep = False  # Remove all conflicts
            elif intent_level in ['specific', 'change_focus']:
                should_keep = weight > (100 * filter_strength)  # Keep only high-weight conflicts
                if should_keep:
                    weight_multiplier = 0.3  # But reduce their weight significantly
            else:
                should_keep = True
                weight_multiplier = 0.6  # Mild reduction for conflicts
        else:
            # Non-conflicting, non-matching keywords (styles, colors, etc.)
            if keep_related:
                # Keep style attributes, colors, etc.
                style_terms = ['casual', 'formal', 'elegant', 'vintage', 'modern', 'minimalist']
                color_terms = ['black', 'white', 'red', 'blue', 'hitam', 'putih']
                
                if any(style in keyword_lower for style in style_terms + color_terms):
                    should_keep = True
                    weight_multiplier = 0.5  # Reduce but keep
                    keep_reason = "style_attribute"
                else:
                    should_keep = weight > (50 * filter_strength)
                    weight_multiplier = 0.4
            else:
                should_keep = weight > (80 * filter_strength)
                weight_multiplier = 0.2
        
        if should_keep:
            final_weight = weight * weight_multiplier
            filtered_keywords.append((keyword, final_weight))
            
            if weight_multiplier > 1.5:
                print(f"   🚀 BOOSTED '{keyword}': {weight:.1f} → {final_weight:.1f} ({keep_reason})")
            elif weight_multiplier < 0.7:
                print(f"   📉 REDUCED '{keyword}': {weight:.1f} → {final_weight:.1f} ({keep_reason})")
            else:
                print(f"   ✅ KEPT '{keyword}': {final_weight:.1f} ({keep_reason})")
        else:
            print(f"   ❌ FILTERED '{keyword}' (conflicts, intent: {intent_level})")
    
    print(f"   📊 FLEXIBLE FILTERING: {len(ranked_keywords)} → {len(filtered_keywords)}")
    print("="*60)
    
    return filtered_keywords

def prioritize_current_clothing_request(ranked_keywords, current_user_input):
    """
    FLEXIBLE: Dynamic prioritization based on query analysis
    """
    if not current_user_input:
        return ranked_keywords
    
    query_analysis = analyze_query_complexity(current_user_input, ranked_keywords)
    user_input_lower = current_user_input.lower()
    
    # Dynamic boost factors based on query type
    boost_factors = {
        'very_specific': {'exact': 10.0, 'related': 1.0, 'other': 0.2},
        'specific': {'exact': 5.0, 'related': 2.0, 'other': 0.5},
        'broad': {'exact': 3.0, 'related': 1.5, 'other': 0.8},
        'normal': {'exact': 2.0, 'related': 1.2, 'other': 1.0}
    }
    
    factors = boost_factors.get(query_analysis['type'], boost_factors['normal'])
    
    prioritized_keywords = []
    
    for keyword, weight in ranked_keywords:
        keyword_lower = keyword.lower()
        
        # Check relationship to current input
        if any(term in keyword_lower for term in query_analysis['primary_terms']):
            # Exact match with mentioned terms
            new_weight = weight * factors['exact']
            priority_type = "EXACT"
        elif any(word in keyword_lower for word in user_input_lower.split() if len(word) > 2):
            # Related to input words
            new_weight = weight * factors['related']
            priority_type = "RELATED"
        else:
            # Other keywords
            new_weight = weight * factors['other']
            priority_type = "OTHER"
        
        prioritized_keywords.append((keyword, new_weight))
        
        if new_weight != weight:
            print(f"   {priority_type} '{keyword}': {weight:.1f} → {new_weight:.1f}")
    
    return sorted(prioritized_keywords, key=lambda x: x[1], reverse=True)

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
    IMPROVED VERSION: Better handling of specificity changes and keyword pollution
    """
    print(f"\n📝 IMPROVED SMART KEYWORD UPDATE")
    print("="*70)
    
    # STEP 0: Handle gender detection FIRST (NEW)
    if is_user_input and user_input:
        detected_gender = handle_gender_in_keyword_update(user_input, user_context)
        if detected_gender:
            print(f"👤 Gender confirmed: {detected_gender}")
    
    # STEP 1: Check for specificity changes FIRST (before any updates)
    if is_user_input and user_input:
        specificity_change, reset_reason, specificity_found, conflicting = detect_specificity_change(
            user_input, user_context
        )
        
        if specificity_change:
            print(f"🎯 SPECIFICITY CHANGE DETECTED - Executing focused reset")
            execute_specificity_reset(user_context, reset_reason, specificity_found, conflicting)
    
    # STEP 2: Check for major category changes (your existing logic)
    if is_user_input and user_input:
        category_changed = detect_and_handle_category_change_improved(user_input, user_context)
        if category_changed:
            print(f"🔄 MAJOR CATEGORY CHANGE - Already handled")
    
    # STEP 3: Apply keyword decay before adding new keywords
    apply_keyword_decay(user_context)
    
    # STEP 4: Add new keywords with smart weighting
    update_accumulated_keywords(new_keywords, user_context, user_input, is_user_input)
    
    # STEP 5: Apply current request boost
    if is_user_input and user_input:
        apply_current_request_boost(user_context, user_input)
    
    # STEP 6: Final cleanup with specificity awareness
    apply_specificity_aware_cleanup(user_context)
    
    print(f"📊 IMPROVED UPDATE COMPLETE")
    print("="*70)

def apply_specificity_aware_cleanup(user_context):
    """
    Cleanup that's aware of specificity and reduces pollution from irrelevant terms
    """
    if "accumulated_keywords" not in user_context:
        return
    
    print(f"\n🧹 SPECIFICITY-AWARE CLEANUP")
    print("="*40)
    
    # Find the highest weight keywords to understand current focus
    sorted_keywords = sorted(
        user_context["accumulated_keywords"].items(),
        key=lambda x: get_weight_compatible(x[1]),
        reverse=True
    )
    
    if not sorted_keywords:
        return
    
    # Identify current focus from top keywords
    top_5_keywords = [kw for kw, _ in sorted_keywords[:5]]
    current_focus = set()
    
    focus_categories = {
        'skirts': ['rok', 'skirt', 'maxi', 'mini', 'midi'],
        'dresses': ['dress', 'gaun'],
        'tops': ['kemeja', 'shirt', 'blouse', 'atasan'],
        'pants': ['celana', 'pants', 'jeans']
    }
    
    for category, terms in focus_categories.items():
        if any(any(term in kw.lower() for term in terms) for kw in top_5_keywords):
            current_focus.add(category)
    
    print(f"   🎯 Current focus detected: {current_focus}")
    
    # Remove or reduce keywords that conflict with current focus
    keywords_to_adjust = {}
    
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        current_weight = get_weight_compatible(data)
        
        # Check if keyword conflicts with current focus
        belongs_to_focus = False
        belongs_to_other = False
        
        for focus_cat in current_focus:
            if focus_cat in focus_categories:
                if any(term in keyword_lower for term in focus_categories[focus_cat]):
                    belongs_to_focus = True
                    break
        
        if not belongs_to_focus:
            for other_cat, terms in focus_categories.items():
                if other_cat not in current_focus:
                    if any(term in keyword_lower for term in terms):
                        belongs_to_other = True
                        break
        
        # Reduce weight of conflicting keywords
        if belongs_to_other and current_weight > 1000:
            new_weight = current_weight * 0.1  # 90% reduction
            keywords_to_adjust[keyword] = new_weight
            print(f"   📉 REDUCING conflicting: '{keyword}' {current_weight:.1f} → {new_weight:.1f}")
    
    # Apply adjustments
    for keyword, new_weight in keywords_to_adjust.items():
        if isinstance(user_context["accumulated_keywords"][keyword], dict):
            user_context["accumulated_keywords"][keyword]["weight"] = new_weight
    
    # Keep only top 25 keywords after cleanup
    if len(user_context["accumulated_keywords"]) > 25:
        sorted_after_cleanup = sorted(
            user_context["accumulated_keywords"].items(),
            key=lambda x: get_weight_compatible(x[1]),
            reverse=True
        )
        user_context["accumulated_keywords"] = dict(sorted_after_cleanup[:25])
        print(f"   🗑️ Kept top 25 keywords after cleanup")
    
    print(f"🧹 Specificity-aware cleanup completed")
    print("="*40)

def extract_ranked_keywords(ai_response: str = None, translated_input: str = None, accumulated_keywords=None):
    """
    ENHANCED: Proper integration of bold headings, translation mapping, and balanced scoring.
    """
    print("\n" + "="*60)
    print("🔤 ENHANCED KEYWORD EXTRACTION WITH BOLD HEADINGS")
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
    
    # Category-based scoring (FIXED: Gender gets appropriate score)
    scoring_categories = {
        'clothing_items': {
            'terms': ['kemeja', 'shirt', 'blouse', 'blus', 'dress', 'gaun', 'rok', 'skirt',
                     'celana', 'pants', 'jeans', 'jacket', 'jaket', 'sweater', 'cardigan',
                     'atasan', 'top', 'kaos', 't-shirt', 'hoodie', 'blazer', 'coat'],
            'user_score': 300,  # Highest for clothing from user
            'ai_score': 120,    # Good for AI clothing items
            'priority': 'HIGHEST'
        },
        'style_attributes': {
            'terms': ['lengan panjang', 'lengan pendek', 'long sleeve', 'short sleeve',
                     'panjang', 'long', 'pendek', 'short', 'slim', 'regular', 'loose', 'ketat',
                     'longgar', 'tight', 'oversized', 'casual', 'formal', 'elegant', 'maxi', 'mini', 'midi'],
            'user_score': 200,
            'ai_score': 80,
            'priority': 'HIGH'
        },
        'colors': {
            'terms': ['white', 'putih', 'black', 'hitam', 'red', 'merah', 'blue', 'biru',
                     'green', 'hijau', 'yellow', 'kuning', 'brown', 'coklat', 'pink',
                     'purple', 'ungu', 'orange', 'oranye', 'grey', 'abu-abu', 'navy', 'beige'],
            'user_score': 150,
            'ai_score': 60,
            'priority': 'MEDIUM'
        },
        'gender_terms': {  # FIXED: Appropriate scoring for gender
            'terms': ['perempuan', 'wanita', 'female', 'woman', 'pria', 'laki-laki', 'male', 'man'],
            'user_score': 50,   # REDUCED: Gender is filter, not primary keyword
            'ai_score': 20,     # REDUCED: Low AI score for gender
            'priority': 'FILTER'
        },
        'occasions': {
            'terms': ['office', 'kantor', 'party', 'pesta', 'wedding', 'pernikahan',
                     'beach', 'pantai', 'sport', 'olahraga', 'work', 'kerja'],
            'user_score': 100,
            'ai_score': 40,
            'priority': 'LOW'
        }
    }
    
    def get_keyword_score(keyword, source, frequency=1):
        """Get appropriate score based on keyword category and source"""
        keyword_lower = keyword.lower()
        
        for category, config in scoring_categories.items():
            if any(term in keyword_lower for term in config['terms']):
                base_score = config['user_score'] if source == 'user' else config['ai_score']
                return base_score * frequency, config['priority']
        
        # Default scoring
        return (100 * frequency) if source == 'user' else (30 * frequency), 'DEFAULT'
    
    # Process user input (HIGHEST PRIORITY)
    if translated_input:
        print(f"📝 USER INPUT: '{translated_input}'")
        
        # Check for simple responses
        input_words = translated_input.lower().split()
        is_simple_response = (
            len(input_words) <= 2 and 
            all(word in simple_responses for word in input_words)
        )
        
        if is_simple_response:
            print(f"   ⚠️  SIMPLE RESPONSE DETECTED - Skipping")
            return []
        
        # Extract keywords using spaCy
        doc = nlp(translated_input)
        user_keywords = {}
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and 
                len(token.text) > 2 and 
                not token.text.isdigit() and
                token.is_alpha and
                token.text.lower() not in simple_responses):
                
                keyword = token.text.lower()
                user_keywords[keyword] = user_keywords.get(keyword, 0) + 1
        
        # Score user keywords with category-aware scoring
        for keyword, frequency in user_keywords.items():
            # MASSIVELY increased base score for current user input
            score, priority = get_keyword_score(keyword, 'user', frequency)
            
            # ADDITIONAL BOOST for current input
            if priority == 'HIGHEST':  # Clothing items
                score *= 10.0  # 10x boost instead of previous multipliers
            elif priority == 'HIGH':   # Style attributes  
                score *= 5.0   # 5x boost
            
            keyword_scores[keyword] = score
            print(f"   📌 CURRENT INPUT '{keyword}' → {score} ({priority})")    
            # Get translation expansion and exclusions
            try:
                search_terms = get_search_terms_for_keyword(keyword)
                if isinstance(search_terms, dict):
                    # Add include terms with reduced score
                    include_terms = search_terms.get('include', [])
                    exclude_terms = search_terms.get('exclude', [])
                    
                    for include_term in include_terms:
                        if include_term != keyword and include_term not in keyword_scores:
                            expansion_score = score * 0.7  # 70% of original score
                            keyword_scores[include_term] = expansion_score
                            print(f"      ➕ Expanded '{keyword}' → '{include_term}' ({expansion_score:.1f})")
                    
                    if exclude_terms:
                        global_exclusions.update(exclude_terms)
                        print(f"      🚫 Will exclude: {exclude_terms}")
            except Exception as e:
                print(f"      ⚠️ Translation mapping error: {e}")
                pass
    
    # Process AI response (UTILIZE BOLD HEADINGS!)
    if ai_response:
        print(f"\n🤖 AI RESPONSE processing...")
        
        # FIXED: Actually use the bold heading extraction function
        bold_headings = extract_bold_headings_from_ai_response(ai_response)
        print(f"   📋 Found {len(bold_headings)} bold headings: {bold_headings}")
        
        # Process bold headings with HIGH priority
        for heading in bold_headings:
            heading_lower = heading.lower()
            cleaned_heading = re.sub(r'[^\w\s-]', '', heading_lower).strip()
            
            if cleaned_heading and len(cleaned_heading) > 2:
                # Bold headings get high AI scores
                score, priority = get_keyword_score(cleaned_heading, 'ai', 2)  # 2x frequency for headings
                
                if cleaned_heading not in keyword_scores or keyword_scores[cleaned_heading] < score:
                    keyword_scores[cleaned_heading] = score
                    print(f"   🔥 BOLD HEADING: '{cleaned_heading}' → {score} ({priority})")
                
                # Expand bold headings too
                try:
                    search_terms = get_search_terms_for_keyword(cleaned_heading)
                    if isinstance(search_terms, dict):
                        include_terms = search_terms.get('include', [])
                        exclude_terms = search_terms.get('exclude', [])
                        
                        for include_term in include_terms:
                            if include_term not in keyword_scores:
                                expansion_score = score * 0.6
                                keyword_scores[include_term] = expansion_score
                                print(f"      ➕ Bold expansion: '{cleaned_heading}' → '{include_term}' ({expansion_score:.1f})")
                        
                        if exclude_terms:
                            global_exclusions.update(exclude_terms)
                except:
                    pass
        
        # Extract general AI keywords (LOWER priority than bold headings)
        doc = nlp(ai_response)
        ai_keywords = {}
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and 
                len(token.text) > 2 and 
                not token.text.isdigit() and
                token.is_alpha and
                token.text.lower() not in simple_responses):
                
                keyword = token.text.lower()
                
                # Only extract if it's fashion-related
                is_fashion_related = any(
                    any(term in keyword for term in config['terms'])
                    for config in scoring_categories.values()
                )
                
                if is_fashion_related:
                    ai_keywords[keyword] = ai_keywords.get(keyword, 0) + 1
        
        # Score AI keywords (lower than bold headings)
        for keyword, frequency in ai_keywords.items():
            if keyword not in keyword_scores:  # Don't override user input or bold headings
                score, priority = get_keyword_score(keyword, 'ai', frequency)
                keyword_scores[keyword] = score
                print(f"   🤖 AI keyword: '{keyword}' → {score} ({priority})")
    
    # Process accumulated keywords (LOWEST priority with decay)
    if accumulated_keywords:
        print(f"\n📚 ACCUMULATED keywords...")
        
        for keyword, old_weight in accumulated_keywords[:10]:
            if (keyword and len(keyword) > 2 and 
                keyword.lower() not in simple_responses and
                not any(char.isdigit() for char in keyword)):
                
                # Apply decay and category-aware scoring
                _, priority = get_keyword_score(keyword, 'accumulated', 1)
                
                # Different decay rates by category
                if priority == 'FILTER':  # Gender terms
                    decay_factor = 0.2  # Heavy decay for gender
                elif priority == 'LOW':   # Occasions
                    decay_factor = 0.3  # Heavy decay for occasions
                else:
                    decay_factor = 0.5  # Normal decay
                
                accumulated_score = old_weight * decay_factor
                
                if keyword not in keyword_scores and accumulated_score > 10:
                    keyword_scores[keyword] = accumulated_score
                    print(f"   📜 '{keyword}' → {accumulated_score:.1f} ({priority}, decay: {decay_factor})")
    
    # Clean up unwanted terms
    excluded_terms = [
        # Budget and numbers
        "rb", "ribu", "rupiah", "budget", "anggaran", "harga", "price",
        "500", "400", "300", "200", "100", "50", "500rb", "400rb", "300rb",
        "jt", "juta", "000", "maksimal", "minimal", "dibawah", "diatas",
        "idr", "rp",
        
        # Generic conversation words
        "yang", "dan", "atau", "dengan", "untuk", "dari", "pada", "akan",
        "dapat", "ada", "adalah", "ini", "itu", "saya", "anda", "kamu",
        "mereka", "dia", "sangat", "lebih", "kurang", "bagus", "baik",
        "indah", "cantik", "menarik", "cocok", "sesuai", "tepat", "bisa",
        "bisa", "dapat", "akan", "juga", "hanya", "sudah", "belum", "masih",
        
        # AI response fillers
        "recommendation", "rekomendasi", "suggestion", "saran", "option",
        "pilihan", "choice", "style", "gaya", "tampilan", "penampilan",
        "memberikan", "melengkapi", "mudah", "dipadukan", "lemari",
        "potongan", "bagian", "warna", "pilihlah", "stylish", "nyaman",
        "rapi", "longgar", "ramping", "sehari", "hari", "pilihan",
        "apakah", "jika", "iya", "ingin", "melihat", "berdasarkan",
        "seperti", "namun", "tetap", "atau", "lebih", "suka", "menjadi",
        "mudah", "lain", "fit", "cocok", "sesuai", "baik", "bagus",
        
        # Physical descriptors that shouldn't be product keywords
        "kulit", "skin", "tubuh", "body", "tinggi", "height", "berat", "weight",
        "kurus", "gemuk", "langsing", "pendek", "tall", "short"
    ]
    
    cleanup_keywords = []
    for keyword in list(keyword_scores.keys()):
        if (keyword in excluded_terms or 
            len(keyword.split()) > 3 or  # Remove long phrases
            len(keyword) <= 2):          # Remove very short terms
            cleanup_keywords.append(keyword)
    
    for keyword in cleanup_keywords:
        del keyword_scores[keyword]
        print(f"   🗑️ Cleaned: '{keyword}'")
    
    # Sort and return
    ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 FINAL ENHANCED KEYWORDS:")
    for i, (keyword, score) in enumerate(ranked_keywords[:15]):
        # Determine category for display
        category_icon = "👕"  # Default clothing
        for cat_name, config in scoring_categories.items():
            if any(term in keyword.lower() for term in config['terms']):
                if cat_name == 'gender_terms':
                    category_icon = "👤"
                elif cat_name == 'colors':
                    category_icon = "🎨"
                elif cat_name == 'style_attributes':
                    category_icon = "✨"
                elif cat_name == 'occasions':
                    category_icon = "🎪"
                break
        
        priority = "🎯 HIGH" if score >= 200 else "📋 MED" if score >= 80 else "📝 LOW"
        print(f"   {i+1:2d}. {category_icon} {priority} '{keyword}' → {score:.1f}")
    
    if global_exclusions:
        print(f"\n🚫 PRODUCT EXCLUSIONS:")
        for term in sorted(global_exclusions):
            print(f"   ❌ '{term}'")
    
    print(f"\n📊 ENHANCED SUMMARY:")
    high_priority = len([k for k, s in ranked_keywords if s >= 200])
    medium_priority = len([k for k, s in ranked_keywords if 80 <= s < 200])
    low_priority = len([k for k, s in ranked_keywords if s < 80])
    
    print(f"   🎯 High priority (200+): {high_priority}")
    print(f"   📋 Medium priority (80-199): {medium_priority}")
    print(f"   📝 Low priority (<80): {low_priority}")
    print(f"   👕 Bold headings found: {len(bold_headings) if ai_response else 0}")
    print(f"   📝 Total keywords: {len(ranked_keywords)}")
    print("="*60)
    
    # Store exclusions for product filtering
    extract_ranked_keywords.last_exclusions = list(global_exclusions)
    
    return ranked_keywords[:15]

# ADDED: Helper function to get the latest exclusions
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

# Enhanced detection for rapid user changes
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

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with smart embedding initialization"""
    print("🚀 Starting Enhanced Fashion Chatbot...")
    
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
    
    # Initialize semantic system (but don't preprocess yet)
    if semantic_system.model:
        print("✅ Enhanced semantic understanding ready")
        print("💡 Embeddings will be created/loaded when first user connects")
    else:
        print("⚠️ Using fallback mode - install sentence-transformers for better results")
    
    print("🎯 Enhanced Fashion Chatbot is ready!")

embeddings_initialized = False

async def initialize_embeddings_once(db: AsyncSession):
    """Initialize embeddings only once per app lifetime"""
    global embeddings_initialized
    
    if not embeddings_initialized and semantic_system.model:
        print(f"🔄 First-time embeddings initialization...")
        
        # Get database session
        try:
            await enhanced_matcher.preprocess_products(db)
            embeddings_initialized = True
            print(f"✅ Embeddings initialized successfully")
        except Exception as e:
            print(f"⚠️ Embeddings initialization failed: {e}")
            print(f"   Continuing with standard keyword search")

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

@app.get("/admin/refresh-embeddings")
async def refresh_embeddings_endpoint(db: AsyncSession = Depends(get_db)):
    """Admin endpoint to manually refresh embeddings"""
    try:
        print("🔄 Manual embeddings refresh requested...")
        await enhanced_matcher.preprocess_products(db, force_refresh=True)
        global embeddings_initialized
        embeddings_initialized = True
        return {"status": "success", "message": "Embeddings refreshed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/admin/clear-embeddings-cache")
async def clear_embeddings_cache_endpoint():
    """Admin endpoint to clear embeddings cache"""
    try:
        enhanced_matcher.clear_cache()
        global embeddings_initialized
        embeddings_initialized = False
        return {"status": "success", "message": "Embeddings cache cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/admin/embeddings-status")
async def embeddings_status_endpoint():
    """Check embeddings status"""
    cache_exists = all(os.path.exists(f) for f in [
        enhanced_matcher.embeddings_file,
        enhanced_matcher.products_file, 
        enhanced_matcher.metadata_file
    ])
    
    metadata = {}
    if cache_exists:
        try:
            with open(enhanced_matcher.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
        except:
            pass
    
    return {
        "embeddings_initialized": embeddings_initialized,
        "cache_exists": cache_exists,
        "semantic_available": semantic_system.model is not None,
        "cache_metadata": metadata
    }

@app.websocket("/ws")
async def chat(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    try:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        await websocket.send_text(f"{session_id}|🔥 Enhanced Fashion Assistant Ready! 🧠\n\nSelamat Datang! Bagaimana saya bisa membantu Anda hari ini?\n\nWelcome! How can I help you today?")
        await initialize_embeddings_once(db)

        # Initialize enhanced system for this session
        if semantic_system.model:
            print(f"✅ Session {session_id}: Semantic understanding ENABLED")
        else:
            print(f"⚠️ Session {session_id}: Using fallback mode")

        # Your existing message setup with enhanced system prompt
        message_objects = [{
            "role": "system",
            "content": (
                "You are an expert fashion consultant with deep understanding of how clothing interacts with different body types, proportions, and personal attributes. Your mission is to provide personalized, thoughtful fashion recommendations that enhance each user's unique features.\n\n"
                
                "CONVERSATION CONTEXT & MEMORY:\n"
                "- Always remember and reference information the user has shared throughout our conversation\n"
                "- Build upon previous recommendations and acknowledge their preferences or concerns\n"
                "- If they mention budget, lifestyle, or specific needs, keep these in mind for all future suggestions\n"
                "- Reference their previous questions or comments to show you're actively listening\n\n"
                
                "ESSENTIAL USER INFORMATION TO GATHER:\n\n"
                
                "Basic Information:\n"
                "- Gender identity and preferred clothing styles\n"
                "- Height and weight (for proportion considerations)\n"
                "- Body shape/type if they're comfortable sharing (apple, pear, hourglass, rectangle, etc.)\n"
                "- Skin tone and undertones (warm, cool, neutral)\n"
                "- Ethnic background (to consider cultural preferences and what colors/styles typically flatter)\n"
                "- Lifestyle (professional, casual, active, etc.)\n"
                "- Any body areas they want to highlight or feel more confident about\n\n"
                
                "Specific Style & Fit Preferences:\n\n"
                
                "Sleeve Preferences:\n"
                "- Short sleeve, long sleeve, 3/4 sleeve, sleeveless, or cap sleeve\n"
                "- Preference for fitted sleeves vs. loose/flowy sleeves\n"
                "- Comfort level with showing arms\n\n"
                
                "Fit Preferences:\n"
                "- Slim/tailored fit vs. regular fit vs. relaxed/oversized fit\n"
                "- Preference for form-fitting vs. loose-fitting clothing\n"
                "- Comfort level with body-hugging vs. flowing silhouettes\n\n"
                
                "Length Preferences:\n"
                "- Top lengths: cropped, regular, tunic, or oversized\n"
                "- Bottom lengths: mini, knee-length, midi, maxi, or ankle-length\n"
                "- Dress lengths and comfort zones\n\n"
                
                "Neckline Preferences:\n"
                "- V-neck, crew neck, scoop neck, off-shoulder, turtleneck, or boat neck\n"
                "- Comfort level with showing décolletage or preferring more coverage\n\n"
                
                "Waist Definition:\n"
                "- Preference for defined waist vs. straight silhouettes\n"
                "- Comfort with belted looks vs. unstructured fits\n"
                "- High-waisted vs. mid-rise vs. low-rise preferences\n\n"
                
                "Fabric & Texture Preferences:\n"
                "- Structured vs. flowy fabrics\n"
                "- Comfort with clingy materials vs. preference for draping fabrics\n"
                "- Texture preferences (smooth, textured, knit, woven)\n\n"
                
                "Pattern & Color Preferences:\n"
                "- Solid colors vs. patterns (florals, stripes, geometric, etc.)\n"
                "- Bold/bright colors vs. neutral/muted tones\n"
                "- Comfort level with prints and how busy they prefer patterns\n\n"
                
                "Occasion-Specific Needs:\n"
                "- Work/professional requirements\n"
                "- Casual everyday wear preferences\n"
                "- Special occasion or evening wear needs\n"
                "- Seasonal considerations and climate\n\n"
                
                "BODY-AWARENESS PRINCIPLES:\n"
                "- Consider how different cuts, fits, and proportions will interact with their specific body type\n"
                "- Explain WHY certain styles work for their body (e.g., 'A-line dresses will balance your proportions by...')\n"
                "- Address fit concerns specific to their measurements (sleeve length for arm proportions, inseam for leg length, etc.)\n"
                "- Consider how fabric drape and structure will look on their body type\n"
                "- Mention styling tricks that enhance their best features\n"
                "- Be sensitive about body image - focus on enhancing and flattering, never 'hiding' or 'fixing'\n"
                "- Explain how their preferred fits will work with their body type and suggest modifications if needed\n\n"
                
                "RECOMMENDATION FORMAT:\n"
                "Provide at least 3 detailed recommendations using this structure:\n"
                "**[Clothing Item Name]**\n"
                "Detailed explanation of why this works for their specific body type, skin tone, lifestyle, AND their stated fit/style preferences. Include styling tips and how it addresses their proportions while honoring their preferred fits and styles.\n\n"
                
                "COMMUNICATION STYLE:\n"
                "- Use warm, encouraging, and body-positive language\n"
                "- Speak conversationally, not in lists or JSON format\n"
                "- Explain the 'why' behind each recommendation, connecting it to both their body type AND their style preferences\n"
                "- Include specific styling suggestions for complete looks\n"
                "- Never mention specific clothing brands\n"
                "- When their preferences might not be the most flattering for their body type, gently suggest modifications or alternatives while respecting their choices\n\n"
                
                "EXAMPLE RESPONSE STRUCTURE:\n"
                "Based on your [height/body type/skin tone] and your preference for [specific fits/styles mentioned], here are styles that will look amazing on you:\n\n"
                "**Wrap Dresses in Midi Length**\n"
                "Since you mentioned loving midi lengths and preferring a defined waist, wrap dresses are perfect for your body type. The wrap style creates that waist definition you enjoy while the A-line skirt balances your proportions beautifully. With your preference for 3/4 sleeves and your warm skin tone, look for wrap dresses in rich jewel tones like emerald or sapphire with sleeves that hit just below your elbow. The V-neckline works well with your comfort level for necklines, and you can style it with the fitted look you prefer by cinching the waist tie.\n\n"
                
                "NATURAL CONSULTATION FLOW:\n"
                "Conduct the consultation like a natural conversation with a human stylist. Follow this process:\n\n"
                
                "1. **Start with a warm greeting** and ask what brings them to you today\n"
                "2. **Ask questions naturally** based on their response - don't bombard them with all questions at once\n"
                "3. **Ask follow-up questions** based on what they tell you, showing genuine interest\n"
                "4. **Gradually gather information** through organic conversation, not like a checklist\n"
                "5. **Reference previous answers** when asking new questions to show you're listening\n"
                "6. **Ask 2-3 questions maximum** per response to keep it conversational\n"
                "7. **Explain why you're asking** certain questions when it helps (e.g., 'I'm asking about your lifestyle so I can suggest pieces that work for your daily routine')\n\n"
                
                "INFORMATION GATHERING PRIORITY:\n"
                "Start with the most important information first:\n"
                "Priority 1: Body type, height, lifestyle, and what they're looking for\n"
                "Priority 2: STYLE AND FIT PREFERENCES (mandatory before summary), comfort zones\n"
                "Priority 3: Specific details like fabric preferences, color preferences, and BUDGET CONSTRAINTS\n\n"
                
                "MANDATORY STYLE & FIT PREFERENCE INQUIRY:\n"
                "Before presenting the consultation summary, you MUST ask about style and fit preferences if the user hasn't mentioned them. Use natural questions like:\n"
                "- 'What's your preference when it comes to fit - do you like more fitted/tailored pieces or do you prefer looser, more relaxed fits?'\n"
                "- 'Are you drawn to any particular sleeve lengths or neckline styles?'\n"
                "- 'Do you prefer shorter or longer hemlines? What lengths make you feel most confident?'\n"
                "- 'When it comes to waist definition, do you like pieces that show your waistline or do you prefer more flowing, unstructured styles?'\n"
                "- 'Are there any specific styles or fits you absolutely love or want to avoid?'\n"
                "These preferences are essential for creating truly personalized recommendations.\n\n"
                
                "BUDGET INQUIRY:\n"
                "Always ask about budget considerations during the consultation. Use natural phrasing such as:\n"
                "- 'Do you have a budget range in mind for these pieces?'\n"
                "- 'Are there any budget considerations I should keep in mind?'\n"
                "- 'What's your comfort zone budget-wise for these recommendations?'\n"
                "- 'Should I focus on more affordable options or are you open to investing in key pieces?'\n"
                "This helps ensure recommendations are practical and achievable for the user.\n\n"
                
                "CONSULTATION CONFIRMATION PROCESS:\n"
                "Only after you've gathered sufficient information through natural conversation, provide a comprehensive confirmation summary using this format:\n\n"
                
                "**CONSULTATION SUMMARY**\n"
                "Let me confirm all the information I've gathered about you to ensure my recommendations will be perfectly tailored:\n\n"
                
                "**Personal Details:**\n"
                "- Gender Identity: [user's response]\n"
                "- Height & Weight: [user's response]\n"
                "- Body Shape: [user's response]\n"
                "- Skin Tone & Undertones: [user's response]\n"
                "- Ethnic Background: [user's response]\n"
                "- Lifestyle: [user's response]\n"
                "- Areas to Highlight: [user's response]\n\n"
                
                "**Style Preferences:**\n"
                "- Sleeve Preferences: [user's response]\n"
                "- Fit Preferences: [user's response]\n"
                "- Length Preferences: [user's response]\n"
                "- Neckline Preferences: [user's response]\n"
                "- Waist Definition: [user's response]\n"
                "- Fabric & Texture: [user's response]\n"
                "- Pattern & Color: [user's response]\n"
                "- Occasion Needs: [user's response]\n"
                "- Budget Considerations: [user's response if mentioned]\n\n"
                
                "**Confirmation Question:**\n"
                "'Is all of this information accurate? Please let me know if anything needs to be corrected or if you'd like to add any additional preferences. Once you confirm this is correct, I'll provide you with personalized style recommendations that perfectly match your body type and preferences!'\n\n"
                
                "IMPORTANT CONSULTATION GUIDELINES:\n"
                "- If a user gives you incomplete information, ask for the missing details naturally\n"
                "- If they seem hesitant to share certain information, respect their boundaries and work with what you have\n"
                "- If they only provide basic information initially, ask gentle follow-up questions to get more details\n"
                "- ALWAYS ask about style and fit preferences before presenting the consultation summary\n"
                "- Don't present the consultation summary until you have at least Priority 1 AND style/fit preferences\n"
                "- You can provide recommendations with incomplete Priority 3 information, but Priority 1 and 2 (including style/fit preferences) are mandatory\n\n"
                
                "MANDATORY SUMMARY BEFORE RECOMMENDATIONS:\n"
                "When you have gathered sufficient information (at minimum Priority 1, ideally Priority 2), you MUST present the consultation summary before giving any style recommendations. Use phrases like:\n"
                "- 'Before I give you my recommendations, let me make sure I have everything right...'\n"
                "- 'Perfect! Let me summarize what I've learned about you to ensure my suggestions are spot-on...'\n"
                "- 'Great! Now that I understand your preferences, let me confirm all the details...'\n\n"
                
                "DO NOT skip the consultation summary step. ALWAYS present it before recommendations, even if the conversation feels ready to move forward.\n\n"
                
                "ONLY proceed with style recommendations AFTER the user confirms the summary is accurate. If they make corrections, update the summary and ask for confirmation again.\n\n"
                
                "AFTER PROVIDING RECOMMENDATIONS: Ask 'Do these style suggestions work with your confirmed preferences? Would you like me to suggest specific products based on these recommendations, or would you like to explore any of these styles further?'\n\n"
                
                "NEVER provide specific product recommendations in your initial response - focus on explaining why certain styles work for their unique attributes AND their confirmed preferences, then wait for their confirmation before suggesting actual items to purchase."
            )
        }]
        
        # Store the most recent AI response for use in confirmation handling
        last_ai_response = ""
        
        # Enhanced user context with semantic capabilities
        user_context = {
            "current_image_url": None,
            "current_text_input": None,
            "pending_image_analysis": False,
            "has_shared_image": False,
            "has_shared_preferences": False,
            "last_query_type": None,
            "awaiting_confirmation": False,
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
            "semantic_enabled": bool(semantic_system.model),  # Track if semantic is available
            "cultural_context": {},  # Store Indonesian cultural preferences
        }

        # Initialize product preprocessing for semantic search
        if semantic_system.model:
            try:
                print(f"🔄 Initializing semantic system for session {session_id}...")
                await enhanced_matcher.preprocess_products(db)
                print(f"✅ Semantic system ready for session {session_id}")
            except Exception as e:
                print(f"⚠️ Semantic initialization failed for session {session_id}: {e}")
                user_context["semantic_enabled"] = False

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
                    user_language = session_manager.detect_or_retrieve_language(user_input, session_id)
                    logging.info(f"User language '{user_language}' for session {session_id}")
                except Exception as e:
                    logging.error(f"Language detection error: {str(e)}")
                    user_language = "en"
                
                # Enhanced confirmation handling
                if user_context["awaiting_confirmation"]:
                    # Process confirmation response with enhanced capabilities
                    is_positive = user_input.strip().lower() in ["yes", "ya", "iya", "sure", "tentu", "ok", "okay"]
                    is_negative = user_input.strip().lower() in ["no", "tidak", "nope", "nah", "tidak usah"]
                    is_more_request = detect_more_products_request(user_input)

                    print(f"\n📋 ENHANCED CONFIRMATION CHECK START")
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
                    
                    if "budget_range" in user_context and user_context.get("budget_range"):
                        budget = user_context["budget_range"]
                        print(f"💰 Budget: {budget}")
                    
                    print(f"🧠 Semantic: {'Enabled' if user_context['semantic_enabled'] else 'Disabled'}")
                    print("="*50)

                    logging.info(f"Enhanced confirmation state - Input: '{user_input}' | Positive: {is_positive}, Negative: {is_negative}, More: {is_more_request}")
                    
                    if is_positive:
                        # User confirmed, show enhanced product recommendations
                        try:
                            # Get accumulated keywords from context
                            accumulated_keywords = []
                            if "accumulated_keywords" in user_context:
                                accumulated_keywords = [(k, v["weight"]) for k, v in user_context["accumulated_keywords"].items()]
                                
                            # Use enhanced keyword extraction if semantic is available
                            last_user_input = user_context.get("current_text_input", "")
                            last_ai_response = user_context.get("last_ai_response", "")
                            
                            if user_context["semantic_enabled"]:
                                print("🧠 Using enhanced semantic keyword extraction")
                                ranked_keywords = enhanced_extract_ranked_keywords(
                                    ai_response=last_ai_response,
                                    translated_input=last_user_input,
                                    accumulated_keywords=accumulated_keywords,
                                    use_semantic=True
                                )
                            else:
                                print("📝 Using standard keyword extraction")
                                ranked_keywords = extract_ranked_keywords(
                                    last_ai_response, last_user_input, accumulated_keywords
                                )
                            
                            # Translate keywords if needed
                            if user_language != "en":
                                keywords_only = [kw for kw, _ in ranked_keywords]
                                combined_text = " ||| ".join(keywords_only)
                                translated_combined = translate_text(combined_text, "en", session_id)
                                translated_keywords = translated_combined.split(" ||| ")
                                
                                if len(translated_keywords) == len(ranked_keywords):
                                    translated_ranked_keywords = [(translated_keywords[i], score) for i, (_, score) in enumerate(ranked_keywords)]
                                else:
                                    print(f"Translation mismatch: got {len(translated_keywords)} items, expected {len(ranked_keywords)}")
                                    translated_ranked_keywords = [(translate_text(kw, "en", session_id), score) for kw, score in ranked_keywords]
                            else:
                                translated_ranked_keywords = ranked_keywords

                            logging.info(f"Using enhanced ranked keywords for product search: {translated_ranked_keywords[:15]}")
                            
                            # Get user gender and budget for filtering
                            user_gender = user_context.get("user_gender", {}).get("category", None)
                            budget_range = user_context.get("budget_range", None)

                            if budget_range:
                                logging.info(f"Using budget filter: {budget_range}")
                                print(f"Using budget filter: {budget_range}")
                            
                            # Enhanced confirmation message with semantic indicators
                            if user_context["semantic_enabled"]:
                                positive_response = "🧠 **Enhanced AI Search Results** - Here are products that perfectly match your style preferences:"
                            else:
                                positive_response = "Great! Based on your preferences and style recommendations, here are some products that might interest you:"
                            
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

                            if user_language != "en":
                                positive_response = translate_text(positive_response, user_language, session_id)

                            print(f"\n🔍 ENHANCED PRODUCT SEARCH INPUTS:")
                            print(f"   🎯 Keywords: {len(translated_ranked_keywords)}")
                            for i, (kw, score) in enumerate(translated_ranked_keywords[:15]):
                                print(f"      {i+1}. '{kw}' → {score:.2f}")
                            print(f"   👤 Gender: {user_gender}")
                            print(f"   💰 Budget: {budget_range}")
                            print(f"   🧠 Semantic: {'Enabled' if user_context['semantic_enabled'] else 'Disabled'}")
                            print()
                            
                            # Use enhanced product search if semantic is available
                            try:
                                recommended_products, budget_status = await enhanced_product_search_with_rebalancing(
                                    websocket, user_context, session_id, db, user_language, user_context["semantic_enabled"]
                                )
                                
                                print(f"Successfully fetched {len(recommended_products)} products with status: {budget_status}")

                                # Handle different budget scenarios (keep your existing logic)
                                if budget_status == "no_products_in_budget":
                                    cheapest_price = recommended_products['price'].min() if not recommended_products.empty else None
                                    most_expensive_price = recommended_products['price'].max() if not recommended_products.empty else None
                                    
                                    budget_messages = generate_budget_message(budget_range, user_language, cheapest_price, most_expensive_price)
                                    budget_response = budget_messages["show_outside_budget"]
                                    
                                    user_context["pending_products"] = recommended_products
                                    user_context["awaiting_budget_decision"] = True
                                    user_context["budget_scenario"] = "show_outside_budget"
                                    
                                    new_ai_message = ChatHistoryDB(
                                        session_id=session_id,
                                        message_type="assistant",
                                        content=budget_response
                                    )
                                    db.add(new_ai_message)
                                    await db.commit()
                                    
                                    await websocket.send_text(f"{session_id}|{budget_response}")
                                    continue
                                    
                                elif budget_status == "no_products_found":
                                    budget_messages = generate_budget_message(budget_range, user_language)
                                    no_products_response = budget_messages["no_products"]
                                    
                                    user_context["awaiting_budget_decision"] = True
                                    user_context["budget_scenario"] = "no_products"
                                    
                                    new_ai_message = ChatHistoryDB(
                                        session_id=session_id,
                                        message_type="assistant",
                                        content=no_products_response
                                    )
                                    db.add(new_ai_message)
                                    await db.commit()
                                    
                                    await websocket.send_text(f"{session_id}|{no_products_response}")
                                    continue
                                
                            except Exception as fetch_error:
                                logging.error(f"Error calling enhanced product search: {str(fetch_error)}")
                                logging.error(f"Parameters passed:")
                                logging.error(f"- db: {type(db)}")
                                logging.error(f"- top_keywords: {type(translated_ranked_keywords)} - {translated_ranked_keywords[:3] if translated_ranked_keywords else 'None'}")
                                logging.error(f"- user_gender: {user_gender}")
                                logging.error(f"- budget_range: {budget_range}")
                                raise
                            
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

                            # Create enhanced response with semantic indicators
                            if not first_page_products.empty:
                                complete_response = positive_response + "\n\n"
                                
                                for _, row in first_page_products.iterrows():
                                    # Enhanced product card with semantic relevance indicator
                                    relevance_indicator = ""
                                    if 'semantic_match' in row and row.get('semantic_match', False):
                                        relevance_score = row.get('relevance', 0)
                                        if relevance_score >= 0.7:
                                            relevance_indicator = "🎯 **Perfect AI Match** | "
                                        elif relevance_score >= 0.5:
                                            relevance_indicator = "✅ **Good AI Match** | "
                                        else:
                                            relevance_indicator = "📝 **Related** | "
                                    elif user_context["semantic_enabled"]:
                                        # Even for keyword matches, show it's enhanced
                                        relevance_indicator = "🔍 **Enhanced Match** | "
                                    
                                    product_card = (
                                        "<div class='product-card'>\n"
                                        f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                        f"<div class='product-info'>\n"
                                        f"<h3>{relevance_indicator}{row['product']}</h3>\n"
                                        f"<p class='price'>IDR {row['price']:,.0f}</p>\n"
                                        f"<p class='description'>{row['description']}</p>\n"
                                        f"<p class='available'>Available in size: {row['size']}, Color: {row['color']}</p>\n"
                                        f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                        "</div>\n"
                                        "</div>\n"
                                    )
                                    complete_response += product_card
                                
                                if has_more:
                                    if user_context["semantic_enabled"]:
                                        complete_response += "\n\n🧠 **Want to see more AI-matched products?** Just ask for 'more products' or 'lainnya'!"
                                    else:
                                        complete_response += "\n\nWould you like to see more options? Just ask for 'more products' or 'lainnya'!"
                            else:
                                complete_response = positive_response + "\n\nI'm sorry, but I couldn't find specific product recommendations at the moment. Would you like me to help you with something else?"

                            # Handle translation while protecting HTML (your existing method)
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
                                    # Fallback to original method
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
                            
                            # Send the enhanced response
                            complete_response_html = render_markdown(translated_response)
                            print(f"📤 Sending enhanced product recommendations to user")
                            await websocket.send_text(f"{session_id}|{complete_response_html}")
                            
                        except Exception as e:
                            logging.error(f"Error during enhanced product recommendation: {str(e)}")
                            error_msg = "I'm sorry, I couldn't fetch product recommendations. Is there something else you'd like to know about fashion?"
                            if user_language != "en":
                                error_msg = translate_text(error_msg, user_language, session_id)
                            await websocket.send_text(f"{session_id}|{error_msg}")
                        
                        # Reset confirmation flag
                        user_context["awaiting_confirmation"] = True

                    elif is_more_request:
                        # Enhanced "more products" handling
                        logging.info("🔄 User requesting MORE products (Enhanced)")
                        
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
                                user_context["product_cache"]["current_page"] = next_page
                                user_context["product_cache"]["has_more"] = has_more
                                
                                # Enhanced responses with semantic awareness
                                if user_context["semantic_enabled"]:
                                    more_responses_en = [
                                        "🧠 Here are more AI-matched options for you:",
                                        "🔍 I found additional styles that perfectly match your preferences:",
                                        "✨ More personalized recommendations from our AI:",
                                    ]
                                    more_responses_id = [
                                        "🧠 Berikut pilihan lain yang cocok dengan AI:",
                                        "🔍 Saya menemukan gaya tambahan yang sesuai preferensi:",
                                        "✨ Rekomendasi personal lainnya dari AI:",
                                    ]
                                else:
                                    more_responses_en = [
                                        "Here are some more options that might interest you:",
                                        "I found some additional styles you might like:",
                                        "Let me show you a few more possibilities:",
                                    ]
                                    more_responses_id = [
                                        "Berikut beberapa pilihan lain yang mungkin menarik:",
                                        "Saya menemukan beberapa gaya tambahan yang mungkin Anda suka:",
                                        "Mari saya tunjukkan beberapa kemungkinan lainnya:",
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
                                
                                # Generate enhanced product cards for next page
                                complete_response = positive_response + "\n\n"
                                
                                for _, row in next_page_products.iterrows():
                                    # Enhanced product card with semantic indicators
                                    relevance_indicator = ""
                                    if 'semantic_match' in row and row.get('semantic_match', False):
                                        relevance_score = row.get('relevance', 0)
                                        if relevance_score >= 0.7:
                                            relevance_indicator = "🎯 **Perfect AI Match** | "
                                        elif relevance_score >= 0.5:
                                            relevance_indicator = "✅ **Good AI Match** | "
                                        else:
                                            relevance_indicator = "📝 **Related** | "
                                    elif user_context["semantic_enabled"]:
                                        relevance_indicator = "🔍 **Enhanced Match** | "
                                    
                                    product_card = (
                                        "<div class='product-card'>\n"
                                        f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                        f"<div class='product-info'>\n"
                                        f"<h3>{relevance_indicator}{row['product']}</h3>\n"
                                        f"<p class='price'>IDR {row['price']:,.0f}</p>\n"
                                        f"<p class='description'>{row['description']}</p>\n"
                                        f"<p class='available'>Available in size: {row['size']}, Color: {row['color']}</p>\n"
                                        f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                        "</div>\n"
                                        "</div>\n"
                                    )
                                    complete_response += product_card
                                
                                # Enhanced footer with semantic awareness
                                if has_more:
                                    if user_language != "en":
                                        if user_context["semantic_enabled"]:
                                            more_hint = translate_text("\n\n🧠 I have even more AI-matched options! Just let me know if you want to continue exploring.", user_language, session_id)
                                        else:
                                            more_hint = translate_text("\n\nI have even more options if you'd like to continue exploring! Just let me know if you want to see more.", user_language, session_id)
                                    else:
                                        if user_context["semantic_enabled"]:
                                            more_hint = "\n\n🧠 I have even more AI-matched options! Just let me know if you want to continue exploring."
                                        else:
                                            more_hint = "\n\nI have even more options if you'd like to continue exploring! Just let me know if you want to see more."
                                    complete_response += more_hint
                                else:
                                    if user_language != "en":
                                        if user_context["semantic_enabled"]:
                                            end_hint = translate_text("\n\n🎯 That's all the AI-matched products I found! Is there anything else I can help you with, or would you like to try a different search?", user_language, session_id)
                                        else:
                                            end_hint = translate_text("\n\nThat's all the products I found based on your preferences. Is there anything else I can help you with, or would you like to try a different search?", user_language, session_id)
                                    else:
                                        if user_context["semantic_enabled"]:
                                            end_hint = "\n\n🎯 That's all the AI-matched products I found! Is there anything else I can help you with, or would you like to try a different search?"
                                        else:
                                            end_hint = "\n\nThat's all the products I found based on your preferences. Is there anything else I can help you with, or would you like to try a different search?"
                                    complete_response += end_hint
                                
                                # Handle translation (same method as before)
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
                                
                                # Log enhanced pagination info
                                total_products = len(user_context["product_cache"]["all_results"])
                                products_shown = (next_page + 1) * products_per_page
                                logging.info(f"📄 Enhanced pagination: Page {next_page + 1}, products {current_page * products_per_page + 1}-{min(products_shown, total_products)} of {total_products}")
                                logging.info(f"📊 Semantic enabled: {user_context['semantic_enabled']}")
                                logging.info(f"📊 Has more pages: {has_more}")
                                
                                # Keep awaiting confirmation for potential more requests
                                user_context["awaiting_confirmation"] = True
                                
                            else:
                                # No more products available
                                if user_language != "en":
                                    if user_context["semantic_enabled"]:
                                        no_more_msg = translate_text("🎯 I've shown you all the best AI matches I could find. Would you like to try a different style or adjust your preferences?", user_language, session_id)
                                    else:
                                        no_more_msg = translate_text("I've shown you all the best matches I could find. Would you like to try a different style or adjust your preferences?", user_language, session_id)
                                else:
                                    if user_context["semantic_enabled"]:
                                        no_more_msg = "🎯 I've shown you all the best AI matches I could find. Would you like to try a different style or adjust your preferences?"
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
                            # No cached results available
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
                        # Handle budget decision responses (keep your existing budget handling logic)
                        budget_response_type = detect_budget_response(user_input)
                        budget_scenario = user_context.get("budget_scenario", "")
                        
                        if budget_response_type == "show_anyway":
                            # User wants to see products outside budget
                            pending_products = user_context.get("pending_products", pd.DataFrame())
                            
                            if not pending_products.empty:
                                # Show the products that were outside budget
                                if user_context["semantic_enabled"]:
                                    positive_response = "🧠 **Enhanced AI Results** - Here are the product recommendations outside your budget range:"
                                else:
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
                                
                                # Generate product cards with enhanced indicators
                                complete_response = positive_response + "\n\n"
                                
                                for _, row in first_page_products.iterrows():
                                    relevance_indicator = ""
                                    if 'semantic_match' in row and row.get('semantic_match', False):
                                        relevance_score = row.get('relevance', 0)
                                        if relevance_score >= 0.7:
                                            relevance_indicator = "🎯 **Perfect AI Match** | "
                                        elif relevance_score >= 0.5:
                                            relevance_indicator = "✅ **Good AI Match** | "
                                        else:
                                            relevance_indicator = "📝 **Related** | "
                                    elif user_context["semantic_enabled"]:
                                        relevance_indicator = "🔍 **Enhanced Match** | "
                                    
                                    product_card = (
                                        "<div class='product-card'>\n"
                                        f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                        f"<div class='product-info'>\n"
                                        f"<h3>{relevance_indicator}{row['product']}</h3>\n"
                                        f"<p class='price'>IDR {row['price']:,.0f}</p>\n"
                                        f"<p class='description'>{row['description']}</p>\n"
                                        f"<p class='available'>Available in size: {row['size']}, Color: {row['color']}</p>\n"
                                        f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                        "</div>\n"
                                        "</div>\n"
                                    )
                                    complete_response += product_card
                                
                                # Handle translation and send response
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
                                    except:
                                        translated_response = complete_response
                                else:
                                    translated_response = complete_response
                                
                                new_ai_message = ChatHistoryDB(
                                    session_id=session_id,
                                    message_type="assistant",
                                    content=complete_response
                                )
                                db.add(new_ai_message)
                                await db.commit()
                                
                                complete_response_html = render_markdown(translated_response)
                                await websocket.send_text(f"{session_id}|{complete_response_html}")
                            
                            # Clear budget decision flags
                            user_context["awaiting_budget_decision"] = False
                            user_context["pending_products"] = pd.DataFrame()
                            user_context["budget_scenario"] = ""
                            user_context["awaiting_confirmation"] = True
                            
                        elif budget_response_type == "adjust_budget":
                            # User wants to adjust budget (keep your existing logic)
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
                        # Extract keywords from the user's additional input and continue
                        if user_language != "en":
                            translated_input = translate_text(user_input, "en", session_id)
                        else:
                            translated_input = user_input
                        
                        # Save current text input for later use
                        user_context["current_text_input"] = user_input
                            
                        # Extract and update keywords from this additional context
                        if user_context["semantic_enabled"]:
                            additional_keywords = enhanced_extract_ranked_keywords("", translated_input, use_semantic=True)
                        else:
                            additional_keywords = extract_ranked_keywords("", translated_input)
                        
                        update_accumulated_keywords(additional_keywords, user_context, user_input, is_user_input=True)
                        
                        # Continue with regular processing
                        user_context["awaiting_confirmation"] = False
                
                # Check if input contains an image URL (keep your existing image processing)
                url_pattern = re.compile(r'(https?://\S+\.(?:jpg|jpeg|png|gif|bmp|webp))', re.IGNORECASE)
                image_url_match = url_pattern.search(user_input)
                
                if not user_context["awaiting_confirmation"] and image_url_match:
                    # Process image input with enhanced capabilities
                    image_url = image_url_match.group(1)
                    text_content = user_input.replace(image_url, "").strip()
                    
                    try:
                        # Update user context for image input
                        user_context["has_shared_image"] = True
                        user_context["last_query_type"] = "mixed" if text_content else "image"
                        user_context["current_image_url"] = image_url
                        user_context["current_text_input"] = text_content

                        # Call your existing image analysis
                        clothing_features = await analyze_uploaded_image(image_url)

                        # Error handling for image analysis
                        if clothing_features.startswith("Error:"):
                            error_message = clothing_features
                            logging.warning(f"Image analysis error: {error_message}")

                            new_error_message = ChatHistoryDB(
                                session_id=session_id,
                                message_type="assistant",
                                content=error_message
                            )
                            db.add(new_error_message)
                            await db.commit()

                            await websocket.send_text(f"{session_id}|{error_message}")
                            continue

                        # Enhanced gender detection from text content
                        if text_content:
                            force_update = detect_gender_change_request(user_input)
                            detect_and_update_gender(user_input, user_context, force_update)

                        # Prepare enhanced prompt with semantic awareness
                        gender_info = get_user_gender(user_context)
                        user_gender = gender_info["category"] if gender_info["is_valid"] else None
                        gender_context = ""
                        if user_gender_info["category"]:
                            gender_context = f" I am {user_gender_info['category']}."
                        
                        semantic_context = ""
                        if user_context["semantic_enabled"]:
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
                        if user_context["semantic_enabled"]:
                            image_keywords = enhanced_extract_ranked_keywords(clothing_features, "", use_semantic=True)
                        else:
                            image_keywords = extract_ranked_keywords(clothing_features, "")
                        update_accumulated_keywords(image_keywords, user_context, user_input, is_user_input=True)
                        
                        # Process text content keywords if available
                        if text_content:
                            if user_context["semantic_enabled"]:
                                text_keywords = enhanced_extract_ranked_keywords("", text_content, use_semantic=True)
                            else:
                                text_keywords = extract_ranked_keywords("", text_content)
                            update_accumulated_keywords(text_keywords, user_context, user_input, is_user_input=True)
                        
                        # Extract keywords from AI style suggestions
                        if user_context["semantic_enabled"]:
                            style_keywords = enhanced_extract_ranked_keywords(ai_response, "", use_semantic=True)
                        else:
                            style_keywords = extract_ranked_keywords(ai_response, "")
                        update_accumulated_keywords(style_keywords, user_context, user_input, is_ai_response=True)
                        
                        # Enhanced logging
                        logging.info(f"Enhanced image analysis - Semantic enabled: {user_context['semantic_enabled']}")
                        logging.info(f"Updated accumulated keywords after enhanced image analysis: {user_context['accumulated_keywords']}")
                        logging.info(f"Current user gender info: {user_context['user_gender']}")

                        # Translate if needed
                        if user_language != "en":
                            translated_ai_response = translate_text(ai_response, user_language, session_id)
                        else:
                            translated_ai_response = ai_response

                        # Save and send enhanced styling recommendations
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
                        logging.error(f"Error during enhanced image processing: {str(input_error)}\n{traceback.format_exc()}")
                        error_msg = "Sorry, there was an issue processing your image. Could you try again?"
                        await websocket.send_text(f"{session_id}|{error_msg}")
                
                # Handle normal text input with enhanced capabilities
                elif not user_context["awaiting_confirmation"]:

                    print(f"\n📋 ENHANCED TEXT PROCESSING START")
                    print("="*50)
                    print(f"📝 User input: '{user_input}'")
                    print(f"🧠 Semantic: {'Enabled' if user_context['semantic_enabled'] else 'Disabled'}")
                    if "budget_range" in user_context and user_context.get("budget_range"):
                        print(f"💰 Current budget: {user_context['budget_range']}")
                    print("="*50)

                    # Check for small talk
                    if await is_small_talk(user_input):
                        if user_context["semantic_enabled"]:
                            ai_response = "🧠 Hello! I'm your enhanced AI fashion assistant. How can I help you with personalized fashion recommendations today? Feel free to share information about your style preferences or upload an image for AI-powered suggestions."
                        else:
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
                    
                    # Enhanced text processing
                    user_context["last_query_type"] = "text"
                    user_context["current_text_input"] = user_input

                    # Translate if needed
                    if user_language != "en":
                        translated_input = translate_text(user_input, "en", session_id)
                    else:
                        translated_input = user_input
                    
                    # Enhanced gender detection
                    force_update = detect_gender_change_request(user_input)
                    detect_and_update_gender(translated_input, user_context, force_update)
                        
                    # Enhanced keyword extraction
                    if user_context["semantic_enabled"]:
                        input_keywords = enhanced_extract_ranked_keywords("", translated_input, use_semantic=True)
                        print(f"🧠 Enhanced semantic keyword extraction completed")
                    else:
                        input_keywords = extract_ranked_keywords("", translated_input)
                        print(f"📝 Standard keyword extraction completed")
                    
                    update_accumulated_keywords(input_keywords, user_context, user_input, is_user_input=True)
                    
                    # Enhanced conversation context
                    enhanced_context = ""
                    if user_context["semantic_enabled"]:
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
                    if user_context["semantic_enabled"]:
                        response_keywords = enhanced_extract_ranked_keywords(ai_response, "", use_semantic=True)
                    else:
                        response_keywords = extract_ranked_keywords(ai_response, "")
                    update_accumulated_keywords(response_keywords, user_context, user_input, is_ai_response=True)
                    
                    # Enhanced logging
                    logging.info(f"Enhanced text conversation - Semantic enabled: {user_context['semantic_enabled']}")
                    logging.info(f"Updated accumulated keywords after enhanced text conversation: {user_context['accumulated_keywords']}")
                    logging.info(f"Current user gender info: {user_context['user_gender']}")
                    
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
                    detect_rapid_preference_changes(user_input, user_context)
                    
                    # Save current text input
                    user_context["current_text_input"] = user_input
                    
                    # Enhanced keyword context update
                    if user_context["semantic_enabled"]:
                        enhanced_input_keywords = enhanced_extract_ranked_keywords(
                            "", translated_input, 
                            accumulated_keywords=[(k, v["weight"]) for k, v in user_context.get("accumulated_keywords", {}).items()],
                            use_semantic=True
                        )
                    else:
                        enhanced_input_keywords = extract_ranked_keywords(
                            "", translated_input, 
                            accumulated_keywords=[(k, v["weight"]) for k, v in user_context.get("accumulated_keywords", {}).items()]
                        )
                    
                    update_accumulated_keywords(enhanced_input_keywords, user_context, user_input, is_user_input=True)
                                        
                    # Set awaiting confirmation flag
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
    Extract budget information from user input text.
    Returns tuple (min_price, max_price) or None if no budget mentioned.
    """
    if not text:
        return None
    
    print(f"\n💰 BUDGET EXTRACTION DEBUG")
    print(f"   📝 Input text: '{text}'")
    
    text_lower = text.lower()
    print(f"   🔤 Lowercase: '{text_lower}'")
    
    def convert_to_rupiah(amount_str, unit):
        """Convert amount string with unit to actual rupiah value"""
        try:
            amount = int(amount_str)
            print(f"      🔢 Converting: {amount_str} + {unit}")
            
            if unit in ['rb', 'ribu', 'k']:
                result = amount * 1000
                print(f"      ✅ {amount_str} {unit} = IDR {result:,}")
                return result
            elif unit in ['jt', 'juta']:
                result = amount * 1000000
                print(f"      ✅ {amount_str} {unit} = IDR {result:,}")
                return result
            elif unit == '000':
                result = amount * 1000
                print(f"      ✅ {amount_str}{unit} = IDR {result:,}")
                return result
            else:
                print(f"      ✅ {amount_str} (no unit) = IDR {amount:,}")
                return amount
        except Exception as e:
            print(f"      ❌ Conversion failed: {e}")
            return None
    
    # Simplified and more accurate patterns
    budget_patterns = [
        # Range patterns: "50rb-200rb", "antara 100rb sampai 300rb"
        (r'(?:antara|between)?\s*(\d+)(?:rb|ribu|k)?\s*(?:-|sampai|hingga|to)\s*(\d+)(?:rb|ribu|k)?', "RANGE"),
        
        # Maximum patterns: "dibawah 300rb", "maksimal 200rb", "under 500000"
        (r'(?:dibawah|under|maksimal|max|kurang\s+dari|less\s+than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|000)?', "MAX"),
        
        # Minimum patterns: "diatas 100rb", "minimal 50rb", "over 200000"
        (r'(?:diatas|over|minimal|min|lebih\s+dari|more\s+than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|000)?', "MIN"),
        
        # Exact budget: "budget 150rb", "anggaran 200rb"
        (r'(?:budget|anggaran)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|000)?', "EXACT"),
    ]
    
    print(f"   🔍 Trying {len(budget_patterns)} patterns...")
    
    # Try each pattern
    for pattern_idx, (pattern, pattern_type) in enumerate(budget_patterns):
        print(f"\n   Pattern {pattern_idx + 1} ({pattern_type}): {pattern}")
        matches = list(re.finditer(pattern, text_lower))
        
        for match_idx, match in enumerate(matches):
            groups = match.groups()
            match_text = match.group(0)
            
            print(f"      Match {match_idx + 1}: '{match_text}' → Groups: {groups}")
            
            # Range pattern (two amounts)
            if pattern_type == "RANGE" and len(groups) >= 2 and groups[0] and groups[1]:
                print(f"      📊 Processing range pattern...")
                
                # Determine units
                unit1 = 'rb' if any(x in match_text for x in ['rb', 'ribu', 'k']) else None
                unit2 = unit1  # Assume same unit for both
                
                print(f"      🏷️  Units: {unit1}, {unit2}")
                
                min_price = convert_to_rupiah(groups[0], unit1)
                max_price = convert_to_rupiah(groups[1], unit2)
                
                if min_price and max_price:
                    result = (min(min_price, max_price), max(min_price, max_price))
                    print(f"      🎯 RANGE BUDGET FOUND: {result}")
                    return result
            
            # Single amount patterns
            elif len(groups) >= 1 and groups[0]:
                print(f"      📊 Processing single amount pattern...")
                
                # Determine unit from match text
                if any(x in match_text for x in ['rb', 'ribu', 'k']):
                    unit = 'rb'
                elif 'jt' in match_text or 'juta' in match_text:
                    unit = 'jt'
                elif '000' in groups[0]:
                    unit = '000'
                else:
                    unit = None
                
                print(f"      🏷️  Unit detected: {unit}")
                
                amount = convert_to_rupiah(groups[0], unit)
                
                if amount:
                    if pattern_type == "MAX":  # Maximum patterns
                        result = (None, amount)
                        print(f"      🎯 MAX BUDGET FOUND: {result}")
                        return result
                    elif pattern_type == "MIN":  # Minimum patterns
                        result = (amount, None)
                        print(f"      🎯 MIN BUDGET FOUND: {result}")
                        return result
                    elif pattern_type == "EXACT":  # Exact budget
                        # Create range ±20%
                        min_range = int(amount * 0.8)
                        max_range = int(amount * 1.2)
                        result = (min_range, max_range)
                        print(f"      🎯 EXACT BUDGET FOUND: {result}")
                        return result
    
    print("   ❌ No budget pattern matched")
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

def update_accumulated_keywords(keywords, user_context, user_input, is_user_input=False, is_ai_response=False):
    """
    IMPROVED: Smarter keyword addition that considers current request context
    """
    if "accumulated_keywords" not in user_context:
        user_context["accumulated_keywords"] = {}
    
    # Extract current request context
    current_specificity = []
    if user_input:
        user_input_lower = user_input.lower()
        specific_modifiers = [
            'maxi', 'mini', 'midi', 'cropped', 'oversized', 'slim', 'only', 'just', 'saja', 'panjang', 'pendek'
        ]
        current_specificity = [mod for mod in specific_modifiers if mod in user_input_lower]
    
    updates_made = 0
    new_keywords_added = 0
    
    for keyword, score in keywords:
        if not keyword or len(keyword) < 2:
            continue
        
        keyword_lower = keyword.lower()
        
        # MASSIVE boost for current user input, especially with specificity
        if is_user_input:
            base_boost = 10.0
            
            # Extra boost for specificity terms in current request
            if any(spec in keyword_lower for spec in current_specificity):
                specificity_boost = 20.0  # HUGE boost for specific requests
                print(f"   🎯 SPECIFICITY BOOST: '{keyword}' gets {specificity_boost}x boost")
            else:
                specificity_boost = 1.0
            
            frequency_boost = base_boost * specificity_boost
            estimated_frequency = max(1, score / 20)  # More sensitive
        else:
            frequency_boost = 1.0
            estimated_frequency = max(1, score / 100)
        
        if keyword_lower in user_context["accumulated_keywords"]:
            # Update existing keyword
            data = user_context["accumulated_keywords"][keyword_lower]
            old_weight = get_weight_compatible(data)
            
            # Calculate new values
            if is_user_input:
                new_weight = old_weight + (score * frequency_boost)
                new_count = data.get("count", 1) + frequency_boost
            else:
                new_weight = old_weight + score
                new_count = data.get("count", 1) + 1
            
            # Apply category-aware boosts
            if is_user_input:
                category = get_keyword_category(keyword_lower)
                if category == 'clothing_items':
                    new_weight *= 3.0  # Boost clothing items
                elif any(spec in keyword_lower for spec in current_specificity):
                    new_weight *= 5.0  # MASSIVE boost for current specificity
            
            # Update with compatible structure
            user_context["accumulated_keywords"][keyword_lower] = {
                "weight": new_weight,
                "total_frequency": new_count,
                "mention_count": new_count,
                "count": new_count,
                "first_seen": data.get("first_seen", datetime.now().isoformat()),
                "last_seen": datetime.now().isoformat(),
                "source": "user_input" if is_user_input else "ai_response",
                "category": get_keyword_category(keyword_lower)
            }
            
            updates_made += 1
            print(f"   📈 UPDATED '{keyword}' weight: {old_weight:.1f} → {new_weight:.1f}")
            
        else:
            # Add new keyword
            initial_frequency = estimated_frequency * frequency_boost
            base_weight = initial_frequency * 100
            
            # MASSIVE boost for current user input specificity
            if is_user_input:
                category = get_keyword_category(keyword_lower)
                if category == 'clothing_items':
                    base_weight *= 5.0
                elif any(spec in keyword_lower for spec in current_specificity):
                    base_weight *= 15.0  # HUGE boost for specificity terms
            
            # Create compatible structure
            user_context["accumulated_keywords"][keyword_lower] = {
                "weight": base_weight,
                "total_frequency": initial_frequency,
                "mention_count": 1,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source": "user_input" if is_user_input else "ai_response",
                "category": get_keyword_category(keyword_lower)
            }
            new_keywords_added += 1
            print(f"   🆕 ADDED '{keyword}' weight: {base_weight:.1f}")
    
    print(f"   📊 Updates: {updates_made}, New: {new_keywords_added}")

def apply_time_based_decay(user_context):
    """Apply time-based decay - compatible version"""
    from datetime import datetime
    
    if "accumulated_keywords" not in user_context:
        return
    
    current_time = datetime.now()
    decay_applied = 0
    
    for keyword, data in user_context["accumulated_keywords"].items():
        try:
            if isinstance(data, dict) and "last_seen" in data:
                last_seen_str = data["last_seen"]
                last_seen = datetime.fromisoformat(last_seen_str)
                minutes_old = (current_time - last_seen).total_seconds() / 60
                
                # Apply stronger decay based on age
                if minutes_old > 60:  # Older than 1 hour
                    decay_factor = max(0.1, 1 - (minutes_old - 60) / 120)
                    old_weight = get_weight_compatible(data)
                    new_weight = old_weight * decay_factor
                    data["weight"] = new_weight
                    
                    if old_weight != new_weight:
                        decay_applied += 1
            else:
                # Old structure or missing timestamp - apply mild decay
                if isinstance(data, dict):
                    data["weight"] = get_weight_compatible(data) * 0.9
                decay_applied += 1
                    
        except Exception:
            # If we can't parse the date, apply decay anyway
            if isinstance(data, dict):
                data["weight"] = get_weight_compatible(data) * 0.7
            decay_applied += 1
    
    if decay_applied > 0:
        print(f"⏰ Applied time decay to {decay_applied} keywords")

def get_keyword_category(keyword_lower):
    """Get the category of a keyword for smart boosting - compatible version"""
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

def detect_and_handle_category_change_improved(user_input, user_context):
    """
    FIXED: More aggressive and accurate category change detection
    """
    print(f"\n🔍 AGGRESSIVE CATEGORY CHANGE DETECTION")
    print("="*60)
    print(f"📝 User input: '{user_input}'")
    
    user_input_lower = user_input.lower()
    
    # Define clothing categories - MORE COMPREHENSIVE
    fashion_categories = {
        'shirts_tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'sweater', 'hoodie', 'cardigan', 'blazer'],
        'pants': ['celana', 'pants', 'jeans', 'trousers', 'legging'],
        'skirts': ['rok', 'skirt'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'coat'],
        'footwear': ['sepatu', 'shoes', 'sneaker', 'heels', 'boots'],
        'accessories': ['tas', 'bag', 'topi', 'hat', 'scarf']
    }
    
    # Find current categories in user input
    current_categories = set()
    current_terms_found = []
    
    for category_name, terms in fashion_categories.items():
        for term in terms:
            if term in user_input_lower:
                current_categories.add(category_name)
                current_terms_found.append(term)
    
    print(f"   🎯 Current categories: {current_categories}")
    print(f"   📝 Terms found: {current_terms_found}")
    
    if not current_categories:
        print("   ⚠️ No clothing categories detected")
        return False
    
    # Check accumulated keywords for dominant categories
    accumulated_keywords = user_context.get("accumulated_keywords", {})
    accumulated_category_weights = {}
    
    for keyword, data in accumulated_keywords.items():
        keyword_lower = keyword.lower()
        weight = get_weight_compatible(data)
        
        for category_name, terms in fashion_categories.items():
            if any(term in keyword_lower for term in terms):
                if category_name not in accumulated_category_weights:
                    accumulated_category_weights[category_name] = 0
                accumulated_category_weights[category_name] += weight
                break
    
    print(f"   📊 Accumulated category weights:")
    for cat, weight in sorted(accumulated_category_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"      {cat}: {weight:.1f}")
    
    # AGGRESSIVE CHANGE DETECTION - Lower thresholds
    dominant_old_categories = set()
    for category_name, total_weight in accumulated_category_weights.items():
        # REDUCED threshold from 5000 to 1000
        if total_weight > 1000 and category_name not in current_categories:
            dominant_old_categories.add(category_name)
    
    print(f"   👑 Dominant old categories: {dominant_old_categories}")
    
    # Trigger category change if:
    change_detected = False
    change_reason = ""
    
    # 1. Different categories with substantial weight difference
    if dominant_old_categories and current_categories:
        if not dominant_old_categories.intersection(current_categories):
            change_detected = True
            change_reason = "different_categories"
    
    # 2. Multiple clothing items mentioned (suggests new search)
    elif len(current_terms_found) >= 2:
        change_detected = True
        change_reason = "multiple_items"
    
    # 3. Explicit change indicators
    change_indicators = ['now', 'sekarang', 'instead', 'ganti', 'different', 'lain', 
                        'show me', 'tunjukkan', 'carikan', 'find', 'cari']
    if any(indicator in user_input_lower for indicator in change_indicators):
        change_detected = True
        change_reason = "explicit_change"
    
    # 4. FORCE change if accumulated weight is very high (> 10000)
    total_accumulated_weight = sum(accumulated_category_weights.values())
    if total_accumulated_weight > 10000 and current_categories:
        change_detected = True
        change_reason = "weight_reset"
    
    print(f"   🔄 Change detected: {change_detected} ({change_reason})")
    
    if change_detected:
        execute_aggressive_category_change(user_context, dominant_old_categories, current_categories, change_reason)
        return True
    
    return False

def execute_aggressive_category_change(user_context, old_categories, new_categories, change_type):
    """
    MUCH MORE AGGRESSIVE: Remove almost everything conflicting
    """
    print(f"\n🔄 EXECUTING VERY AGGRESSIVE CATEGORY CHANGE")
    print("="*50)
    print(f"   📤 Removing categories: {old_categories}")
    print(f"   📥 Focusing on: {new_categories}")
    print(f"   🎯 Change type: {change_type}")
    
    if "accumulated_keywords" not in user_context:
        return
    
    # Define comprehensive clothing categories
    fashion_categories = {
        'shirts_tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'sweater', 'hoodie', 'cardigan', 'blazer', 'tank', 'top'],
        'pants': ['celana', 'pants', 'jeans', 'trousers', 'legging', 'palazzo'],
        'skirts': ['rok', 'skirt'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'coat'],
        'footwear': ['sepatu', 'shoes', 'sneaker', 'heels', 'boots'],
        'accessories': ['tas', 'bag', 'topi', 'hat', 'scarf']
    }
    
    # Only preserve these critical attributes
    critical_preserve = {
        'gender': ['female', 'male', 'woman', 'man', 'perempuan', 'wanita', 'pria'],
        'colors': ['black', 'white', 'red', 'blue', 'hitam', 'putih', 'merah', 'biru'],
        'sizes': ['small', 'medium', 'large', 'kecil', 'sedang', 'besar', 'xl', 'xxl'], # Only basic colors
        'basic_styles': ['casual', 'formal']  # Only basic styles
    }
    
    preserved_keywords = {}
    removed_count = 0
    
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        current_weight = get_weight_compatible(data)
        should_preserve = False
        preserve_reason = "removed"
        new_weight = current_weight
        
        # 1. ALWAYS preserve gender (but reduce weight)
        if any(term in keyword_lower for term in critical_preserve['gender']):
            should_preserve = True
            preserve_reason = "gender"
            new_weight = min(current_weight * 0.2, 500)  # Very low weight
        
        # 2. Check if belongs to NEW categories (BOOST heavily)
        elif new_categories:
            for new_cat in new_categories:
                if new_cat in fashion_categories:
                    if any(term in keyword_lower for term in fashion_categories[new_cat]):
                        should_preserve = True
                        preserve_reason = f"new_{new_cat}"
                        new_weight = current_weight * 5.0  # MASSIVE boost
                        break
        
        # 3. AGGRESSIVELY remove old category keywords
        if not should_preserve:
            belongs_to_old = False
            for old_cat in old_categories:
                if old_cat in fashion_categories:
                    if any(term in keyword_lower for term in fashion_categories[old_cat]):
                        belongs_to_old = True
                        break
            
            # Also remove ANY clothing terms not in new categories
            if not belongs_to_old:
                for cat_name, terms in fashion_categories.items():
                    if cat_name not in new_categories:
                        if any(term in keyword_lower for term in terms):
                            belongs_to_old = True
                            break
            
            if belongs_to_old:
                removed_count += 1
                print(f"   ❌ REMOVING '{keyword}' (old/conflicting category)")
                continue
        
        # 4. Keep only basic colors and styles (very reduced weight)
        if not should_preserve:
            if any(term in keyword_lower for term in critical_preserve['colors']):
                should_preserve = True
                preserve_reason = "basic_color"
                new_weight = min(current_weight * 0.1, 300)
            elif any(term in keyword_lower for term in critical_preserve['basic_styles']):
                should_preserve = True
                preserve_reason = "basic_style"
                new_weight = min(current_weight * 0.1, 200)
        
        # 5. Remove everything else
        if not should_preserve:
            removed_count += 1
            print(f"   ❌ REMOVING '{keyword}' (not essential)")
            continue
        
        # Create preserved keyword entry
        if isinstance(data, dict):
            updated_data = data.copy()
            updated_data["weight"] = new_weight
        else:
            updated_data = {
                "weight": new_weight,
                "total_frequency": 1,
                "mention_count": 1,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source": "preserved",
                "category": preserve_reason
            }
        
        preserved_keywords[keyword] = updated_data
        
        if preserve_reason.startswith("new_"):
            print(f"   🚀 BOOSTING '{keyword}' ({preserve_reason}): {current_weight:.1f} → {new_weight:.1f}")
        else:
            print(f"   📉 PRESERVING '{keyword}' ({preserve_reason}): {current_weight:.1f} → {new_weight:.1f}")
    
    # Update context with drastically reduced keywords
    user_context["accumulated_keywords"] = preserved_keywords
    
    # Clear product cache
    user_context["product_cache"] = {
        "all_results": pd.DataFrame(),
        "current_page": 0,
        "products_per_page": 5,
        "last_search_params": {},
        "has_more": False
    }
    
    print(f"   📊 VERY AGGRESSIVE SUMMARY:")
    print(f"      ❌ Removed: {removed_count} keywords")
    print(f"      ✅ Preserved: {len(preserved_keywords)} essential keywords")
    print("="*50)

def boost_current_request_keywords_aggressively(user_context, current_user_input):
    """
    AGGRESSIVE: Boost keywords that appear in current user input
    """
    if not current_user_input or "accumulated_keywords" not in user_context:
        return
    
    print(f"\n🚀 AGGRESSIVE CURRENT REQUEST BOOST")
    print("="*50)
    print(f"📝 Boosting keywords from: '{current_user_input}'")
    
    user_input_lower = current_user_input.lower()
    boosted_count = 0
    
    # Extract all words from current input
    input_words = set(user_input_lower.split())
    
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        
        # Check if keyword appears in current input (exact match or contains)
        keyword_in_input = (
            keyword_lower in user_input_lower or
            any(word in keyword_lower for word in input_words if len(word) > 2)
        )
        
        if keyword_in_input:
            current_weight = get_weight_compatible(data)
            
            # Determine boost factor based on keyword type
            current_clothing_terms = ['celana', 'pants', 'rok', 'skirt']  # Only current request items
            style_terms = ['casual', 'formal', 'elegant']

            if any(clothing in keyword_lower for clothing in current_clothing_terms):
                boost_factor = 20.0  # MASSIVE boost for current clothing items
            elif any(style in keyword_lower for style in style_terms):
                boost_factor = 1.5   # REDUCED boost for styles
            else:
                boost_factor = 2.0   # REDUCED boost for other terms

            new_weight = current_weight * boost_factor

            # ADDITIONAL: Equalize Indonesian vs English terms
            if keyword_lower in ['celana', 'rok']:
                new_weight = max(new_weight, 300000)  # Ensure Indonesian terms get high weight
            elif keyword_lower in ['pants', 'skirt']:
                new_weight = max(new_weight, 250000)  # Ensure English terms get high weight too
            
            # Update weight
            if isinstance(data, dict):
                data["weight"] = new_weight
            
            print(f"   🚀 BOOSTED '{keyword}': {current_weight:.1f} → {new_weight:.1f} (×{boost_factor})")
            boosted_count += 1
    
    print(f"🚀 Applied aggressive boost to {boosted_count} current request keywords")
    print("="*50)

def detect_fashion_category_change(user_input, user_context):
    """
    COMPATIBLE VERSION: Works with your existing accumulated_keywords structure
    """
    user_input_lower = user_input.lower()
    
    # Define major fashion categories
    fashion_categories = {
        'tops': [
            'kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 
            'sweater', 'hoodie', 'cardigan', 'blazer', 'jacket', 'jaket', 'top'
        ],
        'bottoms': [
            'celana', 'pants', 'jeans', 'rok', 'skirt', 'bawahan', 'bottom',
            'shorts', 'legging', 'trousers'
        ],
        'dresses': [
            'dress', 'gaun', 'terusan', 'overall'
        ],
        'footwear': [
            'sepatu', 'shoes', 'sneaker', 'heels', 'sandal', 'boots', 'flat'
        ],
        'accessories': [
            'tas', 'bag', 'handbag', 'clutch', 'backpack', 'topi', 'hat'
        ]
    }
    
    # Find current category in user input
    current_categories = set()
    current_terms_found = []
    
    for category, terms in fashion_categories.items():
        found_terms = [term for term in terms if term in user_input_lower]
        if found_terms:
            current_categories.add(category)
            current_terms_found.extend(found_terms)
    
    # Find dominant category in accumulated keywords - COMPATIBLE VERSION
    accumulated_categories = {}
    accumulated_keywords = user_context.get("accumulated_keywords", {})
    
    for keyword, data in accumulated_keywords.items():
        keyword_lower = keyword.lower()
        weight = get_weight_compatible(data)  # Handle both old and new structures
        
        for category, terms in fashion_categories.items():
            if any(term in keyword_lower for term in terms):
                if category not in accumulated_categories:
                    accumulated_categories[category] = 0
                accumulated_categories[category] += weight
                break
    
    # Find the dominant category from accumulated keywords
    dominant_old_category = None
    if accumulated_categories:
        dominant_old_category = max(accumulated_categories.items(), key=lambda x: x[1])[0]
        dominant_weight = accumulated_categories[dominant_old_category]
        
        # Only consider it dominant if it has significant weight
        if dominant_weight < 1000:
            dominant_old_category = None
    
    print(f"\n🔍 COMPATIBLE CATEGORY CHANGE DETECTION:")
    print(f"   📝 Current input categories: {current_categories}")
    print(f"   📝 Terms found: {current_terms_found}")
    print(f"   📚 Accumulated categories: {accumulated_categories}")
    print(f"   👑 Dominant old category: {dominant_old_category}")
    
    # Detect major category change
    if current_categories and dominant_old_category:
        # Check if user is asking for completely different category
        if not current_categories.intersection({dominant_old_category}):
            # Additional checks for explicit change indicators
            change_indicators = [
                'now', 'sekarang', 'instead', 'ganti', 'bukan', 'not',
                'different', 'lain', 'berbeda', 'other', 'selain',
                'want', 'mau', 'need', 'butuh', 'looking for', 'cari'
            ]
            
            has_change_indicator = any(indicator in user_input_lower for indicator in change_indicators)
            
            print(f"   🔄 Different categories detected: {dominant_old_category} → {current_categories}")
            print(f"   🗣️ Has change indicator: {has_change_indicator}")
            
            # Trigger reset if categories are different OR explicit change indicators
            if has_change_indicator or len(current_terms_found) >= 2:
                print(f"   ✅ MAJOR CATEGORY CHANGE CONFIRMED!")
                return True, dominant_old_category, current_categories
    
    return False, None, current_categories

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

def detect_specificity_change(user_input, user_context):
    """
    Detect when user is asking for something more specific or different within the same category
    """
    print(f"\n🔍 SPECIFICITY CHANGE DETECTION")
    print("="*50)
    print(f"📝 User input: '{user_input}'")
    
    user_input_lower = user_input.lower()
    
    # Define specific product modifiers that indicate a refinement request
    specificity_indicators = {
        'style_modifiers': [
            'maxi', 'mini', 'midi', 'cropped', 'oversized', 'slim', 'wide leg', 
            'high waist', 'low waist', 'off shoulder', 'long sleeve', 'short sleeve',
            'button up', 'wrap', 'pleated', 'a-line', 'pencil', 'flare', 'straight',
            'tiered', 'asymmetrical', 'halter', 'strapless', 'backless'
        ],
        'specific_requests': [
            'just', 'only', 'saja', 'hanya', 'specifically', 'khusus',
            'show me only', 'tunjukkan hanya', 'carikan yang', 'find only'
        ],
        'pattern_modifiers': [
            'batik', 'floral', 'striped', 'polka dot', 'plaid', 'solid',
            'geometric', 'abstract', 'paisley', 'leopard', 'zebra'
        ],
        'material_modifiers': [
            'cotton', 'silk', 'denim', 'leather', 'chiffon', 'satin',
            'wool', 'linen', 'polyester', 'velvet'
        ]
    }
    
    # Check if user input contains specificity indicators
    specificity_found = []
    for category, indicators in specificity_indicators.items():
        found_indicators = [ind for ind in indicators if ind in user_input_lower]
        if found_indicators:
            specificity_found.extend([(category, ind) for ind in found_indicators])
    
    print(f"   🎯 Specificity indicators found: {specificity_found}")
    
    # Check accumulated keywords for conflicting specificity
    accumulated_keywords = user_context.get("accumulated_keywords", {})
    conflicting_specificity = []
    
    for keyword, data in accumulated_keywords.items():
        keyword_lower = keyword.lower()
        weight = get_weight_compatible(data)
        
        # Skip low weight keywords
        if weight < 1000:
            continue
            
        # Check for conflicting modifiers
        for category, current_indicator in specificity_found:
            if category == 'style_modifiers':
                # Find conflicting style modifiers
                for other_modifier in specificity_indicators['style_modifiers']:
                    if (other_modifier != current_indicator and 
                        other_modifier in keyword_lower and
                        weight > 5000):  # High weight conflicting modifier
                        conflicting_specificity.append((keyword, other_modifier, weight))
    
    print(f"   ⚠️ Conflicting specificity: {conflicting_specificity}")
    
    # Determine if we should reset for specificity
    should_reset = False
    reset_reason = ""
    
    # Reset if we have specific requests with conflicting modifiers
    if specificity_found and conflicting_specificity:
        should_reset = True
        reset_reason = "conflicting_specificity"
    
    # Reset if user explicitly asks for "only" or "just" something
    elif any('specific_requests' == cat for cat, _ in specificity_found):
        should_reset = True
        reset_reason = "explicit_only_request"
    
    # Reset if user mentions multiple specific modifiers (new focused search)
    elif len([cat for cat, _ in specificity_found if cat == 'style_modifiers']) >= 2:
        should_reset = True
        reset_reason = "multiple_modifiers"
    
    print(f"   🔄 Should reset: {should_reset} ({reset_reason})")
    print("="*50)
    
    return should_reset, reset_reason, specificity_found, conflicting_specificity

def execute_specificity_reset(user_context, reset_reason, specificity_found, conflicting_specificity):
    """
    Reset keywords while preserving essential info but removing conflicting specificity
    """
    print(f"\n🔄 EXECUTING SPECIFICITY RESET")
    print("="*50)
    print(f"   🎯 Reason: {reset_reason}")
    print(f"   📝 New specificity: {specificity_found}")
    print(f"   ⚠️ Conflicts: {conflicting_specificity}")
    
    if "accumulated_keywords" not in user_context:
        return
    
    # Always preserve these critical attributes
    always_preserve = {
        'gender': ['female', 'male', 'woman', 'man', 'perempuan', 'wanita', 'pria'],
        'basic_colors': ['black', 'white', 'hitam', 'putih'],  # Only most basic colors
        'budget_related': ['budget', 'price', 'harga']  # Keep budget context
    }
    
    # Get conflicting keywords to remove
    conflicting_keywords = set()
    for keyword, modifier, weight in conflicting_specificity:
        conflicting_keywords.add(keyword.lower())
    
    # Also remove broad category terms that might interfere
    broad_terms_to_reduce = [
        'atasan', 'kemeja', 'wanita', 'tops', 'shirt', 'blouse'  # These interfere with skirt searches
    ]
    
    preserved_keywords = {}
    removed_count = 0
    reduced_count = 0
    
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        current_weight = get_weight_compatible(data)
        should_preserve = False
        preserve_reason = "removed"
        new_weight = current_weight
        
        # 1. ALWAYS preserve gender (but reduce weight)
        if any(term in keyword_lower for term in always_preserve['gender']):
            should_preserve = True
            preserve_reason = "gender"
            new_weight = min(current_weight * 0.1, 100)  # Very low weight
        
        # 2. Remove conflicting specificity keywords
        elif keyword_lower in conflicting_keywords:
            removed_count += 1
            print(f"   ❌ REMOVING conflicting: '{keyword}' (weight: {current_weight:.1f})")
            continue
        
        # 3. Heavily reduce broad category terms that interfere
        elif any(broad in keyword_lower for broad in broad_terms_to_reduce):
            if current_weight > 1000:  # Only if they have significant weight
                should_preserve = True
                preserve_reason = "reduced_broad"
                new_weight = min(current_weight * 0.05, 50)  # Massive reduction
                reduced_count += 1
                print(f"   📉 HEAVILY REDUCING broad term: '{keyword}' {current_weight:.1f} → {new_weight:.1f}")
            else:
                removed_count += 1
                print(f"   ❌ REMOVING low-weight broad: '{keyword}'")
                continue
        
        # 4. Keep basic colors but reduce weight
        elif any(color in keyword_lower for color in always_preserve['basic_colors']):
            should_preserve = True
            preserve_reason = "basic_color"
            new_weight = min(current_weight * 0.2, 200)
        
        # 5. Keep only very high weight keywords with significant reduction
        elif current_weight > 10000:
            should_preserve = True
            preserve_reason = "high_weight_reduced"
            new_weight = current_weight * 0.1  # 90% reduction
            reduced_count += 1
            print(f"   📉 REDUCING high weight: '{keyword}' {current_weight:.1f} → {new_weight:.1f}")
        
        # 6. Remove everything else
        else:
            removed_count += 1
            print(f"   ❌ REMOVING: '{keyword}' (weight: {current_weight:.1f})")
            continue
        
        # Create preserved keyword entry
        if isinstance(data, dict):
            updated_data = data.copy()
            updated_data["weight"] = new_weight
            updated_data["last_seen"] = datetime.now().isoformat()
        else:
            updated_data = {
                "weight": new_weight,
                "total_frequency": 1,
                "mention_count": 1,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source": "preserved_specificity",
                "category": preserve_reason
            }
        
        preserved_keywords[keyword] = updated_data
    
    # Update context with cleaned keywords
    user_context["accumulated_keywords"] = preserved_keywords
    
    # Clear product cache to force new search
    user_context["product_cache"] = {
        "all_results": pd.DataFrame(),
        "current_page": 0,
        "products_per_page": 5,
        "last_search_params": {},
        "has_more": False
    }
    
    print(f"   📊 SPECIFICITY RESET SUMMARY:")
    print(f"      ❌ Removed: {removed_count} keywords")
    print(f"      📉 Reduced: {reduced_count} keywords")
    print(f"      ✅ Preserved: {len(preserved_keywords)} essential keywords")
    print("="*50)

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

def smart_keyword_reset_for_category_change(user_context, old_category, new_categories):
    """
    COMPATIBLE VERSION: Smart reset that works with your existing data structure
    """
    print(f"\n🔄 COMPATIBLE SMART KEYWORD RESET")
    print(f"   📤 Removing: {old_category}")
    print(f"   📥 Keeping for: {new_categories}")
    
    if "accumulated_keywords" not in user_context:
        return
    
    # Categories to preserve (non-conflicting)
    preserve_categories = {
        'style_attributes': ['casual', 'formal', 'elegant', 'vintage', 'modern', 'minimalist'],
        'colors': ['black', 'white', 'red', 'blue', 'green', 'hitam', 'putih', 'merah', 'biru'],
        'user_identity': ['female', 'male', 'woman', 'man', 'perempuan', 'pria', 'wanita'],
        'size_preferences': ['small', 'medium', 'large', 'kecil', 'sedang', 'besar'],
        'fit_preferences': ['slim', 'loose', 'oversized', 'ketat', 'longgar']
    }
    
    # Fashion categories that conflict with each other
    conflicting_categories = {
        'tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'sweater', 'hoodie'],
        'bottoms': ['celana', 'pants', 'jeans', 'rok', 'skirt', 'bawahan'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'footwear': ['sepatu', 'shoes', 'sneaker', 'heels', 'sandal'],
        'accessories': ['tas', 'bag', 'handbag', 'topi', 'hat']
    }
    
    preserved_keywords = {}
    removed_keywords = []
    
    for keyword, data in user_context["accumulated_keywords"].items():
        keyword_lower = keyword.lower()
        should_preserve = False
        preserve_reason = None
        
        # Get current weight in compatible way
        current_weight = get_weight_compatible(data)
        
        # Check if it's a non-conflicting attribute we should preserve
        for preserve_cat, terms in preserve_categories.items():
            if any(term in keyword_lower for term in terms):
                should_preserve = True
                preserve_reason = preserve_cat
                # Reduce weight but keep it
                new_weight = current_weight * 0.3
                break
        
        # Check if it belongs to the old conflicting category (remove it)
        if not should_preserve:
            belongs_to_old_category = False
            if old_category and old_category in conflicting_categories:
                old_terms = conflicting_categories[old_category]
                if any(term in keyword_lower for term in old_terms):
                    belongs_to_old_category = True
            
            if belongs_to_old_category:
                removed_keywords.append(keyword)
                print(f"   ❌ Removing '{keyword}' (old {old_category})")
            else:
                # Keep other keywords but reduce their weight
                should_preserve = True
                preserve_reason = "non_conflicting"
                new_weight = current_weight * 0.5
        
        if should_preserve:
            # Create compatible data structure
            if isinstance(data, dict):
                # Update existing dict structure
                updated_data = data.copy()
                updated_data["weight"] = new_weight
                # Ensure all required fields exist
                if "count" not in updated_data:
                    updated_data["count"] = updated_data.get("mention_count", updated_data.get("total_frequency", 1))
                if "total_frequency" not in updated_data:
                    updated_data["total_frequency"] = updated_data.get("count", 1)
                if "mention_count" not in updated_data:
                    updated_data["mention_count"] = updated_data.get("count", 1)
                preserved_keywords[keyword] = updated_data
            else:
                # Create new dict structure from old simple weight
                preserved_keywords[keyword] = {
                    "weight": new_weight,
                    "total_frequency": 1,
                    "mention_count": 1,
                    "count": 1,
                    "first_seen": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "source": "preserved",
                    "category": get_keyword_category(keyword_lower)
                }
            
            print(f"   ✅ Preserving '{keyword}' ({preserve_reason}) weight: {new_weight:.1f}")
    
    # Update the context
    user_context["accumulated_keywords"] = preserved_keywords
    
    # Reset product cache
    user_context["product_cache"] = {
        "all_results": pd.DataFrame(),
        "current_page": 0,
        "products_per_page": 5,
        "last_search_params": {},
        "has_more": False
    }
    
    print(f"   📊 Removed {len(removed_keywords)} conflicting keywords")
    print(f"   📊 Preserved {len(preserved_keywords)} compatible keywords")

def get_weight_compatible(data):
    """Extract weight from data in a compatible way"""
    if isinstance(data, dict):
        return data.get("weight", 0)
    else:
        # Old structure - assume it's just the weight
        return float(data) if data else 0

def get_source_compatible(data):
    """Extract source from data in a compatible way"""
    if isinstance(data, dict):
        return data.get("source", "unknown")
    else:
        return "legacy"

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
    IMPROVED: More aggressive and accurate gender detection that handles changes
    """
    print(f"\n👤 IMPROVED GENDER DETECTION")
    print("="*50)
    print(f"📝 Input: '{user_input}'")
    print(f"🔄 Force update: {force_update}")
    
    current_gender = user_context.get("user_gender", {})
    has_existing_gender = current_gender.get("category") is not None
    
    print(f"📊 Current gender: {current_gender.get('category', 'None')} (confidence: {current_gender.get('confidence', 0):.1f})")
    
    # Enhanced gender detection patterns with context
    gender_patterns = {
        'male': [
            # Direct statements
            r'\b(i am|i\'m|saya|aku)\s+(a\s+)?(male|man|pria|laki-laki|cowok|cowo)\b',
            r'\b(male|man|pria|laki-laki|cowok|cowo)\s+(here|di sini|nih)\b',
            r'\b(as a|sebagai)\s+(male|man|pria|laki-laki)\b',
            
            # Context-based
            r'\b(for|untuk)\s+(me|saya|aku),?\s+(a\s+)?(male|man|pria|laki-laki)\b',
            r'\b(i am looking|saya mencari)\s+(for|untuk)?\s+(male|man|pria|laki-laki)\s+(clothing|fashion|style)\b',
            r'\b(male|man|pria|laki-laki)\s+(fashion|style|clothing|clothes|outfit)\b',
            
            # Change statements
            r'\b(actually|sebenarnya|correction|koreksi)\s+.*(male|man|pria|laki-laki)\b',
            r'\b(change|ganti|ubah)\s+.*(male|man|pria|laki-laki)\b',
            r'\b(not|bukan)\s+(female|woman|perempuan|wanita).*(male|man|pria|laki-laki)\b',
        ],
        'female': [
            # Direct statements
            r'\b(i am|i\'m|saya|aku)\s+(a\s+)?(female|woman|perempuan|wanita|cewek|cewe)\b',
            r'\b(female|woman|perempuan|wanita|cewek|cewe)\s+(here|di sini|nih)\b',
            r'\b(as a|sebagai)\s+(female|woman|perempuan|wanita)\b',
            
            # Context-based
            r'\b(for|untuk)\s+(me|saya|aku),?\s+(a\s+)?(female|woman|perempuan|wanita)\b',
            r'\b(i am looking|saya mencari)\s+(for|untuk)?\s+(female|woman|perempuan|wanita)\s+(clothing|fashion|style)\b',
            r'\b(female|woman|perempuan|wanita)\s+(fashion|style|clothing|clothes|outfit)\b',
            
            # Change statements
            r'\b(actually|sebenarnya|correction|koreksi)\s+.*(female|woman|perempuan|wanita)\b',
            r'\b(change|ganti|ubah)\s+.*(female|woman|perempuan|wanita)\b',
            r'\b(not|bukan)\s+(male|man|pria|laki-laki).*(female|woman|perempuan|wanita)\b',
        ]
    }
    
    user_input_lower = user_input.lower()
    detected_gender = None
    detected_term = None
    confidence = 0
    detection_type = "none"
    
    # Check for explicit change requests first (highest priority)
    change_patterns = [
        r'\b(actually|sebenarnya|wait|tunggu|correction|koreksi|ralat)\b',
        r'\b(change|ganti|ubah)\s+(to|ke|menjadi)\b',
        r'\b(i\'m|saya)\s+(not|bukan)\b',
        r'\b(wrong|salah|mistake|kesalahan)\b'
    ]
    
    has_change_indicator = any(re.search(pattern, user_input_lower) for pattern in change_patterns)
    
    if has_change_indicator:
        force_update = True
        detection_type = "explicit_change"
        print(f"🔄 Change indicator detected - forcing update")
    
    # Check for gender patterns with context awareness
    for gender, patterns in gender_patterns.items():
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, user_input_lower)
            if match:
                detected_gender = gender
                detected_term = match.group()
                
                # Assign confidence based on pattern type
                if i < 3:  # Direct statements (highest confidence)
                    confidence = 15.0
                elif i < 6:  # Context-based (high confidence)
                    confidence = 12.0
                else:  # Change statements (very high confidence)
                    confidence = 20.0
                    force_update = True
                
                detection_type = f"pattern_{i}"
                print(f"   ✅ Detected: {gender} (pattern: {pattern})")
                print(f"   📏 Confidence: {confidence} ({detection_type})")
                break
        if detected_gender:
            break
    
    # Additional check for gender keywords with high frequency
    if not detected_gender:
        gender_keywords = {
            'male': ['pria', 'laki-laki', 'male', 'man', 'cowok', 'cowo'],
            'female': ['perempuan', 'wanita', 'female', 'woman', 'cewek', 'cewe']
        }
        
        for gender, keywords in gender_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in user_input_lower)
            if keyword_count > 0:
                detected_gender = gender
                detected_term = next(keyword for keyword in keywords if keyword in user_input_lower)
                confidence = 8.0 + (keyword_count * 2.0)  # Base confidence + frequency boost
                detection_type = "keyword_frequency"
                print(f"   🔍 Keyword detection: {gender} (count: {keyword_count})")
                break
    
    # Decision logic for updating gender
    should_update = False
    update_reason = ""
    
    if not has_existing_gender and detected_gender:
        should_update = True
        update_reason = "first_time_detection"
    elif force_update and detected_gender:
        should_update = True
        update_reason = "forced_update"
    elif detected_gender and confidence > current_gender.get("confidence", 0):
        should_update = True
        update_reason = "higher_confidence"
    elif has_change_indicator and detected_gender:
        should_update = True
        update_reason = "explicit_change_request"
    
    print(f"📊 Detection summary:")
    print(f"   🎯 Detected: {detected_gender or 'None'}")
    print(f"   📏 Confidence: {confidence}")
    print(f"   🔄 Should update: {should_update} ({update_reason})")
    
    # Update gender if conditions are met
    if should_update and detected_gender:
        old_gender = current_gender.get("category", "None")
        
        user_context["user_gender"] = {
            "category": detected_gender,
            "term": detected_term,
            "confidence": confidence,
            "last_updated": datetime.now().isoformat(),
            "detection_type": detection_type,
            "update_reason": update_reason
        }
        
        print(f"   ✅ GENDER UPDATED: {old_gender} → {detected_gender}")
        print(f"   📝 Term: '{detected_term}' (confidence: {confidence})")
        
        # Clear product cache when gender changes to force new search
        if old_gender != "None" and old_gender != detected_gender:
            user_context["product_cache"] = {
                "all_results": pd.DataFrame(),
                "current_page": 0,
                "products_per_page": 5,
                "has_more": False
            }
            print(f"   🗑️ Cleared product cache due to gender change")
        
        print("="*50)
        return detected_gender
    
    # Return existing gender if no update needed
    if has_existing_gender:
        existing_gender = current_gender["category"]
        print(f"   📋 Using existing gender: {existing_gender}")
        print("="*50)
        return existing_gender
    
    print(f"   ❌ No gender detected or updated")
    print("="*50)
    return None

def get_user_gender(user_context):
    """
    IMPROVED: Better gender information retrieval with validation
    """
    gender_info = user_context.get("user_gender", {})
    
    # Validate gender data
    category = gender_info.get("category")
    if category not in ["male", "female"]:
        return {
            "category": None,
            "term": None,
            "confidence": 0,
            "last_updated": None,
            "is_valid": False
        }
    
    return {
        "category": category,
        "term": gender_info.get("term"),
        "confidence": gender_info.get("confidence", 0),
        "last_updated": gender_info.get("last_updated"),
        "detection_type": gender_info.get("detection_type", "unknown"),
        "update_reason": gender_info.get("update_reason", "unknown"),
        "is_valid": True
    }

def handle_gender_in_keyword_update(user_input, user_context):
    """
    Handle gender detection during keyword updates
    """
    # Check if this is a gender change request
    is_gender_change = detect_gender_change_request(user_input)
    
    # Detect and update gender
    detected_gender = detect_and_update_gender(
        user_input, 
        user_context, 
        force_update=is_gender_change
    )
    
    # If gender was updated, add it to accumulated keywords with appropriate weight
    if detected_gender and "accumulated_keywords" in user_context:
        gender_terms = {
            'male': ['male', 'man', 'pria', 'laki-laki'],
            'female': ['female', 'woman', 'perempuan', 'wanita']
        }
        
        # Add gender terms to keywords with moderate weight (not too high to dominate)
        for term in gender_terms[detected_gender]:
            user_context["accumulated_keywords"][term] = {
                "weight": 500.0,  # Moderate weight - enough to filter but not dominate
                "total_frequency": 1,
                "mention_count": 1,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source": "gender_detection",
                "category": "gender_terms"
            }
    
    return detected_gender

def detect_gender_change_request(user_input):
    """
    IMPROVED: Better detection of explicit gender change requests
    """
    change_patterns = [
        # Explicit corrections
        r'\b(actually|sebenarnya)\s+(i am|saya|i\'m)\s+(male|female|pria|wanita|perempuan|laki-laki)',
        r'\b(correction|koreksi|ralat)\s*[:-]?\s*(i am|saya|i\'m)\s+(male|female|pria|wanita|perempuan|laki-laki)',
        
        # Gender switches
        r'\b(change|ganti|ubah)\s+(to|ke|menjadi)\s+(male|female|pria|wanita|perempuan|laki-laki)',
        r'\b(switch|alih)\s+(to|ke)\s+(male|female|pria|wanita|perempuan|laki-laki)',
        
        # Negation corrections
        r'\b(i\'m|saya)\s+(not|bukan)\s+(male|female|pria|wanita|perempuan|laki-laki)',
        r'\b(not|bukan)\s+(male|female|pria|wanita|perempuan|laki-laki).*(i am|saya)',
        
        # Mistake corrections
        r'\b(wrong|salah|mistake|kesalahan)',
        r'\b(that\'s wrong|itu salah)',
        
        # Context switches
        r'\b(for|untuk)\s+(male|female|pria|wanita|perempuan|laki-laki)\s+(now|sekarang)',
        r'\b(show|tunjukkan)\s+(male|female|pria|wanita|perempuan|laki-laki)\s+(fashion|clothes|style)',
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