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
from deep_translator import GoogleTranslator
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
    "always", "usually", "great", "very", "really", "sure",
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
    "sebuah", "pilih", "menarik"
])

# Constant for scoring
USER_KEYWORD_BOOST = 3.0
MULTI_WORD_USER_BOOST = 2.0
MULTI_WORD_AI_BOOST = 0.5
POSITION_WEIGHT = 0.3
FREQUENCY_WEIGHT = 0.5
POS_WEIGHT = 0.2

def extract_ranked_keywords(translated_response: str, translated_input: str = None):
    if not translated_response:
        return []

    keyword_scores = {}
    doc_ai = nlp(translated_response)

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

    # Initialize keyword scores with AI keywords
    for i, token in enumerate(doc_ai):
        token_text = token.text.lower()
        if len(token_text) < 3 or token_text in stop_words:
            continue

        # Frequency score
        freq_score = ai_keyword_freq.get(token_text, 0)
        # Position score
        position_score = 1.0 - (i / len(doc_ai))
        # POS score
        pos_score = {'PROPN': 3, 'NOUN': 2, 'ADJ': 1}.get(token.pos_, 0)
        # Final score
        ai_score = (freq_score * FREQUENCY_WEIGHT) + (position_score * POSITION_WEIGHT) + (pos_score * POS_WEIGHT)
        # Update keyword scores
        keyword_scores[token_text] = keyword_scores.get(token_text, 0) + ai_score

    # Boost multi-word AI phrases
    for chunk in ai_noun_chunks:
        if ' ' in chunk and len(chunk.split()) <= 3:  # Ensure capturing multi-word keywords like "wrap tops"
            freq_score = ai_keyword_freq.get(chunk, 0)
            keyword_scores[chunk] = keyword_scores.get(chunk, 0) + freq_score * FREQUENCY_WEIGHT + MULTI_WORD_AI_BOOST

    # Process user input if provided
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

        # Update keyword scores with user keywords and apply boost
        for word, count in user_keyword_freq.items():
            keyword_scores[word] = keyword_scores.get(word, 0) + (count * USER_KEYWORD_BOOST)
        
        # Boost multi-word user phrases
        for chunk in user_noun_chunks:
            if ' ' in chunk and len(chunk.split()) <= 3:
                keyword_scores[chunk] = keyword_scores.get(chunk, 0) + MULTI_WORD_USER_BOOST

    # Sort keywords by score
    ranked_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_keywords

async def fetch_products_from_db(db: AsyncSession, ranked_keywords: list[str], max_results=5):
    # Filter out irrelevant keywords
    relevant_keywords = [(kw, score) for kw, score in ranked_keywords if len(kw) > 2 and kw.lower() not in stop_words]

    if not relevant_keywords:
        query = (
            select(Product, ProductVariant, ProductPhoto)
            .join(ProductVariant, Product.product_id == ProductVariant.product_id)
            .join(ProductPhoto, Product.product_id == ProductPhoto.product_id)
            .where(ProductVariant.stock > 0)
            .order_by(func.rand())
            .limit(max_results)
        )
    else:
        match_conditions = []
        for keyword, score in relevant_keywords[:10]:  # Consider only top 10 keywords for performance
            exact_match = Product.product_name.ilike(f"{keyword}")
            starts_with = Product.product_name.ilike(f"{keyword}%")
            contains = Product.product_name.ilike(f"%{keyword}%")
            desc_match = Product.product_detail.ilike(f"%{keyword}%")

            match_conditions.extend([
                (exact_match, score * 2.0),         # Exact matches get highest weight
                (starts_with, score * 1.5),         # Starting with keyword gets good weight
                (contains, score * 1.0),            # Contains keyword gets base weight
                (desc_match, score * 0.5)           # Description matches get lowest weight
            ])

        ranking_case = case(*[(condition, weight) for condition, weight in match_conditions], else_=0)

        query = (
            select(Product, ProductVariant, ProductPhoto, ranking_case.label("relevance"))
            .join(ProductVariant, Product.product_id == ProductVariant.product_id)
            .join(ProductPhoto, Product.product_id == ProductPhoto.product_id)
            .where(
                or_(*[condition for condition, _ in match_conditions]),  # Ensure at least one keyword match
                ProductVariant.stock > 0  # Ensure product is available
            )
            .order_by(desc("relevance"), func.rand())  # Order by relevance, random for ties
            .limit(max_results)
        )

    result = await db.execute(query)
    products = result.fetchall()

    product_list = []
    seen_products = set()
    
    for row in products:
        if len(row) == 4:  # Query with relevance score
            product, variant, photo, relevance = row
        else:  # Fallback query without relevance
            product, variant, photo = row
            relevance = 0
            
        if product.product_id not in seen_products:
            product_data = {
                "product_id": product.product_id,
                "product": product.product_name,
                "description": product.product_detail,
                "price": variant.product_price,
                "size": variant.size,
                "color": variant.color,
                "stock": variant.stock,
                "link": f"http://localhost/e-commerce-main/product-{product.product_seourl}-{product.product_id}",
                "photo": photo.productphoto_path, 
                "relevance": relevance
            }
            product_list.append(product_data)
            seen_products.add(product.product_id)

    return pd.DataFrame(product_list)

def search_products_in_dataframe(keywords, product_df):
    pattern = '|'.join(map(re.escape, keywords))
    matching_products = product_df[product_df['product'].str.contains(pattern, case=False, na=False)]
    return matching_products

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
    html_content = markdown(text)
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

# Function to detect the language of the text
def detect_language(text):
    try:
        if not text or not text.strip():
            raise ValueError("Input text is empty or invalid.")
        return detect(text)  # Detect the language using langdetect
    except Exception as e:
        print(f"Language detection error: {e}")
        return "unknown"

# Function to translate text using Deep Translator
def translate_text(text, target_language):
    try:
        # Detect the source language of the input text
        source_language = detect_language(text)
        print(f"Detected source language: {source_language}")

        # If the source and target languages are the same, no translation needed
        if source_language == target_language:
            return text

        # Use GoogleTranslator from Deep Translator to perform the translation
        translated_text = GoogleTranslator(source=source_language, target=target_language).translate(text)
        return translated_text

    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # Return the original text as a fallback

def extract_intent(user_input, target_language="en"):
    """Extracts keywords and entities, translating input if necessary."""
    doc = nlp(user_input)

    try:
        detected_language = detect_language(user_input)
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

        # Initial system prompt for fashion consultation
        message_objects = [{
            "role": "system",
            "content": (
                "You are a fashion consultant. Your task is to provide detailed fashion recommendations "
                "for users based on their appearance and style preferences. Respond in a friendly, natural tone "
                "and avoid using structured JSON or code format. Instead, communicate recommendations in conversational sentences.\n\n"
                "Always ask for their gender, weight and height, skin tone, their ethnical background and use this information as a base for your recommendations."
                "When giving recommendations, mention specific clothing items and how they would suit the user's attributes, "
                "such as height, weight, and skin tone. Use descriptive phrases and consider mentioning outfit ideas for casual occasions, "
                "unless the user specifies a different occasion.\n"
                "give at least 3 items recommendation.\n\n"
                "make each clothing item as a bold text and for the explanation make it as a paragraph and in different line from the title, for new or different item make it in different line."
                "Example response:\n"
                "For casual wear, here are some styles that would look great on you:  \n"
                "- Cropped Tees or Tank Tops  \n– Perfect for a laid-back look, and they can help accentuate your waist and balance out your height.  \n"
                "- Oversized T-shirts  \n– A relaxed fit in earthy or jewel tones would be flattering. You could tuck them into high-waisted jeans or shorts to add some shape.  \n"
                "- Off-Shoulder Tops  \n– These are cute and can show off some skin while keeping things casual. They also come in various styles, like long-sleeved or with ruffles, which add a nice detail.  \n"
                "- Button-Up Shirts (Linen or Cotton)  \n– A lightweight button-up, especially in tan, beige, or pastel shades, could be a staple. You can wear it loose or tied at the waist.  \n"
                "- Graphic Tees  \n– These are always in style and add personality to any casual outfit. You could go for retro or minimalist designs in colors that complement your skin tone.  \n\n"
                "Ask the user if these styles align with their preferences or if they have any specific style they would like to focus on."
                "Do not mention any specific brand of clothing"
                "Always ask for user opinion after each suggestion, always use a yes or no question to ask the user opinion")
            }]
        
        # Store the most recent AI response for use in confirmation handling
        last_ai_response = ""
        
        # Store user context for better recommendations
        user_context = {
            "current_image_url": None,
            "current_text_input": None,
            "pending_image_analysis": False,  # Add a flag to track pending image analysis
            "has_shared_image": False,
            "has_shared_preferences": False,
            "last_query_type": None,
            "preferences": {},
            "known_attributes": {}
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
                    user_language = detect_language(user_input)
                except:
                    user_language = "en"
                
                # Process Text and Image Input with Keyword Extraction and Recommendations
                if user_input.startswith(("http://", "https://")) and any(ext in user_input.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
                    try:
                        # Update user context for combined input
                        user_context["has_shared_image"] = True
                        user_context["last_query_type"] = "image"

                        # Call image analysis
                        clothing_features = await analyze_uploaded_image(user_input)

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
                            return

                        # Combine text and image input for styling recommendations
                        text_analysis = user_context.get("text_input", "")
                        combined_features = f"{clothing_features} {text_analysis}"

                        # Extract ranked keywords from the combined features
                        ranked_keywords = extract_ranked_keywords(combined_features, text_analysis)
                        logging.info(f"Extracted Ranked Keywords: {ranked_keywords}")

                        # Generate automatic styling recommendations
                        message_objects.append({
                            "role": "user",
                            "content": combined_features,
                        })

                        # Get AI response
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=message_objects,
                            temperature=0.5
                        )
                        
                        ai_response = response.choices[0].message.content.strip()
                        last_ai_response = ai_response
                        
                        message_objects.append({
                            "role": "assistant",
                            "content": ai_response
                        })

                        # Extract ranked keywords from the combined features
                        ranked_keywords = extract_ranked_keywords(combined_features, translated_input)
                        logging.info(f"Extracted Ranked Keywords: {ranked_keywords}")
                        print(f"Extracted Ranked Keywords: {ranked_keywords}")

                        # Translate if needed
                        if user_language != "en":
                            translated_ai_response = translate_text(last_ai_response, user_language)
                        else:
                            translated_ai_response = last_ai_response

                        # Save and send styling recommendations
                        new_ai_response = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=ai_response
                        )
                        db.add(new_ai_message)
                        await db.commit()

                        ai_response_html = render_markdown(translated_ai_response)
                        await websocket.send_text(f"{session_id}|{ai_response_html}")

                        # Prompt for product recommendations with extracted keywords
                        ai_response = f"Would you like product recommendations based on this analysis and styling suggestions? (Yes/No)"
                        await websocket.send_text(f"{session_id}|{ai_response}")

                    except Exception as input_error:
                        logging.error(f"Error during input handling: {str(input_error)}")
                        error_msg = "Sorry, there was an issue processing your input. Could you try again?"
                        await websocket.send_text(f"{session_id}|{error_msg}")
                
                # Handle yes/no responses to previous recommendations
                elif user_input.strip().lower() in ["yes", "ya", "iya", "no", "tidak", "nope", "nah"]:
                    is_positive = user_input.strip().lower() in ["yes", "ya", "iya"]
                    
                    if is_positive:
                        try:
                            # Extract keywords from previous conversation
                            ranked_keywords = extract_ranked_keywords(last_ai_response, translated_input)
                            logging.info(f"Ranked keywords: {ranked_keywords}")
                            
                            # Adjust recommendations based on positive feedback
                            positive_response = "Great! I'm glad you like these recommendations. Here are some additional items that might complement your style:"
                            if user_language != "en":
                                positive_response = translate_text(positive_response, user_language)
                                
                            # Fetch alternative products based on same keywords
                            alt_recommended_products = await fetch_products_from_db(db, ranked_keywords, max_results=5)
                            
                            if not alt_recommended_products.empty:
                                complete_response = positive_response + "\n\n"
                                for _, row in alt_recommended_products.iterrows():
                                    complete_response += (
                                        f"\n![Product Image]({row['photo']})\n"
                                        f"\n**{row['product']}**\n for IDR{row['price']}\n"
                                        f"\n{row['description']}\n"
                                        f"\nAvailable in size: {row['size']}, Color: {row['color']}\n"
                                        f"\n<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                    )
                                    print("Photo Path:", row['photo'])  # Check the exact value
                                complete_response += "\n\nIs there anything specific you'd like to know about these items?"
                            else:
                                complete_response = positive_response + "\n\nHowever, I couldn't find additional items at the moment. Would you like me to help you with something else?"
                                
                            if user_language != "en":
                                translated_complete_response = translate_text(complete_response, user_language)
                                
                            # Save and send response
                            new_ai_message = ChatHistoryDB(
                                session_id=session_id,
                                message_type="assistant",
                                content=complete_response
                            )
                            db.add(new_ai_message)
                            await db.commit()
                            
                            complete_response_html = render_markdown(translated_complete_response)
                            await websocket.send_text(f"{session_id}|{complete_response_html}")
                            
                        except Exception as e:
                            logging.error(f"Error during recommendation processing: {str(e)}\n{traceback.format_exc()}")
                            error_msg = "I'm sorry, I couldn't fetch additional recommendations. Is there something else you'd like to know about fashion?"
                            if user_language != "en":
                                error_msg = translate_text(error_msg, user_language)
                            await websocket.send_text(f"{session_id}|{error_msg}")
                    else:
                        # Handle negative feedback - ask for more specific preferences
                        negative_response = "I understand these recommendations don't match what you're looking for. Could you share more specific preferences? For example, what colors, styles, or occasions are you interested in?"
                        if user_language != "en":
                            translated_negative_response = translate_text(negative_response, user_language)
                            
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=negative_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        await websocket.send_text(f"{session_id}|{translated_negative_response}")
                    
                    # Continue to next iteration
                    continue
                # Handle normal text input
                else:
                    # Check for small talk
                    if await is_small_talk(user_input):
                        ai_response = "Hello! How can I assist you with fashion recommendations today? Feel free to share information about your style preferences or upload an image for personalized suggestions."
                        if user_language != "en":
                            ai_response = translate_text(ai_response, user_language)
                            
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=ai_response
                        )
                        db.add(new_ai_message)
                        await db.commit()
                        
                        await websocket.send_text(f"{session_id}|{ai_response}")
                        continue
                    
                    # Extract any preferences from message
                    user_context["last_query_type"] = "text"
                    
                    # Translate if needed
                    if user_language != "en":
                        translated_input = translate_text(user_input, "en")
                    else:
                        translated_input = user_input
                    
                    # Add to message history
                    message_objects.append({
                        "role": "user",
                        "content": translated_input,
                    })
                    
                    # Get AI response
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=message_objects,
                        temperature=0.5
                    )
                    
                    ai_response = response.choices[0].message.content.strip()
                    last_ai_response = ai_response
                    
                    message_objects.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                    
                    # Check if we should add product recommendations based on text
                    should_add_products = any(keyword in translated_input.lower() for keyword in [
                        "recommend", "suggestion", "product", "clothes", "clothing", "fashion", 
                        "wear", "outfit", "dress", "shirt", "pant", "skirt", "style", "rekomen", 
                        "saran", "produk", "pakaian", "busana", "gaya"
                    ])
                    
                    if should_add_products:
                        # Extract keywords for product search
                        ranked_keywords = extract_ranked_keywords(last_ai_response, translated_input)
                        recommended_products = await fetch_products_from_db(db, ranked_keywords, max_results=3)
                        
                        if not recommended_products.empty:
                            ai_response += "\n\n**Here are some products recommendations:**\n\n"
                            for _, row in recommended_products.iterrows():
                                ai_response += (
                                    f"\n![Product Image]({row['photo']})\n"
                                    f"\n**{row['product']}**\n for IDR{row['price']}\n"
                                    f"\n{row['description']}\n"
                                    f"\nAvailable in size: {row['size']}, Color: {row['color']}\n"
                                    f"\n<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                )
                                print("Photo Path:", row['photo'])  # Check the exact value
                    
                    # Translate back if needed
                    if user_language != "en":
                        translated_response = translate_text(ai_response, user_language)
                    else:
                        translated_response = ai_response
                    
                    # Save response
                    new_ai_message = ChatHistoryDB(
                        session_id=session_id,
                        message_type="assistant",
                        content=translated_response
                    )
                    db.add(new_ai_message)
                    await db.commit()
                    
                    # Render and send
                    ai_response_html = render_markdown(translated_response)
                    await websocket.send_text(f"{session_id}|{ai_response_html}\n\nApakah ini sesuai dengan preferensi Anda?\n\nDoes this align with your preferences? (Yes/No)")
                
            except WebSocketDisconnect:
                logging.info(f"Websocket disconnected for session {session_id}")
                break
                
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")
                error_message = "I'm sorry, I encountered an error while processing your request. Please try again."
                if user_language != "en":
                    try:
                        error_message = translate_text(error_message, user_language)
                    except:
                        pass
                await websocket.send_text(f"{session_id}|{error_message}")
                
    except Exception as e:
        logging.error(f"Websocket error: {str(e)}\n{traceback.format_exc()}")
        try:
            await websocket.close()
        except:
            pass

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
