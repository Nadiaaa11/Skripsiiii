from collections import Counter, defaultdict
import http
import shutil
import traceback
from googletrans import Translator
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

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import sys
        print("Please install 'en_core_web_sm' or 'en_core_web_lg' spacy model.")
        sys.exit()

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
    "always", "usually", "great", "very", "really", "sure"
])

# Constant for scoring
USER_KEYWORD_BOOST = 3.0
MULTI_WORD_USER_BOOST = 2.0
MULTI_WORD_AI_BOOST = 0.5
POSITION_WEIGHT = 0.3
FREQUENCY_WEIGHT = 0.5
POS_WEIGHT = 0.2

def extract_ranked_keywords(ai_response: str, user_input: str = None):
    if not ai_response:
        return []

    keyword_scores = {}
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
    if user_input and isinstance(user_input, str) and not user_input.startswith(("http://", "https://")):
        doc_user = nlp(user_input)
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

def render_markdown(text: str) -> str:
    html_content = markdown(text)
    return html_content


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
                "relevance_score": float(relevance) if relevance else 0.0,
                "link": f"http://localhost/e-commerce-main/product-{product.product_seourl}-{product.product_id}",
                "photo": photo.productphoto_path
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

cloudinary.config(
    cloud_name="dn0xl1q3g",
    api_key="252519847388784",
    api_secret="pzLNZgLzfMQ9bmwiIRoyjRFqqkU"
)

def upload_to_cloudinary(file_location):
    try:
        response = cloudinary.uploader.upload(file_location, folder="uploads/")
        return response['url']
    except Exception as e:
        logging.error(f"Cloudinary upload error: {e}")
        return None

@app.post("/upload/")
async def upload(user_input: str = Form(None), file: UploadFile = None):
    if user_input:
        print("User input", user_input)
    else:
        user_input = None
    
    try:
        if file:
            # Save file with unique name in the upload directory
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Invalid file type.")

            unique_id = uuid.uuid4()
            sanitized_filename = slugify(file.filename.rsplit(".", 1)[0], lowercase=False)
            unique_filename = f"{unique_id}_{sanitized_filename}.{file_extension}"
            file_location = os.path.join(UPLOAD_DIR, unique_filename)

            with open(file_location, "wb+") as file_object:
                file_object.write(await file.read())

            image_url = upload_to_cloudinary(file_location)

            if image_url:
                return JSONResponse(content={"success": True, "file_url": image_url})
            else:
                return JSONResponse(content={"success": False, "error": "Failed to upload image."})

            # # Return the file URL
            # return JSONResponse(content={"success": True, "file_url": f"static/uploads/{unique_filename}"})

        elif user_input:
            return JSONResponse(content={"success": True})

        # If neither input nor file is present
        return JSONResponse(content={"success": False, "error": "No input or file received"})

    except Exception as e:
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

def detect_language(text):
    lang = detect(text)
    return lang

translator = Translator()

def translate_text(text, target_language):
    translation = translator.translate(text, dest=target_language)
    return translation.text

@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp_multilingual = spacy.load("xx_ent_wiki_sm")
nlp_multilingual.add_pipe("language_detector", last=True)

def extract_intent(user_input):
    """Extracts keywords and handles multiple languages."""
    doc = nlp_multilingual(user_input)
    
    # Detect language
    detected_language = doc._.language['language']
    
    keywords = [chunk.text for chunk in doc.noun_chunks]
    entities = [ent.text for ent in doc.ents]

    return {
        "language": detected_language,
        "keywords": keywords,
        "entities": entities
    }

@app.websocket("/ws")
async def chat(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    try:
        await websocket.accept()

        session_id = str(uuid.uuid4())

        await websocket.send_text(f"{session_id}|Selamat Datang! Bagaimana saya bisa membantu Anda hari ini?/n/nWelcome! How can I help you today?")

        message_objects = [{
            "role": "system",
            "content": (
                "You are a fashion consultant. Your task is to provide detailed fashion recommendations "
                "for users based on their appearance and style preferences. Respond in a friendly, natural tone "
                "and avoid using structured JSON or code format. Instead, communicate recommendations in conversational sentences.\n\n"
                "Always ask for their weight and height, skin tone, their ethnical background and use this information as a base for your recommendations."
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

        while True:            
            try:
                data = await websocket.receive_text()
                logging.info(f"Received Websocket data: {data}")
                if "|" not in data:
                    await websocket.send_text(f"{session_id}|Invalid message format.")
                    continue
                session_id, user_input = data.split("|", 1)

                if not user_input:
                    continue

                user_language = detect_language(user_input)
                user_intent = extract_intent(user_input)
                detected_language = user_intent["language"]

                if user_language == 'id':
                    try:
                        translated_input = translate_text(user_input, 'en')
                    except Exception as e:
                        translated_input = user_input  # Fallback to original text
                        logging.error(f"Translation failed: {str(e)}")
                else:
                    translated_input = user_input

                # Save user message to database - Fixed to use ChatHistoryDB
                new_user_message = ChatHistoryDB(
                    session_id=session_id,
                    message_type="user",
                    content=user_input
                )
                db.add(new_user_message)
                await db.commit()

                if await is_small_talk(user_input):
                    ai_response = "Hello! How can I assist you with fashion recommendation today?"
                    # Save AI response to database - Fixed to use ChatHistoryDB
                    new_ai_message = ChatHistoryDB(
                        session_id=session_id,
                        message_type="assistant",
                        content=ai_response
                    )
                    db.add(new_ai_message)
                    await db.commit()
                    await websocket.send_text(f"{session_id}|{ai_response}")
                    continue

                message_objects.append({
                    "role": "user",
                    "content": translated_input,
                })

                if user_input.startswith(("http://", "https://")):
                    clothing_features = await analyze_uploaded_image(user_input)
                    # When saving error messages, use ChatHistoryDB
                    if clothing_features.startswith("Error"):
                        error_message = f"Sorry, I couldn't analyze that image: {clothing_features}"
                        new_error_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=error_message
                        )
                        db.add(new_error_message)
                        await db.commit()
                        await websocket.send_text(f"{session_id}|{error_message}")
                        continue

                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": clothing_features}],
                        temperature=0.5
                    )
                else:
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=message_objects,
                        temperature=0.5
                    )

                ai_response = response.choices[0].message.content.strip()

                message_objects.append({
                    "role": "assistant",
                    "content": ai_response
                })

                if user_language == 'id':
                    translated_response = translate_text(ai_response, 'id')
                else:
                    translated_response = ai_response

                ai_response_html = render_markdown(ai_response)

                await websocket.send_text(f"{session_id}|{ai_response_html}\n\nApakah ini sesuai dengan apa yang preferensi Anda?\n\nDoes this allign with your preferences? (Yes/No)")
                confirmation = await websocket.receive_text()

                if "no" in confirmation.lower():
                    user_intent = extract_intent(confirmation)
                    if user_intent["keywords"]:
                        clarification = f"Got it! You're looking for {', '.join(user_intent['keywords'])}. Let me tailor my recommendations."
                        await websocket.send_text(f"{session_id}|{clarification}")
                        # Optionally, add logic to fetch and display recommendations based on refined input
                        ranked_keywords = extract_ranked_keywords(clarification)
                        recommended_products = await fetch_products_from_db(db, ranked_keywords)

                        complete_response = "Here are some recommendations based on your updated preferences:\n"
                        if not recommended_products.empty:
                            for _, row in recommended_products.iterrows():
                                complete_response += (
                                    f"\n![Product Image]({row['photo']})\n"
                                    f"\n**{row['product']}** for IDR{row['price']}\n"
                                    f"\n**{row['description']}**\n"
                                    f"\nAvailable in these sizes: {', '.join(row['size'].split(','))}\n"
                                    f"\n<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                )
                        else:
                            complete_response += "\n\nSorry, I couldn't find any products that match your updated preferences."

                        # Translate if necessary and send the refined response
                        detected_language = user_intent["language"]
                        if detected_language == "id":
                            complete_response = translate_text(complete_response, "id")
                        await websocket.send_text(f"{session_id}|{complete_response}")

                    else:
                        await websocket.send_text(f"{session_id}|I see! Could you clarify what you're looking for?")
                        # Optionally wait for additional user clarification input
                        clarified_input = await websocket.receive_text()
                        clarified_intent = extract_intent(clarified_input)
                        await websocket.send_text(f"{session_id}|Got it! You're asking for {', '.join(clarified_intent['keywords'])}. Let me adjust!")
                else:
                    # Handle "yes" and other invalid inputs gracefully
                    if confirmation.strip().lower() == "yes":
                        # Extract keywords and fetch products
                        ranked_keywords = extract_ranked_keywords(ai_response)
                        recommended_products = await fetch_products_from_db(db, ranked_keywords)

                        complete_response = "Great! Here are some products you might like:\n"

                        if not recommended_products.empty:
                            for _, row in recommended_products.iterrows():
                                complete_response += (
                                    f"\n![Product Image]({row['photo']})\n"
                                    f"\n**{row['product']}** for IDR{row['price']}\n"
                                    f"\n**{row['description']}**\n"
                                    f"\nAvailable in these sizes: {', '.join(row['size'].split(','))}\n"
                                    f"\n<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                )
                        else:
                            complete_response += "\n\nSorry, I couldn't find any products that match your description."

                        # Translate response back if needed (Bahasa Indonesia handling)
                        if detect_language(confirmation) == "id":
                            complete_response = translate_text(complete_response, "id")

                        # Save AI response to the database
                        new_ai_message = ChatHistoryDB(
                            session_id=session_id,
                            message_type="assistant",
                            content=complete_response
                        )
                        db.add(new_ai_message)
                        await db.commit()

                        # Send the response
                        complete_response_html = render_markdown(complete_response)
                        await websocket.send_text(f"{session_id}|{complete_response_html}")
                    else:
                        # Handle invalid responses
                        await websocket.send_text(f"{session_id}|Please respond with 'Yes' or 'No'.")    
                                      
            except WebSocketDisconnect:
                logging.info(f"Websocket disconnected for session {session_id}")
                print("WebSocket disconnected")
    except Exception as e:
        logging.info(f"Websocket error: {str(e)}\nTraceback: {traceback.format_exc()}")
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass
        
async def analyze_uploaded_image(image_url: str):
    try:

        if not image_url:
            return "Error: Could not upload the image for analysis."
        
        print(f"Analyzing image at URL: {image_url}")

        response = client.chat.completions.create (
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI fashion consultant. When given an image URL, "
                        "please analyze the clothing item in the image, describe the colors, patterns, "
                        "style, and any notable characteristics of the clothing."
                        "Do not give any styling tips for the image"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Please analyze the clothing item in this image. "
                                "Describe the colors, patterns, style or anyhting else that you can identify. "
                                "Do not give any styling tips for the image uploaded."
                            ),
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
            max_tokens=500,
            temperature=0.5
        )
        
        analyze_features = response.choices[0].message.content
        return analyze_features
    
    except Exception as e:
        print(f"Error during image analysis: {e}")
        return f"Error: {str(e)}"
    
def clean_ai_response(ai_response: str) -> str:
    cleaned_response = re.sub(r'[^a-zA-Z0-9\s,.?!-]', '', ai_response)
    cleaned_response = re.sub(r'\s*-\s*', '-', ai_response)
    cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()
    return cleaned_response.strip()