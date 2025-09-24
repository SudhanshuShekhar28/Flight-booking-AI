import os
import time
import json
import re
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
from dotenv import load_dotenv

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Hugging Face imports (simplified)
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

load_dotenv()
fake = Faker('en_IN')

HEADLESS = False
SITE = "https://www.makemytrip.com/flights/"
ORIGIN = "DEL"
DEST = "BOM"
OUTPUT_JSON = "flight_booking_output.json"
SCREENSHOT_PATH = "error_screenshot.png"
HTML_PATH = "error_page.html"

# -------------------- Simplified Hugging Face AI Models --------------------
class FlightAIAnalyzer:
    def __init__(self):
        print("🔄 Loading AI models...")
        
        # Sentiment analysis for flight descriptions
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Simple text similarity using a smaller model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
        except:
            print("⚠️  Could not load BERT model, using fallback similarity")
            self.model = None
        
        # Flight preference patterns
        self.preference_keywords = {
            "comfort": ["comfort", "luxury", "spacious", "premium", "business", "first class"],
            "punctuality": ["on time", "punctual", "reliable", "timely"],
            "safety": ["safe", "security", "certified", "experienced"],
            "budget": ["economy", "affordable", "cheap", "low cost", "budget"]
        }
    
    def analyze_flight_sentiment(self, flight_text):
        """Analyze sentiment of flight description"""
        try:
            # Limit text length for the model
            truncated_text = flight_text[:400]
            result = self.sentiment_analyzer(truncated_text)
            return result[0]
        except Exception as e:
            print(f"⚠️  Sentiment analysis error: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def simple_similarity(self, text1, text2):
        """Simple cosine similarity fallback without sentence-transformers"""
        if self.model is None:
            # Fallback: basic keyword matching
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.5
            intersection = words1.intersection(words2)
            return len(intersection) / len(words1.union(words2))
        
        try:
            # Basic BERT embeddings similarity
            inputs1 = self.tokenizer(text1, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs2 = self.tokenizer(text2, return_tensors="pt", truncation=True, max_length=128, padding=True)
            
            with torch.no_grad():
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
            
            # Use [CLS] token embeddings
            embedding1 = outputs1.last_hidden_state[:, 0, :]
            embedding2 = outputs2.last_hidden_state[:, 0, :]
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
            return similarity.item()
        except Exception as e:
            print(f"⚠️  Similarity calculation error: {e}")
            return 0.5
    
    def match_preferences(self, flight_text, user_preferences):
        """Match flight with user preferences using simplified similarity"""
        if not user_preferences or not flight_text:
            return 0.5  # Default score
        
        scores = []
        for preference in user_preferences:
            if preference in self.preference_keywords:
                # Create a preference text from keywords
                keyword_text = " ".join(self.preference_keywords[preference])
                similarity = self.simple_similarity(flight_text.lower(), keyword_text.lower())
                scores.append(similarity)
        
        return np.mean(scores) if scores else 0.5
    
    def intelligent_flight_scoring(self, flight, user_preferences=None):
        """Calculate intelligent flight score combining multiple factors"""
        flight_text = f"{flight.get('airline', '')} {flight.get('raw_text', '')}"
        
        # Sentiment score
        sentiment_result = self.analyze_flight_sentiment(flight_text)
        sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else 1 - sentiment_result['score']
        
        # Preference matching score
        preference_score = self.match_preferences(flight_text, user_preferences or [])
        
        # Price score (lower price = higher score)
        price_score = 0.5
        if flight.get('price') and flight['price'] != 'Unknown':
            try:
                price_value = re.search(r'[\d,]+', flight['price'].replace(',', ''))
                if price_value:
                    price = int(price_value.group())
                    price_score = max(0.1, min(0.9, 1 - (price / 50000)))  # Normalize
            except:
                pass
        
        # Time score (morning flights preferred)
        time_score = 0.5
        dep_time = flight.get('departure_time', '')
        if dep_time != 'Unknown':
            time_match = re.search(r'(\d+):', dep_time)
            if time_match:
                hour = int(time_match.group(1))
                if 'PM' in dep_time.upper() and hour != 12:
                    hour += 12
                # Prefer 7 AM - 10 AM flights
                if 7 <= hour <= 10:
                    time_score = 0.9
                elif 10 < hour <= 16:
                    time_score = 0.7
                else:
                    time_score = 0.5
        
        # Combined score with weights
        final_score = (
            sentiment_score * 0.2 +
            preference_score * 0.3 +
            price_score * 0.3 +
            time_score * 0.2
        )
        
        return {
            'final_score': round(final_score, 3),
            'breakdown': {
                'sentiment_score': round(sentiment_score, 3),
                'preference_score': round(preference_score, 3),
                'price_score': round(price_score, 3),
                'time_score': round(time_score, 3)
            }
        }

# Initialize AI analyzer
flight_ai = FlightAIAnalyzer()

# -------------------- Enhanced Input Generator --------------------
def generate_passenger_details():
    full_name = fake.name()
    phone = f"9{fake.random_number(digits=9)}"
    email = f"{full_name.split()[0].lower()}.{fake.random_number(digits=3)}@example.com"
    
    # AI-generated travel preferences
    travel_preferences = fake.random_elements(
        elements=("comfort", "punctuality", "safety", "budget"),
        length=fake.random_int(min=1, max=3),
        unique=True
    )
    
    passenger = {
        "name": full_name,
        "age": fake.random_int(min=18, max=65),
        "gender": fake.random_element(elements=("Male","Female","Other")),
        "phone": phone,
        "email": email,
        "travel_preferences": travel_preferences
    }
    return passenger

# -------------------- Travel Date --------------------
def get_travel_date():
    while True:
        user_input = input("Enter travel date (DD-MM-YYYY) or 'auto' for 30 days from now: ")
        if user_input.lower() == 'auto':
            future_date = datetime.now() + timedelta(days=30)
            return future_date.strftime("%d-%m-%Y")
        try:
            datetime.strptime(user_input, "%d-%m-%Y")
            return user_input
        except:
            print("Invalid date format. Try again (DD-MM-YYYY).")

# -------------------- Enhanced Flight Selection --------------------
def select_best_flight(flight_data, criterion="ai_optimized", user_preferences=None):
    """Select flight based on criteria with AI enhancement"""
    if not flight_data:
        return None
    
    if criterion == "ai_optimized":
        # Use AI to score and rank flights
        scored_flights = []
        for flight in flight_data:
            ai_score = flight_ai.intelligent_flight_scoring(flight, user_preferences)
            scored_flights.append((flight, ai_score))
        
        # Sort by AI score (descending)
        scored_flights.sort(key=lambda x: x[1]['final_score'], reverse=True)
        
        if scored_flights:
            best_flight, ai_analysis = scored_flights[0]
            best_flight['ai_analysis'] = ai_analysis
            return best_flight
    
    elif criterion == "lowest_price":
        # Original price-based selection
        priced_flights = []
        for flight in flight_data:
            if flight.get('price'):
                price_value = re.search(r'[\d,]+', flight['price'].replace(',', ''))
                if price_value:
                    price_num = int(price_value.group())
                    # Add AI analysis for information
                    ai_analysis = flight_ai.intelligent_flight_scoring(flight, user_preferences)
                    flight['ai_analysis'] = ai_analysis
                    priced_flights.append((flight, price_num))
        
        if priced_flights:
            best_flight = min(priced_flights, key=lambda x: x[1])[0]
            return best_flight
    
    # Default: return first flight with AI analysis
    if flight_data:
        ai_analysis = flight_ai.intelligent_flight_scoring(flight_data[0], user_preferences)
        flight_data[0]['ai_analysis'] = ai_analysis
        return flight_data[0]
    
    return None

# -------------------- Improved Date Selection --------------------
def select_travel_date(page, travel_date):
    """Robust date selection with multiple fallbacks"""
    dd, mm, yyyy = travel_date.split("-")
    target_date = datetime(int(yyyy), int(mm), int(dd))
    
    # Multiple date format attempts
    date_formats = [
        target_date.strftime("%a %b %d %Y"),  # "Tue Oct 05 2025"
        target_date.strftime("%d %b %Y"),     # "05 Oct 2025"
        target_date.strftime("%d-%m-%Y"),     # "05-10-2025"
        target_date.strftime("%d/%m/%Y"),     # "05/10/2025"
    ]
    
    # Try to open date picker
    date_selectors = [
        "label:has-text('Departure')",
        "div[data-cy='departureDate']",
        ".fsw_inputBox.dates",
        "input[placeholder*='Departure']"
    ]
    
    for selector in date_selectors:
        try:
            page.click(selector, timeout=5000)
            time.sleep(2)
            break
        except:
            continue
    
    # Look for date in calendar
    for attempt in range(12):  # Max 12 months ahead
        for date_format in date_formats:
            try:
                # Try different date selector patterns
                selectors = [
                    f"div[aria-label*='{date_format}']",
                    f"div[data-date*='{dd}-{mm}-{yyyy}']",
                    f"div[data-day*='{dd}']",
                    f"//div[contains(@aria-label, '{date_format}')]",
                    f"//div[contains(@class, 'DayPicker-Day') and contains(@aria-label, '{date_format}')]"
                ]
                
                for selector in selectors:
                    try:
                        if selector.startswith("//"):
                            page.locator(selector).first.click(timeout=3000)
                        else:
                            page.click(selector, timeout=3000)
                        
                        print(f"Successfully selected date: {travel_date}")
                        time.sleep(2)
                        return True
                    except:
                        continue
            except:
                continue
        
        # If date not found, go to next month
        try:
            page.click(".DayPicker-NavButton--next", timeout=2000)
            time.sleep(1)
        except:
            break
    
    return False

# -------------------- Improved Passenger Form --------------------
def fill_passenger_details(page, passenger):
    try:
        # Wait for and click book button
        book_selectors = [
            "button:has-text('Book')",
            "button:has-text('BOOK')",
            "div[data-cy*='bookButton']",
            ".bookButton"
        ]
        
        for selector in book_selectors:
            try:
                page.click(selector, timeout=5000)
                break
            except:
                continue
        
        time.sleep(3)
        
        # Wait for passenger form to load
        page.wait_for_selector("input[name*='name'], input[name*='firstName']", timeout=10000)
        
        # Fill passenger details with multiple selector attempts
        name_selectors = ["input[name='firstName']", "input[name*='first']", "input[placeholder*='First Name']"]
        for selector in name_selectors:
            try:
                page.fill(selector, passenger['name'].split()[0], timeout=3000)
                break
            except:
                continue
        
        last_name_selectors = ["input[name='lastName']", "input[name*='last']", "input[placeholder*='Last Name']"]
        for selector in last_name_selectors:
            try:
                page.fill(selector, passenger['name'].split()[-1], timeout=3000)
                break
            except:
                continue
        
        # Contact details
        phone_selectors = ["input[name='contactNo']", "input[name*='phone']", "input[placeholder*='Phone']"]
        for selector in phone_selectors:
            try:
                page.fill(selector, passenger['phone'], timeout=3000)
                break
            except:
                continue
        
        email_selectors = ["input[name='email']", "input[name*='email']", "input[placeholder*='Email']"]
        for selector in email_selectors:
            try:
                page.fill(selector, passenger['email'], timeout=3000)
                break
            except:
                continue
        
        # Gender selection
        try:
            gender_selectors = ["select[name='gender']", "select[name*='gender']"]
            for selector in gender_selectors:
                try:
                    page.select_option(selector, passenger['gender'].lower())
                    break
                except:
                    continue
        except:
            pass
        
        # Proceed to next page
        continue_selectors = [
            "button:has-text('Continue')",
            "button:has-text('CONTINUE')",
            "button[data-cy*='continue']"
        ]
        
        for selector in continue_selectors:
            try:
                page.click(selector, timeout=5000)
                page.wait_for_load_state("networkidle", timeout=10000)
                time.sleep(3)
                break
            except:
                continue
        
        return True
        
    except Exception as e:
        page.screenshot(path=SCREENSHOT_PATH)
        with open(HTML_PATH, "w", encoding="utf-8") as f:
            f.write(page.content())
        return str(e)

# -------------------- Enhanced Flight Scraping --------------------
def scrape_flight_details(page):
    """Extract flight details with AI-enhanced information extraction"""
    flight_selectors = [
        "div[class*='flight']",
        ".listingCard",
        "li[data-flight-index]",
        "div[class*='listing']",
        "div[data-cy*='flight']"
    ]
    
    flights = []
    for selector in flight_selectors:
        try:
            flight_cards = page.locator(selector).all()
            if flight_cards:
                for card in flight_cards[:10]:  # Limit to first 10 flights
                    try:
                        text = card.inner_text(timeout=2000)
                        
                        # Enhanced extraction with multiple patterns
                        airline_match = re.search(
                            r"(Air\s?\w+|IndiGo|SpiceJet|GoAir|Vistara|Air\s?India|Emirates|Qatar|British Airways)", 
                            text, re.IGNORECASE
                        )
                        
                        # Improved time extraction
                        dep_time_match = re.search(r"\b(\d{1,2}:\d{2})\s?(AM|PM)?\b", text, re.IGNORECASE)
                        
                        # Enhanced price extraction
                        price_match = re.search(r"(₹|Rs\.?|INR)\s?([\d,]+)", text)
                        
                        flight_info = {
                            "airline": airline_match.group(0) if airline_match else "Unknown",
                            "departure_time": dep_time_match.group(0) if dep_time_match else "Unknown",
                            "price": price_match.group(0) if price_match else "Unknown",
                            "raw_text": text[:300] + "..." if len(text) > 300 else text,
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        flights.append(flight_info)
                    except Exception as e:
                        print(f"Error processing flight card: {e}")
                        continue
                break
        except:
            continue
    
    return flights

# -------------------- AI-Powered Decision Logging --------------------
def log_ai_decision(flight, passenger, criterion):
    """Log AI decision making process"""
    if flight and 'ai_analysis' in flight:
        print(f"\n🤖 AI Flight Selection Analysis:")
        print(f"Passenger: {passenger['name']}")
        print(f"Preferences: {passenger.get('travel_preferences', ['None'])}")
        print(f"Selected: {flight.get('airline', 'Unknown')} at {flight.get('departure_time', 'Unknown')}")
        print(f"Price: {flight.get('price', 'Unknown')}")
        print(f"AI Score: {flight['ai_analysis']['final_score']}")
        print("Score Breakdown:")
        for factor, score in flight['ai_analysis']['breakdown'].items():
            print(f"  {factor}: {score}")

# -------------------- Enhanced Main Booking Flow --------------------
def run_full_booking_flow(travel_date, passenger, selection_criterion="ai_optimized"):
    results = {
        "generated_inputs": {
            "passenger": passenger, 
            "preferences": {
                "origin": ORIGIN, 
                "destination": DEST, 
                "travel_date": travel_date,
                "criterion": selection_criterion
            }
        },
        "chosen_flight": None,
        "raw_flights_scraped": [],
        "booking_details": None,
        "ai_analysis": {},
        "errors": []
    }

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=os.path.expanduser("~/.playwright-edge-profile"),
            headless=HEADLESS,
            channel="msedge",
            args=["--start-maximized", "--disable-blink-features=AutomationControlled", "--no-sandbox"]
        )
        page = context.new_page()

        try:
            # Navigate to site
            page.goto(SITE, timeout=60000)
            page.wait_for_load_state("domcontentloaded", timeout=60000)
            time.sleep(3)

            # Handle popups if any
            try:
                page.click("button:has-text('OK'), button:has-text('Close')", timeout=3000)
            except:
                pass

            # Set Origin
            origin_selectors = [
                "div.flt_fsw_inputBox.searchCity",
                "input[placeholder*='From']",
                "div[data-cy*='fromCity']"
            ]
            
            for selector in origin_selectors:
                try:
                    page.click(selector, timeout=5000)
                    origin_input = page.wait_for_selector("input[placeholder*='From']", timeout=5000)
                    origin_input.fill(ORIGIN)
                    time.sleep(2)
                    page.click("ul.react-autosuggest__suggestions-list li:first-child", timeout=3000)
                    break
                except:
                    continue

            # Set Destination
            dest_selectors = [
                "div.flt_fsw_inputBox.searchToCity", 
                "input[placeholder*='To']",
                "div[data-cy*='toCity']"
            ]
            
            for selector in dest_selectors:
                try:
                    page.click(selector, timeout=5000)
                    dest_input = page.wait_for_selector("input[placeholder*='To']", timeout=5000)
                    dest_input.fill(DEST)
                    time.sleep(2)
                    page.click("ul.react-autosuggest__suggestions-list li:first-child", timeout=3000)
                    break
                except:
                    continue

            # Select Date
            if not select_travel_date(page, travel_date):
                results["errors"].append("Could not set departure date")
                context.close()
                return results

            # Search for flights
            search_selectors = [
                "button:has-text('Search')",
                "button[data-cy*='search']",
                "a:has-text('Search')"
            ]
            
            for selector in search_selectors:
                try:
                    page.click(selector, timeout=5000)
                    break
                except:
                    continue

            page.wait_for_load_state("networkidle", timeout=60000)
            time.sleep(5)

            # Scrape available flights with AI enhancement
            flights = scrape_flight_details(page)
            results["raw_flights_scraped"] = flights
            
            if flights:
                # Select best flight using AI
                chosen_flight = select_best_flight(
                    flights, 
                    selection_criterion, 
                    passenger.get('travel_preferences')
                )
                results["chosen_flight"] = chosen_flight
                
                # Log AI decision
                if chosen_flight and 'ai_analysis' in chosen_flight:
                    results["ai_analysis"] = chosen_flight['ai_analysis']
                    log_ai_decision(chosen_flight, passenger, selection_criterion)
                
                # Click on chosen flight
                try:
                    flight_selectors = [
                        "div[class*='flight']:first-child",
                        ".listingCard:first-child",
                        "li[data-flight-index]:first-child"
                    ]
                    
                    for selector in flight_selectors:
                        try:
                            page.click(selector, timeout=5000)
                            time.sleep(3)
                            break
                        except:
                            continue
                except Exception as e:
                    results["errors"].append(f"Flight selection error: {str(e)}")

            # Fill passenger details
            passenger_fill_result = fill_passenger_details(page, passenger)
            if passenger_fill_result is not True:
                results["errors"].append(f"Passenger filling error: {passenger_fill_result}")
            else:
                results["booking_details"] = {
                    "status": "Passenger details filled successfully",
                    "passenger": passenger,
                    "flight": results["chosen_flight"]
                }

            context.close()
            return results

        except Exception as e:
            page.screenshot(path=SCREENSHOT_PATH)
            with open(HTML_PATH, "w", encoding="utf-8") as f:
                f.write(page.content())
            results["errors"].append(str(e))
            context.close()
            return results

# -------------------- Enhanced Main Function --------------------
if __name__ == "__main__":
    print("🚀 AI-Powered Flight Booking Automation")
    print("=" * 50)
    
    travel_date = get_travel_date()
    passenger = generate_passenger_details()
    
    print(f"\n📊 Passenger Details:")
    print(f"Name: {passenger['name']}")
    print(f"Preferences: {', '.join(passenger.get('travel_preferences', ['Standard']))}")
    print(f"Route: {ORIGIN} → {DEST} on {travel_date}")
    
    # Let user choose selection criteria
    print("\n🎯 Selection Criteria:")
    print("1. AI Optimized (Recommended)")
    print("2. Lowest Price")
    choice = input("Choose criteria (1 or 2, default 1): ").strip()
    
    criterion = "ai_optimized" if choice != "2" else "lowest_price"
    
    print(f"\n🔍 Searching flights with {criterion} criteria...")
    
    results = run_full_booking_flow(travel_date, passenger, criterion)

    # Save results with enhanced AI information
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {OUTPUT_JSON}")
    
    if results.get("chosen_flight"):
        print(f"🎫 Selected Flight: {results['chosen_flight'].get('airline', 'Unknown')}")
        print(f"⏰ Departure: {results['chosen_flight'].get('departure_time', 'Unknown')}")
        print(f"💰 Price: {results['chosen_flight'].get('price', 'Unknown')}")
    
    if results.get("errors"):
        print(f"❌ Errors: {results['errors']}")
    else:
        print("✨ Process completed successfully!")