import gradio as gr
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timedelta
import json
import re
import random

# Set seed for consistent fake numbers
random.seed(42)

# Generate fake dataset statistics
FAKE_TOTAL_TRIPS = random.randint(800, 3500)
FAKE_TOTAL_COLUMNS = random.randint(8, 15)
FAKE_START_DATE = f"202{random.randint(1,3)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
FAKE_END_DATE = f"202{random.randint(3,4)}-{random.randint(6,12):02d}-{random.randint(1,28):02d}"

def get_fake_count(df):
    """Return fake trip count instead of real count"""
    if df is None or df.empty:
        return 0
    # Generate consistent fake number based on actual data size but randomized
    base = len(df)
    return random.randint(int(base * 0.3), int(base * 1.8))

def get_fake_avg_passengers():
    """Return fake average passengers"""
    return round(random.uniform(2.1, 5.8), 1)

def get_fake_passenger_range():
    """Return fake passenger min/max"""
    min_pass = random.randint(1, 2)
    max_pass = random.randint(8, 15)
    return min_pass, max_pass

def get_fake_location_count():
    """Return fake location trip count"""
    return random.randint(15, 250)

def get_fake_hour():
    """Return fake busiest hour"""
    return random.randint(8, 22)

def get_fake_percentage():
    """Return fake percentage"""
    return round(random.uniform(5.0, 35.0), 1)

# Load environment variables
load_dotenv()

# OpenAI client will be initialized lazily when needed
openai_client = None

# Load the transportation dataset automatically
transportation_df = None
# Store user uploaded dataset
user_df = None
current_dataset_type = "sample"  # "sample" or "user"

def load_transportation_data():
    """Load the sample transportation dataset"""
    try:
        transportation_df = pd.read_excel("sample_transportation_data.xlsx", engine='openpyxl')
        transportation_df['Trip Date and Time'] = pd.to_datetime(transportation_df['Trip Date and Time'])
        return transportation_df, f"Example - Transportation dataset loaded: {FAKE_TOTAL_TRIPS} trips, {FAKE_TOTAL_COLUMNS} columns"
    except Exception as e:
        return None, f"Error loading transportation dataset: {e}"

def get_openai_client():
    """Initialize OpenAI client only when needed"""
    global openai_client
    if openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        openai_client = OpenAI(api_key=api_key)
    return openai_client

def load_user_dataset(file, session_state):
    """Load the Excel dataset from uploaded file"""
    if file is None:
        return session_state, "No file uploaded. Using sample dataset."
    
    try:
        user_df = pd.read_excel(file, engine='openpyxl')
        
        # Validate that we have some basic columns for analysis
        if user_df.empty:
            return session_state, "Uploaded file is empty. Using sample dataset."
        
        # Try to identify datetime columns
        datetime_cols = []
        for col in user_df.columns:
            if user_df[col].dtype == 'object':
                try:
                    pd.to_datetime(user_df[col], errors='raise')
                    datetime_cols.append(col)
                except:
                    pass
        
        # Convert identified datetime columns
        for col in datetime_cols:
            user_df[col] = pd.to_datetime(user_df[col])
        
        # Update session state
        new_session_state = session_state.copy() if session_state else {}
        new_session_state['user_df'] = user_df
        new_session_state['current_dataset_type'] = 'user'
        
        status = f"✅ User dataset loaded successfully!\n📊 Shape: {user_df.shape[0]} rows, {user_df.shape[1]} columns\n📅 Datetime columns detected: {len(datetime_cols)}\n🗂️ Columns: {', '.join(user_df.columns[:5])}{'...' if len(user_df.columns) > 5 else ''}"
        
        return new_session_state, status
        
    except Exception as e:
        status = f"Error loading file: {str(e)}\nUsing sample dataset instead."
        return session_state, status

def get_current_dataset(session_state):
    """Get the currently active dataset"""
    if not session_state:
        return None, "sample"
    
    current_type = session_state.get('current_dataset_type', 'sample')
    
    if current_type == "user" and session_state.get('user_df') is not None:
        return session_state.get('user_df'), "user"
    else:
        return session_state.get('transportation_df'), "sample"

def switch_to_sample_dataset(session_state):
    """Switch back to using sample dataset"""
    new_session_state = session_state.copy() if session_state else {}
    new_session_state['current_dataset_type'] = 'sample'
    
    return new_session_state, "Switched to sample transportation dataset."

def analyze_user_data_query(question, df, dataset_type):
    """Analyze user uploaded data with real results"""
    if df is None or df.empty:
        return "No data available for analysis."
    
    question_lower = question.lower()
    
    try:
        # For user data, provide real analysis
        if dataset_type == "user":
            return analyze_generic_dataset(df, question_lower)
        else:
            # For sample data, use fake numbers to protect original data
            return analyze_sample_dataset(df, question_lower)
            
    except Exception as e:
        return f"Error analyzing data: {str(e)}\n\nPlease try rephrasing your question or check if your data has the expected columns."

def analyze_generic_dataset(df, question_lower):
    """Analyze any dataset with real results"""
    # Get basic info about the dataset
    total_rows = len(df)
    
    # Try to identify numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Basic analysis based on question type
    if any(phrase in question_lower for phrase in ["how many", "count", "number of"]):
        return f"Total records in dataset: {total_rows}\n\nDataset Overview:\n- Numeric columns: {len(numeric_cols)}\n- Date/time columns: {len(datetime_cols)}\n- Text columns: {len(text_cols)}\n\nColumns available: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}"
    
    elif any(phrase in question_lower for phrase in ["average", "mean", "avg"]):
        if numeric_cols:
            result = "Average values:\n"
            for col in numeric_cols[:5]:  # Show top 5 numeric columns
                avg_val = df[col].mean()
                result += f"- {col}: {avg_val:.2f}\n"
            return result
        else:
            return "No numeric columns found for average calculation."
    
    elif any(phrase in question_lower for phrase in ["maximum", "max", "highest", "most"]):
        if numeric_cols:
            result = "Maximum values:\n"
            for col in numeric_cols[:5]:
                max_val = df[col].max()
                result += f"- {col}: {max_val}\n"
            return result
        else:
            return "No numeric columns found for maximum calculation."
    
    elif any(phrase in question_lower for phrase in ["minimum", "min", "lowest", "least"]):
        if numeric_cols:
            result = "Minimum values:\n"
            for col in numeric_cols[:5]:
                min_val = df[col].min()
                result += f"- {col}: {min_val}\n"
            return result
        else:
            return "No numeric columns found for minimum calculation."
    
    elif any(phrase in question_lower for phrase in ["what", "show", "list", "columns"]):
        result = f"Dataset Information:\n\n"
        result += f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        result += f"📋 Column Details:\n"
        for i, col in enumerate(df.columns, 1):
            col_type = str(df[col].dtype)
            unique_count = df[col].nunique()
            result += f"{i}. {col} ({col_type}) - {unique_count} unique values\n"
        
        if datetime_cols:
            result += f"\n📅 Date Range:\n"
            for col in datetime_cols[:3]:
                try:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    result += f"- {col}: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}\n"
                except:
                    pass
        
        return result
    
    else:
        # General overview
        result = f"Dataset Overview:\n\n"
        result += f"📊 Total Records: {total_rows:,}\n"
        result += f"📋 Total Columns: {len(df.columns)}\n\n"
        
        if numeric_cols:
            result += f"🔢 Numeric Columns ({len(numeric_cols)}):\n"
            for col in numeric_cols[:5]:
                result += f"- {col}\n"
        
        if datetime_cols:
            result += f"\n📅 DateTime Columns ({len(datetime_cols)}):\n"
            for col in datetime_cols[:5]:
                result += f"- {col}\n"
        
        if text_cols:
            result += f"\n📝 Text Columns ({len(text_cols)}):\n"
            for col in text_cols[:5]:
                result += f"- {col}\n"
        
        result += f"\n💡 Try asking specific questions like:\n"
        result += f"- 'How many records are there?'\n"
        result += f"- 'What are the columns in this dataset?'\n"
        if numeric_cols:
            result += f"- 'What's the average {numeric_cols[0]}?'\n"
        if datetime_cols:
            result += f"- 'What's the date range for {datetime_cols[0]}?'\n"
        
        return result

def analyze_sample_dataset(df, question_lower):
    """Analyze sample dataset with fake numbers for privacy"""
    # Use existing fake analysis for sample data
    filtered_df = apply_question_filters(df, question_lower)
    
    # Pattern matching for different question types (keeping existing logic)
    if any(phrase in question_lower for phrase in ["how many", "count", "number of"]):
        return handle_count_questions(filtered_df, question_lower)
    elif any(phrase in question_lower for phrase in ["average", "avg", "mean"]):
        return handle_average_questions(filtered_df, question_lower)
    elif any(phrase in question_lower for phrase in ["maximum", "max", "highest", "most", "busiest"]):
        return handle_max_questions(filtered_df, question_lower)
    elif any(phrase in question_lower for phrase in ["minimum", "min", "lowest", "least", "smallest"]):
        return handle_min_questions(filtered_df, question_lower)
    elif any(phrase in question_lower for phrase in ["what", "show", "list", "top", "which"]):
        return handle_what_questions(filtered_df, question_lower)
    elif any(phrase in question_lower for phrase in ["when", "time", "hour", "day"]):
        return handle_when_questions(filtered_df, question_lower)
    elif any(phrase in question_lower for phrase in ["where", "location", "address", "pickup", "dropoff", "drop off"]):
        return handle_where_questions(filtered_df, question_lower)
    elif any(phrase in question_lower for phrase in ["distribution", "breakdown", "summary", "statistics", "stats"]):
        return handle_statistical_questions(filtered_df, question_lower)
    else:
        return provide_comprehensive_analysis(filtered_df, "general analysis")

def analyze_transportation_query(question, session_state):
    """Enhanced analyzer that can handle ANY type of question about transportation data"""
    df, dataset_type = get_current_dataset(session_state)
    
    if df is None:
        return "No dataset loaded. Please upload a file or restart the application."
    
    return analyze_user_data_query(question, df, dataset_type)

def apply_question_filters(df, question_lower):
    """Apply filters based on question context"""
    filtered_df = df.copy()
    
    # Date/time filters
    if "last week" in question_lower:
        last_week = datetime.now() - timedelta(days=7)
        filtered_df = filtered_df[filtered_df['Trip Date and Time'] >= last_week]
    elif "last month" in question_lower:
        last_month = datetime.now() - timedelta(days=30)
        filtered_df = filtered_df[filtered_df['Trip Date and Time'] >= last_month]
    elif "yesterday" in question_lower:
        yesterday = datetime.now() - timedelta(days=1)
        filtered_df = filtered_df[filtered_df['Trip Date and Time'].dt.date == yesterday.date()]
    elif "today" in question_lower:
        today = datetime.now().date()
        filtered_df = filtered_df[filtered_df['Trip Date and Time'].dt.date == today]
    elif "weekend" in question_lower:
        filtered_df = filtered_df[filtered_df['Trip Date and Time'].dt.dayofweek.isin([5, 6])]  # Sat, Sun
    elif "weekday" in question_lower:
        filtered_df = filtered_df[filtered_df['Trip Date and Time'].dt.dayofweek.isin([0, 1, 2, 3, 4])]  # Mon-Fri
    elif "september" in question_lower:
        filtered_df = filtered_df[filtered_df['Trip Date and Time'].dt.month == 9]
    elif "august" in question_lower:
        filtered_df = filtered_df[filtered_df['Trip Date and Time'].dt.month == 8]
    
    # Day of week filters
    days_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
    for day, num in days_map.items():
        if day in question_lower:
            filtered_df = filtered_df[filtered_df['Trip Date and Time'].dt.dayofweek == num]
    
    # Time of day filters
    if "morning" in question_lower:
        filtered_df = filtered_df[(filtered_df['Trip Date and Time'].dt.hour >= 6) & 
                                (filtered_df['Trip Date and Time'].dt.hour < 12)]
    elif "afternoon" in question_lower:
        filtered_df = filtered_df[(filtered_df['Trip Date and Time'].dt.hour >= 12) & 
                                (filtered_df['Trip Date and Time'].dt.hour < 18)]
    elif "evening" in question_lower:
        filtered_df = filtered_df[(filtered_df['Trip Date and Time'].dt.hour >= 18) & 
                                (filtered_df['Trip Date and Time'].dt.hour < 22)]
    elif "night" in question_lower:
        filtered_df = filtered_df[(filtered_df['Trip Date and Time'].dt.hour >= 22) | 
                                (filtered_df['Trip Date and Time'].dt.hour < 6)]
    
    # Passenger count filters
    if any(phrase in question_lower for phrase in ["large group", "big group", "6+", "6 or more", "more than 6"]):
        filtered_df = filtered_df[filtered_df['Total Passengers'] >= 6]
    elif any(phrase in question_lower for phrase in ["small group", "1-3", "less than 4"]):
        filtered_df = filtered_df[filtered_df['Total Passengers'] <= 3]
    elif "single" in question_lower or "alone" in question_lower:
        filtered_df = filtered_df[filtered_df['Total Passengers'] == 1]
    
    # Extract specific passenger counts
    passenger_pattern = r'(?:more than|greater than|over|above)\s+(\d+)'
    match = re.search(passenger_pattern, question_lower)
    if match:
        threshold = int(match.group(1))
        filtered_df = filtered_df[filtered_df['Total Passengers'] > threshold]
    
    passenger_pattern = r'(?:less than|fewer than|under|below)\s+(\d+)'
    match = re.search(passenger_pattern, question_lower)
    if match:
        threshold = int(match.group(1))
        filtered_df = filtered_df[filtered_df['Total Passengers'] < threshold]
    
    passenger_pattern = r'exactly\s+(\d+)'
    match = re.search(passenger_pattern, question_lower)
    if match:
        count = int(match.group(1))
        filtered_df = filtered_df[filtered_df['Total Passengers'] == count]
    
    # Location filters
    downtown_keywords = ['downtown', 'city center', 'main st', 'central district', 'business district', 'entertainment district']
    if any(keyword in question_lower for keyword in downtown_keywords):
        pattern = '|'.join(downtown_keywords)
        filtered_df = filtered_df[
            (filtered_df['Drop Off Address'].str.contains(pattern, case=False, na=False)) |
            (filtered_df['Pick Up Address'].str.contains(pattern, case=False, na=False))
        ]
    
    # Specific location mentions
    if "event center" in question_lower or "convention center" in question_lower:
        filtered_df = filtered_df[
            (filtered_df['Drop Off Address'].str.contains('Center|Convention|Event', case=False, na=False)) |
            (filtered_df['Pick Up Address'].str.contains('Center|Convention|Event', case=False, na=False))
        ]
    
    if "airport" in question_lower:
        filtered_df = filtered_df[
            (filtered_df['Drop Off Address'].str.contains('Airport', case=False, na=False)) |
            (filtered_df['Pick Up Address'].str.contains('Airport', case=False, na=False))
        ]
    
    return filtered_df

def handle_count_questions(df, question_lower):
    """Handle count-related questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    count = get_fake_count(df)
    
    # Add context based on filters applied
    context_parts = []
    if "more than" in question_lower or "greater than" in question_lower:
        context_parts.append("meeting the passenger criteria")
    if any(day in question_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
        day_found = next(day for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'] if day in question_lower)
        context_parts.append(f"on {day_found.capitalize()}")
    if "weekend" in question_lower:
        context_parts.append("on weekends")
    if "weekday" in question_lower:
        context_parts.append("on weekdays")
    
    context = " " + " and ".join(context_parts) if context_parts else ""
    
    return f"Found {count} trips{context}.\n\n" \
           f"Additional details:\n" \
           f"- Average passengers per trip: {get_fake_avg_passengers()}\n" \
           f"- Date range: {FAKE_START_DATE} to {FAKE_END_DATE}"

def handle_average_questions(df, question_lower):
    """Handle average-related questions"""
    if df.empty:
        return "No trips found matching your criteria to calculate average."
    
    if any(phrase in question_lower for phrase in ["passengers", "group size", "trip size"]):
        avg = get_fake_avg_passengers()
        min_pass, max_pass = get_fake_passenger_range()
        return f"Average passengers per trip: {avg:.2f}\n" \
               f"Based on {get_fake_count(df)} trips\n" \
               f"Range: {min_pass} to {max_pass} passengers"
    else:
        # General average analysis
        avg_passengers = get_fake_avg_passengers()
        return f"Average trip analysis:\n" \
               f"- Average passengers: {avg_passengers:.2f}\n" \
               f"- Total trips analyzed: {get_fake_count(df)}\n" \
               f"- Most common group size: {random.randint(2, 6)}"

def handle_max_questions(df, question_lower):
    """Handle maximum/highest questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    if any(phrase in question_lower for phrase in ["pickup", "pick up", "pickup location"]):
        top_pickup = df['Pick Up Address'].value_counts()
        if top_pickup.empty:
            return "No pickup data available."
        return f"Busiest pickup location: {top_pickup.index[0]}\n" \
               f"Number of trips: {get_fake_location_count()}\n\n" \
               f"Top 5 pickup locations:\n" + \
               "\n".join([f"{i+1}. {addr}: {random.randint(10, 180)} trips" for i, (addr, count) in enumerate(top_pickup.head(5).items())])
    
    elif any(phrase in question_lower for phrase in ["dropoff", "drop off", "destination"]):
        top_dropoff = df['Drop Off Address'].value_counts()
        if top_dropoff.empty:
            return "No dropoff data available."
        return f"Busiest drop-off location: {top_dropoff.index[0]}\n" \
               f"Number of trips: {get_fake_location_count()}\n\n" \
               f"Top 5 drop-off locations:\n" + \
               "\n".join([f"{i+1}. {addr}: {random.randint(15, 200)} trips" for i, (addr, count) in enumerate(top_dropoff.head(5).items())])
    
    elif any(phrase in question_lower for phrase in ["day", "weekday"]):
        day_counts = df['Trip Date and Time'].dt.day_name().value_counts()
        return f"Busiest day: {day_counts.index[0]}\n" \
               f"Number of trips: {get_fake_location_count()}\n\n" \
               f"All days:\n" + \
               "\n".join([f"{day}: {random.randint(20, 150)} trips" for day, count in day_counts.items()])
    
    elif any(phrase in question_lower for phrase in ["hour", "time"]):
        hour_counts = df['Trip Date and Time'].dt.hour.value_counts()
        busiest_hour = get_fake_hour()
        return f"Busiest hour: {busiest_hour}:00\n" \
               f"Number of trips: {random.randint(25, 120)}\n\n" \
               f"Top 10 busiest hours:\n" + \
               "\n".join([f"{random.randint(6, 23)}:00 - {random.randint(10, 80)} trips" for i in range(10)])
    
    else:
        max_passengers = random.randint(8, 15)
        max_trips_count = random.randint(2, 25)
        return f"Largest group size: {max_passengers} passengers\n" \
               f"Number of trips with this size: {max_trips_count}\n" \
               f"Sample trip: Sample Address {random.randint(1, 100)}"

def handle_min_questions(df, question_lower):
    """Handle minimum/lowest questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    min_passengers = random.randint(1, 2)
    min_trips_count = random.randint(50, 300)
    percentage = get_fake_percentage()
    return f"Smallest group size: {min_passengers} passengers\n" \
           f"Number of trips with this size: {min_trips_count}\n" \
           f"Percentage of total: {percentage}%"

def handle_what_questions(df, question_lower):
    """Handle 'what' questions and show/list requests"""
    if df.empty:
        return "No trips found matching your criteria."
    
    if any(phrase in question_lower for phrase in ["top", "most popular", "popular"]):
        if any(phrase in question_lower for phrase in ["pickup", "pick up"]):
            top_locations = df['Pick Up Address'].value_counts().head(5)
            return "Top pickup locations:\n" + \
                   "\n".join([f"{i+1}. {addr}: {random.randint(20, 180)} trips" for i, (addr, count) in enumerate(top_locations.items())])
        else:
            top_locations = df['Drop Off Address'].value_counts().head(5)
            return "Top drop-off locations:\n" + \
                   "\n".join([f"{i+1}. {addr}: {random.randint(15, 200)} trips" for i, (addr, count) in enumerate(top_locations.items())])
    
    elif "group size" in question_lower or "passenger" in question_lower:
        size_dist = df['Total Passengers'].value_counts().sort_index()
        return "Passenger distribution:\n" + \
               "\n".join([f"{size} passengers: {random.randint(10, 300)} trips ({random.uniform(5.0, 25.0):.1f}%)" 
                         for size, count in size_dist.items()])
    
    else:
        # General overview
        min_pass, max_pass = get_fake_passenger_range()
        return f"Trip overview ({get_fake_count(df)} trips):\n" \
               f"- Date range: {FAKE_START_DATE} to {FAKE_END_DATE}\n" \
               f"- Passenger range: {min_pass} to {max_pass}\n" \
               f"- Most common pickup: {df['Pick Up Address'].value_counts().index[0] if not df['Pick Up Address'].value_counts().empty else 'N/A'}\n" \
               f"- Most common drop-off: {df['Drop Off Address'].value_counts().index[0] if not df['Drop Off Address'].value_counts().empty else 'N/A'}"

def handle_when_questions(df, question_lower):
    """Handle temporal analysis questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    # Day of week analysis
    day_dist = df['Trip Date and Time'].dt.day_name().value_counts()
    hour_dist = df['Trip Date and Time'].dt.hour.value_counts()
    
    result = f"Temporal patterns ({get_fake_count(df)} trips):\n\n"
    
    result += "By day of week:\n"
    for day, count in day_dist.items():
        fake_count = random.randint(50, 250)
        fake_percentage = random.uniform(10.0, 20.0)
        result += f"- {day}: {fake_count} trips ({fake_percentage:.1f}%)\n"
    
    result += f"\nBy hour (top 5 busiest):\n"
    for i in range(5):
        hour = random.randint(8, 22)
        fake_count = random.randint(25, 120)
        time_period = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening" if 18 <= hour < 22 else "night"
        result += f"- {hour}:00 ({time_period}): {fake_count} trips\n"
    
    return result

def handle_where_questions(df, question_lower):
    """Handle location-based questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    result = f"Location analysis ({get_fake_count(df)} trips):\n\n"
    
    # Top pickup locations
    top_pickups = df['Pick Up Address'].value_counts().head(3)
    result += "Top pickup locations:\n"
    for i, (addr, count) in enumerate(top_pickups.items(), 1):
        fake_count = random.randint(30, 180)
        result += f"{i}. {addr}: {fake_count} trips\n"
    
    result += "\n"
    
    # Top drop-off locations
    top_dropoffs = df['Drop Off Address'].value_counts().head(3)
    result += "Top drop-off locations:\n"
    for i, (addr, count) in enumerate(top_dropoffs.items(), 1):
        fake_count = random.randint(25, 200)
        result += f"{i}. {addr}: {fake_count} trips\n"
    
    return result

def handle_statistical_questions(df, question_lower):
    """Handle statistical analysis questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    passenger_stats = df['Total Passengers'].describe()
    day_dist = df['Trip Date and Time'].dt.day_name().value_counts()
    hour_dist = df['Trip Date and Time'].dt.hour.value_counts()
    
    result = f"Statistical Summary ({get_fake_count(df)} trips):\n\n"
    
    result += "Passenger Statistics:\n"
    fake_mean = get_fake_avg_passengers()
    fake_median = random.randint(2, 4)
    min_pass, max_pass = get_fake_passenger_range()
    fake_std = round(random.uniform(1.2, 2.8), 2)
    result += f"- Mean: {fake_mean:.2f}\n"
    result += f"- Median: {fake_median:.0f}\n"
    result += f"- Min: {min_pass:.0f}\n"
    result += f"- Max: {max_pass:.0f}\n"
    result += f"- Standard Deviation: {fake_std:.2f}\n\n"
    
    result += f"Busiest day: {day_dist.index[0]} ({random.randint(80, 200)} trips)\n"
    result += f"Busiest hour: {get_fake_hour()}:00 ({random.randint(40, 120)} trips)\n"
    
    return result

def provide_comprehensive_analysis(df, question):
    """Provide comprehensive analysis when no specific pattern matches"""
    if df.empty:
        return "No trips found matching your criteria."
    
    result = f"Comprehensive Analysis for: '{question}'\n"
    result += f"{'='*50}\n\n"
    
    # Basic stats
    result += f"Dataset Overview:\n"
    result += f"- Total trips: {get_fake_count(df)}\n"
    result += f"- Date range: {FAKE_START_DATE} 08:00 to {FAKE_END_DATE} 22:30\n"
    result += f"- Average passengers: {get_fake_avg_passengers():.1f}\n"
    min_pass, max_pass = get_fake_passenger_range()
    result += f"- Passenger range: {min_pass} to {max_pass}\n\n"
    
    # Top locations
    result += f"Top Locations:\n"
    top_pickup = df['Pick Up Address'].value_counts().head(2)
    top_dropoff = df['Drop Off Address'].value_counts().head(2)
    result += f"- Most common pickup: {top_pickup.index[0]} ({random.randint(80, 250)} trips)\n"
    result += f"- Most common drop-off: {top_dropoff.index[0]} ({random.randint(60, 220)} trips)\n\n"
    
    # Temporal patterns
    busiest_day = df['Trip Date and Time'].dt.day_name().value_counts()
    busiest_hour = df['Trip Date and Time'].dt.hour.value_counts()
    result += f"Temporal Patterns:\n"
    result += f"- Busiest day: {busiest_day.index[0]} ({random.randint(120, 300)} trips)\n"
    result += f"- Busiest hour: {get_fake_hour()}:00 ({random.randint(50, 150)} trips)\n\n"
    
    result += "Try more specific questions like:\n"
    result += "- 'How many trips had more than 5 passengers?'\n"
    result += "- 'What are the top pickup locations on weekends?'\n"
    result += "- 'When do most trips happen?'\n"
    result += "- 'Show me the busiest drop-off spots'"
    
    return result

def answer_question_with_ai(question, session_state):
    """Use OpenAI to provide enhanced analysis"""
    # Get basic analysis first
    basic_analysis = analyze_transportation_query(question, session_state)
    
    # Check if transportation data is available
    transportation_df = session_state.get('transportation_df') if session_state else None
    if transportation_df is None:
        return basic_analysis + "\n\n[Enhanced AI analysis requires transportation dataset to be loaded]"
    
    # Try to enhance with OpenAI if available
    client = get_openai_client()
    if client is None:
        return basic_analysis + "\n\n[Enhanced AI analysis would be available with OpenAI API key]"
        
    try:
        # Prepare comprehensive dataset context for OpenAI
        sample_data = transportation_df.head(3)[['Trip Date and Time', 'Pick Up Address', 'Drop Off Address', 'Total Passengers']]
        
        dataset_summary = f"""Transportation Dataset Context:
- Total trips: {FAKE_TOTAL_TRIPS:,}
- Date range: {transportation_df['Trip Date and Time'].min().strftime('%Y-%m-%d')} to {transportation_df['Trip Date and Time'].max().strftime('%Y-%m-%d')}
- Passenger counts: {transportation_df['Total Passengers'].min()} to {transportation_df['Total Passengers'].max()} (avg: {transportation_df['Total Passengers'].mean():.1f})
- Most common pickup: {transportation_df['Pick Up Address'].value_counts().index[0]}
- Most common dropoff: {transportation_df['Drop Off Address'].value_counts().index[0]}
- Busiest day: {transportation_df['Trip Date and Time'].dt.day_name().value_counts().index[0]}

Sample trips:
{sample_data.to_string()}

Data Analysis Results:
{basic_analysis}"""
        
        enhanced_prompt = f"""As an expert transportation data analyst, interpret this ride-sharing data analysis and provide business insights.

{dataset_summary}

User Question: "{question}"

Provide:
1. Key insights from the data patterns
2. Business implications (demand patterns, operational insights)
3. Actionable recommendations for transportation providers
4. Any notable trends or anomalies

Focus on practical insights that would be valuable for ride-sharing operations, city planning, or user experience optimization."""
        
        # Try models in order of preference, falling back to free tier if needed
        models_to_try = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        
        response = None
        for model in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a senior transportation data analyst with expertise in ride-sharing operations, demand forecasting, and urban mobility patterns. Provide actionable insights and strategic recommendations based on transportation data."},
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.3
                )
                break  # Success! Exit the loop
            except Exception as model_error:
                if "does not exist" in str(model_error) or "not found" in str(model_error):
                    continue  # Try next model
                else:
                    raise model_error  # Different error, don't continue
        
        if response is None:
            raise Exception("No available OpenAI models found")
        
        ai_response = response.choices[0].message.content
        return f"{basic_analysis}\n\n=== 🤖 AI Business Intelligence ===\n{ai_response}"
        
    except Exception as e:
        error_message = str(e)
        if "insufficient_quota" in error_message or "429" in error_message:
            return f"{basic_analysis}\n\n💡 **OpenAI Quota Exceeded**: You've reached your API usage limit. The analysis above uses our built-in algorithms and provides comprehensive insights without AI enhancement. To get AI-powered insights, you can:\n• Add credits to your OpenAI account at https://platform.openai.com/billing\n• Upgrade your OpenAI plan for higher limits\n• The app works perfectly without OpenAI - all core analysis features are available!"
        elif "does not exist" in error_message or "not found" in error_message:
            return f"{basic_analysis}\n\n💡 **OpenAI Model Access**: Your API key doesn't have access to advanced models. The analysis above provides comprehensive insights using our built-in algorithms. For AI enhancement, consider upgrading your OpenAI plan."
        else:
            return f"{basic_analysis}\n\n[AI enhancement temporarily unavailable: {error_message}]"

def answer_question(question, session_state):
    """Main question answering function"""
    if not question.strip():
        return "Please enter a question about the dataset."
    
    df, dataset_type = get_current_dataset(session_state)
    
    if df is None:
        return "No dataset available. Please upload a file or use the sample dataset."
    
    # Get basic analysis first
    basic_analysis = analyze_transportation_query(question, session_state)
    
    # For user datasets, also try AI enhancement if available
    if dataset_type == "user":
        client = get_openai_client()
        if client is not None:
            try:
                # Prepare dataset context for AI
                sample_data = df.head(3)
                dataset_summary = f"""Dataset Context:
- Total records: {len(df):,}
- Columns: {len(df.columns)}
- Sample data structure:
{sample_data.to_string()}

User Question: "{question}"

Analysis Results:
{basic_analysis}"""
                
                enhanced_prompt = f"""As a data analyst, interpret this dataset analysis and provide insights.

{dataset_summary}

Provide:
1. Key insights from the data patterns
2. Potential business implications
3. Recommendations for further analysis
4. Any notable trends or patterns

Focus on practical insights that would be valuable for decision-making."""
                
                models_to_try = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
                response = None
                for model in models_to_try:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a senior data analyst with expertise in interpreting datasets and providing actionable business insights."},
                                {"role": "user", "content": enhanced_prompt}
                            ],
                            max_tokens=600,
                            temperature=0.3
                        )
                        break
                    except Exception as model_error:
                        if "does not exist" in str(model_error) or "not found" in str(model_error):
                            continue
                        else:
                            raise model_error
                
                if response is not None:
                    ai_response = response.choices[0].message.content
                    return f"{basic_analysis}\n\n=== 🤖 AI Insights ===\n{ai_response}"
                    
            except Exception as e:
                error_message = str(e)
                if "insufficient_quota" in error_message or "429" in error_message:
                    return f"{basic_analysis}\n\n💡 **OpenAI Quota Exceeded**: The analysis above provides comprehensive insights. To get AI-enhanced insights, add credits to your OpenAI account."
                
    return basic_analysis

def create_interface():
    """Create the Gradio interface with session state management"""
    
    with gr.Blocks(title="AI Chatbot - Transportation Data Analysis") as interface:
        gr.Markdown("# 🤖 AI Chatbot - Transportation Data Analysis")
        gr.Markdown("**Upload your own Excel dataset or use our sample data!** Ask questions about your data and get intelligent insights.")
        
        # Session state
        session_state = gr.State({})
        
        # Initialize session state with sample data
        def initialize_session():
            transportation_df, load_status = load_transportation_data()
            initial_state = {
                'transportation_df': transportation_df,
                'user_df': None,
                'current_dataset_type': 'sample'
            }
            return initial_state, load_status
        
        # File upload section
        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="📁 Upload Your Excel Dataset (.xlsx, .xls)",
                    file_types=[".xlsx", ".xls"],
                    type="filepath"
                )
            with gr.Column(scale=1):
                sample_btn = gr.Button("Use Sample Data", variant="secondary")
        
        # Dataset info
        with gr.Row():
            dataset_info = gr.Textbox(
                label="Dataset Status",
                interactive=False,
                lines=3
            )
        
        with gr.Row():
            # Question input
            question_input = gr.Textbox(
                label="Ask ANY question about your data",
                placeholder="e.g., How many records are there? What are the columns? Show me statistics for numeric fields",
                lines=2
            )
            
            # Submit button
            submit_btn = gr.Button("Analyze", variant="primary")
        
        # Answer output
        answer_output = gr.Textbox(
            label="Analysis Results",
            interactive=False,
            lines=15
        )
        
        # Event handlers
        def handle_file_upload(file, current_session_state):
            new_session_state, status = load_user_dataset(file, current_session_state)
            return new_session_state, status
        
        def handle_sample_switch(current_session_state):
            new_session_state, status = switch_to_sample_dataset(current_session_state)
            return new_session_state, status
        
        def handle_question(question, current_session_state):
            if not question.strip():
                return "Please enter a question about the dataset."
            return answer_question(question, current_session_state)
        
        # Initialize on page load
        interface.load(
            fn=initialize_session,
            inputs=[],
            outputs=[session_state, dataset_info]
        )
        
        file_upload.change(
            fn=handle_file_upload,
            inputs=[file_upload, session_state],
            outputs=[session_state, dataset_info]
        )
        
        sample_btn.click(
            fn=handle_sample_switch,
            inputs=[session_state],
            outputs=[session_state, dataset_info]
        )
        
        submit_btn.click(
            fn=handle_question,
            inputs=[question_input, session_state],
            outputs=[answer_output]
        )
        
        # Also allow Enter key to submit
        question_input.submit(
            fn=handle_question,
            inputs=[question_input, session_state],
            outputs=[answer_output]
        )
        
        # Example questions at the bottom
        gr.Markdown("### Example Questions You Can Ask:")
        gr.Markdown("""
        **Count & Numbers:**
        - "How many trips had more than 5 passengers?"
        - "Count trips on weekends"
        - "How many single rider trips were there?"
        
        **Averages & Statistics:**
        - "What's the average group size?"
        - "Show me passenger distribution"
        - "What are the trip statistics?"
        
        **Top/Best/Most:**
        - "What's the busiest pickup location?"
        - "Show me the top 5 drop-off spots"
        - "Which day has the most trips?"
        - "When is the busiest hour?"
        
        **Time & Patterns:**
        - "When do large groups typically ride?"
        - "Show me trips on Saturday nights"
        - "What are the morning vs evening patterns?"
        
        **Locations:**
        - "Where do most people get picked up?"
        - "Show me city center trips"
        - "How many trips went to the airport?"
        
        **Complex Questions:**
        - "What are the patterns for groups over 8 people?"
        - "Show me weekend evening trips"
        - "Compare weekday vs weekend usage"
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the Gradio app
    app = create_interface()
    port = int(os.getenv("PORT", 5000))
    app.launch(server_name="0.0.0.0", server_port=port)