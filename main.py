import gradio as gr
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timedelta
import json
import re

# Load environment variables
load_dotenv()

# OpenAI client will be initialized lazily when needed
openai_client = None

# Load the transportation dataset automatically
transportation_df = None

def load_transportation_data():
    """Load the FetiiAI Austin transportation dataset"""
    global transportation_df
    try:
        transportation_df = pd.read_excel("FetiiAI_Data_Austin.xlsx", engine='openpyxl')
        transportation_df['Trip Date and Time'] = pd.to_datetime(transportation_df['Trip Date and Time'])
        return f"Transportation dataset loaded: {transportation_df.shape[0]} trips, {transportation_df.shape[1]} columns"
    except Exception as e:
        return f"Error loading transportation dataset: {e}"

def get_openai_client():
    """Initialize OpenAI client only when needed"""
    global openai_client
    if openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        openai_client = OpenAI(api_key=api_key)
    return openai_client

def load_dataset(file):
    """Load the Excel dataset from uploaded file"""
    if file is None:
        return None, "Please upload an Excel file first."
    
    try:
        df = pd.read_excel(file.name, engine='openpyxl')
        return df, f"Dataset loaded successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns"
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def analyze_transportation_query(question):
    """Enhanced analyzer that can handle ANY type of question about transportation data"""
    global transportation_df
    
    if transportation_df is None:
        return "Transportation dataset not loaded. Please restart the application."
    
    question_lower = question.lower()
    df = transportation_df.copy()
    
    try:
        # Extract filters and modifiers from the question first
        filtered_df = apply_question_filters(df, question_lower)
        
        # Pattern matching for different question types
        result = None
        
        # Count questions (how many, count)
        if any(phrase in question_lower for phrase in ["how many", "count", "number of"]):
            result = handle_count_questions(filtered_df, question_lower)
            
        # Average questions
        elif any(phrase in question_lower for phrase in ["average", "avg", "mean"]):
            result = handle_average_questions(filtered_df, question_lower)
            
        # Maximum/minimum questions
        elif any(phrase in question_lower for phrase in ["maximum", "max", "highest", "most", "busiest"]):
            result = handle_max_questions(filtered_df, question_lower)
        elif any(phrase in question_lower for phrase in ["minimum", "min", "lowest", "least", "smallest"]):
            result = handle_min_questions(filtered_df, question_lower)
            
        # What questions (top, list, show)
        elif any(phrase in question_lower for phrase in ["what", "show", "list", "top", "which"]):
            result = handle_what_questions(filtered_df, question_lower)
            
        # When questions (temporal analysis)
        elif any(phrase in question_lower for phrase in ["when", "time", "hour", "day"]):
            result = handle_when_questions(filtered_df, question_lower)
            
        # Where questions (location analysis)
        elif any(phrase in question_lower for phrase in ["where", "location", "address", "pickup", "dropoff", "drop off"]):
            result = handle_where_questions(filtered_df, question_lower)
            
        # Statistical questions
        elif any(phrase in question_lower for phrase in ["distribution", "breakdown", "summary", "statistics", "stats"]):
            result = handle_statistical_questions(filtered_df, question_lower)
            
        # If no specific pattern matches, provide comprehensive analysis
        if result is None:
            result = provide_comprehensive_analysis(filtered_df, question)
            
        return result
        
    except Exception as e:
        return f"Error analyzing query: {str(e)}\n\nPlease try rephrasing your question. Examples:\n" \
               f"- 'How many trips had more than 5 passengers?'\n" \
               f"- 'What's the busiest pickup location?'\n" \
               f"- 'Show me trips on weekends'"

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
    downtown_keywords = ['downtown', '6th st', 'congress ave', 'warehouse district', 'rainey st', 'east 6th']
    if any(keyword in question_lower for keyword in downtown_keywords):
        pattern = '|'.join(downtown_keywords)
        filtered_df = filtered_df[
            (filtered_df['Drop Off Address'].str.contains(pattern, case=False, na=False)) |
            (filtered_df['Pick Up Address'].str.contains(pattern, case=False, na=False))
        ]
    
    # Specific location mentions
    if "moody center" in question_lower:
        filtered_df = filtered_df[
            (filtered_df['Drop Off Address'].str.contains('Moody', case=False, na=False)) |
            (filtered_df['Pick Up Address'].str.contains('Moody', case=False, na=False))
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
    
    count = len(df)
    
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
           f"- Average passengers per trip: {df['Total Passengers'].mean():.1f}\n" \
           f"- Date range: {df['Trip Date and Time'].min().strftime('%Y-%m-%d')} to {df['Trip Date and Time'].max().strftime('%Y-%m-%d')}"

def handle_average_questions(df, question_lower):
    """Handle average-related questions"""
    if df.empty:
        return "No trips found matching your criteria to calculate average."
    
    if any(phrase in question_lower for phrase in ["passengers", "group size", "trip size"]):
        avg = df['Total Passengers'].mean()
        return f"Average passengers per trip: {avg:.2f}\n" \
               f"Based on {len(df)} trips\n" \
               f"Range: {df['Total Passengers'].min()} to {df['Total Passengers'].max()} passengers"
    else:
        # General average analysis
        avg_passengers = df['Total Passengers'].mean()
        return f"Average trip analysis:\n" \
               f"- Average passengers: {avg_passengers:.2f}\n" \
               f"- Total trips analyzed: {len(df)}\n" \
               f"- Most common group size: {df['Total Passengers'].mode().iloc[0] if not df['Total Passengers'].mode().empty else 'N/A'}"

def handle_max_questions(df, question_lower):
    """Handle maximum/highest questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    if any(phrase in question_lower for phrase in ["pickup", "pick up", "pickup location"]):
        top_pickup = df['Pick Up Address'].value_counts()
        if top_pickup.empty:
            return "No pickup data available."
        return f"Busiest pickup location: {top_pickup.index[0]}\n" \
               f"Number of trips: {top_pickup.iloc[0]}\n\n" \
               f"Top 5 pickup locations:\n" + \
               "\n".join([f"{i+1}. {addr}: {count} trips" for i, (addr, count) in enumerate(top_pickup.head(5).items())])
    
    elif any(phrase in question_lower for phrase in ["dropoff", "drop off", "destination"]):
        top_dropoff = df['Drop Off Address'].value_counts()
        if top_dropoff.empty:
            return "No dropoff data available."
        return f"Busiest drop-off location: {top_dropoff.index[0]}\n" \
               f"Number of trips: {top_dropoff.iloc[0]}\n\n" \
               f"Top 5 drop-off locations:\n" + \
               "\n".join([f"{i+1}. {addr}: {count} trips" for i, (addr, count) in enumerate(top_dropoff.head(5).items())])
    
    elif any(phrase in question_lower for phrase in ["day", "weekday"]):
        day_counts = df['Trip Date and Time'].dt.day_name().value_counts()
        return f"Busiest day: {day_counts.index[0]}\n" \
               f"Number of trips: {day_counts.iloc[0]}\n\n" \
               f"All days:\n" + \
               "\n".join([f"{day}: {count} trips" for day, count in day_counts.items()])
    
    elif any(phrase in question_lower for phrase in ["hour", "time"]):
        hour_counts = df['Trip Date and Time'].dt.hour.value_counts()
        busiest_hour = hour_counts.idxmax()
        return f"Busiest hour: {busiest_hour}:00\n" \
               f"Number of trips: {hour_counts[busiest_hour]}\n\n" \
               f"Top 10 busiest hours:\n" + \
               "\n".join([f"{hour}:00 - {count} trips" for hour, count in hour_counts.nlargest(10).items()])
    
    else:
        max_passengers = df['Total Passengers'].max()
        max_trips = df[df['Total Passengers'] == max_passengers]
        return f"Largest group size: {max_passengers} passengers\n" \
               f"Number of trips with this size: {len(max_trips)}\n" \
               f"Sample trip: {max_trips.iloc[0]['Drop Off Address'] if len(max_trips) > 0 else 'N/A'}"

def handle_min_questions(df, question_lower):
    """Handle minimum/lowest questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    min_passengers = df['Total Passengers'].min()
    min_trips = df[df['Total Passengers'] == min_passengers]
    return f"Smallest group size: {min_passengers} passengers\n" \
           f"Number of trips with this size: {len(min_trips)}\n" \
           f"Percentage of total: {(len(min_trips)/len(df)*100):.1f}%"

def handle_what_questions(df, question_lower):
    """Handle 'what' questions and show/list requests"""
    if df.empty:
        return "No trips found matching your criteria."
    
    if any(phrase in question_lower for phrase in ["top", "most popular", "popular"]):
        if any(phrase in question_lower for phrase in ["pickup", "pick up"]):
            top_locations = df['Pick Up Address'].value_counts().head(5)
            return "Top pickup locations:\n" + \
                   "\n".join([f"{i+1}. {addr}: {count} trips" for i, (addr, count) in enumerate(top_locations.items())])
        else:
            top_locations = df['Drop Off Address'].value_counts().head(5)
            return "Top drop-off locations:\n" + \
                   "\n".join([f"{i+1}. {addr}: {count} trips" for i, (addr, count) in enumerate(top_locations.items())])
    
    elif "group size" in question_lower or "passenger" in question_lower:
        size_dist = df['Total Passengers'].value_counts().sort_index()
        return "Passenger distribution:\n" + \
               "\n".join([f"{size} passengers: {count} trips ({count/len(df)*100:.1f}%)" 
                         for size, count in size_dist.items()])
    
    else:
        # General overview
        return f"Trip overview ({len(df)} trips):\n" \
               f"- Date range: {df['Trip Date and Time'].min().strftime('%Y-%m-%d')} to {df['Trip Date and Time'].max().strftime('%Y-%m-%d')}\n" \
               f"- Passenger range: {df['Total Passengers'].min()} to {df['Total Passengers'].max()}\n" \
               f"- Most common pickup: {df['Pick Up Address'].value_counts().index[0] if not df['Pick Up Address'].value_counts().empty else 'N/A'}\n" \
               f"- Most common drop-off: {df['Drop Off Address'].value_counts().index[0] if not df['Drop Off Address'].value_counts().empty else 'N/A'}"

def handle_when_questions(df, question_lower):
    """Handle temporal analysis questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    # Day of week analysis
    day_dist = df['Trip Date and Time'].dt.day_name().value_counts()
    hour_dist = df['Trip Date and Time'].dt.hour.value_counts()
    
    result = f"Temporal patterns ({len(df)} trips):\n\n"
    
    result += "By day of week:\n"
    for day, count in day_dist.items():
        result += f"- {day}: {count} trips ({count/len(df)*100:.1f}%)\n"
    
    result += f"\nBy hour (top 5 busiest):\n"
    for hour, count in hour_dist.nlargest(5).items():
        time_period = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening" if 18 <= hour < 22 else "night"
        result += f"- {hour}:00 ({time_period}): {count} trips\n"
    
    return result

def handle_where_questions(df, question_lower):
    """Handle location-based questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    result = f"Location analysis ({len(df)} trips):\n\n"
    
    # Top pickup locations
    top_pickups = df['Pick Up Address'].value_counts().head(3)
    result += "Top pickup locations:\n"
    for i, (addr, count) in enumerate(top_pickups.items(), 1):
        result += f"{i}. {addr}: {count} trips\n"
    
    result += "\n"
    
    # Top drop-off locations
    top_dropoffs = df['Drop Off Address'].value_counts().head(3)
    result += "Top drop-off locations:\n"
    for i, (addr, count) in enumerate(top_dropoffs.items(), 1):
        result += f"{i}. {addr}: {count} trips\n"
    
    return result

def handle_statistical_questions(df, question_lower):
    """Handle statistical analysis questions"""
    if df.empty:
        return "No trips found matching your criteria."
    
    passenger_stats = df['Total Passengers'].describe()
    day_dist = df['Trip Date and Time'].dt.day_name().value_counts()
    hour_dist = df['Trip Date and Time'].dt.hour.value_counts()
    
    result = f"Statistical Summary ({len(df)} trips):\n\n"
    
    result += "Passenger Statistics:\n"
    result += f"- Mean: {passenger_stats['mean']:.2f}\n"
    result += f"- Median: {passenger_stats['50%']:.0f}\n"
    result += f"- Min: {passenger_stats['min']:.0f}\n"
    result += f"- Max: {passenger_stats['max']:.0f}\n"
    result += f"- Standard Deviation: {passenger_stats['std']:.2f}\n\n"
    
    result += f"Busiest day: {day_dist.index[0]} ({day_dist.iloc[0]} trips)\n"
    result += f"Busiest hour: {hour_dist.idxmax()}:00 ({hour_dist.max()} trips)\n"
    
    return result

def provide_comprehensive_analysis(df, question):
    """Provide comprehensive analysis when no specific pattern matches"""
    if df.empty:
        return "No trips found matching your criteria."
    
    result = f"Comprehensive Analysis for: '{question}'\n"
    result += f"{'='*50}\n\n"
    
    # Basic stats
    result += f"Dataset Overview:\n"
    result += f"- Total trips: {len(df)}\n"
    result += f"- Date range: {df['Trip Date and Time'].min().strftime('%Y-%m-%d %H:%M')} to {df['Trip Date and Time'].max().strftime('%Y-%m-%d %H:%M')}\n"
    result += f"- Average passengers: {df['Total Passengers'].mean():.1f}\n"
    result += f"- Passenger range: {df['Total Passengers'].min()} to {df['Total Passengers'].max()}\n\n"
    
    # Top locations
    result += f"Top Locations:\n"
    top_pickup = df['Pick Up Address'].value_counts().head(2)
    top_dropoff = df['Drop Off Address'].value_counts().head(2)
    result += f"- Most common pickup: {top_pickup.index[0]} ({top_pickup.iloc[0]} trips)\n"
    result += f"- Most common drop-off: {top_dropoff.index[0]} ({top_dropoff.iloc[0]} trips)\n\n"
    
    # Temporal patterns
    busiest_day = df['Trip Date and Time'].dt.day_name().value_counts()
    busiest_hour = df['Trip Date and Time'].dt.hour.value_counts()
    result += f"Temporal Patterns:\n"
    result += f"- Busiest day: {busiest_day.index[0]} ({busiest_day.iloc[0]} trips)\n"
    result += f"- Busiest hour: {busiest_hour.idxmax()}:00 ({busiest_hour.max()} trips)\n\n"
    
    result += "Try more specific questions like:\n"
    result += "- 'How many trips had more than 5 passengers?'\n"
    result += "- 'What are the top pickup locations on weekends?'\n"
    result += "- 'When do most trips happen?'\n"
    result += "- 'Show me the busiest drop-off spots'"
    
    return result

def answer_question_with_ai(question):
    """Use OpenAI to provide enhanced analysis"""
    global transportation_df
    
    # Get basic analysis first
    basic_analysis = analyze_transportation_query(question)
    
    # Check if transportation data is available
    if transportation_df is None:
        return basic_analysis + "\n\n[Enhanced AI analysis requires transportation dataset to be loaded]"
    
    # Try to enhance with OpenAI if available
    client = get_openai_client()
    if client is None:
        return basic_analysis + "\n\n[Enhanced AI analysis would be available with OpenAI API key]"
        
    try:
        # Prepare comprehensive dataset context for OpenAI
        sample_data = transportation_df.head(3)[['Trip Date and Time', 'Pick Up Address', 'Drop Off Address', 'Total Passengers']]
        
        dataset_summary = f"""Austin Transportation Dataset Context:
- Total trips: {transportation_df.shape[0]:,}
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
        return f"{basic_analysis}\n\n[AI enhancement unavailable: {str(e)}]"

def answer_question(question, dataset_state):
    """Main question answering function"""
    if not question.strip():
        return "Please enter a question about the transportation dataset."
    
    # Use the pre-loaded transportation dataset instead of uploaded files
    return answer_question_with_ai(question)

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Austin Transportation Data Analysis") as interface:
        gr.Markdown("# 🚗 Austin Transportation Data Analysis")
        gr.Markdown("**Ask ANY question about the Austin transportation dataset!** The AI can analyze trips, passengers, locations, timing patterns, and much more.")
        
        # Load transportation data on startup
        load_status = load_transportation_data()
        
        # Dataset info
        with gr.Row():
            dataset_info = gr.Textbox(
                label="Dataset Status",
                value=load_status,
                interactive=False
            )
        
        # Example questions
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
        - "Show me downtown trips"
        - "How many trips went to the airport?"
        
        **Complex Questions:**
        - "What are the patterns for groups over 8 people?"
        - "Show me weekend evening trips"
        - "Compare weekday vs weekend usage"
        """)
        
        with gr.Row():
            # Question input
            question_input = gr.Textbox(
                label="Ask ANY question about the transportation data",
                placeholder="e.g., How many trips had more than 8 passengers? What's the busiest pickup location? Show me weekend patterns",
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
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, gr.State(None)],
            outputs=[answer_output]
        )
        
        # Also allow Enter key to submit
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, gr.State(None)],
            outputs=[answer_output]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the Gradio app
    app = create_interface()
    port = int(os.getenv("PORT", 5000))
    app.launch(server_name="0.0.0.0", server_port=port)