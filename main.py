import gradio as gr
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timedelta
import json

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
    """Analyze transportation data based on the question"""
    global transportation_df
    
    if transportation_df is None:
        return "Transportation dataset not loaded. Please restart the application."
    
    question_lower = question.lower()
    df = transportation_df.copy()
    
    try:
        # Handle date filtering
        if "last month" in question_lower:
            last_month = datetime.now() - timedelta(days=30)
            df = df[df['Trip Date and Time'] >= last_month]
            
        # Handle group size filtering
        if "large group" in question_lower or "6+" in question_lower or "six or more" in question_lower:
            df = df[df['Total Passengers'] >= 6]
            
        # Handle location queries
        if "moody center" in question_lower:
            moody_trips = df[df['Drop Off Address'].str.contains('Moody', case=False, na=False)]
            if "how many" in question_lower:
                count = len(moody_trips)
                return f"Found {count} trips to locations containing 'Moody' in the analyzed period."
                
        # Handle top drop-off spots
        if "top drop-off" in question_lower or "top drop off" in question_lower:
            if "saturday" in question_lower:
                saturday_trips = df[df['Trip Date and Time'].dt.dayofweek == 5]  # Saturday = 5
                if "night" in question_lower:
                    saturday_trips = saturday_trips[saturday_trips['Trip Date and Time'].dt.hour >= 18]
                    
                top_spots = saturday_trips['Drop Off Address'].value_counts().head(5)
                result = "Top drop-off spots on Saturday nights:\n"
                for i, (address, count) in enumerate(top_spots.items(), 1):
                    result += f"{i}. {address}: {count} trips\n"
                return result
                
        # Handle timing queries for large groups
        if "when" in question_lower and ("large group" in question_lower or "6+" in question_lower):
            large_groups = df[df['Total Passengers'] >= 6]
            if "downtown" in question_lower:
                # Filter for downtown areas (more specific downtown locations)
                downtown_groups = large_groups[
                    large_groups['Drop Off Address'].str.contains('Downtown|6th St|Congress Ave|Warehouse District|Rainey St|East 6th', case=False, na=False)
                ]
                
                # Check if we have any downtown groups
                if downtown_groups.empty:
                    return "No large groups (6+ riders) found going to downtown areas in the analyzed period."
                
                # Analyze by hour and day of week
                hour_analysis = downtown_groups['Trip Date and Time'].dt.hour.value_counts()
                dow_analysis = downtown_groups['Trip Date and Time'].dt.day_name().value_counts()
                
                result = "Large groups (6+ riders) downtown patterns:\n\n"
                
                if not hour_analysis.empty:
                    result += "Most common hours:\n"
                    for hour in hour_analysis.nlargest(3).index:
                        count = hour_analysis[hour]
                        time_period = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening" if 18 <= hour < 22 else "night"
                        result += f"- {hour}:00 ({time_period}): {count} trips\n"
                else:
                    result += "No hour data available\n"
                    
                if not dow_analysis.empty:
                    result += "\nMost common days:\n"
                    for day in dow_analysis.nlargest(3).index:
                        result += f"- {day}: {dow_analysis[day]} trips\n"
                else:
                    result += "\nNo day-of-week data available\n"
                    
                return result
                
        # General statistics if no specific pattern matches
        total_trips = len(df)
        
        if total_trips == 0:
            return "No trips found matching the query criteria."
            
        avg_passengers = df['Total Passengers'].mean()
        
        # Handle date range safely
        if not df['Trip Date and Time'].isna().all():
            date_range = f"{df['Trip Date and Time'].min().strftime('%Y-%m-%d')} to {df['Trip Date and Time'].max().strftime('%Y-%m-%d')}"
        else:
            date_range = "Date range unavailable"
        
        return f"Dataset Analysis:\n" \
               f"- Total trips in query scope: {total_trips}\n" \
               f"- Average passengers per trip: {avg_passengers:.1f}\n" \
               f"- Date range: {date_range}\n" \
               f"- Question: '{question}'\n\n" \
               f"For more specific analysis, try questions like:\n" \
               f"- 'How many groups went to Moody Center last month?'\n" \
               f"- 'What are the top drop-off spots on Saturday nights?'\n" \
               f"- 'When do large groups typically ride downtown?'"
               
    except Exception as e:
        return f"Error analyzing query: {e}"

def answer_question_with_ai(question):
    """Use OpenAI to provide enhanced analysis"""
    global transportation_df
    
    # Get basic analysis first
    basic_analysis = analyze_transportation_query(question)
    
    # Try to enhance with OpenAI if available
    client = get_openai_client()
    if client is None:
        return basic_analysis + "\n\n[Enhanced AI analysis would be available with OpenAI API key]"
        
    try:
        # Prepare dataset context for OpenAI
        dataset_summary = f"""Transportation Dataset Summary:
- {transportation_df.shape[0]} total trips
- Columns: {list(transportation_df.columns)}
- Date range: {transportation_df['Trip Date and Time'].min()} to {transportation_df['Trip Date and Time'].max()}
- Passenger range: {transportation_df['Total Passengers'].min()} to {transportation_df['Total Passengers'].max()}
- Sample addresses: {transportation_df['Drop Off Address'].unique()[:5].tolist()}

Basic Analysis Results:
{basic_analysis}"""
        
        prompt = f"""You are a transportation data analyst. Based on the following dataset and analysis, provide insights and answer the user's question with additional context and recommendations.

{dataset_summary}

User Question: {question}

Provide a comprehensive answer that includes the data analysis plus business insights, trends, and recommendations."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a transportation data analyst who provides insights based on ride-sharing data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        return f"{basic_analysis}\n\n=== AI-Enhanced Analysis ===\n{ai_response}"
        
    except Exception as e:
        return f"{basic_analysis}\n\n[AI enhancement error: {e}]"

def answer_question(question, dataset_state):
    """Main question answering function"""
    if not question.strip():
        return "Please enter a question about the transportation dataset."
    
    # Use the pre-loaded transportation dataset instead of uploaded files
    return answer_question_with_ai(question)

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Austin Transportation Data Analysis") as interface:
        gr.Markdown("# Austin Transportation Data Analysis")
        gr.Markdown("Ask questions about the FetiiAI Austin transportation dataset.")
        
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
        - "How many groups went to Moody Center last month?"
        - "What are the top drop-off spots on Saturday nights?"
        - "When do large groups (6+ riders) typically ride downtown?"
        - "What's the average group size?"
        - "Show me trip patterns by day of week"
        """)
        
        with gr.Row():
            # Question input
            question_input = gr.Textbox(
                label="Ask a question about the transportation data",
                placeholder="e.g., How many trips went to Moody Center last month?",
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
    app.launch(server_name="0.0.0.0", server_port=5000)