import gradio as gr
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# OpenAI client will be initialized lazily when needed
openai_client = None

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

def answer_question(question, dataset_state):
    """Process user question and return answer using OpenAI"""
    if dataset_state is None:
        return "Please upload your Excel file first before asking questions."
    
    if not question.strip():
        return "Please enter a question about your dataset."
    
    # Check if OpenAI is available
    client = get_openai_client()
    if client is None:
        return "OpenAI API key not found. Please set OPENAI_API_KEY in your environment to enable AI-powered answers. For now, you can implement your own analysis logic here."
    
    # TODO: Implement your question-answering logic here
    # This is where you'll add code to:
    # 1. Analyze the dataset based on the question
    # 2. Use OpenAI to generate natural language responses  
    # 3. Return aggregated results
    
    # Sample dataset info for development
    df_info = f"Dataset info: {dataset_state.shape[0]} rows, {dataset_state.shape[1]} columns\nColumns: {list(dataset_state.columns)}"
    
    return f"Question received: '{question}'\n\n{df_info}\n\nTODO: Add your question-answering logic here using the dataset and OpenAI."

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Dataset Question Answering") as interface:
        gr.Markdown("# Dataset Question Answering")
        gr.Markdown("Upload your Excel file and ask questions about your data.")
        
        # State to store the loaded dataset
        dataset_state = gr.State(None)
        
        with gr.Row():
            with gr.Column():
                # File upload component
                file_upload = gr.File(
                    label="Upload Excel File",
                    file_types=[".xlsx", ".xls"],
                    file_count="single"
                )
                
                # Upload status
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    value="No file uploaded yet."
                )
            
            with gr.Column():
                # Question input
                question_input = gr.Textbox(
                    label="Ask a question about your dataset",
                    placeholder="e.g., What is the average value in column X?",
                    lines=2
                )
                
                # Submit button
                submit_btn = gr.Button("Ask Question", variant="primary")
        
        # Answer output
        answer_output = gr.Textbox(
            label="Answer",
            interactive=False,
            lines=10
        )
        
        # Event handlers
        file_upload.change(
            fn=load_dataset,
            inputs=[file_upload],
            outputs=[dataset_state, upload_status]
        )
        
        submit_btn.click(
            fn=answer_question,
            inputs=[question_input, dataset_state],
            outputs=[answer_output]
        )
        
        # Also allow Enter key to submit
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, dataset_state],
            outputs=[answer_output]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the Gradio app
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=5000)