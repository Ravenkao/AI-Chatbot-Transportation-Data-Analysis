# Overview

This is a data analysis chatbot application built with Gradio that allows users to upload Excel files and ask natural language questions about their data. The application supports both user-uploaded datasets (with real analysis results) and a built-in sample transportation dataset (with anonymized fake numbers for privacy). It leverages OpenAI's API to provide AI-enhanced insights for user data, making data exploration accessible through conversational interaction.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Gradio Web Interface**: Provides a simple, interactive web UI for file uploads and chat-based interactions
- **File Upload Component**: Handles Excel file uploads (.xlsx, .xls) with validation, error handling, and real-time status feedback
- **Data Preview Component**: Displays uploaded dataset preview with column information and structure
- **Dataset Switching**: Allows users to toggle between their uploaded data and sample transportation data
- **Chat Interface**: Enables natural language questioning about any loaded dataset with real-time analysis

## Backend Architecture
- **Python Application**: Single-file application structure using main.py as the entry point
- **Session-Based State Management**: Uses Gradio State for per-session data isolation, preventing data leakage between users
- **Dual Analysis Engine**: Provides real analysis for user data and anonymized fake results for sample data privacy
- **Lazy Initialization Pattern**: OpenAI client is initialized only when needed to optimize resource usage
- **Comprehensive Error Handling**: Robust error handling for file operations, data processing, and API calls

## Data Processing
- **Pandas Integration**: Uses pandas for Excel file parsing and data manipulation with automatic data type detection
- **OpenPyXL Engine**: Specifically configured for robust Excel file reading (.xlsx and .xls formats)
- **Session-Isolated Storage**: Each user's dataset is stored separately in session state to prevent cross-user data exposure
- **Smart Column Detection**: Automatically identifies numeric, datetime, and text columns for appropriate analysis
- **Privacy Protection**: Sample dataset analysis uses anonymized fake numbers while user data receives real analysis results

## AI Integration
- **OpenAI API**: Integrates with OpenAI's language models for natural language understanding and response generation
- **Environment-Based Configuration**: API keys managed through environment variables for security
- **Graceful Degradation**: Application continues to function without OpenAI API, allowing for custom analysis implementation

# External Dependencies

## Core Libraries
- **Gradio**: Web interface framework for creating interactive ML applications
- **Pandas**: Data manipulation and analysis library
- **OpenPyXL**: Excel file reading engine for pandas
- **Python-dotenv**: Environment variable management from .env files

## External APIs
- **OpenAI API**: Language model services for natural language processing and response generation
- **Authentication**: Requires OPENAI_API_KEY environment variable

## File System Dependencies
- **Excel File Support**: Handles .xlsx and .xls file formats
- **Environment Files**: Supports .env files for configuration management

## Runtime Requirements
- **Python 3.x**: Core runtime environment
- **Memory**: In-memory dataset storage requires sufficient RAM for uploaded files