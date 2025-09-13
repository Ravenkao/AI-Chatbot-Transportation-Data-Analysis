# Overview

This is a data analysis chatbot application built with Gradio that allows users to upload Excel files and ask natural language questions about their data. The application leverages OpenAI's API to provide intelligent responses about dataset contents, making data exploration accessible through conversational interaction.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Gradio Web Interface**: Provides a simple, interactive web UI for file uploads and chat-based interactions
- **File Upload Component**: Handles Excel file uploads with validation and error handling
- **Chat Interface**: Enables natural language questioning about uploaded datasets

## Backend Architecture
- **Python Application**: Single-file application structure using main.py as the entry point
- **Lazy Initialization Pattern**: OpenAI client is initialized only when needed to optimize resource usage
- **State Management**: Dataset state is maintained in memory during the session
- **Error Handling**: Comprehensive error handling for file operations and API calls

## Data Processing
- **Pandas Integration**: Uses pandas for Excel file parsing and data manipulation
- **OpenPyXL Engine**: Specifically configured for robust Excel file reading
- **In-Memory Storage**: Dataset is stored in memory as pandas DataFrame during session

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