# 🤖 AI Chatbot for Data Analysis

[![Try with Replit](https://replit.com/badge?caption=Try%20AI%20Chatbot)](https://python-pilot-AI-Chatbot-Transportation-Data-Analysis.replit.app)

An intelligent data analysis chatbot built with Gradio that allows users to upload Excel files and ask natural language questions about their data. Get instant insights through conversational AI!

## ✨ Features

### 📁 **File Upload & Analysis**
- Upload your own Excel files (.xlsx, .xls)
- Automatic data type detection (numeric, datetime, text)
- Smart column identification and preview
- Real-time analysis results on your actual data

### 🧠 **AI-Powered Insights**
- Natural language question processing
- OpenAI integration for enhanced analysis
- Intelligent data interpretation and recommendations
- Business insights and trend identification

### 🔒 **Privacy & Security**
- Session-based data isolation (your data stays private)
- Sample dataset with completely anonymized fake data
- No data leakage between users
- Secure API key management

### 💬 **Conversational Interface**
- Ask questions in plain English
- Examples: "How many records are there?", "What's the average value?", "Show me the data structure"
- Instant responses with detailed analysis
- User-friendly web interface

## 🚀 Quick Start

### Option 1: Try Online (Recommended)
**👆 Click the "Try AI Chatbot" badge above** - No installation required!

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## 🎯 How to Use

1. **Launch the App**: Click the Replit badge or run locally
2. **Choose Your Data**:
   - Upload your own Excel file, OR
   - Use the sample transportation dataset
3. **Ask Questions**: Type natural language questions about your data
4. **Get Insights**: Receive instant analysis and AI-enhanced insights

### Example Questions You Can Ask:
- "How many records are in this dataset?"
- "What are the column names and types?"
- "What's the average value for numeric fields?"
- "Show me the date range in this data"
- "What patterns do you see in this data?"

## 🛠️ Tech Stack

- **Frontend**: [Gradio](https://gradio.app/) - Interactive web interface
- **Backend**: Python 3.11+
- **AI Integration**: OpenAI API for enhanced insights
- **Data Processing**: 
  - Pandas for data manipulation
  - OpenPyXL for Excel file reading
- **Deployment**: Replit for instant online access

## 📊 Sample Dataset

The included sample dataset features completely anonymized transportation data with:
- **2,000 trip records** with randomized IDs and timestamps
- **Fictional addresses** in imaginary cities (Springfield, Riverside, Greenfield, Oakville)
- **Privacy-safe data** - all sensitive information has been replaced with fake values
- **Full functionality** - demonstrates all analysis capabilities

## 🔐 Privacy & Data Protection

### Your Data Security:
✅ **Session Isolation**: Each user's data is completely separate  
✅ **No Data Storage**: Files are processed in memory only  
✅ **Real Analysis**: Your uploaded data receives accurate analysis  
✅ **API Security**: OpenAI keys managed through secure environment variables  

### Sample Data Privacy:
✅ **Fully Anonymized**: All real addresses, IDs, and timestamps replaced  
✅ **Fictional Locations**: Uses made-up cities and addresses  
✅ **Privacy Compliance**: Safe for public demonstration and sharing  

## 🌟 Key Benefits

- **Zero Setup**: Run instantly in your browser
- **Universal Compatibility**: Works with any Excel file format
- **Smart Analysis**: Automatically detects data types and patterns
- **AI Enhancement**: Get deeper insights with OpenAI integration
- **Educational**: Perfect for learning data analysis concepts
- **Open Source**: Full source code available for customization

## 🔧 Configuration

### OpenAI Integration (Optional)
For enhanced AI insights on your uploaded data:
1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/)
2. Add it as `OPENAI_API_KEY` environment variable
3. Enjoy AI-powered analysis and recommendations!

*Note: The chatbot works perfectly without OpenAI - you'll still get comprehensive analysis using built-in algorithms.*

## 📝 Use Cases

- **Business Analytics**: Analyze sales, customer, or operational data
- **Academic Research**: Explore datasets for research projects
- **Data Exploration**: Quickly understand new datasets
- **Learning Tool**: Understand data analysis concepts through conversation
- **Prototype Development**: Test data analysis workflows

## 🤝 Contributing

Feel free to fork this project and submit pull requests! Areas for improvement:
- Additional file format support (CSV, JSON)
- More advanced statistical analysis
- Custom visualization generation
- Multi-language support

---

**Ready to analyze your data?** 👆 **[Click here to try the AI Chatbot now!](https://python-pilot-AI-Chatbot-Transportation-Data-Analysis.replit.app)**
