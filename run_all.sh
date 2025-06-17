#!/bin/bash
# run_all.sh - Script to run both FastAPI backend and Streamlit frontend
# Export Gemini API Key if not already set
if [ -z "$GEMINI_API_KEY" ]; then
  echo "⚠️  GEMINI_API_KEY not set. Set it using export GEMINI_API_KEY=your-key"
  exit 1
fi

# Run FastAPI backend
echo "🚀 Starting FastAPI backend on http://localhost:8000"
uvicorn backend.main:app --reload &

# Wait for backend to initialize
sleep 2

# Run Streamlit frontend
echo "🌐 Launching Streamlit frontend on http://localhost:8501"
streamlit run frontend/app.py
