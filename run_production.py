import os
from waitress import serve
from app.app import app, ensure_generated_dirs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Ensure necessary directories exist
    ensure_generated_dirs()
    
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 Starting Production Server on http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop.")
    
    serve(app, host="0.0.0.0", port=port)
