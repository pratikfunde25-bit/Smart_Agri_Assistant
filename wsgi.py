from app.app import app, ensure_generated_dirs

if __name__ == "__main__":
    ensure_generated_dirs()
    app.run()
