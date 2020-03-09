from app.routers import app
# from app.config import Config

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5555)
