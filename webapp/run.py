import webbrowser
import uvicorn
import time

if __name__ == "__main__":
    # Give the server a little time before opening browser
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
