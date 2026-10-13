from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

html_content = """
<!DOCTYPE html>
<html>
    <head>
        <title> CardioAgent Gateway</title>
        <style>
            body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
            .container { max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 20px; border-radius: 10px; }
            .result { margin-top: 20px; padding: 10px; background: #f4f4f4; border-radius: 5px; white-space: pre-wrap; }
            button { cursor: pointer; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> CardioAgent Gateway</h1>
            <p>Upload a PCG audio file for analysis</p>
            <form id="uploadForm">
                <input type="file" name="file" accept=".wav" required><br><br>
                <input type="text" name="prompt" placeholder="Optional prompt..." style="width: 100%; padding: 8px; margin-bottom: 10px;"><br>
                <button type="submit">Analyze</button>
            </form>
            <div id="result" class="result" style="display:none;"></div>
        </div>

        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const resDiv = document.getElementById('result');
                resDiv.style.display = 'block';
                resDiv.innerText = 'Analyzing...';

                try {
                    const response = await fetch('/analyze_pcg', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    resDiv.innerText = data.message;
                } catch (err) {
                    resDiv.innerText = 'Error: ' + err.message;
                }
            };
        </script>
    </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content


@app.get("/health")
async def health():
    return {"service": "api-gateway", "status": "ok"}

@app.post("/analyze_pcg")
async def analyze_pcg(file: UploadFile = File(...), prompt: str = Form(None)):
    # Simple mock response for now
    return {
        "status": "success",
        "message": f"Received file: {file.filename}. Prompt: {prompt if prompt else 'No prompt provided'}. \n\n(This is a simple mock response. Integration with orchestrator coming soon!)"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
