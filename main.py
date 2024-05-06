from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import utils

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def home_predict(request: Request, file: UploadFile = File(...)):
    result = None
    error = None
    try:
        result = utils.get_result(image_file=file)
    except Exception as ex:
        error = ex
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "error": error})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return utils.get_result(image_file=file, is_api=True)
