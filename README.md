## Installation
Run my Project

```bash
  git clone https://github.com/KormazovaVer/ML_Project/Image-Recognition-App-using-FastAPI-and-PyTorch
  cd Image-Recognition-App-using-FastAPI-and-PyTorch
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  uvicorn main:app --reload
```
## Image Recognition App using FastAPI and PyTorch: TODO
- [x] Create a virtual environment
- [x] Create FastAPI App
    - Install fast API
    - Install Uvicorn
    - Install Pytest
    - Install Jinja2
    - Install python-multipart
    - Install requests
    - Create a main file with some routes
    - Create a main test file to test the home page route

- [x] Pytorch Setup
    - Install torch & torchvision (use cpu version for small size)

- [x] Prediction 
    - Create a predict post route
    - Create a file utils.py
    - Test predict route
    - Create some helper function
    - Put some test images inside static folder
    - Create a test to upload an image in predict route
    - Predict

- [x] Create a home page for prediction
    - Create an index.html file inside the templates directory
    - Setup template and static directory in the main app
    - Initial HTML for home page
    - Use Tailwind CSS cdn link for css
    - Google Fonts
    - Create a form to predict
    - update homepage route for prediction
    - Update UI of the page
    - Add Some javascript to autoload the image
    - Add logo and favicon
    - Add meta tags
    - Add response image for preview as base64 data

- [x] Docker Support
    - The API has built-in support for running in a Docker container
    - Build the Docker image by running the following command in the project directory:
    - docker-compose up --build
    - Run the Docker image using the following command:
    - docker-compose run -p 8000:8000
    - The API server will be running inside the Docker container



