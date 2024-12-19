from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import numpy as np
import io
import sys
import logging
import joblib

# Création de l'API
app = FastAPI()

@app.get("/")
def read_root():
    print("print : test pendant le get")
    logging.info("log : test pendant le get")
    return {"message": "Hello World!"}