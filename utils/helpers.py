from nixtla import NixtlaClient
import os
from dotenv import load_dotenv

def API():
    llave = os.getenv('api_key')
    nixtla_client = NixtlaClient(api_key=llave)
    return nixtla_client