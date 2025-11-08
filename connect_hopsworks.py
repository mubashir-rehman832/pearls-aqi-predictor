import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
print("✅ Connected to Hopsworks project:", project.name)
