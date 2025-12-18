import nltk
nltk.download('punkt_tab')
import psutil
print("RAM usage:", psutil.virtual_memory().percent)
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["disaster_hub"]
# db["documents"].delete_many({})
print(db["documents"].count_documents({}))
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["disaster_hub"]

doc = db["documents"].find_one()
print("embedding" in doc)
