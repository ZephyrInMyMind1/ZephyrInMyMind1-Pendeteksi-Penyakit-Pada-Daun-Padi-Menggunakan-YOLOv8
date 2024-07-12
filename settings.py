from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

Base = declarative_base()

class DetectionResult(Base):
    __tablename__ = 'detection_results'
    id = Column(Integer, primary_key=True, index=True)
    object_name = Column(String, index=True)
    confidence = Column(String)
    image_base64 = Column(Text)  # Pastikan kolom ini ada
    timestamp = Column(DateTime, default=datetime.utcnow)

# URL database SQLite
DATABASE_URL = "sqlite:///./test.db"

# Buat engine dan session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Fungsi untuk mendapatkan session database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Buat tabel jika belum ada
Base.metadata.create_all(bind=engine)

# Path ke model deteksi
MODEL_DIR = Path(__file__).resolve().parent / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

# Sumber data yang didukung
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
YOUTUBE = "YouTube"
SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, YOUTUBE]

# Webcam path
WEBCAM_PATH = 0  # Biasanya 0 untuk webcam default, sesuaikan jika perlu