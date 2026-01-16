# Instalasi BADAK AI Worker - Panduan Khusus macOS

## Masalah Kompilasi di macOS

Pada macOS dengan Python 3.12, beberapa package memiliki masalah kompilasi C++:
- `insightface==0.7.3` - error `'cmath' file not found`
- `chroma-hnswlib==0.7.3` - error `unsupported argument 'native' to option '-march='`

## Solusi Instalasi

### Solusi 1: Gunakan LLVM via Homebrew (Paling Efektif)

Ini adalah solusi terbaik untuk mengatasi error `'cmath' file not found`.

1. Install LLVM:
   ```bash
   brew install llvm
   ```

2. Install dengan environment variables yang mengarah ke LLVM:
   ```bash
   export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
   export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
   export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
   export CC="/opt/homebrew/opt/llvm/bin/clang"
   export CXX="/opt/homebrew/opt/llvm/bin/clang++"
   
   pip install insightface==0.7.3
   pip install onnxruntime
   ```

### Solusi 2: Downgrade ke Python 3.11 (Alternatif)

Python 3.11 memiliki kompatibilitas lebih baik dengan packages ini:

```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Buat virtual environment dengan Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Opsi 2: Install Packages Pre-built

Install packages yang bermasalah dengan versi pre-built:

```bash
source venv/bin/activate

# Install hnswlib tanpa marching native
pip install hnswlib

# Install chromadb dengan hnswlib sudah terinstall
pip install chromadb==0.4.22

# Untuk insightface, gunakan alternatif:
# 1. Install dari source dengan fix manual, atau
# 2. Gunakan package alternatif seperti face_recognition
```

### Opsi 3: Skip Problematic Packages (Development)

Untuk development/testing tanpa perlu full functionality:

```bash
# Install semua kecuali insightface dan chromadb
pip install fastapi uvicorn pydantic onnxruntime open-clip-torch
pip install transformers torch pillow numpy opencv-python
pip install httpx python-dotenv pytest pytest-asyncio pytest-cov
```

### Opsi 4: Gunakan Docker (Paling Stabil)

```bash
cd python-worker

# Build Docker image
docker build -t badak-ai-worker .

# Run container
docker run -p 8000:8000 --env-file .env badak-ai-worker
```

## Instalasi yang Berhasil

Jika berhasil install, verifikasi dengan:

```bash
source venv/bin/activate
python -c "import insightface; print('InsightFace OK')"
python -c "import chromadb; print('ChromaDB OK')"
python -c "import open_clip; print('OpenCLIP OK')"
python -c "import transformers; print('Transformers OK')"
```

## Workaround untuk ChromaDB

Jika chromadb tidak bisa diinstall, gunakan alternatif vector database:

### Gunakan FAISS

```bash
pip install faiss-cpu  # atau faiss-gpu untuk GPU
```

Lalu modify `services/vectordb.py` untuk menggunakan FAISS instead of ChromaDB.

### Gunakan Qdrant

```bash
pip install qdrant-client
```

## Hardware Requirements

- **macOS**: 10.15+
- **RAM**: 16GB minimum
- **Storage**: 20GB free (untuk AI models)
- **GPU**: Optional (MPS untuk Mac M1/M2/M3, atau CPU fallback)

## Troubleshooting

### Error: clang++ failed
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Atau install Xcode dari App Store
```

### Error: numpy version conflict
```bash
# Uninstall semua numpy versions
pip uninstall numpy -y
pip install numpy==1.26.3
```

### Error: symbol not found in flat namespace
```bash
# Reinstall dengan no binary
pip install --no-binary :all: package-name
```

## Rekomendasi Final

**Untuk Production**: Gunakan Python 3.11 di macOS, atau deploy di Linux/Docker

**Untuk Development**: Skip packages bermasalah, test dengan mocks

**Untuk Mac M1/M2/M3**: Pastikan gunakan Rosetta 2 jika ada masalah ARM compatibility

---

Jika masih ada masalah, hubungi tim development atau buat issue di GitHub repository.
