
import sys
import os

# Add python-worker to path
sys.path.append(os.path.join(os.getcwd(), "python-worker"))

import logging
from PIL import Image
from models.blip_model import BLIPModel
from services.translator import TranslatorService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_blip_description():
    # Initialize BLIP
    logger.info("Initializing BLIP Model...")
    blip = BLIPModel(device_type="cpu")
    
    # Create a dummy image (black square)
    img = Image.new('RGB', (224, 224), color='red')
    
    # Get context
    logger.info("Generating context...")
    result = blip.get_context_comprehensive(img)
    
    print("\n=== BLIP RESULT ===")
    print(f"English Caption: {result['english_caption']}")
    print(f"Indonesian Phrase: {result['indonesian_phrase']}")
    print(f"Indonesian Description: {result['indonesian_description']}")
    
    # Verify current behavior (template based)
    # The current implementation constructs description from elements
    # logic: phrase + elements
    
    # Also test translator directly
    logger.info("\nTesting Translator Service...")
    translator = TranslatorService()
    translation = translator.translate(result['english_caption'])
    print(f"Direct Translation: {translation}")

if __name__ == "__main__":
    test_blip_description()
