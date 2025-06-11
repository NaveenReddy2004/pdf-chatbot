from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preinstall_models():
    try:
        # Pre-install embedding model
        logger.info("Loading all-MiniLM-L6-v2...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_model.encode(["Test sentence"])  # Trigger download
        logger.info("all-MiniLM-L6-v2 loaded successfully")

        # Pre-install BART model
        logger.info("Loading facebook/bart-large...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        test_input = tokenizer("Test input", return_tensors="pt")
        model.generate(test_input.input_ids)  # Trigger download
        logger.info("facebook/bart-large loaded successfully")

    except Exception as e:
        logger.error(f"Error pre-installing models: {e}")

if __name__ == "__main__":
    preinstall_models()