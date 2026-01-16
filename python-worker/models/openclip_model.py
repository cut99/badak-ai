"""
OpenCLIP model for vision tagging using zero-shot classification.
Uses ViT-B-32 model for predefined tag classification.
"""

import logging
import torch
from typing import List, Dict, Any
from PIL import Image
import open_clip

logger = logging.getLogger(__name__)


class OpenCLIPModel:
    """Vision tagging using OpenCLIP zero-shot classification."""

    # Expanded predefined tags for government/official context (100+ tags)
    TAGS_EN = [
        # Environment (15 tags)
        "indoor", "outdoor", "office", "meeting room", "auditorium", "garden",
        "conference hall", "lobby", "government building", "park", "street",
        "ceremony venue", "banquet hall", "courtyard", "balcony",

        # Formality & Clothing (12 tags)
        "formal", "informal", "casual", "business attire", "traditional clothing",
        "uniform", "official attire", "ceremonial dress", "suit and tie",
        "formal dress", "business casual", "traditional outfit",

        # Activities (40 tags)
        "meeting", "ceremony", "presentation", "conference", "discussion",
        "handshake", "signing document", "award ceremony", "speech", "interview",
        "press conference", "official visit", "inauguration", "opening ceremony",
        "closing ceremony", "ribbon cutting", "flag raising", "swearing in",
        "group discussion", "panel discussion", "workshop", "seminar",
        "reception", "banquet", "dinner", "lunch meeting", "breakfast meeting",
        "site visit", "inspection", "monitoring", "evaluation", "review",
        "greeting", "welcoming", "farewell", "departure", "arrival",
        "celebration", "commemoration", "memorial", "tribute",

        # People & Composition (15 tags)
        "single person", "two people", "small group", "large group", "crowd",
        "portrait", "group photo", "candid", "posed photo", "selfie",
        "standing", "sitting", "walking", "gathering", "lineup",

        # Objects & Setting (20 tags)
        "desk", "podium", "microphone", "flag", "banner",
        "document", "certificate", "award", "trophy", "plaque",
        "table", "chairs", "stage", "lectern", "screen",
        "projector", "flowers", "decorations", "signage", "backdrop",

        # Government Context (15 tags)
        "official event", "government function", "state ceremony", "public service",
        "community event", "civic engagement", "policy meeting", "coordination meeting",
        "working visit", "official inspection", "field visit", "stakeholder meeting",
        "interagency meeting", "bilateral meeting", "multilateral meeting"
    ]

    # Indonesian translations for all tags
    TAG_TRANSLATIONS = {
        # Environment
        "indoor": "dalam ruangan",
        "outdoor": "luar ruangan",
        "office": "kantor",
        "meeting room": "ruang rapat",
        "auditorium": "auditorium",
        "garden": "taman",
        "conference hall": "ruang konferensi",
        "lobby": "lobi",
        "government building": "gedung pemerintah",
        "park": "taman",
        "street": "jalan",
        "ceremony venue": "tempat upacara",
        "banquet hall": "ruang jamuan",
        "courtyard": "halaman",
        "balcony": "balkon",

        # Formality & Clothing
        "formal": "formal",
        "informal": "informal",
        "casual": "kasual",
        "business attire": "pakaian bisnis",
        "traditional clothing": "pakaian tradisional",
        "uniform": "seragam",
        "official attire": "pakaian resmi",
        "ceremonial dress": "pakaian upacara",
        "suit and tie": "jas dan dasi",
        "formal dress": "pakaian formal",
        "business casual": "semi formal",
        "traditional outfit": "busana tradisional",

        # Activities
        "meeting": "rapat",
        "ceremony": "upacara",
        "presentation": "presentasi",
        "conference": "konferensi",
        "discussion": "diskusi",
        "handshake": "bersalaman",
        "signing document": "penandatanganan dokumen",
        "award ceremony": "upacara penghargaan",
        "speech": "pidato",
        "interview": "wawancara",
        "press conference": "konferensi pers",
        "official visit": "kunjungan resmi",
        "inauguration": "pelantikan",
        "opening ceremony": "upacara pembukaan",
        "closing ceremony": "upacara penutupan",
        "ribbon cutting": "gunting pita",
        "flag raising": "pengibaran bendera",
        "swearing in": "pengambilan sumpah",
        "group discussion": "diskusi kelompok",
        "panel discussion": "diskusi panel",
        "workshop": "lokakarya",
        "seminar": "seminar",
        "reception": "resepsi",
        "banquet": "jamuan",
        "dinner": "makan malam",
        "lunch meeting": "rapat makan siang",
        "breakfast meeting": "rapat sarapan",
        "site visit": "kunjungan lapangan",
        "inspection": "inspeksi",
        "monitoring": "monitoring",
        "evaluation": "evaluasi",
        "review": "tinjauan",
        "greeting": "menyambut",
        "welcoming": "penyambutan",
        "farewell": "perpisahan",
        "departure": "keberangkatan",
        "arrival": "kedatangan",
        "celebration": "perayaan",
        "commemoration": "peringatan",
        "memorial": "memorial",
        "tribute": "penghormatan",

        # People & Composition
        "single person": "satu orang",
        "two people": "dua orang",
        "small group": "kelompok kecil",
        "large group": "kelompok besar",
        "crowd": "kerumunan",
        "portrait": "potret",
        "group photo": "foto bersama",
        "candid": "candid",
        "posed photo": "foto berpose",
        "selfie": "swafoto",
        "standing": "berdiri",
        "sitting": "duduk",
        "walking": "berjalan",
        "gathering": "berkumpul",
        "lineup": "berbaris",

        # Objects & Setting
        "desk": "meja kerja",
        "podium": "podium",
        "microphone": "mikrofon",
        "flag": "bendera",
        "banner": "spanduk",
        "document": "dokumen",
        "certificate": "sertifikat",
        "award": "penghargaan",
        "trophy": "trofi",
        "plaque": "plakat",
        "table": "meja",
        "chairs": "kursi",
        "stage": "panggung",
        "lectern": "mimbar",
        "screen": "layar",
        "projector": "proyektor",
        "flowers": "bunga",
        "decorations": "dekorasi",
        "signage": "papan nama",
        "backdrop": "latar belakang",

        # Government Context
        "official event": "acara resmi",
        "government function": "acara pemerintah",
        "state ceremony": "upacara negara",
        "public service": "pelayanan publik",
        "community event": "acara komunitas",
        "civic engagement": "keterlibatan sipil",
        "policy meeting": "rapat kebijakan",
        "coordination meeting": "rapat koordinasi",
        "working visit": "kunjungan kerja",
        "official inspection": "inspeksi resmi",
        "field visit": "kunjungan lapangan",
        "stakeholder meeting": "rapat pemangku kepentingan",
        "interagency meeting": "rapat antar lembaga",
        "bilateral meeting": "pertemuan bilateral",
        "multilateral meeting": "pertemuan multilateral"
    }

    # Backward compatibility - use TAGS_EN
    TAGS = TAGS_EN

    # Extensive Object List for "Open-Ended" feel
    OBJECTS_EN = [
        # Electronics & Office
        "laptop", "computer", "monitor", "keyboard", "mouse", "phone", "smartphone", "tablet",
        "projector", "screen", "television", "microphone", "camera", "tripod", "cable",
        "headphone", "earphone", "speaker", "remote", "printer", "scanner",
        "whiteboard", "blackboard", "marker", "pen", "pencil", "notebook", "stappler",
        "paper", "document", "folder", "binder", "envelope", "card", "id card",

        # Furniture & Interior
        "desk", "table", "chair", "sofa", "couch", "bench", "cabinet", "shelf",
        "bookshelf", "cupboard", "drawer", "podium", "lectern", "lamp", "light",
        "ceiling fan", "air conditioner", "clock", "curtain", "blind", "carpet", "rug",
        "trash bin", "vase", "mirror", "painting", "poster", "door", "window",

        # Clothing & Accessories
        "shirt", "t-shirt", "polo shirt", "blouse", "jacket", "coat", "blazer", "suit",
        "vest", "pants", "trousers", "jeans", "shorts", "skirt", "dress", "uniform",
        "shoe", "sneaker", "boot", "sandal", "sock", "hat", "cap", "helmet", "hijab",
        "glasses", "sunglasses", "watch", "smartwatch", "bracelet", "ring", "necklace",
        "tie", "bowtie", "belt", "mask", "glove", "bag", "backpack", "handbag", "suitcase",

        # Food & Drink
        "bottle", "water bottle", "cup", "glass", "mug", "plate", "bowl", "fork", "spoon",
        "knife", "napkin", "food", "snack", "drink", "coffee", "tea", "water",

        # Vehicles & Outdoor
        "car", "motorcycle", "bicycle", "bus", "truck", "van", "wheel", "tire",
        "tree", "flower", "plant", "grass", "road", "pavement", "sidewalk",
        "building", "house", "gate", "fence", "sign", "traffic light", "flag", "banner"
    ]

    def __init__(self, device_type: str = "cpu", threshold: float = 0.25, top_k: int = 10, language: str = "id"):
        """
        Initialize OpenCLIP model.

        Args:
            device_type: Device type ("cuda", "mps", or "cpu")
            threshold: Minimum confidence threshold for tag inclusion (default: 0.25)
            top_k: Number of top tags to return (default: 10)
            language: Language for tags ("en" or "id" for Indonesian, default: "id")
        """
        self.device_type = device_type
        self.threshold = threshold
        self.top_k = top_k
        self.language = language
        self.device = torch.device(device_type)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the OpenCLIP ViT-B-32 model."""
        try:
            logger.info(f"Loading OpenCLIP ViT-B-32 model on device: {self.device_type}")

            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k',
                device=self.device
            )

            # Get tokenizer
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

            # Set model to evaluation mode
            self.model.eval()

            logger.info("OpenCLIP model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            raise

    def get_objects(self, image: Image.Image, threshold: float = 0.20, top_k: int = 15) -> List[str]:
        """
        Get concrete list of objects detected in the image using OpenCLIP.
        Uses a separate, extensive list of nouns (OBJECTS_EN).
        
        Args:
            image: PIL Image
            threshold: Confidence threshold (default slightly lower for objects: 0.20)
            top_k: Max objects to return
            
        Returns:
            List of English object names
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            with torch.no_grad():
                # Preprocess image
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                # Create text prompts for objects
                text_prompts = [f"a photo of a {obj}" for obj in self.OBJECTS_EN]
                text_input = self.tokenizer(text_prompts).to(self.device)

                # Encode
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)

                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get all scores
                values, indices = similarity[0].topk(len(self.OBJECTS_EN))

                selected_objects = []
                for value, index in zip(values, indices):
                    confidence = value.item()
                    if confidence >= threshold and len(selected_objects) < top_k:
                        obj = self.OBJECTS_EN[index]
                        selected_objects.append(obj)
                    elif len(selected_objects) >= top_k:
                        break
                
                return selected_objects

        except Exception as e:
            logger.error(f"Error extracting objects with OpenCLIP: {e}")
            return []

    def get_tags(self, image: Image.Image, threshold: float = None, top_k: int = None, language: str = None) -> List[str]:
        """
        Get tags for an image using zero-shot classification.

        Args:
            image: PIL Image object
            threshold: Confidence threshold (uses instance threshold if not specified)
            top_k: Number of top tags to return (uses instance top_k if not specified)
            language: Language for tags - "en" or "id" (uses instance language if not specified)

        Returns:
            List of tag strings (in English or Indonesian based on language parameter)

        Example:
            >>> model = OpenCLIPModel()
            >>> img = Image.open("photo.jpg")
            >>> tags = model.get_tags(img)
            >>> print(tags)
            ['dalam ruangan', 'formal', 'foto bersama']
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        threshold = threshold if threshold is not None else self.threshold
        top_k = top_k if top_k is not None else self.top_k
        language = language if language is not None else self.language

        try:
            with torch.no_grad():
                # Preprocess image
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                # Create text prompts for each tag (always use English for inference)
                text_prompts = [f"a photo of {tag}" for tag in self.TAGS_EN]
                text_input = self.tokenizer(text_prompts).to(self.device)

                # Get image and text features
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)

                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Calculate similarity (cosine similarity)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(self.TAGS_EN))

                # Select top K tags above threshold
                selected_tags_en = []
                for value, index in zip(values, indices):
                    confidence = value.item()
                    if confidence >= threshold and len(selected_tags_en) < top_k:
                        tag = self.TAGS_EN[index]
                        selected_tags_en.append(tag)
                        logger.debug(f"Tag '{tag}' selected with confidence {confidence:.3f}")
                    elif len(selected_tags_en) >= top_k:
                        break

                # Translate to Indonesian if needed
                if language == "id":
                    selected_tags = [self.TAG_TRANSLATIONS.get(tag, tag) for tag in selected_tags_en]
                else:
                    selected_tags = selected_tags_en

            logger.debug(f"Selected {len(selected_tags)} tags from image (language: {language})")
            return selected_tags

        except Exception as e:
            logger.error(f"Error getting tags: {e}")
            raise

    def get_tags_with_scores(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Get tags with their confidence scores.

        Args:
            image: PIL Image object

        Returns:
            List of dictionaries with 'tag' and 'score' keys, sorted by score descending

        Example:
            >>> model = OpenCLIPModel()
            >>> img = Image.open("photo.jpg")
            >>> tags = model.get_tags_with_scores(img)
            >>> for item in tags:
            ...     print(f"{item['tag']}: {item['score']:.3f}")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            with torch.no_grad():
                # Preprocess image
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                # Create text prompts for each tag
                text_prompts = [f"a photo of {tag}" for tag in self.TAGS]
                text_input = self.tokenizer(text_prompts).to(self.device)

                # Get image and text features
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)

                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len(self.TAGS))

                # Create results
                results = []
                for value, index in zip(values, indices):
                    results.append({
                        "tag": self.TAGS[index],
                        "score": value.item()
                    })

            return results

        except Exception as e:
            logger.error(f"Error getting tags with scores: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "framework": "OpenCLIP",
            "device_type": self.device_type,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "language": self.language,
            "num_tags": len(self.TAGS_EN),
            "tags_en": self.TAGS_EN,
            "has_indonesian_translation": True,
            "is_loaded": self.model is not None
        }
