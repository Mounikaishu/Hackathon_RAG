import os
import base64
from groq import Groq
from app.config import settings

class VisionTool:
    """
    Encodes local chart images to base64 and dispatches multimodal query prompts
    to Groq's high-speed Llama 3.2 Vision model.
    """

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.client = None
        if self.api_key:
            self.client = Groq(api_key=self.api_key)

    def _encode_image(self, image_path: str) -> str:
        """Reads local image file and converts bytes to a base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _find_chart_image(self) -> str:
        """Finds any rendered chart PNG image in the processed charts directory."""
        if not os.path.exists(settings.CHARTS_DIR):
            return None
        files = [f for f in os.listdir(settings.CHARTS_DIR) if f.endswith(".png")]
        if not files:
            return None
        # Return the first available chart page png
        return os.path.join(settings.CHARTS_DIR, files[0])

    def query_chart_vision(self, question: str, image_path: str = None) -> str:
        """
        Sends the base64 encoded chart PNG image alongside the user question
        to Llama 3.2 Vision for visual reasoning.
        """
        if not self.client:
            return (
                "⚠️ Groq Vision API Error: The GROQ_API_KEY is not configured in your .env file.\n"
                "Please configure GROQ_API_KEY to activate true multimodal chart analysis."
            )

        # Locate correct chart image
        target_img = image_path or self._find_chart_image()
        if not target_img or not os.path.exists(target_img):
            return (
                "⚠️ Vision Extraction Warning: Could not find any extracted chart image assets in "
                f"'{settings.CHARTS_DIR}'. Please run the indexing pipeline first."
            )

        try:
            # Encode image
            base64_image = self._encode_image(target_img)
            
            # Construct standard OpenAI-compatible multimodal content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                "You are a Placement Analytics Vision Assistant. The attached image contains "
                                "the 'Section 3: Hiring Distribution by Role' bar charts. "
                                f"Please analyze the charts and answer the user question accurately: {question}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # Request completion
            response = self.client.chat.completions.create(
                model=settings.GROQ_VISION_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1024
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ Groq Multimodal API Error: {str(e)}"
