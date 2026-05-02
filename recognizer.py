import easyocr
import re
class PlateRecognizer:
    def __init__(self,languages=['en']):
        self.reader = easyocr.Reader(languages)
    def extract_text(self,image):
        preprocessed = self._preprocess(image)
        results = self.read.readtext(preprocessed,detail=0)
        text = " ".join(results)
        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text)
        return cleaned_text
    def _preprocess(self,image):
        return image