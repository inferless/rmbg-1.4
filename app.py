import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from transformers import pipeline
from io import BytesIO
import base64

class InferlessPythonModel:
    def initialize(self):
        model_id = "briaai/RMBG-1.4"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = pipeline("image-segmentation", model=model_id, trust_remote_code=True)

    def infer(self, inputs):
        image_url = inputs["image_url"]
        pillow_mask = self.pipe(image_url, return_mask = True) # outputs a pillow mask
        pillow_image = self.pipe(image_url) # applies mask on input and returns a pillow image
        
        buff = BytesIO()
        pillow_image.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode()
        return { "generated_image_base64" : img_str }

    def finalize(self):
        self.pipe = None
