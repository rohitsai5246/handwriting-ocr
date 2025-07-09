# Step 1: Import libraries
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Step 2: Load the TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Step 3: Load your image
image_path = "/content/handwritten text.png"
image = Image.open(image_path).convert("RGB")

# Step 4: Preprocess and predict
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Step 5: Output the result
print("Extracted Handwritten Text:\n")
print(generated_text)
