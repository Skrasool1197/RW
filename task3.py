from PIL import Image
import pytesseract

# Load the image
image_path = "D:/COLLEGE/ML/Tasks/RandomWalk/flag.png"
img = Image.open(image_path)

# Use OCR to extract text from the image
extracted_text = pytesseract.image_to_string(img)

# Display the extracted text
print(extracted_text)
