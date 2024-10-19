import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import os

# If Tesseract is not in your system's PATH, uncomment the following line and provide the path
# pytesseract.pytesseract.tesseract_cmd = r'YOUR_TESSERACT_PATH'

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results.
    Steps:
    - Convert to grayscale
    - Apply thresholding
    - Remove noise
    """
    # Read image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Dilate to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    return dilated

def extract_text_with_data(preprocessed_image):
    """
    Extract text using pytesseract and get detailed data including bounding boxes and font sizes.
    """
    # Convert OpenCV image to PIL Image
    pil_image = Image.fromarray(preprocessed_image)

    # Use Tesseract to do OCR on the image
    data = pytesseract.image_to_data(pil_image, output_type=Output.DICT)

    return data

def organize_text(data):
    """
    Organize extracted text into a dictionary where headings are keys and subheadings are values.
    Assumption: Headings have larger font sizes than subheadings.
    """
    n_boxes = len(data['level'])
    texts = []

    # Collect all text elements with their font sizes and positions
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Filter out weak confidence text
            text = data['text'][i].strip()
            if text:
                texts.append({
                    'text': text,
                    'font_size': data['height'][i],
                    'y': data['top'][i]
                })

    # Sort texts based on y-coordinate to maintain order
    texts = sorted(texts, key=lambda x: x['y'])

    # Determine a threshold to differentiate between headings and subheadings
    # For simplicity, assume the top 30% font sizes are headings
    font_sizes = [item['font_size'] for item in texts]
    if not font_sizes:
        return {}
    threshold = sorted(font_sizes, reverse=True)[max(1, len(font_sizes)//3)]

    organized_dict = {}
    current_heading = None

    for item in texts:
        if item['font_size'] >= threshold:
            current_heading = item['text']
            organized_dict[current_heading] = ""
        else:
            if current_heading:
                if organized_dict[current_heading]:
                    organized_dict[current_heading] += " " + item['text']
                else:
                    organized_dict[current_heading] = item['text']

    return organized_dict

def extract_headings_subheadings(image_path):
    """
    Main function to extract headings and subheadings from an image and return as a dictionary.
    """
    preprocessed_image = preprocess_image(image_path)
    data = extract_text_with_data(preprocessed_image)
    organized_dict = organize_text(data)
    return organized_dict

if __name__ == "__main__":
    # Example usage
    image_path = input("Enter the path to the image: ").strip()

    if not os.path.isfile(image_path):
        print("The provided image path does not exist.")
    else:
        result = extract_headings_subheadings(image_path)
        print("\nExtracted Headings and Subheadings:")
        print(result)
