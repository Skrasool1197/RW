import cv2
import pytesseract
from pytesseract import Output
from PIL import Image
import os
from collections import defaultdict

# If Tesseract is not in your system's PATH, uncomment the following line and provide the path
# pytesseract.pytesseract.tesseract_cmd = r'YOUR_TESSERACT_PATH'

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR results.
    Steps:
    - Convert to grayscale
    - Apply adaptive thresholding
    - Invert colors
    - Dilate to connect text regions
    """
    # Read image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Invert the image: text becomes black, background becomes white
    thresh = cv2.bitwise_not(thresh)

    # Dilate to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
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

def group_words_into_lines(data):
    """
    Group words into lines based on block_num, par_num, and line_num.
    Returns a list of dictionaries with 'text', 'font_size', and 'y' (top position).
    """
    n_boxes = len(data['level'])
    lines_dict = defaultdict(list)

    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Filter out weak confidence text
            block_num = data['block_num'][i]
            par_num = data['par_num'][i]
            line_num = data['line_num'][i]
            word = data['text'][i].strip()
            if word:
                key = (block_num, par_num, line_num)
                lines_dict[key].append({
                    'text': word,
                    'font_size': data['height'][i],
                    'y': data['top'][i]
                })

    lines = []
    for key in sorted(lines_dict.keys(), key=lambda x: (x[0], x[1], x[2])):
        words = lines_dict[key]
        line_text = ' '.join([word['text'] for word in words])
        # Use the maximum font size in the line as its font size
        max_font_size = max(word['font_size'] for word in words)
        # Use the minimum y-coordinate in the line as its y-position
        min_y = min(word['y'] for word in words)
        lines.append({
            'text': line_text,
            'font_size': max_font_size,
            'y': min_y
        })

    return lines

def determine_font_size_threshold(lines, threshold_ratio=0.8):
    """
    Determine the font size threshold to differentiate headings from subheadings.
    Uses a ratio of the maximum font size.
    """
    if not lines:
        return None
    max_font_size = max(line['font_size'] for line in lines)
    threshold = max_font_size * threshold_ratio
    return threshold

def organize_into_dictionary(lines, threshold):
    """
    Organize lines into a dictionary where headings are keys and subheadings are values.
    """
    organized_dict = {}
    current_heading = None

    for line in lines:
        text = line['text']
        font_size = line['font_size']

        if font_size >= threshold:
            # Treat as heading
            current_heading = text
            organized_dict[current_heading] = ""
        else:
            # Treat as subheading
            if current_heading:
                # Assign the subheading to the current heading
                if organized_dict[current_heading]:
                    organized_dict[current_heading] += " " + text
                else:
                    organized_dict[current_heading] = text
            else:
                # If no heading has been set yet, skip or handle differently
                pass

    return organized_dict

def extract_headings_subheadings(image_path):
    """
    Main function to extract headings and subheadings from an image and return as a dictionary.
    """
    try:
        preprocessed_image = preprocess_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return {}

    data = extract_text_with_data(preprocessed_image)
    lines = group_words_into_lines(data)

    if not lines:
        print("No text found in the image.")
        return {}

    threshold = determine_font_size_threshold(lines, threshold_ratio=0.8)

    if threshold is None:
        print("Unable to determine font size threshold.")
        return {}

    organized_dict = organize_into_dictionary(lines, threshold)
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
