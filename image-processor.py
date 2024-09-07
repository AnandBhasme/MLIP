import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import io
import zipfile
from streamlit.runtime.state import SessionState

st.title("Machine Learning Image Processing App")

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

# Function to convert image to black and white
def convert_to_bw(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to denoise image
def denoise_image(image, h=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)

# Function to erode image
def erode_image(image, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

# Function to dilate image
def dilate_image(image, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

# Function to blur image
def blur_image(image, ksize=10):
    return cv2.blur(image, (ksize, ksize))

# Function to adjust image contrast
def adjust_contrast(image, contrast=1.05):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=0)

# Function to adjust image saturation
def adjust_saturation(image, saturation=1.0):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = hsv_image[:, :, 1] * saturation
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Function to ensure image is in color
def ensure_color(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

# Function to detect and draw contours on image
def detect_and_draw_contours(image, threshold1=100, threshold2=200, create_mask=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    if create_mask:
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        return result, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        return result, None

# Function to convert image to binary
def convert_to_binary(image, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

# Function to convert image to grayscale
def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Function for color correction
def color_correction(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Function to draw defect contours on image
def draw_defect_contours(image, min_area=261121.0):
    image = ensure_color(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    
    result = image.copy()
    defect_detected = False
    
    if (dilation == 0).sum() > 1:
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.drawContours(result, [contour], -1, (0, 0, 255), 3)
                defect_detected = True
        
        if defect_detected:
            cv2.putText(result, "Defective fabric", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(result, "Good fabric", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return result, defect_detected

# Function to get processing parameters based on selected option
def get_processing_params(option, step_number):
    key_prefix = f"step_{step_number}_"
    params = {}

    if option == "Convert to Binary":
        params['threshold'] = st.slider("Threshold", 0, 255, 127, key=f"{key_prefix}binary_threshold")
    elif option == "Color Correction":
        params['alpha'] = st.slider("Contrast", 0.0, 3.0, 1.0, key=f"{key_prefix}color_alpha")
        params['beta'] = st.slider("Brightness", -100, 100, 0, key=f"{key_prefix}color_beta")
    elif option == "Denoise":
        params['h'] = st.slider("Denoising strength (h)", 1, 30, 10, key=f"{key_prefix}denoise_h")
        params['templateWindowSize'] = st.slider("Template Window Size", 1, 10, 7, key=f"{key_prefix}denoise_template")
        params['searchWindowSize'] = st.slider("Search Window Size", 1, 30, 21, key=f"{key_prefix}denoise_search")
    elif option == "Erode" or option == "Dilate":
        params['kernel_size'] = st.slider("Kernel Size", 1, 29, 5, key=f"{key_prefix}{option.lower()}_kernel")
        params['kernel_size'] = params['kernel_size'] if params['kernel_size'] % 2 == 1 else params['kernel_size'] + 1
        params['iterations'] = st.slider("Iterations", 1, 10, 1, key=f"{key_prefix}{option.lower()}_iterations")
    elif option == "Blur":
        params['ksize'] = st.slider("Kernel Size", 1, 29, 5, key=f"{key_prefix}blur_kernel")
        params['ksize'] = params['ksize'] if params['ksize'] % 2 == 1 else params['ksize'] + 1
    elif option == "Adjust Contrast":
        params['contrast'] = st.slider("Contrast", 0.5, 3.0, 1.05, key=f"{key_prefix}contrast")
    elif option == "Adjust Saturation":
        params['saturation'] = st.slider("Saturation", 0.0, 3.0, 1.0, key=f"{key_prefix}saturation")
    elif option == "Detect Contours":
        params['threshold1'] = st.slider("Lower Threshold", 0, 255, 100, key=f"{key_prefix}contour_lower")
        params['threshold2'] = st.slider("Upper Threshold", 0, 255, 200, key=f"{key_prefix}contour_upper")
        params['create_mask'] = st.checkbox("Create contour mask", value=False, key=f"{key_prefix}create_mask")
    elif option == "Detect Defects":
        params['min_area'] = st.slider("Minimum contour area", 1000.0, 500000.0, 261121.0, key=f"{key_prefix}min_area")

    return params

# Function to process image based on selected option and parameters
def process_image(image, option, params):
    if option == "Convert to Binary":
        return cv2.threshold(image, params['threshold'], 255, cv2.THRESH_BINARY)[1]
    elif option == "Convert to Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif option == "Color Correction":
        return cv2.convertScaleAbs(image, alpha=params['alpha'], beta=params['beta'])
    elif option == "Black and White":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif option == "Denoise":
        if len(image.shape) == 2 or image.shape[2] == 1:
            return cv2.fastNlMeansDenoising(image, None, params['h'], params['templateWindowSize'], params['searchWindowSize'])
        else:
            return cv2.fastNlMeansDenoisingColored(image, None, params['h'], params['h'], params['templateWindowSize'], params['searchWindowSize'])
    elif option == "Erode":
        kernel = np.ones((params['kernel_size'], params['kernel_size']), np.uint8)
        return cv2.erode(image, kernel, iterations=params['iterations'])
    elif option == "Dilate":
        kernel = np.ones((params['kernel_size'], params['kernel_size']), np.uint8)
        return cv2.dilate(image, kernel, iterations=params['iterations'])
    elif option == "Blur":
        return cv2.blur(image, (params['ksize'], params['ksize']))
    elif option == "Adjust Contrast":
        return cv2.convertScaleAbs(image, alpha=params['contrast'], beta=0)
    elif option == "Adjust Saturation":
        image = ensure_color(image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = hsv_image[:, :, 1] * params['saturation']
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    elif option == "Detect Contours":
        image = ensure_color(image)
        result, mask = detect_and_draw_contours(image, params['threshold1'], params['threshold2'], params['create_mask'])
        if params['create_mask']:
            return np.hstack((result, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
        else:
            return result
    elif option == "Detect Defects":
        result, _ = draw_defect_contours(image, params['min_area'])
        return result
    else:
        return image

# Main processing logic
if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    
    st.subheader("Original Images")
    cols = st.columns(3)
    for idx, image in enumerate(images):
        cols[idx % 3].image(image, caption=f"Image {idx+1}", use_column_width=True)
    
    # Add JSON upload option here
    st.subheader("Import Processing Steps (Optional)")
    uploaded_json = st.file_uploader("Upload processing steps JSON file", type=['json'])

    imported_steps = None
    if uploaded_json is not None:
        imported_steps = json.load(uploaded_json)
        st.success(f"Successfully imported {len(imported_steps)} processing steps.")
    
    processing_options = [
        "Original", "Convert to Binary", "Convert to Grayscale", "Color Correction",
        "Black and White", "Denoise", "Erode", "Dilate", "Blur", "Adjust Contrast",
        "Adjust Saturation", "Detect Contours", "Detect Defects"
    ]
    
    # Initialize session state
    if 'processing_steps' not in st.session_state:
        st.session_state.processing_steps = []
    if 'steps_to_remove' not in st.session_state:
        st.session_state.steps_to_remove = set()

    step_number = 1
    
    # Apply imported steps if available
    if imported_steps:
        st.subheader("Imported Processing Steps")
        
        for idx, step in enumerate(imported_steps):
            if idx not in st.session_state.steps_to_remove:
                option = step['option']
                params = step['params']
                
                st.write(f"Step {idx+1}: {option}")
                
                # Allow changing parameters
                new_params = get_processing_params(option, f"imported_{idx}")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("Updated Parameters:", new_params)
                with col2:
                    if st.button(f"Remove Step {idx+1}", key=f"remove_step_{idx}"):
                        st.session_state.steps_to_remove.add(idx)
                        st.rerun()
                
                if new_params != params:
                    imported_steps[idx]['params'] = new_params
        
        # Remove steps marked for removal
        imported_steps = [step for idx, step in enumerate(imported_steps) if idx not in st.session_state.steps_to_remove]
        
        # Apply remaining imported steps
        for step in imported_steps:
            option = step['option']
            params = step['params']
            
            st.subheader(f"Processing Step {step_number}")
            st.write(f"Applied option: {option}")
            st.write("Parameters:", params)
            
            processed_images = [process_image(img.copy(), option, params) for img in images]
            
            cols = st.columns(3)
            for idx, image in enumerate(processed_images):
                cols[idx % 3].image(image, caption=f"Image {idx+1} after Step {step_number}", use_column_width=True)
            
            images = processed_images
            st.session_state.processing_steps.append({"option": option, "params": params})
            step_number += 1
    
    # Continue with manual steps
    continue_processing = st.checkbox("Apply additional processing steps?", key="continue_processing")
    
    while continue_processing:
        st.subheader(f"Processing Step {step_number}")
        
        option = st.selectbox(f"Choose processing option for step {step_number}", 
                              processing_options, 
                              key=f"step_{step_number}_option")
        
        if option != "Original":
            params = get_processing_params(option, step_number)
            processed_images = [process_image(img.copy(), option, params) for img in images]
            
            cols = st.columns(3)
            for idx, image in enumerate(processed_images):
                cols[idx % 3].image(image, caption=f"Image {idx+1} after Step {step_number}", use_column_width=True)
            
            images = processed_images
            st.session_state.processing_steps.append({"option": option, "params": params})
        
        continue_processing = st.checkbox("Apply additional processing step?", key=f"step_{step_number}_continue")
        
        step_number += 1
    
    st.subheader("Final Processed Images")
    cols = st.columns(3)
    for idx, image in enumerate(images):
        cols[idx % 3].image(image, caption=f"Final Image {idx+1}", use_column_width=True)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for idx, image in enumerate(images):
            img_byte_arr = io.BytesIO()
            Image.fromarray(image).save(img_byte_arr, format='PNG')
            zip_file.writestr(f"processed_image_{idx+1}.png", img_byte_arr.getvalue())
    
    st.download_button(
        label="Download All Processed Images",
        data=zip_buffer.getvalue(),
        file_name="processed_images.zip",
        mime="application/zip"
    )

    # Export processing steps as JSON for download
    if st.session_state.processing_steps:
        st.subheader("Export Processing Steps")
        json_steps = json.dumps(st.session_state.processing_steps, indent=2)
        st.download_button(
            label="Download Processing Steps",
            data=json_steps,
            file_name="processing_steps.json",
            mime="application/json"
        )
st.markdown("---")  # This adds a horizontal line for separation
st.markdown(
    "<h6 style='text-align: center; color: gray;'>Created by <a href='https://github.com/AnandBhasme' target='_blank' style='color: #4A90E2; text-decoration: none;'>Anand Bhasme</a></h6>", 
    unsafe_allow_html=True
)