import cv2
import numpy as np

def dark_channel(image, size=15):
    """Calculate the dark channel of the image."""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel_img = cv2.erode(min_channel, kernel)
    return dark_channel_img

def atmospheric_light(image, dark_channel_img, top_percent=0.001):
    """Estimate atmospheric light using the brightest pixels in the dark channel."""
    flat_img = image.reshape(-1, 3)
    flat_dark = dark_channel_img.flatten()
    num_pixels = len(flat_dark)
    top_pixels = int(max(top_percent * num_pixels, 1))  
    indices = np.argsort(flat_dark)[-top_pixels:]
    brightest = flat_img[indices]
    atm_light = np.mean(brightest, axis=0)  
    return np.clip(atm_light, 0, 255)

def transmission_map(image, atmospheric_light, omega=0.9, size=15):
    """Estimate the transmission map."""
    atmospheric_light = np.maximum(atmospheric_light, 1e-6)  
    norm_image = image / atmospheric_light
    dark_channel_norm = dark_channel(norm_image, size)
    transmission = 1 - omega * dark_channel_norm
    
    # Apply bilateral filter or guided filter (if ximgproc is available)
    transmission = cv2.bilateralFilter(transmission.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
    return transmission

def recover_scene_radiance(image, transmission, atmospheric_light, t0=0.2):
    """Recover the dehazed image."""
    transmission = np.clip(transmission, t0, 1.0)[:, :, np.newaxis]
    scene_radiance = (image - atmospheric_light) / transmission + atmospheric_light
    return np.clip(scene_radiance, 0, 255).astype(np.uint8)

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Slightly higher clipLimit
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

def apply_gamma_correction(image, gamma=1.1):
    """Apply gamma correction to adjust brightness."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_sharpening(image):
    """Apply sharpening filter to make the image clearer."""
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def highlight_fire(image):
    """Highlight fire regions using color detection (red/orange regions)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([10, 100, 100])
    upper_fire = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_highlight = cv2.bitwise_and(image, image, mask=mask)
    enhanced_image = cv2.addWeighted(image, 0.8, fire_highlight, 0.2, 0)
    return enhanced_image

def apply_dcp(image):
    """Apply the Dark Channel Prior algorithm to an image."""
    dark_channel_img = dark_channel(image)
    atm_light = atmospheric_light(image, dark_channel_img)
    transmission = transmission_map(image, atm_light)
    dehazed_image = recover_scene_radiance(image, transmission, atm_light)
    
    # Post-processing: enhance contrast, fire, and brightness
    dehazed_image = apply_clahe(dehazed_image)
    dehazed_image = apply_gamma_correction(dehazed_image, gamma=1.1)
    dehazed_image = apply_sharpening(dehazed_image)
    dehazed_image = highlight_fire(dehazed_image)
    return dehazed_image

def process_video(input_path, output_path):
    """Process a video by applying DCP to each frame."""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Processing frame {frame_idx + 1}/{frame_count}")
        dehazed_frame = apply_dcp(frame)
        out.write(dehazed_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Video processing complete!")

# Paths to input and output video
input_video_path = 'input_video3.mp4'
output_video_path = 'output_video2.mp4'

# Run the video processing function
process_video(input_video_path, output_video_path)
