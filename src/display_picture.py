# display_image.py
#   Intelligently crop and scale supplied image
#   and then display on Waveshare e-paper display.
#   This script takes an input image, processes it to fit the display dimensions,
#   and can intelligently crop based on the most interesting part of the image.

import argparse
import cv2  # OpenCV for image processing
import numpy as np  # Numerical operations for image analysis
from PIL import Image  # PIL for final image conversion for Waveshare display

def load_image(image_path):
    """Load an image from file using OpenCV (in BGR format)."""
    return cv2.imread(image_path)

def save_image(image_path, image):
    """Save the processed image to a file."""
    cv2.imwrite(image_path, image)

def crop(image, disp_w, disp_h, intelligent=True):
    """
    Resize and crop image to match display dimensions while preserving aspect ratio.
    If intelligent=True, uses saliency detection to keep the most interesting part of the image.
    
    Args:
        image: Input image (numpy array in BGR format)
        disp_w: Target display width
        disp_h: Target display height
        intelligent: Whether to use saliency detection for cropping
    
    Returns:
        Cropped and resized image
    """
    # Get input image dimensions
    img_h, img_w, img_c = image.shape
    print(f"Input WxH: {img_w} x {img_h}")

    # Calculate aspect ratios
    img_aspect = img_w / img_h
    disp_aspect = disp_w / disp_h

    print(f"Image aspect ratio {img_aspect} ({img_w} x {img_h})")
    print(f"Display aspect ratio {disp_aspect} ({disp_w} x {disp_h})")

    # Determine how to scale the image to match display dimensions
    if img_aspect < disp_aspect:
        # Image is too tall - scale to match width and crop height
        resize = (disp_w, int(disp_w / img_aspect))
    else:
        # Image is too wide - scale to match height and crop width
        resize = (int(disp_h * img_aspect), disp_h)

    print(f"Resizing to {resize}")
    image = cv2.resize(image, resize)
    img_h, img_w, img_c = image.shape

    # Calculate crop offsets - one will always be 0
    x_off = int((img_w - disp_w) / 2)
    y_off = int((img_h - disp_h) / 2)
    assert x_off == 0 or y_off == 0, "My logic is broken"

    if intelligent:
        # Use OpenCV's saliency detection to find the most interesting part of the image
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        if x_off == 0:
            # If we're cropping height, find the most interesting vertical region
            vert = np.max(saliencyMap, axis=1)  # Collapse to 1D array
            vert = np.convolve(vert, np.ones(64)/64, "same")  # Smooth the data
             
            # Find the center of the most interesting region
            sal_centre = int(np.argmax(vert))
            img_centre = int(img_h / 2)
            # Calculate how far we can shift the crop window
            shift_y = max(min(sal_centre - img_centre, y_off), -y_off)
            y_off += shift_y
        else:
            # If we're cropping width, find the most interesting horizontal region
            horiz = np.max(saliencyMap, axis=0)  # Collapse to 1D array
            horiz = np.convolve(horiz, np.ones(64)/64, "same")  # Smooth the data
            
            # Find the center of the most interesting region
            sal_centre = int(np.argmax(horiz))
            img_centre = int(img_w / 2)
            # Calculate how far we can shift the crop window
            shift_x = max(min(sal_centre - img_centre, x_off), -x_off)
            x_off += shift_x

    # Crop the image to final display size
    image = image[y_off:y_off + disp_h, x_off:x_off + disp_w]

    img_h, img_w, img_c = image.shape
    print(f"Cropped WxH: {img_w} x {img_h}")
    return image

def display_waveshare(image, saturation=1.0):
    """
    Display the image on Waveshare e-paper display.
    
    Args:
        image: Input image (numpy array in BGR format)
        saturation: Color saturation adjustment (not used by Waveshare)
    """
    from waveshare_epd import epd7in3f  # Import Waveshare display driver
    epd = epd7in3f.EPD()
    
    # Rotate image if it's taller than wide
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Convert from BGR to RGB color space for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize display, show image, and put display to sleep
    epd.init()
    epd.display(epd.getbuffer(Image.fromarray(image)))
    epd.sleep()

if __name__ == "__main__":
    # Set up command line argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="input image")
    ap.add_argument("-o", "--output", default="",
                    help="name to save cropped display image if provided")
    ap.add_argument("-p", "--portrait", action="store_true",
                    default=False, help="Portrait orientation")
    ap.add_argument("-c", "--centre_crop", action="store_true",
                    default=False, help="Simple centre cropping")
    ap.add_argument("-r", "--resize_only", action="store_true",
                    default=False, help="Simply resize image to display ignoring aspect ratio")
    ap.add_argument("--width", type=int, default=800, help="The width of the display")
    ap.add_argument("--height", type=int, default=480, help="The height of the display")
    args = vars(ap.parse_args())

    # Get display dimensions
    disp_w, disp_h = args["width"], args["height"]

    # Swap dimensions if portrait mode is requested
    if args["portrait"]:
        disp_w, disp_h = disp_h, disp_w

    # Load and process the image
    image = load_image(args["image"])
    if args["resize_only"]:
        print(f"Resizing to {disp_w}x{disp_h}")
        image = cv2.resize(image, (disp_w, disp_h))
    else:
        image = crop(image, disp_w, disp_h, intelligent=(not args["centre_crop"]))

    # Display the image
    display_waveshare(image)

    # Save the processed image if requested
    if args["output"]:
        save_image(args["output"], image)