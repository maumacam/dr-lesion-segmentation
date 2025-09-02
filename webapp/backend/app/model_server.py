from PIL import Image, ImageDraw
import os

RESULT_DIR = "results"

def process_image_placeholder(input_path, filename):
    """
    Simulates processing by creating fake SegNet and U-Net++ outputs
    by drawing rectangles on the original image.
    """
    img = Image.open(input_path)
    width, height = img.size

    # SegNet fake
    segnet_img = img.copy()
    draw = ImageDraw.Draw(segnet_img)
    draw.rectangle([width*0.1, height*0.1, width*0.5, height*0.5], outline="red", width=5)
    segnet_path = os.path.join(RESULT_DIR, "segnet", filename)
    segnet_img.save(segnet_path)

    # U-Net++ fake
    unetpp_img = img.copy()
    draw = ImageDraw.Draw(unetpp_img)
    draw.ellipse([width*0.3, height*0.3, width*0.7, height*0.7], outline="blue", width=5)
    unetpp_path = os.path.join(RESULT_DIR, "unetpp", filename)
    unetpp_img.save(unetpp_path)

    return segnet_path, unetpp_path
