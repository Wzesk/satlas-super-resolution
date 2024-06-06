"""
This module adds upsampling functionality to the SDS_preprocess module
# additional requirements: opencv-python, diffusers, pillow   ( pip install -qq diffusers==0.11.1 accelerate )
"""

# load modules
import os
import numpy as np

from PIL import Image
import math
import cv2  
import skimage.io

import torch
#from diffusers import LDMSuperResolutionPipeline

from ssr.utils.infer_utils import format_s2naip_data
from ssr.utils.options import yaml_load
from ssr.utils.model_utils import build_network
np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

###################################################################################################
# UPSAMPLING
###################################################################################################


def extract_sub_images(original_image, sub_image_size, min_overlap,folder_path):
    # Open the original image
    img = Image.open(original_image)
    width, height = img.size

    # extend the original image if needed to fit the sub_image_size using the pixel color at the edge
    if width < sub_image_size:
        img = img.crop((0, 0, sub_image_size, height))
        width = sub_image_size
        x_steps = 1
        x_step_size = width
    else:
        x_steps = math.ceil((width) / (sub_image_size-min_overlap))
        x_step_size = math.ceil((width - sub_image_size) / (x_steps-1))

    if height < sub_image_size:
        img = img.crop((0, 0, width, sub_image_size))
        height = sub_image_size
        y_steps = 1
        y_step_size = height      
    else:
        y_steps = math.ceil((height) / (sub_image_size-min_overlap))
        y_step_size = math.ceil((height - sub_image_size) / (y_steps-1))

    #inpaint missing pixels
    img_array = infill_missing_pixels(np.array(img), threshold=10, inpaint_radius=3, inpaint_method=cv2.INPAINT_TELEA)

    print('x_steps='+str(x_steps)+', y_steps='+str(y_steps)+', x_step_size='+str(x_step_size)+', y_step_size='+str(y_step_size))
    img = Image.fromarray(img_array)

    # List to hold sub-images
    sub_images = []
    padding_list = []
    # Loop over the image to extract sub-images
    print("Extracting sub-images...")
    print("original image size: "+str(img.size))
    print("sub-image size: "+str(sub_image_size))

    x = 0
    while x < x_steps:#(width-sub_image_size):
        y = 0
        while y < y_steps:#(height-sub_image_size):
            if((x*x_step_size)+sub_image_size > width):
                xpad = width - sub_image_size
            else:
                xpad = x*x_step_size
            if((y*y_step_size)+sub_image_size > height):
                ypad = height - sub_image_size
            else:
                ypad = y*y_step_size

            # Compute the dimensions of the sub-image
            right = min(xpad + sub_image_size, width)
            bottom = min(ypad + sub_image_size, height)

            # Extract and add the sub-image to the list
            sub_image = img.crop((xpad, ypad, right, bottom))
            sub_images.append(sub_image)

            # Compute the padding needed to make the sub-image the same size as the others
            padding = {
                        "left":xpad,
                        "right":width - (xpad+sub_image_size),
                        "top":ypad,
                        "bottom":height - (ypad+sub_image_size)
                       }
            print(padding)
            padding_list.append(padding)
            # Save the sub-image to the folder
            sub_image.save(os.path.join(folder_path, f"lr/sub_image_{x}_{y}.png"))
            y += 1
        x+=1#x_step_size
    #save the padding list as a txt file to the folder
    with open(os.path.join(folder_path, "padding_list.txt"), "w") as f:
        for item in padding_list:
            f.write("%s\n" % item)
    return sub_images, padding_list,img.size

def infill_missing_pixels(image, threshold=10, inpaint_radius=3, inpaint_method=cv2.INPAINT_TELEA):
    """
    Replaces black or near-black pixels in an image with the color of the nearest non-black pixel using OpenCV inpainting.

    Parameters:
    image_path: the image file.
    threshold (int): The threshold below which a pixel is considered black (default 10).
    inpaint_radius (int): Radius of a circular neighborhood of each point inpainted.
    inpaint_method (cv2 constant): Inpainting method (cv2.INPAINT_TELEA or cv2.INPAINT_NS).

    Returns:
    numpy.ndarray: The modified image with black pixels infilled.
    """


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask where black pixels are marked
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Apply inpainting
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)

    return inpainted_image

def upsample_subimages(folder_path):
    """
    upsaple set of sub-images.

    Parameters:
    folder_path: path to the folder containing the sub-images

    Returns:
    string status.
    """

    # List to hold the upsampled images
    upsampled_images = []

    #load the sub-images
    sub_images = []
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            sub_images.append(Image.open(os.path.join(folder_path, file)))

    # Loop over the sub-images
    for sub_image in sub_images:
        # Upsample the sub-image
        upsampled_image = superresolve(sub_image)
        #save the upsampled image
        upsampled_image.save(os.path.join(folder_path, f"sr/{sub_image.filename}"))

    return 'complete'

def superresolve(sub_image,model_name='satlas'):
    """
    upsamples each sub-image using the specified model

    Parameters:
    sub-image to upsample
    model name to use

    Returns:
    upsamped image
    """

    #select the model to use based on the model_name
    if model_name == 'satlas':
        upsampled_image = satlas_superresolve(sub_image)
    elif model_name == 'stable_diffusion':
        upsampled_image = stable_diffusion_superresolve(sub_image)
    else:
        print("Model not found, using nearest naighbor upscaling")
        #upscale image using nearest neighbor
        upsampled_image = sub_image.resize((sub_image.size[0]*4,sub_image.size[1]*4), Image.NEAREST)     

    return upsampled_image   


    return upsampled_image

def satlas_superresolve(sub_image,size=32,project_path='D:/Github/satlas-super-resolution/ssr/options/'):
    """
    upsamples each image

    Parameters:
    image to upsample

    Returns:
    upsamped image
    """
    #device = torch.device('cuda')
    device = torch.device('cpu')

    # Load the configuration file.
    opt = yaml_load(project_path + 'infer_example.yml')

    n_lr_images = opt['n_lr_images']  # number of low-res images as input to the model; must be the same as when the model was trained

    # Define the generator model, based on the type and parameters specified in the config.
    model = build_network(opt)

    # Load the pretrained weights into the model
    if not 'pretrain_network_g' in opt['path']:
        print("WARNING: Model weights are not specified in configuration file.")
    else:
        weights = opt['path']['pretrain_network_g']  # path to the generator weights
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict[opt['path']['param_key_g']], strict=opt['path']['strict_load_g'])
    model = model.to(device).eval()

    im = sub_image

    # Feed the low-res images through the super-res model.
    input_tensor, s2_image = format_s2naip_data(im, n_lr_images, device)
    output = model(input_tensor)

    # Convert the model output back to a numpy array and adjust shape and range.
    output = torch.clamp(output, 0, 1)
    output = output.squeeze().cpu().detach().numpy()
    output = np.transpose(output, (1, 2, 0))  # transpose to [h, w, 3] to save as image
    output = (output * 255).astype(np.uint8)

    return output

# def stable_diffusion_superresolve(sub_image,size=128):
#     """
#     upsamples each image

#     Parameters:
#     image to upsample

#     Returns:
#     upsamped image
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     pipe = LDMSuperResolutionPipeline.from_pretrained( "CompVis/ldm-super-resolution-4x-openimages")
#     pipe = pipe.to(device)

#     sub_image = sub_image.crop((0, 0, size, size))

#     upsampled_image = pipe(sub_image, num_inference_steps=100, eta=1).images[0]

#     return upsampled_image

def pad_sub_images(sub_images, padding_list,target_size,scale=4):
    """
    Pads each sub-image to make it the same size as the others.

    Parameters:
    sub_images (list of PIL.Image): List of sub-images to pad.
    padding_list (list of dict): List of padding dictionaries for each sub-image.

    Returns:
    list of PIL.Image: List of padded sub-images.
    """
    target_size = (target_size[0]*scale,target_size[1]*scale)
    print(target_size)
    padded_images = []
    for i in range(len(sub_images)):
        padded_image = Image.new("RGB", target_size, (0, 0, 0))
        padded_image.paste(sub_images[i], box=(padding_list[i]["left"]*scale,padding_list[i]["top"]*scale) )
        padded_images.append(padded_image)  

    return padded_images

def reassemble_images(padded_sub_images):
    """
    Reassembles sub-images into a full-size image, averaging overlapping areas.

    Parameters:
    sub_images (list of PIL.Image): The list of sub-images.

    Returns:
    PIL.Image: The reassembled image.
    """
    # Get the size of the full image
    full_size = padded_sub_images[0].size

    # Overlay and average the sub-images
    
    # create a temporary image to hold the count of non-black pixels for the current sub-image
    temp_img = Image.new("RGB", full_size, (0, 0, 0))

    for x in range(full_size[0]):
        for y in range(full_size[1]):
            pixel_array = []
            #get the pixel values from each sub-image
            for img in padded_sub_images:
                #if the pixel is black, skip it
                if img.getpixel((x,y)) == (0,0,0):
                    continue
                else:
                    pixel_array.append(img.getpixel((x,y)))
            #average the pixel values
            if len(pixel_array) > 0:
                pixel_average = tuple([int(sum(x) / len(x)) for x in zip(*pixel_array)])
            else:
                pixel_average = (0,0,0)
            #set the pixel value in the temporary image
            temp_img.putpixel((x,y), pixel_average)
    return temp_img

