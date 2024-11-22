import cv2
import os

# Path to the directory containing images
image_dir = '/home/cao/DEFORM/test_results'

# Video output settings
output_video_path = 'DEFORM_flatcable.mp4'
frame_rate = 30  # frames per second

# Get sorted list of image file paths
image_files = [os.path.join(image_dir, f"{i}.png") for i in range(435)]

# Check if the list has images
if not image_files:
    raise ValueError("No images found in the directory.")

# Read the first image to get dimensions
first_frame = cv2.imread(image_files[0])
height, width, _ = first_frame.shape

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

# Write each image to the video
for image_file in image_files:
    frame = cv2.imread(image_file)
    if frame is None:
        print(f"Warning: {image_file} could not be read and was skipped.")
        continue
    video_writer.write(frame)

# Release the video writer
video_writer.release()
print("Video created successfully!")
