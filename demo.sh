#!/bin/bash

# runs object detection sequentially on all images in the Images directory
# and outputs the image for each with a bounding box if an object was detected

for image in Images/*; do
	./detection.py "$image"
done
