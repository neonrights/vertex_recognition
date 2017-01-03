import os
import cv2
import numpy as np

input_path = 'test_images/'
output_path = 'test_unlabeled/'

file_names = os.listdir(input_path)
for file_name in file_names:
	# load in image and convert to binary
	img = cv2.imread(file_name)
	gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	binary = cv2.threshold(gray_scale, cv2.THRESH_BINARY)

	# display image to make sure its good
	cv2.imshow("binary conversion", binary)
	# if image is good continue processing, otherwise reprocess with different method
	# select locations of centered nodes

	cv2.waitkey(0)
	cv2.destroyAllWidows()

	# label all other locations as 0, no node
	# label locations as 1, centered node
	filter_size = 50
	count = 0
	width, height = img.shape
	for i in range(width-filter_size):
		for j in range(height-filter_size):
			count += 1
			cv2.imwrite(output_path+'sample-'+count+'.png', binary[i:(i+filter_size),j:(j+filter_size)])

# finished message
print("completed processing all images")