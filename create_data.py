import os
import cv2
import numpy as np
from scipy import sparse

input_path = 'test_images/'
output_path = 'test_unlabeled/'

file_names = os.listdir(input_path)
for file_name in file_names:
	# load in image and convert to binary
	print(file_name)
	img = cv2.imread(input_path + file_name, cv2.IMREAD_GRAYSCALE)
	_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

	# display image to make sure its good
	cv2.imshow("graph", img)
	# if image is good continue processing, otherwise reprocess with different method
	# select locations of centered nodes

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	output_name = output_path + file_name[:-4] + '-50x50.dat'
	os.makedirs(os.path.dirname(output_name), exist_ok=True)
	#with open(output_name, 'w+') as output_file:
	to_save = list()
	filter_size = 50
	count = 0
	width, height = img.shape
	for i in range(width-filter_size):
		for j in range(height-filter_size):
			count += 1
			stencil = binary[i:(i+filter_size),j:(j+filter_size)].flatten()
			stencil[stencil == 0] = 1
			stencil[stencil == 255] = 0
			sparse = stencil.nonzero()[0]
			if sparse.size != 0:
				to_save.append(sparse)
	
	np.savez(output_name, tuple(to_save))

# finished message
print("completed processing all images")
