import math
import cv2
import numpy as np


# region Direct Mapping 0-Order

# Create a function for Resizing Direct Mapping: 0-Order algorithm
# First parameter: Input image that will be resized
# Second parameter: resizing factor along the horizontal and the vertical axes
def direct_mapping_0Order(image, factor):
    if len(image.shape) == 3:
        # Seeing the shape width (row) and height (column) and channels
        # of the image using shape attribute.
        [row, column, channel] = image.shape

        # calculate the new dimensions of the resized image using `factor`.
        new_row = row * factor
        new_column = column * factor

        # Creating a new matrix of zeros with the new dimensions of the resized image.
        # We use the zeros function of NumPy to create this new matrix.
        # We also specify the data type of the matrix as unsigned 8-bit integers using the dtype argument.
        new_image = np.zeros([new_row, new_column, channel], dtype=np.uint8)

        # We iterate over each channel of the image using k
        # Then for each pixel in the image using i and j
        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    new_image[i * factor:i * factor + factor, j * factor:j * factor + factor, k] = image[i, j, k]
        return new_image

    elif len(image.shape) == 2:
        [row, column] = image.shape

        new_row = row * factor
        new_column = column * factor

        new_image = np.zeros([new_row, new_column], dtype=np.uint8)

        for i in range(row):
            for j in range(column):
                new_image[i * factor:i * factor + factor, j * factor:j * factor + factor] = image[i, j]

        return new_image


# endregion

# region Direct Mapping 1-Order

# Create a function for Resizing Direct Mapping: 1-Order algorithm
# First parameter: Input image that will be resized
# Second parameter: resizing factor along the horizontal and the vertical axes
def direct_mapping_1Order(old_image, factor):
    if len(old_image.shape) == 3:
        # Saving the shape width (row) and height (column) and channels
        # of the image using shape attribute.
        [row, column, channel] = old_image.shape

        # calculate the new dimensions of the resized image using `factor`.
        new_row, new_column = row * factor, column * factor

        # Creating a new matrix of zeros with the new dimensions of the resized image.
        # We use the zeros function of NumPy to create this new matrix.
        # We also specify the data type of the matrix as unsigned 8-bit integers using the dtype argument.
        resized_image = np.zeros([new_row, new_column, channel], dtype=np.uint8)

        # Copying an old values of the old image into the new image that will be resized
        # We iterate over each channel of the image using k
        # Then for each pixel in the image using i and j multiplying by the factor
        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    resized_image[i * factor, j * factor, k] = old_image[i, j, k]

        # We iterate throw each row to fill the gaps between the pixels in the resized image.
        # Filling in the rows by comparing adjacent pixel values in each row using linear interpolation
        for k in range(channel):
            for i in range(0, new_row, factor):
                for j in range(0, new_column - factor, factor):
                    # Saving the maximum and minimum pixels from the resized image
                    x = resized_image[i, j, k]
                    y = resized_image[i, j + factor, k]
                    maximum = max(y, x)
                    minimum = min(y, x)

                    # Case of from top (minimum) to bottom (maximum)
                    for pixel in range(1, factor):
                        # Pixel(i) = Round(((Max - Min)/Fact)*i + Min))
                        resized_image[i, j + pixel, k] = round(((maximum - minimum) / factor) * pixel + minimum)

                # The rest of pixels that not between min and max pixels
                resized_image[i, new_column - factor + 1:new_column, k] = resized_image[i, new_column - factor, k]

        # We iterate throw each column to fill the gaps between the pixels in the resized image.
        # Filling in the column by comparing adjacent pixel values in each column using linear interpolation
        for k in range(channel):
            for j in range(0, new_column):
                for i in range(0, new_row - factor, factor):
                    # Saving the maximum and minimum pixels from the resized image
                    x = resized_image[i, j, k]
                    y = resized_image[i + factor, j, k]
                    maximum = max(y, x)
                    minimum = min(y, x)

                    # Case of from right (minimum) to left (maximum)
                    for pixel in range(1, factor):
                        # Pixel(i)= Round(((Max - Min)/Fact)*i + Min))
                        resized_image[i + pixel, j, k] = int(
                            round(((maximum - minimum) / factor) * pixel + minimum))

                # The rest of pixels that not between min and max pixels
                resized_image[new_row - factor:new_row, j, k] = resized_image[new_row - factor, j, k]

        return resized_image

    elif len(old_image.shape) == 2:
        [row, column] = old_image.shape
        new_row, new_column = row * factor, column * factor

        resized_image = np.zeros([new_row, new_column], dtype=np.uint8)

        for i in range(row):
            for j in range(column):
                resized_image[i * factor, j * factor] = old_image[i, j]

        for i in range(0, new_row, factor):
            for j in range(0, new_column - factor, factor):
                x = resized_image[i, j]
                y = resized_image[i, j + factor]
                maximum = max(y, x)
                minimum = min(y, x)

                for pixel in range(1, factor):
                    resized_image[i, j + pixel] = round(((maximum - minimum) / factor) * pixel + minimum)

            resized_image[i, new_column - factor + 1:new_column] = resized_image[i, new_column - factor]

        for j in range(0, new_column):
            for i in range(0, new_row - factor, factor):
                x = resized_image[i, j]
                y = resized_image[i + factor, j]
                maximum = max(y, x)
                minimum = min(y, x)

                for pixel in range(1, factor):
                    resized_image[i + pixel, j] = int(
                        round(((maximum - minimum) / factor) * pixel + minimum))

            resized_image[new_row - factor:new_row, j] = resized_image[new_row - factor, j]

        return resized_image


# endregion

# region Reverse Mapping 0-Order

# Create a function for Resizing Reverse Mapping: 0-Order algorithm
# First parameter: Input image that will be resized
# Second parameter: resizing factor along the horizontal axis
# Third parameter: resizing factor along the vertical axis
def reverse_mapping_0Order(image, row_factor, column_factor):
    if len(image.shape) == 3:
        # Seeing the shape width (row) and height (column) and channels
        # of the image using shape attribute.
        [row, column, channel] = image.shape

        # calculate the new dimensions of the resized image using `factor`.
        new_row = int(row * row_factor)
        new_column = int(column * column_factor)

        # calculate the ratio to access the pixels in old image.
        row_ratio = row / new_row
        column_ratio = column / new_column

        # Creating a new matrix of zeros with the new dimensions of the resized image.
        # We use the zeros function of NumPy to create this new matrix.
        # We also specify the data type of the matrix as unsigned 8-bit integers using the dtype argument.
        new_image = np.zeros([new_row, new_column, channel], dtype=np.uint8)

        # We iterate over each channel of the image using k
        # Then for each pixel in the image using i and j
        for k in range(channel):
            for i in range(new_row):
                old_x = int(i * row_ratio)
                for j in range(new_column):
                    old_y = int(j * column_ratio)
                    new_image[i, j, k] = image[old_x, old_y, k]

        return new_image

    if len(image.shape) == 2:
        [row, column] = image.shape

        new_row = int(row * row_factor)
        new_column = int(column * column_factor)

        row_ratio = row / new_row
        column_ratio = column / new_column

        new_image = np.zeros([new_row, new_column], dtype=np.uint8)

        for i in range(new_row):
            old_x = int(i * row_ratio)
            for j in range(new_column):
                old_y = int(j * column_ratio)
                new_image[i, j] = image[old_x, old_y]

        return new_image


# endregion

# region Reverse Mapping 1-Order (Bilinear Resizing)

# Create a function for Resizing Reverse Mapping: 1-Order algorithm
# First parameter: Input image that will be resized
# Second parameter: resizing factor along the horizontal axis
# Third parameter: resizing factor along the vertical axis
def reverse_mapping_1Order(image, row_factor, column_factor):
    if len(image.shape) == 3:
        # Seeing the shape width (row) and height (column) and channels
        # of the image using shape attribute.
        [row, column, channel] = image.shape

        # calculate the new dimensions of the resized image using `factor`.
        new_row = row * row_factor
        new_column = column * column_factor

        # calculate the ratio to access the pixels in old image.
        row_ratio = row / new_row
        column_ratio = column / new_column

        # Creating a new matrix of zeros with the new dimensions of the resized image.
        # We use the zeros function of NumPy to create this new matrix.
        # We also specify the data type of the matrix as unsigned 8-bit integers using the dtype argument.
        new_image = np.zeros([new_row, new_column, channel], dtype=np.uint8)

        # We iterate over each channel of the image using k
        # Then for each pixel in the image using new_x and new_y
        for k in range(channel):
            for new_x in range(new_row):
                old_x = new_x * row_ratio
                x1 = int(old_x)
                if x1 == 0:
                    x1 = 1
                x2 = x1 + 1
                if x2 >= row:
                    x2 = row - 1
                x_fraction = abs(old_x - x1)

                for new_y in range(new_column):
                    old_y = new_y * column_ratio
                    y1 = int(old_y)
                    if y1 == 0:
                        y1 = 1
                    y2 = y1 + 1
                    if y2 >= column:
                        y2 = column - 1
                    y_fraction = abs(old_y - y1)

                    p1 = image[x1, y1, k]
                    p2 = image[x2, y1, k]
                    p3 = image[x1, y2, k]
                    p4 = image[x2, y2, k]

                    z1 = p1 * (1 - x_fraction) + p2 * x_fraction
                    z2 = p3 * (1 - x_fraction) + p4 * x_fraction

                    new_pixel = z1 * (1 - y_fraction) + z2 * y_fraction
                    new_image[new_x, new_y, k] = int(new_pixel)

        return new_image

    if len(image.shape) == 2:
        [row, column] = image.shape

        new_row = row * row_factor
        new_column = column * column_factor

        row_ratio = row / new_row
        column_ratio = column / new_column

        new_image = np.zeros([new_row, new_column], dtype=np.uint8)

        for new_x in range(new_row):
            old_x = new_x * row_ratio
            x1 = int(old_x)
            if x1 == 0:
                x1 = 1
            x2 = x1 + 1
            if x2 >= row:
                x2 = row - 1
            x_fraction = abs(old_x - x1)

            for new_y in range(new_column):
                old_y = new_y * column_ratio
                y1 = int(old_y)
                if y1 == 0:
                    y1 = 1
                y2 = y1 + 1
                if y2 >= column:
                    y2 = column - 1
                y_fraction = abs(old_y - y1)

                p1 = image[x1, y1]
                p2 = image[x2, y1]
                p3 = image[x1, y2]
                p4 = image[x2, y2]

                z1 = p1 * (1 - x_fraction) + p2 * x_fraction
                z2 = p3 * (1 - x_fraction) + p4 * x_fraction

                new_pixel = z1 * (1 - y_fraction) + z2 * y_fraction
                new_image[new_x, new_y] = int(new_pixel)

        return new_image


# endregion

# region Convert to Gray

# Create a function for converting RGB image to Gray image
# First parameter: Input image that will be converted

def convert_to_gray(rgb_image):
    # Convert the RGB image to grayscale using the formula: gray = 0.3RED + 0.59GREEN + 0.11BLUE
    gray_image = (0.3 * rgb_image[:, :, 0]) + (0.59 * rgb_image[:, :, 1]) + (0.11 * rgb_image[:, :, 2])

    return gray_image.astype(np.uint8)


# endregion

# region Drawing The Histogram

# Create a function for histogram plot
# First parameter: Input image that will be adjustment
def histogram_plot(image):
    global histogram

    # Calculate histogram values for grayscale image
    if len(image.shape) == 2:
        histogram = np.zeros(256, dtype=np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                histogram[image[i, j]] += 1

    # Calculate histogram values for RGB color image
    elif len(image.shape) == 3:
        [row, column, channel] = image.shape
        histogram = np.zeros((256, channel), dtype=np.uint8)
        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    histogram[image[i, j, k], k] += 1

    return histogram


# endregion

# region Contrast Adjustment

# Create a function for contrast adjustment (histogram stretch/shrink)
# First parameter: Input image that will be adjustment
# Second parameter: New value as maximum
# Third parameter: New value as minimum
def contrast_adjustment(image, new_min, new_max):
    if len(image.shape) == 3:
        [row, column, channel] = image.shape

        # Find the minimum and maximum intensity values across the image.
        old_min = np.amin(image, axis=(0, 1))
        old_max = np.amax(image, axis=(0, 1))

        new_image = np.zeros([row, column, channel], dtype=np.uint8)

        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    # New Value = [(Old Value – Old Min)/(Old Max – Old Min)] × (New Max – New Min) + New Min
                    new_value = ((image[i, j, k] - old_min[k]) / (old_max[k] - old_min[k])) * (
                            new_max - new_min) + new_min
                    if new_value > 255:
                        new_value = 255
                    if new_value < 0:
                        new_value = 0
                    new_image[i, j, k] = new_value

        return new_image

    elif len(image.shape) == 2:
        [row, column] = image.shape

        # Find the minimum and maximum intensity values across the image.
        old_min = np.amin(image, axis=(0, 1))
        old_max = np.amax(image, axis=(0, 1))

        new_image = np.zeros([row, column], dtype=np.uint8)

        for i in range(row):
            for j in range(column):
                # New Value = [(Old Value – Old Min)/(Old Max – Old Min)] × (New Max – New Min) + New Min
                new_value = ((image[i, j] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                if new_value > 255:
                    new_value = 255
                if new_value < 0:
                    new_value = 0
                new_image[i, j] = new_value

        return new_image


# endregion

# region Normalization (0 : 255)

def normalization(image):
    if len(image.shape) == 3:
        [row, column, channel] = image.shape

        # Find the minimum and maximum intensity values across the image.
        old_min = np.amin(image, axis=(0, 1))
        old_max = np.amax(image, axis=(0, 1))

        new_image = np.zeros([row, column, channel], dtype=np.uint8)

        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    # New Value = [(Old Value – Old Min)/(Old Max – Old Min)] × (New Max – New Min) + New Min
                    new_value = ((image[i, j, k] - old_min[k]) / (old_max[k] - old_min[k])) * (255 - 0) + 0
                    if new_value > 255:
                        new_value = 255
                    if new_value < 0:
                        new_value = 0
                    new_image[i, j, k] = new_value

        return new_image

    elif len(image.shape) == 2:
        [row, column] = image.shape

        # Find the minimum and maximum intensity values across the image.
        old_min = np.amin(image, axis=(0, 1))
        old_max = np.amax(image, axis=(0, 1))

        new_image = np.zeros([row, column], dtype=np.uint8)

        for i in range(row):
            for j in range(column):
                # New Value = [(Old Value – Old Min)/(Old Max – Old Min)] × (New Max – New Min) + New Min
                new_value = ((image[i, j] - old_min) / (old_max - old_min)) * (255 - 0) + 0
                if new_value > 255:
                    new_value = 255
                if new_value < 0:
                    new_value = 0
                new_image[i, j] = new_value

        return new_image


# endregion

# region Brightness Adjustment

# Create a function for brightness adjustment
# First parameter: Input image that will be adjustment
# Second parameter: Offset that specify you want brightness or darkness image
def brightness_adjustment(image, offset):
    if len(image.shape) == 3:
        [row, column, channel] = image.shape
        new_image = np.zeros([row, column, channel], dtype=np.uint8)

        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    new_value = image[i, j, k] + offset
                    if new_value > 255:
                        new_value = 255
                    if new_value < 0:
                        new_value = 0
                    new_image[i, j, k] = new_value
        return new_image

    elif len(image.shape) == 2:
        [row, column] = image.shape
        new_image = np.zeros([row, column], dtype=np.uint8)

        for i in range(row):
            for j in range(column):
                new_value = image[i, j] + offset
                if new_value > 255:
                    new_value = 255
                if new_value < 0:
                    new_value = 0
                new_image[i, j] = new_value
        return new_image


# endregion

# region Power Law

# Create a function for power law transformations
# First parameter: Input image that will be adjustment
# Second parameter: gamma value
def power_law(image, gamma):
    if len(image.shape) == 3:
        [row, column, channel] = image.shape
        new_image = np.zeros([row, column, channel], dtype=np.uint8)

        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    # New Value = (Old Value) ^ Gamma
                    new_value = image[i, j, k] ** gamma
                    new_image[i, j, k] = new_value

        return normalization(new_image)

    if len(image.shape) == 2:
        [row, column] = image.shape
        new_image = np.zeros([row, column], dtype=np.uint8)

        for i in range(row):
            for j in range(column):
                # New Value = (Old Value) ^ Gamma
                new_value = image[i, j] ** gamma
                new_image[i, j] = new_value

        return normalization(new_image)


# endregion

# region Histogram Equalization

# Create a function for histogram plot
# First parameter: Input image that will be adjustment
def histogram_equalization(image):
    number_of_pixels = histogram_plot(image)

    # Calculate running sum
    running_sum = np.zeros(256, dtype=int)
    running_sum[0] = number_of_pixels[0]
    for i in range(1, 256):
        running_sum[i] = running_sum[i - 1] + number_of_pixels[i]

    # Calculate histogram equalization
    pixels_sum = number_of_pixels.sum()
    equalized_values = np.zeros(256, dtype=int)
    for i in range(256):
        equalized_values[i] = round((255 * running_sum[i]) / pixels_sum)

    return equalized_values, number_of_pixels


# endregion

# region Histogram Matching

def histogram_matching(first_image, second_image):
    # Compute histograms equalization of the input images
    equalization1, number_of_pixels_of_image1 = histogram_equalization(first_image)
    equalization2, number_of_pixels_of_image2 = histogram_equalization(second_image)

    # Create a lookup table to map intensity levels from img to img2
    matching = np.zeros((256,), dtype=np.uint8)
    for i in range(256):
        diff = np.abs(equalization1[i] - equalization2)
        ind = np.argmin(diff)
        matching[i] = ind

    # Apply the matching to the input image
    new_image = matching[first_image]

    # Compute histogram of the output image
    new_image_histogram = histogram_plot(new_image)

    return new_image, new_image_histogram


# endregion

# region Add Two Images

# Create a function for adding image to another image
# First parameter: Input image that will be the original
# Second parameter: Input image that will be the added image as a watermark
def add_two_images(first_image, second_image):
    if len(first_image.shape) == 3 and len(second_image.shape) == 3:

        [row, column, channel] = first_image.shape
        [row2, column2, channel2] = second_image.shape

        # Make sure the two images have the same dimensions
        second_image = reverse_mapping_1Order(
            second_image,
            max(row, row2) // min(row, row2),
            max(column, column2) // min(column, column)
        )

        # Convert the images to NumPy arrays
        first_image = np.array(first_image, dtype=np.uint8)
        second_image = np.array(second_image, dtype=np.uint8)

        # Create a new NumPy array with the same dimensions as the images
        new_image = np.zeros((row, column, 3), dtype=np.uint8)

        for k in range(channel):
            for i in range(row):
                for j in range(column):
                    new_image[i, j, k] = first_image[i, j, k] + second_image[i, j, k]

        new_image = normalization(new_image)
        return new_image

    elif len(first_image.shape) == 2 and len(second_image.shape) == 2:
        # Grayscale images
        row, column = first_image.shape
        row2, column2 = second_image.shape

        # Make sure the two images have the same dimensions
        second_image = reverse_mapping_1Order(
            second_image,
            max(row, row2) // min(row, row2),
            max(column, column2) // min(column, column)
        )

        # Convert the images to NumPy arrays
        first_image = np.array(first_image, dtype=np.uint8)
        second_image = np.array(second_image, dtype=np.uint8)

        # Create a new NumPy array with the same dimensions as the images
        new_image = np.zeros((row, column), dtype=np.uint8)

        for i in range(row):
            for j in range(column):
                new_image[i, j] = first_image[i, j] + second_image[i, j]

        new_image = normalization(new_image)
        return new_image


# endregion

# region Subtract Two Images

# Create a function for adding image to another image
# First parameter: Input image that will be the original
# Second parameter: Input image that will be the added image as a watermark
def subtract_two_images(first_image, second_image):
    [row, column, channel] = first_image.shape

    # Make sure the two images have the same dimensions
    second_image = cv2.resize(second_image, (first_image.shape[1], first_image.shape[0]))

    # Convert the images to NumPy arrays
    first_image = np.array(first_image, dtype=np.uint8)
    second_image = np.array(second_image, dtype=np.uint8)

    # Create a new NumPy array with the same dimensions as the images
    new_image = np.zeros((first_image.shape[0], first_image.shape[1], 3), dtype=np.uint8)

    for k in range(channel):
        for i in range(row):
            for j in range(column):
                new_image[i, j, k] = np.abs(first_image[i, j, k] - second_image[i, j, k])

    new_image = contrast_adjustment(new_image, new_min=0, new_max=255)

    return new_image


# endregion

# region Image Negatives

# Create a function for adding image to another image
# First parameter: Input image that will be the original
# Second parameter: Input image that will be the added image as a watermark
def negative_image(image):
    [row, column, channel] = image.shape
    new_image = np.zeros([row, column, channel], dtype=np.uint8)

    for k in range(channel):
        for i in range(row):
            for j in range(column):
                new_image[i, j, k] = 255 - image[i, j, k]

    return new_image


# endregion

# region Quantization

def quantization(image, number_of_bits):
    [row, column, channel] = image.shape
    new_image = np.zeros([row, column, channel], dtype=np.uint8)

    gray_level = 2 ** number_of_bits
    gap = int(256 / gray_level)
    colors = range(0, 256, gap)

    for k in range(0, channel):
        for r in range(0, row):
            for c in range(0, column):
                new_pixel = image[r, c, k] // gap
                new_image[r, c, k] = colors[new_pixel]

    return new_image


# endregion

# region Average Filter

def average_filter(image, mask_size):
    [row, column] = image.shape
    new_image = np.zeros([row, column], dtype=np.uint8)

    padding = mask_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Get the current 3x3 pixel mask
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            mask_mean = np.sum(mask) / (mask_size ** 2)
            new_image[i - padding, j - padding] = mask_mean

    return new_image


# endregion

# region Weighted (Gaussian) Filter

def gaussian_kernel(size, sigma):
    x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1), np.arange(-size // 2 + 1, size // 2 + 1))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    return kernel / np.sum(kernel)


def gaussian_filter(image, size, sigma):
    [row, column] = image.shape
    new_image = np.zeros([row, column], dtype=np.float32)

    kernel = gaussian_kernel(size, sigma)

    padding = size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    # Apply the filter to each pixel in the image
    for i in range(row):
        for j in range(column):
            neighbor_pixels = padded_image[i:i + size, j:j + size]
            weighted_pixels = neighbor_pixels * kernel

            new_image[i, j] = np.sum(weighted_pixels)

    return new_image.astype(np.uint8)


# endregion

# region Max Filter

def max_filter(image, mask_size):
    [row, column] = image.shape
    new_image = np.zeros([row, column], dtype=np.uint8)

    padding = mask_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Get the current 3x3 pixel mask
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            new_pixel = int(np.max(mask))
            new_image[i - padding, j - padding] = new_pixel

    return new_image


# endregion

# region Min Filter

def min_filter(image, mask_size):
    [row, column] = image.shape
    new_image = np.zeros([row, column], dtype=np.uint8)

    padding = mask_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Get the current 3x3 pixel mask
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            new_pixel = int(np.min(mask))
            new_image[i - padding, j - padding] = new_pixel

    return new_image


# endregion

# region Median Filter

def median_filter(image, mask_size):
    [row, column] = image.shape
    new_image = np.zeros([row, column], dtype=np.uint8)

    padding = mask_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Get the current 3x3 pixel mask
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            mask = np.sort(mask)
            new_pixel = int(np.median(mask))
            new_image[i - padding, j - padding] = new_pixel

    return new_image


# endregion

# region Sharpening Filter

def sharpening(image):
    [row, column] = image.shape

    new_image1 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image2 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image3 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image4 = np.zeros((row, column, 1), dtype=np.uint8)

    # Line Edge Detection
    horizontal_filter = [[0, 1, 0],
                         [0, 1, 0],
                         [0, -1, 0]]

    vertical_filter = [[0, 0, 0],
                       [1, 1, -1],
                       [0, 0, 0]]

    diagonal1_filter = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1]]

    diagonal2_filter = [[0, 0, 1],
                        [0, 1, 0],
                        [-1, 0, 0]]

    padding = 1
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Get the current 3x3 pixel mask
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # Apply each filter to the window and calculate the gradient
            horizontal_gradient = np.sum(np.multiply(mask, horizontal_filter))
            if horizontal_gradient > 255:
                horizontal_gradient = 255
            if horizontal_gradient < 0:
                horizontal_gradient = 0

            vertical_gradient = np.sum(np.multiply(mask, vertical_filter))
            if vertical_gradient > 255:
                vertical_gradient = 255
            if vertical_gradient < 0:
                vertical_gradient = 0

            diagonal1_gradient = np.sum(np.multiply(mask, diagonal1_filter))
            if diagonal1_gradient > 255:
                diagonal1_gradient = 255
            if diagonal1_gradient < 0:
                diagonal1_gradient = 0

            diagonal2_gradient = np.sum(np.multiply(mask, diagonal2_filter))
            if diagonal2_gradient > 255:
                diagonal2_gradient = 255
            if diagonal2_gradient < 0:
                diagonal2_gradient = 0

            # Set the value of the new images based on the gradient values
            new_image1[i - padding, j - padding] = np.abs(horizontal_gradient)
            new_image2[i - padding, j - padding] = np.abs(vertical_gradient)
            new_image3[i - padding, j - padding] = np.abs(diagonal1_gradient)
            new_image4[i - padding, j - padding] = np.abs(diagonal2_gradient)

    # combining four different line edge detection filters (horizontal, vertical, and two diagonal filters)
    # using bitwise OR operations.
    sharpened_image = cv2.bitwise_or(new_image1, new_image2)
    sharpened_image = cv2.bitwise_or(sharpened_image, new_image3)
    sharpened_image = cv2.bitwise_or(sharpened_image, new_image4)

    return sharpened_image


# endregion

# region Edge Detection Filter

def edge_detection(image):
    [row, column] = image.shape

    new_image1 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image2 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image3 = np.zeros((row, column, 1), dtype=np.uint8)
    new_image4 = np.zeros((row, column, 1), dtype=np.uint8)

    # Line Edge Detection
    horizontal_filter = [[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]

    vertical_filter = [[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]]

    diagonal1_filter = [[0, 1, 2],
                        [-1, 0, 1],
                        [-2, -1, 0]]

    diagonal2_filter = [[2, 1, 0],
                        [1, 0, -1],
                        [0, -1, -2]]

    padding = 1
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    for i in range(padding, row + padding):
        for j in range(padding, column + padding):
            # Get the current 3x3 pixel mask
            mask = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # Apply each filter to the window and calculate the gradient
            horizontal_gradient = np.sum(np.multiply(mask, horizontal_filter))
            vertical_gradient = np.sum(np.multiply(mask, vertical_filter))
            diagonal1_gradient = np.sum(np.multiply(mask, diagonal1_filter))
            diagonal2_gradient = np.sum(np.multiply(mask, diagonal2_filter))

            # Set the value of the new images based on the gradient values
            new_image1[i - padding, j - padding] = np.abs(horizontal_gradient)
            new_image2[i - padding, j - padding] = np.abs(vertical_gradient)
            new_image3[i - padding, j - padding] = np.abs(diagonal1_gradient)
            new_image4[i - padding, j - padding] = np.abs(diagonal2_gradient)

    # Combine the new images into one image using the maximum gradient
    final_image = np.maximum(np.maximum(np.maximum(new_image1, new_image2), new_image3), new_image4)

    final_image = final_image.astype(np.uint8)

    return normalization(final_image)


# endregion

# region Unsharpened Filter

def unsharpened(image):
    gaussian_image = gaussian_filter(image, 5, 2)
    subtract_image = subtract_two_images(first_image=image, second_image=gaussian_image)

    return add_two_images(first_image=image, second_image=subtract_image)


# endregion

# region Ideal Low Pass Filter

def ideal_low_pass(image, radius):
    [row, column, channel] = image.shape

    dft = np.fft.fft2(image, axes=(0, 1))
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros_like(image)

    for k in range(0, channel):
        for i in range(0, row):
            for j in range(0, column):
                distance = int(((((i - (row / 2)) ** 2) + ((j - (column / 2)) ** 2)) ** 0.5))
                if distance > radius:
                    mask[i, j, k] = 0
                else:
                    mask[i, j, k] = 255

    dft_shift_masked = np.multiply(dft_shift, mask) / 255

    back_is_shift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_is_shift_masked, axes=(0, 1))
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

    return img_filtered


# endregion

# region Ideal High Pass Filter

def ideal_high_pass(image, radius):
    [row, column, channel] = image.shape

    dft = np.fft.fft2(image, axes=(0, 1))
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros_like(image)

    for k in range(0, channel):
        for i in range(0, row):
            for j in range(0, column):
                distance = int(((((i - (row / 2)) ** 2) + ((j - (column / 2)) ** 2)) ** 0.5))
                if distance > radius:
                    mask[i, j, k] = 255
                else:
                    mask[i, j, k] = 0

    dft_shift_masked = np.multiply(dft_shift, mask) / 255

    back_is_shift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_is_shift_masked, axes=(0, 1))
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

    return img_filtered


# endregion

# region Butterworth Low Pass Filter

def butterworth_low_pass(image, radius, n):
    [row, column, channel] = image.shape

    dft = np.fft.fft2(image, axes=(0, 1))
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros_like(image)

    for k in range(0, channel):
        for i in range(0, row):
            for j in range(0, column):
                distance = int(((((i - (row / 2)) ** 2) + ((j - (column / 2)) ** 2)) ** 0.5))
                temp = (1 / (1 + (distance / radius) ** (2 * n))) * 255
                mask[i, j, k] = temp

    dft_shift_masked = np.multiply(dft_shift, mask) / 255

    back_is_shift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_is_shift_masked, axes=(0, 1))
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

    return img_filtered


# endregion

# region Butterworth High Pass Filter

def butterworth_high_pass(image, radius, n):
    [row, column, channel] = image.shape

    dft = np.fft.fft2(image, axes=(0, 1))
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros_like(image)

    for k in range(0, channel):
        for i in range(0, row):
            for j in range(0, column):
                distance = int(((((i - (row / 2)) ** 2) + ((j - (column / 2)) ** 2)) ** 0.5))
                temp = (1 / (1 + (distance / radius) ** (2 * n))) * 255
                mask[i, j, k] = 255 - temp

    dft_shift_masked = np.multiply(dft_shift, mask) / 255

    back_is_shift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_is_shift_masked, axes=(0, 1))
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

    return img_filtered


# endregion

# region Gaussian Low Pass Filter

def gaussian_low_pass(image, radius):
    [row, column, channel] = image.shape

    dft = np.fft.fft2(image, axes=(0, 1))
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros_like(image)

    for k in range(0, channel):
        for i in range(0, row):
            for j in range(0, column):
                distance = int(((((i - (row / 2)) ** 2) + ((j - (column / 2)) ** 2)) ** 0.5))
                mask[i, j, k] = int((math.exp(int(-pow(distance, 2) / (2 * pow(radius, 2))))))
                mask[i, j, k] = mask[i, j, k] * 255

    dft_shift_masked = np.multiply(dft_shift, mask) / 255

    back_is_shift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_is_shift_masked, axes=(0, 1))
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

    return img_filtered


# endregion

# region Gaussian High Pass Filter

def gaussian_high_pass(image, radius):
    [row, column, channel] = image.shape

    dft = np.fft.fft2(image, axes=(0, 1))
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros_like(image)

    for k in range(0, channel):
        for i in range(0, row):
            for j in range(0, column):
                distance = int(((((i - (row / 2)) ** 2) + ((j - (column / 2)) ** 2)) ** 0.5))
                mask[i, j, k] = int((math.exp(int(-pow(distance, 2) / (2 * pow(radius, 2))))))
                mask[i, j, k] = 255 - mask[i, j, k] * 255

    dft_shift_masked = np.multiply(dft_shift, mask) / 255

    back_is_shift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_is_shift_masked, axes=(0, 1))
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

    return img_filtered

# endregion
