import glob
import os
import numpy as np
import cv2


def define_brightness_mask(input_directory):
    filenames = glob.glob(f"{input_directory}//*.png")
    first_image = cv2.imread(filenames[0], cv2.IMREAD_GRAYSCALE)
    h, w = first_image.shape
    image_data = np.zeros((h, w, len(filenames)), dtype=np.uint8)

    for i, filename in enumerate(filenames):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        image[image == 0] = np.finfo(float).eps
        image_data[:, :, i] = image

    max_brightness = np.max(image_data, axis=2)
    max_brightness[max_brightness < np.quantile(max_brightness, 0.85)] = 255
    count_array = np.zeros_like(max_brightness)
    brightness_threshold = max_brightness * 0.95
    brightness_threshold = brightness_threshold.astype(np.uint8)
    for i in range(image_data.shape[2]):
        idx = image_data[:, :, i] > brightness_threshold
        count_array[idx] += 1
    count_array[count_array > 2] = 255
    binary_image = count_array
    binary_image = cv2.dilate(binary_image, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8), iterations=10)
    binary_image = cv2.erode(binary_image, np.ones((3, 3), np.uint8), iterations=5)
    edges = cv2.Canny(binary_image, 100, 200)
    edges[1, binary_image[1, :] == 255] = 1
    edges[-1, binary_image[-1, :] == 255] = 1
    edges[binary_image[:, 1] == 255, 1] = 1
    edges[binary_image[:, -1] == 255, -1] = 1
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_image, np.uint8)
    for cnt in contours:
        _, radius = cv2.minEnclosingCircle(cnt)
        if radius > 15.0:
            cv2.drawContours(mask, [cnt], 0, 255, -1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_path = os.path.join(input_directory, "mask.npy")
    response = input("Enter y to save the created mask to " + save_path + ":\n")
    if response == "y":
        np.save(save_path, mask == 255)
    new_mask = np.load(save_path)
    first_image[new_mask] = 0
    cv2.imshow("Masked Image", first_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    define_brightness_mask(r"C:\Users\JohannesBurger\AIIS\3D_Ice_Shedding_Trajectory_Reconstruction_on_a_Full"
                           r"-Scale_Propeller\02_Data\Calib2\10\ChronosMono\SE_01\PNG")
