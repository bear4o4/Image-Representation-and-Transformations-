import cv2
import matplotlib.pyplot as plt
import numpy as np



grayscale_image = cv2.imread('grey_scale_image.jpg', cv2.IMREAD_GRAYSCALE)
rgb_image = cv2.imread('RGB_image.jpg', cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rgb_image)
plt.title('RGB Image')
plt.axis('off')

plt.show()


print(f"Grayscale image dimensions: {grayscale_image.shape}")
print(f"RGB image dimensions: {rgb_image.shape}")

print("##############################################")

#task 2


red_channel = rgb_image[:, :, 0]
green_channel = rgb_image[:, :, 1]
blue_channel = rgb_image[:, :, 2]


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(red_channel, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.imshow(green_channel, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(blue_channel, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.show()


print("##############################################")

#task 3
manual_grayscale_image = rgb_image.mean(axis=2).astype('uint8')

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(manual_grayscale_image, cmap='gray')
plt.title('Manual Grayscale Image')
plt.axis('off')

builtin_grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

plt.subplot(1, 2, 2)
plt.imshow(builtin_grayscale_image, cmap='gray')
plt.title('Built-in Grayscale Image')
plt.axis('off')

plt.show()


print(f"Manual grayscale image dimensions: {manual_grayscale_image.shape}")
print(f"Built-in grayscale image dimensions: {builtin_grayscale_image.shape}")


print("##############################################")

#task 4


manual_resized_image = grayscale_image[::2, ::2]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(manual_resized_image, cmap='gray')
plt.title('Manual Resized Image')
plt.axis('off')

builtin_resized_image = cv2.resize(grayscale_image, (grayscale_image.shape[1] // 2, grayscale_image.shape[0] // 2))

plt.subplot(1, 2, 2)
plt.imshow(builtin_resized_image, cmap='gray')
plt.title('Built-in Resized Image')
plt.axis('off')

plt.show()

print(f"Manual resized image dimensions: {manual_resized_image.shape}")
print(f"Built-in resized image dimensions: {builtin_resized_image.shape}")


print("##############################################")

#task 5

manual_rotated_image = np.rot90(grayscale_image, k=-1)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(manual_rotated_image, cmap='gray')
plt.title('Manual Rotated Image')
plt.axis('off')

builtin_rotated_image = cv2.rotate(grayscale_image, cv2.ROTATE_90_CLOCKWISE)

plt.subplot(1, 2, 2)
plt.imshow(builtin_rotated_image, cmap='gray')
plt.title('Built-in Rotated Image')
plt.axis('off')

plt.show()

print(f"Manual rotated image dimensions: {manual_rotated_image.shape}")
print(f"Built-in rotated image dimensions: {builtin_rotated_image.shape}")

print("##############################################")

#task 6

translation_matrix = np.float32([[1, 0, 300], [0, 1, 200]])


translated_image = cv2.warpAffine(grayscale_image, translation_matrix, (grayscale_image.shape[1], grayscale_image.shape[0]))


plt.figure(figsize=(10, 5))
plt.imshow(translated_image, cmap='gray')
plt.title('Translated Image')
plt.axis('off')
plt.show()


print(f"Translated image dimensions: {translated_image.shape}")



print("##############################################")

#task 7


cv2.imwrite('manual_grayscale_image.jpg', manual_grayscale_image)


cv2.imwrite('builtin_grayscale_image.jpg', builtin_grayscale_image)


cv2.imwrite('manual_resized_image.jpg', manual_resized_image)


cv2.imwrite('builtin_resized_image.jpg', builtin_resized_image)


cv2.imwrite('manual_rotated_image.jpg', manual_rotated_image)


cv2.imwrite('builtin_rotated_image.jpg', builtin_rotated_image)


cv2.imwrite('translated_image.jpg', translated_image)