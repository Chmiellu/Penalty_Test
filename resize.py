import cv2
import os

# Ścieżka do głównego folderu
base_path = 'data/test'

# Pobranie listy wszystkich plików w folderze
image_names = os.listdir(base_path)

for image in image_names:

    image_path = os.path.join(base_path, image)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Nie udało się odczytać obrazu: {image}")
        continue

    resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    cv2.imwrite(image_path, resized_img)

print("FINISHED")
