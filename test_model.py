from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
import numpy as np
import cv2
from math import ceil
from sklearn.metrics import classification_report

# Ścieżki do modelu i danych testowych
model_dir = 'model'
#model_name = 'modelBASIC.h5'
#model_name = 'modelMIDDLE.h5'
model_name = 'modelPRO.h5'
image_path = 'data/test'

# Załaduj model
model = load_model(os.path.join(model_dir, model_name))

# Pobierz listę obrazów testowych i ich etykiety
image_names = os.listdir(image_path)
true_labels = [0 if 'center' in name else 1 if 'left' in name else 2 for name in image_names]

# Mapowanie predykcji na etykiety
label_map = {0: "center", 1: "left", 2: "right"}

# Rozmiary obrazu i siatki
image_size = 128  # Rozmiar obrazu w pikselach (zakładamy kwadratowe)
grid_columns = 3  # Liczba obrazów w wierszu
grid_rows = ceil(len(image_names) / grid_columns)

# Oblicz rozmiar dodatkowego obszaru na statystyki
stats_width = 300
grid_image = np.ones(((image_size + 50) * grid_rows, image_size * grid_columns + stats_width, 3), dtype=np.uint8) * 255

# Statystyki
predictions = []
correct_predictions = 0
class_correct = {0: 0, 1: 0, 2: 0}
class_total = {0: 0, 1: 0, 2: 0}

# Przetwarzanie obrazów
for idx, image_name in enumerate(image_names):
    # Wczytaj obraz
    img_path = os.path.join(image_path, image_name)
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized.astype('float') / 255.0
    img_array = img_to_array(img_normalized)
    img_array = np.expand_dims(img_array, axis=0)

    # Predykcja
    pred = model.predict(img_array)
    pred_label = pred.argmax(axis=1)[0]
    predictions.append(pred_label)

    # Sprawdź poprawność
    true_label = true_labels[idx]
    class_total[true_label] += 1
    if pred_label == true_label:
        correct_predictions += 1
        class_correct[true_label] += 1

    # Wyznacz pozycję obrazu w siatce
    row = idx // grid_columns
    col = idx % grid_columns
    x_offset = col * image_size
    y_offset = row * (image_size + 50)

    # Wklej obraz do siatki
    grid_image[y_offset:y_offset + image_size, x_offset:x_offset + image_size] = img_resized

    # Dodaj etykietę predykcji
    pred_text = f"Pred: {label_map[pred_label]}"
    true_text = f"True: {label_map[true_label]}"
    color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)

    cv2.putText(grid_image, pred_text, (x_offset, y_offset + image_size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(grid_image, true_text, (x_offset, y_offset + image_size + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# Oblicz dokładność
accuracy = correct_predictions / len(image_names) * 100

# Oblicz accuracy dla każdej klasy
class_accuracies = {label_map[k]: (class_correct[k] / class_total[k] * 100) if class_total[k] > 0 else 0 for k in class_total}

# Dodaj statystyki do obrazu po prawej stronie
start_x = image_size * grid_columns + 20  # Początek przestrzeni na statystyki
cv2.putText(grid_image, f"Overall Accuracy: {accuracy:.2f}%", (start_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Dodaj accuracy dla każdej klasy
y_offset_stats = 70
for class_name, class_acc in class_accuracies.items():
    cv2.putText(grid_image, f"{class_name.capitalize()} Accuracy: {class_acc:.2f}%", (start_x, y_offset_stats), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_offset_stats += 30

# Wyświetl i zapisz wynik
cv2.imshow("Results", grid_image)
cv2.imwrite("test_results_with_class_accuracy_PRO.png", grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
