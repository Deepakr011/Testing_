import cv2
from ultralytics import YOLO 

def run_yolov8(model_path, image_path, output_path):
    model = YOLO(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    results = model.predict(image, conf=0.1)

    annotated_image = results[0].plot()

    cv2.imwrite(output_path, annotated_image)
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    print("welcome to the project")
    model_path = r"D:\detection_project\best (14).pt"
    image_path = r"D:\detection_project\036_0_jpg.rf.02f5cd69c73723649bbf9c74c3bb276a.jpg"
    output_path = r"D:\detection_project\output_image.jpg"


    run_yolov8(model_path, image_path, output_path)
