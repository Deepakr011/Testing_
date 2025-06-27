import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), text_thickness)
    return img, results

def load_image():
    global img_path, input_img

    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not img_path:
        return

    input_img = cv2.imread(img_path)
    show_image(input_img, input_label)

def detect_objects():
    global result_img
    if input_img is None:
        messagebox.showerror("Error", "Please load an image first!")
        return

    result_img, results = predict_and_detect(model, input_img.copy(), classes=[], conf=0.5)

    # Check for specific tumor detection
    detected_tumors = []
    for result in results:
        for box in result.boxes:
            label_index = int(box.cls[0])
            label_name = result.names[label_index]
            if label_name in ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]:
                detected_tumors.append(label_name)

    if detected_tumors:
        message = f"Tumor(s) detected: {', '.join(detected_tumors)}.\nPlease consult a medical professional."
        messagebox.showinfo("Detection Result", message)

    show_image(result_img, result_label)


    result_img, results = predict_and_detect(model, input_img.copy(), classes=[], conf=0.5)

    # Check for brain tumor detection
    detection_made = False
    for result in results:
        for box in result.boxes:
            label_index = int(box.cls[0])
            label_name = result.names[label_index]
            if label_name.lower() == "brain tumor":  # Assuming "brain tumor" is the label name
                detection_made = True
                break

    if detection_made:
        messagebox.showinfo("Detection Result", "Brain tumor detected. Please consult a medical professional.")

    show_image(result_img, result_label)

def save_image():
    if result_img is None:
        messagebox.showerror("Error", "No detection results to save!")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
    if save_path:
        cv2.imwrite(save_path, result_img)
        messagebox.showinfo("Success", "Image saved successfully!")

def show_image(img, label):
    max_width, max_height = 400, 400  # Define max dimensions for the label
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Resize image to fit within max dimensions while preserving aspect ratio
    img_pil.thumbnail((max_width, max_height))

    img_tk = ImageTk.PhotoImage(img_pil)

    label.img_tk = img_tk  # Keep reference to avoid garbage collection
    label.config(image=img_tk)

# Load YOLO model
model = YOLO(r"D:\detection_project\best (15).pt")

# Initialize Tkinter window
root = tk.Tk()
root.title("Brain Tumor Detection GUI")
root.configure(bg="#2E3440")

# Style variables
button_style = {"bg": "#88C0D0", "fg": "#2E3440", "font": ("Arial", 12, "bold"), "relief": "raised"}
label_style = {"bg": "#4C566A", "fg": "#ECEFF4", "font": ("Arial", 12, "bold"), "relief": "solid"}

# Variables
img_path = None
input_img = None
result_img = None

# Frames
frame_top = Frame(root, bg="#2E3440")
frame_top.grid(row=0, column=0, columnspan=2, pady=10)

frame_bottom = Frame(root, bg="#2E3440")
frame_bottom.grid(row=1, column=0, columnspan=2, pady=10)

# Input image label
input_label = Label(frame_top, text="Input Image", **label_style)
input_label.grid(row=0, column=0, padx=10, pady=10)

# Result image label
result_label = Label(frame_top, text="Result Image", **label_style)
result_label.grid(row=0, column=1, padx=10, pady=10)

# Buttons
load_button = Button(frame_bottom, text="Load Image", command=load_image, **button_style)
load_button.grid(row=1, column=0, padx=20, pady=10)

detect_button = Button(frame_bottom, text="Detect Objects", command=detect_objects, **button_style)
detect_button.grid(row=1, column=1, padx=20, pady=10)

save_button = Button(frame_bottom, text="Save Result", command=save_image, **button_style)
save_button.grid(row=2, column=1, padx=20, pady=10)

# Start the Tkinter event loop
root.mainloop()
