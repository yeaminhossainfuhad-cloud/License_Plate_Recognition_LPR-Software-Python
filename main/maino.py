# --- Your existing imports ---
import cv2
import tkinter as tk
from tkinter import Label, Button, Entry, Frame
from PIL import Image, ImageTk
import easyocr
from ultralytics import YOLO
import numpy as np
import imutils
import pandas as pd
import re
import json
import os

# --- Model loading ---
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['bn', 'en'])

# --- Utility functions ---
def extract_digits(text):
    return ''.join(re.findall(r'[০-৯0-9]', text))

def bangla_to_english_digits(text):
    bangla = '০১২৩৪৫৬৭৮৯'
    english = '0123456789'
    return ''.join(english[bangla.index(ch)] if ch in bangla else ch for ch in text)

def clean_plate_text(text):
    corrections = {
        'O': '0', 'I': '1', '|': '1', 'l': '1', 'B': '8',
        '১': '1', '২': '2', '৩': '3', '৪': '4', '৫': '5',
        '৬': '6', '৭': '7', '৮': '8', '৯': '9', '০': '0'
    }
    return ''.join(corrections.get(c, c) for c in text).strip()

# --- Main App ---
class LicensePlateApp:
    def __init__(self, window):
        self.window = window
        self.window.title("LPR - License Plate Recognition")
        self.settings_file = "settings.json"
        self.dark_mode = self.load_theme_setting()
        self.detected_plate = ""

        self.video = cv2.VideoCapture(0)

        self.main_frame = Frame(window)
        self.info_frame = Frame(window)

        self.auto_capture = True
        self.frame_counter = 0
        self.last_detected_plate = ""

        self.setup_main_frame()
        self.setup_info_frame()
        self.apply_theme()

        self.main_frame.pack(fill="both", expand=True)
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_theme_setting(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    return settings.get("dark_mode", False)
            except Exception:
                return False
        return False

    def save_theme_setting(self):
        try:
            with open(self.settings_file, "w") as f:
                json.dump({"dark_mode": self.dark_mode}, f)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def setup_main_frame(self):
        self.main_nav = Frame(self.main_frame, height=50)
        self.main_nav.pack(fill="x", side="top")

        self.logo = Label(self.main_nav, text="LPR", font=("Helvetica", 16, "bold"))
        self.logo.pack(side="left", padx=10)

        self.toggle_button = Button(self.main_nav, text="🌙", command=self.toggle_dark_mode)
        self.toggle_button.pack(side="left")

        self.search_button = Button(self.main_nav, text="Search", command=self.lookup_manual_plate)
        self.search_button.pack(side="right", padx=5, pady=10)

        self.manual_entry = Entry(self.main_nav, font=("Helvetica", 12), width=20)
        self.manual_entry.pack(side="right", padx=5, pady=10)

        self.canvas = Label(self.main_frame)
        self.canvas.pack()

        self.auto_button = Button(self.main_frame, text="Auto Capture: ON", command=self.toggle_auto_capture)
        self.auto_button.pack(side="right", pady=5)

        self.capture_button = Button(self.main_frame, text="Capture Plate", command=self.capture_plate)
        self.capture_button.pack(pady=5)

        self.plate_label = Label(self.main_frame, text="Detected Plate-Number:", font=("Helvetica", 12))
        self.plate_label.pack()

        self.plate_entry = Entry(self.main_frame, font=("Helvetica", 14), width=20, justify="center")
        self.plate_entry.pack(pady=5)

        self.continue_button = Button(self.main_frame, text="Continue", command=self.lookup_corrected_plate)
        self.continue_button.pack(pady=10)

    def setup_info_frame(self):
        self.info_nav = Frame(self.info_frame, height=50)
        self.info_nav.pack(fill="x", side="top")

        self.back_button = Button(self.info_nav, text="← Back", command=self.back_to_main)
        self.back_button.pack(side="left", padx=10, pady=10)

        self.logo_info = Label(self.info_nav, text="LPR", font=("Helvetica", 16, "bold"))
        self.logo_info.pack(side="left", padx=10)

        self.info_toggle_button = Button(self.info_nav, text="🌙", command=self.toggle_dark_mode)
        self.info_toggle_button.pack(side="right", padx=10)

        self.info_label = Label(self.info_frame, text="", font=("Helvetica", 14), justify="left")
        self.info_label.pack(pady=10)

        # Add image labels
        self.number_plate_image_label = Label(self.info_frame)
        self.number_plate_image_label.pack(pady=5)

        self.vehicle_image_label = Label(self.info_frame)
        self.vehicle_image_label.pack(pady=5)

    def switch_to_info_frame(self):
        self.main_frame.pack_forget()
        self.info_frame.pack(fill="both", expand=True)
        self.apply_theme()

    def back_to_main(self):
        self.info_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)
        self.apply_theme()

    def update(self):
        ret, frame = self.video.read()
        if ret and self.main_frame.winfo_ismapped():
            self.current_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            self.canvas.configure(image=photo)
            self.canvas.image = photo

            if self.auto_capture:
                self.frame_counter += 1
                if self.frame_counter % 15 == 0:
                    self.auto_detect_plate(self.current_frame)

        self.window.after(10, self.update)

    def preprocess_for_ocr(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        _, thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def capture_plate(self):
        self.auto_detect_plate(self.current_frame)

    def auto_detect_plate(self, frame):
        frame = imutils.resize(frame, width=640)
        results = model(frame)
        plates = results[0].boxes

        plate_img = None
        for box in plates:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            break

        if plate_img is None:
            return

        processed = self.preprocess_for_ocr(plate_img)
        results = reader.readtext(processed)

        texts = []
        for (_, text, prob) in results:
            if prob > 0.5:
                cleaned = clean_plate_text(text)
                if len(extract_digits(cleaned)) == 6:
                    texts.append(cleaned)

        current_detected = " ".join(texts) if texts else ""

        if current_detected and current_detected != self.last_detected_plate:
            self.last_detected_plate = current_detected
            self.plate_entry.delete(0, tk.END)
            self.plate_entry.insert(0, current_detected)

    def toggle_auto_capture(self):
        self.auto_capture = not self.auto_capture
        state = "ON" if self.auto_capture else "OFF"
        self.auto_button.config(text=f"Auto Capture: {state}")

    def lookup_corrected_plate(self):
        plate_text = self.plate_entry.get().strip()
        self.lookup_plate(plate_text)

    def lookup_manual_plate(self):
        plate_text = self.manual_entry.get().strip()
        self.lookup_plate(plate_text)

    def lookup_plate(self, plate_text):
        raw_digits = extract_digits(plate_text)
        numeric_plate = bangla_to_english_digits(raw_digits)

        try:
            df = pd.read_excel("vehicle_database.xlsx")
            df['Plate Number'] = df['Plate Number'].astype(str).str.strip()

            match = df[df['Plate Number'] == numeric_plate]

            if match.empty:
                self.info_label.config(text=f"No record found for: {numeric_plate}")
                self.number_plate_image_label.config(image="")
                self.vehicle_image_label.config(image="")
            else:
                info = match.iloc[0].to_dict()
                display_text = "\n".join([f"{key}: {value}" for key, value in info.items()])
                self.info_label.config(text=f"Vehicle Info:\n\n{display_text}")

                # Load and show number plate image
                np_path = info.get("Number plate Image", "")
                vehicle_path = info.get("Vehicle Image", "")

                def show_image(label, path):
                    if os.path.exists(path):
                        img = Image.open(path).resize((300, 180))
                        photo = ImageTk.PhotoImage(img)
                        label.config(image=photo)
                        label.image = photo
                    else:
                        label.config(text="Image not found", image="")

                show_image(self.number_plate_image_label, np_path)
                show_image(self.vehicle_image_label, vehicle_path)

            self.switch_to_info_frame()

        except FileNotFoundError:
            self.info_label.config(text="Database file not found: vehicle_database.xlsx")
            self.switch_to_info_frame()
        except Exception as e:
            self.info_label.config(text=f"Error: {str(e)}")
            self.switch_to_info_frame()

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.save_theme_setting()

    def apply_theme(self):
        dark_bg = "#1e1e1e"
        dark_fg = "#f0f0f0"
        light_bg = "#ffffff"
        light_fg = "#000000"
        entry_bg = "#2c2c2c" if self.dark_mode else "#ffffff"

        bg = dark_bg if self.dark_mode else light_bg
        fg = dark_fg if self.dark_mode else light_fg

        self.window.configure(bg=bg)
        for frame in [self.main_frame, self.info_frame, self.main_nav, self.info_nav]:
            frame.configure(bg=bg)

        widgets = [
            self.logo, self.capture_button, self.auto_button, self.continue_button, self.search_button,
            self.plate_label, self.plate_entry, self.manual_entry, self.logo_info,
            self.info_label, self.back_button, self.toggle_button, self.info_toggle_button
        ]
        for widget in widgets:
            if isinstance(widget, (Button, Label)):
                widget.configure(bg=bg, fg=fg, activebackground=bg, activeforeground=fg)
            elif isinstance(widget, Entry):
                widget.configure(bg=entry_bg, fg=fg, insertbackground=fg)

        self.toggle_button.configure(text="☀" if self.dark_mode else "🌙")
        self.info_toggle_button.configure(text="☀" if self.dark_mode else "🌙")

    def on_closing(self):
        self.video.release()
        self.window.destroy()

# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
