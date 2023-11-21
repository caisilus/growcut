import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np

from growcut import GrowCut

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GrowCut Interactive Segmentation")

        self.original_image = None
        self.segmantation_mask = None
        self.segmentation_output = None

        self.drawing = None
        self.object_color = (255, 0, 0)
        self.background_color = (0, 0, 255)
        self.last_x = None
        self.last_y = None
        
        self.growcut = GrowCut()
        
        self.setup_ui()

    def setup_ui(self):
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, pady=10, padx=10)

        self.growcut_button = tk.Button(root, text="GrowCut", command=self.growcut_segmentation)
        self.growcut_button.grid(row=0, column=1, pady=10, padx=10)

        self.original_image_canvas = tk.Canvas(root)
        self.original_image_canvas.grid(row=1, column=0, columnspan=2)

        self.original_image_canvas.bind("<B1-Motion>", self.draw_object_brush)
        self.original_image_canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        self.original_image_canvas.bind("<B3-Motion>", self.draw_background_brush)
        self.original_image_canvas.bind("<ButtonRelease-3>", self.stop_drawing)

        self.segmentation_output_label = tk.Label(root)
        self.segmentation_output_label.grid(row=2, column=0, columnspan=2)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])

        if not file_path:
            return

        self.original_image = cv.imread(file_path)
        self.original_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2RGB)
        self.segmantation_mask = np.zeros_like(self.original_image)
        self.update_canvas()

    def update_canvas(self):
        self.original_image_tk = ImageTk.PhotoImage(Image.fromarray(self.original_image))
        self.original_image_canvas.config(width=self.original_image.shape[1], height=self.original_image.shape[0])
        self.original_image_canvas.create_image(0, 0, anchor=tk.NW, image=self.original_image_tk)

    def growcut_segmentation(self):
        self.segmentation_output = self.growcut.growcut(self.original_image, self.segmantation_mask)
        segmentation_output_tk = ImageTk.PhotoImage(Image.fromarray(self.segmentation_output))
        self.segmentation_output_label.config(image=segmentation_output_tk)
        self.segmentation_output_label.image = segmentation_output_tk

    def draw_object_brush(self, event):
        self.draw_with_brush(event, "object")

    def draw_background_brush(self, event):
        self.draw_with_brush(event, "background")

    def draw_with_brush(self, event, brush_name):
        if (self.drawing != brush_name) and (self.drawing is not None):
            return

        self.drawing = brush_name
        x, y = event.x, event.y
        img_x = int((x / self.original_image_canvas.winfo_width()) * self.original_image.shape[1])
        img_y = int((y / self.original_image_canvas.winfo_height()) * self.original_image.shape[0])
        if self.last_x and self.last_y:
            # Update the original image with the drawn brush
            color = self.object_color if self.drawing == "object" else self.background_color
            cv.line(self.original_image, (self.last_x, self.last_y), (x, y), color, 5)
            cv.line(self.segmantation_mask, (self.last_x, self.last_y), (x, y), color, 5)
            self.update_canvas()

        self.last_x, self.last_y = img_x, img_y

    def stop_drawing(self, event):
        self.drawing = None
        self.last_x, self.last_x = None, None

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()