import os
import cv2
import numpy as np
import openslide
# from openslide.deepzoom import DeepZoomGenerator
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import filters
from skimage.measure import shannon_entropy
import torch

class ProcessSVS:
    def __init__(self, slide_dir, output_dir):
        self.slide_dir = slide_dir
        self.output_dir = output_dir
        self.model = self.load_model()

    def load_model(self):
        # Load a pretrained YOLO model from Ultralytics' hub
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def detect_and_save_cells(self, image_path, output_dir):
        image = cv2.imread(image_path)
        if image is None:
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb, size=640)  # Adjust size according to your requirements
        results = results.pandas().xyxy[0]  # Extract bounding boxes as DataFrame

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        for index, row in results.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            segmented_cell = cv2.bitwise_and(image, image, mask=mask)
            cell_filename = output_dir / f"/Test_cells_using_yolo/cell_{index}.jpg"
            cv2.imwrite(str(cell_filename), segmented_cell)
            print(f"Saved: {cell_filename}")

    def is_low_contrast_and_edges(self, tile, edge_threshold=100, contrast_threshold=0.005):
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        if np.max(gray) - np.min(gray) < contrast_threshold * 255 or np.sum(filters.sobel(gray) > 0.01) < edge_threshold:
            return True
        return False

    def is_empty_tile(self, tile, std_dev_threshold=10, entropy_threshold=3.5):
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        if np.std(gray) < std_dev_threshold or shannon_entropy(gray) < entropy_threshold:
            return True
        if np.mean(tile, axis=(0, 1))[0] > 200:  # Adjust based on your data
            return True
        return False

    # draw polygons around the cells in a tile
    def process_tile(self, tile):
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(tile, markers)
        tile[markers == -1] = [255, 0, 0]  # Draw boundaries in red

        for label in np.unique(markers):
            if label == 1:
                continue
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[markers == label] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tile, contours, -1, (0, 255, 0), 2)

        return tile
    
    # extracts tiles from each slide 
    def process_slide(self, slide_path, slide_name, level=0, tile_size=(800, 800)):
        slide = openslide.OpenSlide(slide_path)
        dims = slide.level_dimensions[level]
        x_tiles = dims[0] // tile_size[0]
        y_tiles = dims[1] // tile_size[1]

        slide_save_dir = f"{self.output_dir}/{slide_name}_test"
        if not os.path.exists(slide_save_dir):
            os.makedirs(slide_save_dir)

        for x in range(x_tiles):
            for y in range(y_tiles):
                x_coord = x * tile_size[0]
                y_coord = y * tile_size[1]
                tile = slide.read_region((x_coord, y_coord), level, tile_size)
                tile = np.array(tile.convert('RGB'))
                if not (self.is_empty_tile(tile) or self.is_low_contrast_and_edges(tile)):
                    processed_tile = self.process_tile(tile)
                    cv2.imwrite(f"{slide_save_dir}/SUB_{slide_name}_tile_({x}_{y}).png", processed_tile)
                # Optionally display the tile
                # plt.imshow(processed_tile)
                # plt.title(f'Tile ({x}, {y}) Processed')
                # plt.show()

    def run(self):
        for slide_path in os.listdir(self.slide_dir):
            if slide_path.endswith('.svs'):
                slide_name = os.path.splitext(slide_path)[0]
                print(f"Processing {slide_path} at level 0...")
                self.process_slide(slide_path=f"{self.slide_dir}/{slide_path}", slide_name=slide_name)

# Example usage:
main_dir = "/media/ist/drive2/MANSOOR/Neuroimaging-Project/Breast_Cancer_Classification_Project"
SVS_dir = f"{main_dir}/SVS_Data"
Tiles_dir = f"{main_dir}/Tiles_Data"
Output_dir = f"{main_dir}/Output_Data"
processor = ProcessSVS(SVS_dir, Output_dir)
# processor = ProcessSVS(Tiles_dir, Output_dir)
processor.run()
