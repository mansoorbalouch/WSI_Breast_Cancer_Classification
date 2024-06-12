import json
import openslide
from shapely.geometry import Polygon, Point
import cv2, os
from PIL import Image, ImageDraw
import numpy as np

class ProcessSVS:
    def __init__(self, slide_dir, output_dir, target_size=(256, 256)):
        self.slide_dir = slide_dir
        self.output_dir = output_dir
        self.target_size = target_size  # Uniform target size for all patches

    def load_annotations(self, geojson_path):
        with open(geojson_path, 'r') as file:
            geojson_data = json.load(file)
        annotations = []
        for feature in geojson_data['features']:
            coords = feature['geometry']['coordinates'][0]
            if isinstance(coords[0][0], list):  # Handle extra nested coordinates
                coords = coords[0]
            polygon = Polygon(coords)
            label = feature['properties']['classification']['name']
            annotations.append((polygon, label))
        return annotations

    def process_slide(self, slide_path, geojson_path, level=0):
        annotations = self.load_annotations(geojson_path)
        slide = openslide.OpenSlide(slide_path)

        for polygon, label in annotations:
            try:
                bounds = polygon.bounds
                x_start, y_start, x_end, y_end = map(int, bounds)
                region_width = x_end - x_start
                region_height = y_end - y_start

                # Read the region containing the polygon
                region = slide.read_region((x_start, y_start), level, (region_width, region_height))
                region = np.array(region.convert('RGB'))

                # Create a mask for the polygon
                mask = Image.new("L", (region_width, region_height), 0)
                draw = ImageDraw.Draw(mask)
                scaled_polygon = [(x - x_start, y - y_start) for x, y in polygon.exterior.coords]
                draw.polygon(scaled_polygon, outline=1, fill=1)
                mask = np.array(mask)

                # Apply the mask to the region
                segmented_region = cv2.bitwise_and(region, region, mask=mask)

                # Resize or pad the image to ensure uniform size
                h, w, _ = segmented_region.shape
                if h > self.target_size[0] or w > self.target_size[1]:  # Resize if larger
                    segmented_region = cv2.resize(segmented_region, self.target_size, interpolation=cv2.INTER_AREA)
                else:  # Pad if smaller
                    pad_h = (self.target_size[0] - h) // 2
                    pad_w = (self.target_size[1] - w) // 2
                    segmented_region = cv2.copyMakeBorder(segmented_region, pad_h, self.target_size[0] - h - pad_h,
                                                        pad_w, self.target_size[1] - w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

                # Save the segmented and uniform-sized region
                tile_path = f"{self.output_dir}/{label}_uniform_tile_({x_start}_{y_start}).png"
                cv2.imwrite(tile_path, segmented_region)
            except:
                print("error in extracting polygon")


    def run(self):
        svs_files = [f for f in os.listdir(self.slide_dir) if f.lower().endswith(".svs")]
        geojson_files = [f for f in os.listdir(self.slide_dir) if f.lower().endswith(".geojson")]

        for svs_file, geojson_file in zip(svs_files, geojson_files):
            print(f"Processing {svs_file} with {geojson_file}...")
            self.process_slide(slide_path=os.path.join(self.slide_dir, svs_file),
                               geojson_path=os.path.join(self.slide_dir, geojson_file))
        print("All slides processed and saved.")

# Usage
main_dir = "/media/ist/Drive2/MANSOOR/Neuroimaging-Project/Breast_Cancer_Classification_Project"
SVS_dir = os.path.join(main_dir, "SVS_Data_Annotations")
Tiles_dir = os.path.join(main_dir, "Tiles_Data_Cell_Category_Classification")
processor = ProcessSVS(SVS_dir, Tiles_dir)
processor.run()
