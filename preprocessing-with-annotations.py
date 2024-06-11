import json
import openslide
from shapely.geometry import Polygon, Point
import cv2, os
import numpy as np

class ProcessSVS:
    def __init__(self, slide_dir, output_dir):
        self.slide_dir = slide_dir
        self.output_dir = output_dir

    def load_annotations(self):
        with open(self.geojson_path, 'r') as file:
            geojson_data = json.load(file)
        annotations = []
        for feature in geojson_data['features']:
            coords = feature['geometry']['coordinates'][0]
            # Print out the coordinates for debugging
            # print("Original coordinates:", coords)

            # Ensure the coordinates are in the correct format
            if isinstance(coords, list) and all(isinstance(c, list) and len(c) == 2 for c in coords):
                polygon = Polygon(coords)
            elif isinstance(coords, list) and all(isinstance(c, list) for c in coords[0]):
                # Handle extra nesting
                polygon = Polygon(coords[0])
            else:
                print("Error in coordinate format:", coords)
                continue  # Skip this feature

            label = feature['properties']['classification']['name']
            annotations.append((polygon, label))
        return annotations



    def process_slide(self, slide_path, geojson_path, level=0, tile_size=(600, 600)):

        self.geojson_path = geojson_path
        annotations = self.load_annotations()
        slide = openslide.OpenSlide(slide_path)
        for polygon, label in annotations:
            bounds = polygon.bounds
            x_start, y_start, x_end, y_end = map(int, bounds)
            for x in range(x_start, x_end, tile_size[0]):
                for y in range(y_start, y_end, tile_size[1]):
                    if polygon.intersects(Point(x, y)):
                        tile = slide.read_region((x, y), level, tile_size)
                        tile = np.array(tile.convert('RGB'))
                        tile_path = f"{self.output_dir}/{label}_tile_({x}_{y}).png"
                        cv2.imwrite(tile_path, tile)
        print("All slides preprocessed and saved..")

    def run(self):
        svs_files = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f)) and f.lower().endswith(".svs")]
        geojson_files = [f for f in os.listdir(self.slide_dir) if os.path.isfile(os.path.join(self.slide_dir, f)) and f.lower().endswith(".geojson")]

        for i in range(len(svs_files)):
            print(f"Processing {svs_files[i]}...")
            self.process_slide(slide_path=f"{self.slide_dir}/{svs_files[i]}", geojson_path=f"{self.slide_dir}/{geojson_files[i]}")


main_dir = "/media/ist/Drive2/MANSOOR/Neuroimaging-Project/Breast_Cancer_Classification_Project"
SVS_dir = os.path.join(main_dir, "SVS_Data_Annotations")
Tiles_dir = os.path.join(main_dir, "Tiles_Data_Cell_Category_Classification")
processor = ProcessSVS(SVS_dir, Tiles_dir)
processor.run()
