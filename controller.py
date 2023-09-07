import os
import pathlib
import re
import pandas as pd
import concurrent.futures
import json
from model import Field, Raster, DOWNLOAD_DIR
from pdf_generator import generate_pdf


TABLE_FILE_PATH = "C:\\Users\\Arthur Machado\\Documents\\Soybean Farming Systems\\PlanetPDFGenerator\\locations.csv"
PDF_PATH = os.path.join(DOWNLOAD_DIR, "report.pdf")


class RastersSet:
    def __init__(self, fields: list, directory: str or None = DOWNLOAD_DIR):
        self.fields = fields
        if type(directory) == str or pathlib.WindowsPath or pathlib.Path:
            self.get_fields_rasters(directory)

    @property
    def rasters(self):
        return self._rasters

    @rasters.setter
    def rasters(self, val):
        self._rasters = val

    def get_fields_rasters(self, dir_path: str or pathlib.WindowsPath or pathlib.Path):
        raster_paths = [os.path.join(str(dir_path), f"{field.field_id}.tif") for field in self.fields]
        self.rasters = [Raster(raster_path, field) for raster_path, field in zip(raster_paths, self.fields)]

    def calculate_excess_green(self) -> list:
        list_excess_green = []
        for raster in self.rasters:
            avg_excess_green, excess_green = raster.calculate_excess_green()
            list_excess_green.append({"avg excess green": avg_excess_green,
                                      "excess green image": excess_green})
        return list_excess_green


class FieldsOrder:
    MAX_WORKERS = 12

    def __is_valid_filename(self, value):
        allowed_pattern = r'^[a-zA-Z][a-zA-Z0-9\s\-_]*$'
        return re.match(allowed_pattern, value) is not None

    def __check_valid_csv(self, df: pd.DataFrame):
        assert all(df['field_id'].apply(self.__is_valid_filename))

    def get_fields_from_csv(self, dir_path=TABLE_FILE_PATH):
        if os.path.splitext(dir_path)[-1] in [".xlsx", ".xls"]:
            df = pd.read_excel(dir_path, usecols=["field_id", "region", "agent", "latitude", "longitude", "geojson"],
                               dtype={"geojson": str}, engine="openpyxl")
        elif os.path.splitext(dir_path)[-1] == ".csv":
            df = pd.read_csv(dir_path, usecols=["field_id", "region", "agent", "latitude", "longitude", "geojson"],
                             dtype={"geojson": str})
        else:
            raise Exception("Incompatible tabular file.")
        self.__check_valid_csv(df)
        self.fields = []
        for idx, row in df.iterrows():
            field = Field(row["field_id"], row["region"], row["agent"], row["latitude"], row["longitude"],
                          json.loads(row["geojson"]))
            self.fields.append(field)

    def _sort_fields_by(self, instance: str) -> list:
        sorted_dataset = sorted(self.fields, key=lambda field: getattr(field, instance))
        return sorted_dataset

    async def run_in_parallel(self):
        with concurrent.futures.ThreadPoolExecutor(self.MAX_WORKERS) as executor:
            sorted_dataset = sorted(self.fields, key=lambda field: field.field_id)
            futures = [executor.submit(obj.call_download_image) for obj in sorted_dataset]
            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

            rasters = RastersSet(self.fields, DOWNLOAD_DIR)
            # rasters.calculate_excess_green()

            for field, raster in zip(self.fields, rasters.rasters):
                # field.avg_excess_green = raster.avg_excess_green
                # field.excess_green_image = raster.excess_green_image
                # field.excess_green_path = raster.npy_to_png(field.excess_green_image, f"Excess Green {field.field_id}")
                field.image_path = raster.save_rgb_png(field.field_id)

            # sorted_fields = self._sort_fields()
            # generate_pdf(fields=sorted_fields, output_file=PDF_PATH)
            generate_pdf(fields=self.fields, output_file=PDF_PATH)

