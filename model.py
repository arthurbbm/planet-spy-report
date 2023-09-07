import asyncio
import os
import planet
import pathlib
import rasterio as rs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.colors import ListedColormap
from PIL import Image
from planet_auth import auth
from datetime import date, datetime, timedelta
from pyproj import Transformer


DOWNLOAD_DIR = pathlib.Path(f"{os.getcwd()}\\{date.today().strftime('%Y-%m-%d')}")
B = 0
G = 1
R = 2
NIR = 3

class Field:
    def __init__(self, field_id, region, agent, latitude, longitude, geojson: dict, distance=300):
        self.field_id = field_id
        self.region = region
        self.agent = agent
        self.latitude = round(latitude, 4)
        self.longitude = round(longitude, 4)
        self.geojson = geojson
        self.bbox = self._generate_bbox(self.latitude, self.longitude, distance)


    @property
    def avg_excess_green(self):
        return round(self._avg_excess_green, 5)

    @avg_excess_green.setter
    def avg_excess_green(self, val):
        self._avg_excess_green = val

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, val):
        self._image_path = val

    @property
    def excess_green_image(self):
        return self._excess_green_image

    @excess_green_image.setter
    def excess_green_image(self, val):
        self._excess_green_image = val

    def _generate_bbox(self, lat, lon, distance):
        lat_change = distance / 111000
        lon_change = distance / (111000 * np.cos(np.radians(lat)))

        min_lat = lat - lat_change
        max_lat = lat + lat_change
        min_lon = lon - lon_change
        max_lon = lon + lon_change

        return {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat]
            ]]
        }

    async def _check_available_image(self, aoi: dict) -> str:
        sdate = datetime.combine(date.today(), datetime.min.time())
        async with planet.Session(auth=auth) as sess:
            client = sess.client('data')
            filters = planet.data_filter.and_filter([
                planet.data_filter.geometry_filter(aoi),
                planet.data_filter.date_range_filter('acquired', lte=sdate, gte=sdate - timedelta(days=7)),
                planet.data_filter.range_filter("cloud_cover", lt=0.1)
            ])
            self.items = [i async for i in client.search(['PSScene'], filters)]
            self.items_ids = [item['id'] for item in self.items]
            self.item_id = self.items_ids[0]
        return self.item_id

    def _create_request(self, item_id: str, geojson: dict) -> dict:
        order = planet.order_request.build_request(
            name=self.field_id,
            products=[
                # planet.order_request.product(item_ids=[item_id],
                #                              product_bundle='analytic_sr_udm2',
                #                              item_type='PSScene'),
                planet.order_request.product(item_ids=[item_id],
                                             product_bundle='visual',
                                             item_type='PSScene')
            ],
            tools=[planet.order_request.clip_tool(aoi=geojson)])
        return order

    def _check_existance_of(self, directory: str or pathlib.Path):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __rename_field_dir(self, src_dir):
        old_dir = os.path.join(src_dir, self.order_id + '/PSSCene')
        for file in os.listdir(old_dir):
            old_path = os.path.join(old_dir, file)
            ext = os.path.splitext(file)[-1]
            new_file_name = self.field_id + ext
            new_path = os.path.join(src_dir, new_file_name)
            os.rename(old_path, new_path)
        os.remove(os.path.join(src_dir, self.order_id))

    async def _create_and_download(self, order_detail: dict, directory: pathlib.Path) -> str:
        async with planet.Session(auth=auth) as sess:
            client = sess.client('orders')
            with planet.reporting.StateBar(state='creating') as reporter:
                order = await client.create_order(order_detail)
                reporter.update(state='created', order_id=order['id'])
                await client.wait(order['id'], callback=reporter.update_state)
                self.order_id = order['id']

            self._check_existance_of(directory)
            await client.download_order(order['id'], directory, progress_bar=True)
            self.__rename_field_dir(directory)
        return self.order_id

    async def download_image(self, directory=DOWNLOAD_DIR):
        item_id = await self._check_available_image(self.bbox)
        request = self._create_request(item_id, self.bbox)
        order_id = await self._create_and_download(request, directory)

    def call_download_image(self):
        asyncio.run(self.download_image())


class Raster:
    def __init__(self, raster_path: str or None, field: Field):
        self.field = field
        if type(raster_path) == str:
            self.open(raster_path)

    def _open_xml(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        band_coefficients = {}
        for band_specific_metadata in root.findall(
                './/{http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level}bandSpecificMetadata'):
            band_number_element = band_specific_metadata.find(
                '{http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level}bandNumber')
            reflectance_coefficient_element = band_specific_metadata.find(
                '{http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level}reflectanceCoefficient')

            if band_number_element is not None and reflectance_coefficient_element is not None and reflectance_coefficient_element.text is not None:
                band_number = int(band_number_element.text) - 1
                reflectance_coefficient = float(reflectance_coefficient_element.text)
                band_coefficients[band_number] = reflectance_coefficient

        self.reflectance_coefficients = band_coefficients

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, val):
        green = val[G]
        blue = val[B]
        red = val[R]
        self._image = np.array([blue, green, red])

    def __image(self, val):
        """
            Correction made according to https://assets.planet.com/docs/Planet_Combined_Imagery_Product_Specs_letter_screen.pdf
            RAD(i) = DN(i) * radiometricScaleFactor(i), where radiometricScaleFactor(i) = 0.01
            The resulting value is the at sensor radiance of that pixel in watts per steradian per square meter (W/m²*sr*μm)

            REF(i) = DN(i) * reflectanceCoefficient(i)
        """
        green = val[G] * self.reflectance_coefficients[G]
        blue = val[B] * self.reflectance_coefficients[B]
        red = val[R] * self.reflectance_coefficients[R]
        nir = val[NIR] * self.reflectance_coefficients[NIR]
        self._image = np.array([blue, green, red, nir])

    def open(self, raster_path: str):
        with rs.open(raster_path) as dataset_reader:
            nlayers = dataset_reader.count
            image_list = [dataset_reader.read(nlayer) for nlayer in range(1, nlayers + 1)]
            self.image = np.array(image_list)
            transformer = Transformer.from_crs("EPSG:4326", dataset_reader.crs)
            x_utm, y_utm = transformer.transform(self.field.latitude, self.field.longitude)
            px, py = dataset_reader.index(x_utm, y_utm)
            self.px, self.py = int(px), int(py)

    def __open(self, raster_path: str):
        xml_path = os.path.splitext(raster_path)[-2] + '.xml'
        self._open_xml(xml_path)
        with rs.open(raster_path) as dataset_reader:
            nlayers = dataset_reader.count
            image_list = [dataset_reader.read(nlayer) for nlayer in range(1, nlayers + 1)]
            self.image = np.array(image_list)

    def __calculate_excess_green(self) -> tuple:
        """
        :return: tuple[float, np.ndarray]. It returns tuple of average excess green and image of excess green.
        """
        blue = self.image[0]
        green = self.image[1]
        red = self.image[2]
        nir = self.image[3]
        self.excess_green_image = 2 * green - (blue + red)
        self.avg_excess_green = np.average(self.excess_green_image)
        return self.avg_excess_green, self.excess_green_image

    def save_rgb_png(self, field_id):
        width, height = self.image.shape[1:]
        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        image_rgb = self.image[:3].transpose((1, 2, 0))
        ax.imshow(image_rgb)
        ax.plot(self.px, self.py, 'r+', markersize=10)
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        rgb_png_path = os.path.join(DOWNLOAD_DIR, f'{field_id}.png')
        plt.savefig(rgb_png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        with Image.open(rgb_png_path) as im:
            im = im.crop((0, 0, width, height))
            im.save(rgb_png_path)
        return rgb_png_path

    @staticmethod
    def npy_to_png(data: np.ndarray, variable_name: str):
        n_colors = 4
        green_palette = sns.color_palette("Greens", n_colors=n_colors)
        cmap = ListedColormap(green_palette)
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap=cmap, vmin=np.min(data), vmax=np.max(data))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(variable_name)
        excess_green_path = os.path.join(DOWNLOAD_DIR, f'{variable_name}.png')
        plt.savefig(excess_green_path)
        return excess_green_path
