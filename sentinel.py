from sentinelhub import SHConfig, DataCollection, SentinelHubRequest, MimeType, BBox, CRS, bbox_to_dimensions
# Добавляем импорты для работы с датами
from datetime import datetime, timedelta
# Импортируем relativedelta для точной работы с месяцами
from dateutil.relativedelta import relativedelta 

# Словарь, определяющий параметры для каждого возможного слоя
LAYER_DEFINITIONS = {
    'true_color': {
        'inputs': ['B02', 'B03', 'B04'],
        'output': {'id': 'true_color', 'bands': 3, 'sampleType': 'FLOAT32'},
        'eval_code': 'let gain = 2.5; let true_color = [gain * sample.B04, gain * sample.B03, gain * sample.B02];'
    },
    'ndvi': {
        'inputs': ['B04', 'B08'],
        'output': {'id': 'ndvi', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let ndvi = [(sample.B08 - sample.B04) / (sample.B08 + sample.B04)];'
    },
    'B01': {
        'inputs': ['B01'],
        'output': {'id': 'B01', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B01 = [sample.B01];'
    },
    'B05': {
        'inputs': ['B05'],
        'output': {'id': 'B05', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B05 = [sample.B05];'
    },
    'B06': {
        'inputs': ['B06'],
        'output': {'id': 'B06', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B06 = [sample.B06];'
    },
    'B07': {
        'inputs': ['B07'],
        'output': {'id': 'B07', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B07 = [sample.B07];'
    },
    'B08': {
        'inputs': ['B08'],
        'output': {'id': 'B08', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B08 = [sample.B08];'
    },
    'B8A': {
        'inputs': ['B8A'],
        'output': {'id': 'B8A', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B8A = [sample.B8A];'
    },
    'B09': {
        'inputs': ['B09'],
        'output': {'id': 'B09', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B09 = [sample.B09];'
    },
    'B11': {
        'inputs': ['B11'],
        'output': {'id': 'B11', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B11 = [sample.B11];'
    },
    'B12': {
        'inputs': ['B12'],
        'output': {'id': 'B12', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let B12 = [sample.B12];'
    },
    'dataMask': {
        'inputs': ['dataMask'],
        'output': {'id': 'dataMask', 'bands': 1, 'sampleType': 'UINT8'},
        'eval_code': 'let dataMask = [sample.dataMask];'
    },
    'Aerosol': {
        'inputs': ['AOT'],
        'output': {'id': 'Aerosol', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let Aerosol = [sample.AOT];'
    },
    'SceneCLassification': {
        'inputs': ['SCL'],
        'output': {'id': 'SceneCLassification', 'bands': 1, 'sampleType': 'UINT8'},
        'eval_code': 'let SceneCLassification = [sample.SCL];'
    },
    'CloudProbability1': {
        'inputs': ['CLD'],
        'output': {'id': 'CloudProbability1', 'bands': 1, 'sampleType': 'UINT8'},
        'eval_code': 'let CloudProbability1 = [sample.CLD];'
    },
    'CloudProbability2': {
        'inputs': ['CLP'],
        'output': {'id': 'CloudProbability2', 'bands': 1, 'sampleType': 'UINT8'},
        'eval_code': 'let CloudProbability2 = [sample.CLP];'
    },
    'CloudMask': {
        'inputs': ['CLM'],
        'output': {'id': 'CloudMask', 'bands': 1, 'sampleType': 'UINT8'},
        'eval_code': 'let CloudMask = [sample.CLM];'
    },
    'SnowProbability': {
        'inputs': ['SNW'],
        'output': {'id': 'SnowProbability', 'bands': 1, 'sampleType': 'UINT8'},
        'eval_code': 'let SnowProbability = [sample.SNW];'
    },
    'sunAzimuthAngles': {
        'inputs': ['sunAzimuthAngles'],
        'output': {'id': 'sunAzimuthAngles', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let sunAzimuthAngles = [sample.sunAzimuthAngles];'
    },
    'sunZenithAngles': {
        'inputs': ['sunZenithAngles'],
        'output': {'id': 'sunZenithAngles', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let sunZenithAngles = [sample.sunZenithAngles];'
    },
    'viewAzimuthMean': {
        'inputs': ['viewAzimuthMean'],
        'output': {'id': 'viewAzimuthMean', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let viewAzimuthMean = [sample.viewAzimuthMean];'
    },
    'viewZenithMean': {
        'inputs': ['viewZenithMean'],
        'output': {'id': 'viewZenithMean', 'bands': 1, 'sampleType': 'FLOAT32'},
        'eval_code': 'let viewZenithMean = [sample.viewZenithMean];'
    },
}

class Sentinel:
    def __init__(self, client_id: str, client_secret: str):
        self.config = SHConfig()
        self.config.sh_client_id=client_id
        self.config.sh_client_secret=client_secret
        self.bbox = None
        self.bbox_size = None
        self.time_interval = None
        self.resolution = None
    def set_aoi(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float):
        """
        Устанавливает интересующую область (Bounding Box).

        :param min_lon: Минимальная долгота.
        :param min_lat: Минимальная широта.
        :param max_lon: Максимальная долгота.
        :param max_lat: Максимальная широта.
        """
        self.bbox = BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)
    
    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_date(self, date: str = 'latest', num_days_back = 7):
        """
        Устанавливает временной интервал для запроса.

        :param date: Дата в формате 'YYYY-MM-DD' и сколько дней назад учитывать или 'latest' для получения самых новых данных.
        """
        if date.lower() == 'latest':
            # Создаем интервал за последние 7 дней для поиска самого свежего снимка
            end = datetime.utcnow()
            start = end - timedelta(days=num_days_back)
        else:
            end = datetime.strptime(date, '%Y-%m-%d').date()
            start = end - timedelta(days=num_days_back)
        self.time_interval = (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        
    def create_evalscript(self, layers: list) -> str:
        """
        Генерирует evalscript для Sentinel Hub на основе списка запрашиваемых слоев.

        :param layers: Список строк с названиями слоев (например, ['ndvi', 'true_color']).
        :return: Строка с готовым evalscript.
        """
        # Используем set для автоматического удаления дубликатов каналов
        input_bands = set()
        output_definitions = []
        evaluate_pixel_code = []
        return_statements = []

        for layer_name in layers:
            if layer_name not in LAYER_DEFINITIONS:
                raise ValueError(f"Слой '{layer_name}' не определен. Доступные слои: {list(LAYER_DEFINITIONS.keys())}")
            
            layer = LAYER_DEFINITIONS[layer_name]
            
            # Собираем все необходимые входные каналы
            for band in layer['inputs']:
                input_bands.add(band)
                
            # Формируем блок output
            output_def = f"""
            {{
                id: "{layer['output']['id']}",
                bands: {layer['output']['bands']},
                sampleType: "{layer['output']['sampleType']}"
            }}"""
            output_definitions.append(output_def)
            
            # Добавляем код для вычисления
            evaluate_pixel_code.append(layer['eval_code'])
            
            # Формируем возвращаемое значение
            return_statements.append(f"{layer['output']['id']}: {layer['output']['id']}")

        # Собираем итоговый evalscript из частей
        evalscript = f"""
        //VERSION=3

        function setup() {{
        return {{
            input: {list(input_bands)},
            output: [{', '.join(output_definitions)}]
        }};
        }}

        function evaluatePixel(sample) {{
        {'; '.join(evaluate_pixel_code)}
        
        return {{
            {', '.join(return_statements)}
        }};
        }}
        """
        return evalscript.replace("'", '"') # Заменяем одинарные кавычки на двойные для соответствия формату JSON
    
    def get_data(self, requested_layers:list):
        bbox_size = bbox_to_dimensions(self.bbox, resolution=self.resolution)
        data_collection = DataCollection.SENTINEL2_L2A
        evalscript = self.create_evalscript(requested_layers)
        responses = [SentinelHubRequest.output_response(layer_id, MimeType.TIFF) for layer_id in requested_layers]
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=self.time_interval, 
                    mosaicking_order='leastCC',
                )
            ],
            responses=responses,
            bbox=self.bbox,
            size=bbox_size,
            config=self.config,
        )
        data = request.get_data(decode_data=True)
        return data[0]

  


