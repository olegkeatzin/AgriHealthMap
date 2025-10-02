import numpy as np
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
    MosaickingOrder
)
# Добавляем импорты для работы с датами
from datetime import datetime, timedelta
# Импортируем relativedelta для точной работы с месяцами
from dateutil.relativedelta import relativedelta    

class SentinelHubProcessor:
    """
    Класс для упрощенной работы с API Sentinel Hub.
    """

    def __init__(self, client_id: str, client_secret: str):
        """
        Инициализирует класс с учетными данными Sentinel Hub.

        :param client_id: ID клиента Sentinel Hub.
        :param client_secret: Секретный ключ клиента Sentinel Hub.
        """
        self.config = SHConfig()
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret
        self.bbox = None
        self.resolution = None
        self.time_interval = None
        self.evalscript = ""

    def set_area_of_interest(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float):
        """
        Устанавливает интересующую область (Bounding Box).

        :param min_lon: Минимальная долгота.
        :param min_lat: Минимальная широта.
        :param max_lon: Максимальная долгота.
        :param max_lat: Максимальная широта.
        """
        self.bbox = BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

    def set_date(self, date: str = 'latest'):
        """
        Устанавливает временной интервал для запроса.

        :param date: Дата в формате 'YYYY-MM-DD' или 'latest' для получения самых новых данных.
        """
        if date.lower() == 'latest':
            # Создаем интервал за последние 30 дней для поиска самого свежего снимка
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=10)
            self.time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        else:
            # Для конкретной даты создаем интервал в один день
            self.time_interval = (date, date)

    def set_resolution(self, resolution_meters: int):
        """
        Устанавливает разрешение изображения.

        :param resolution_meters: Разрешение в метрах.
        """
        self.resolution = resolution_meters

    def set_channels(self, channel_alias: str):
        """
        Устанавливает, какие каналы будут запрошены.

        :param channel_alias: Псевдоним для набора каналов ('true_color', 'ndvi').
        """
        if channel_alias.lower() == 'true_color':
            self.evalscript = """
                //VERSION=3
                function setup() {
                    return {
                        input: ["B04", "B03", "B02"],
                        output: { bands: 3, sampleType: "FLOAT32"}
                    };
                }

                function evaluatePixel(sample) {
                    // Увеличиваем яркость для лучшей визуализации
                    return [3.5 * sample.B04, 3.5 * sample.B03, 3.5 * sample.B02];
                }
            """
        elif channel_alias.lower() == 'ndvi':
            self.evalscript = """
                //VERSION=3
                function setup() {
                    return {
                        input: ["B04", "B08"],
                        output: { id: "default", bands: 1, sampleType: "FLOAT32" }
                    };
                }

                function evaluatePixel(sample) {
                    let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
                    return [ndvi];
                }
            """
        else:
            raise ValueError("Неизвестный псевдоним канала. Доступные варианты: 'true_color', 'ndvi'.")

    def get_data(self) -> np.ndarray:
        """
        Выполняет запрос и получает данные из Sentinel Hub.

        :return: Массив NumPy с данными изображения.
        """
        if not all([self.bbox, self.resolution, self.time_interval, self.evalscript]):
            raise ValueError("Необходимо установить область интереса, разрешение, дату и каналы перед запросом данных.")

        bbox_size = bbox_to_dimensions(self.bbox, resolution=self.resolution)
        
        # Выбираем коллекцию данных. Для "latest" лучше всего подходит L2A
        data_collection = DataCollection.SENTINEL2_L2A

        request = SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=self.time_interval,
                    mosaicking_order=MosaickingOrder.LEAST_CC, # Выбираем снимок с наименьшим количеством облаков
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=self.bbox,
            size=bbox_size,
            config=self.config,
        )

        # get_data() может вернуть список изображений, если за указанный период их было несколько.
        # Берем первый (и, как правило, единственный) элемент.
        data = request.get_data()
        if not data:
            raise Exception("Не удалось найти данные за указанный период. Попробуйте расширить диапазон дат.")
        
        return data[0]
    
    def get_monthly_images(self, n_months: int) -> list[np.ndarray]:
        """
        Получает список лучших изображений за каждый из последних N месяцев,
        выполняя по одному запросу на каждый месяц.

        :param n_months: Количество месяцев для запроса.
        :return: Список массивов NumPy, где каждый массив - это изображение за один месяц.
        """
        if not all([self.bbox, self.resolution, self.evalscript]):
            raise ValueError("Необходимо установить область интереса, разрешение и каналы перед запросом данных.")

        if not isinstance(n_months, int) or n_months <= 0:
            raise ValueError("Количество месяцев (n_months) должно быть положительным целым числом.")

        images_list = []
        time_intervals = []
        # Начинаем с текущей даты и идем в прошлое
        end_date = datetime.utcnow()

        print(f"Запуск сбора данных за последние {n_months} месяцев...")

        for i in range(n_months):
            # Вычисляем начало периода (месяц назад от end_date)
            # relativedelta(months=1) работает точнее, чем timedelta(days=30)
            start_date = end_date - relativedelta(months=1)
            
            # Устанавливаем временной интервал для следующего вызова get_data()
            self.time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            time_intervals.append(self.time_interval)
            print(f"Запрос данных за период: {self.time_interval[0]} по {self.time_interval[1]}")
            
            try:
                # Вызываем существующий метод get_data()
                monthly_image = self.get_data()
                images_list.append(monthly_image)
                print("-> Успешно. Изображение добавлено.")
            except Exception as e:
                # Если get_data() не нашел снимков за этот месяц, он вызовет исключение.
                # Мы его ловим, выводим сообщение и продолжаем цикл.
                print(f"-> Предупреждение: {e}")

            # Сдвигаем конечную дату на начало предыдущего периода для следующей итерации
            end_date = start_date

        return images_list, time_intervals