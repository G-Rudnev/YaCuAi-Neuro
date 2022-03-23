# YaCuAi-Neuro
Мы YaCu и мы занимаемся беспилотниками. 
Наш основной продукт - автономный поломоечный робот, который решен в настоящее время классическими методами измерения и управленмя (векторная и матричная алгебры, ТАУ, ЦОС и т.п.)

Мы хотим заблаговременно понять, могут ли методы с обучением по аналогии, т.е нейронные сети, качественно улучшить наши решения и 
бесшовно расширить наш стек технологий, не вступив в конфликт с имеющимися достижениями. 
Мы также задумываемся и о развитии новых продуктов, поэтому предлагаем к исследованию данные, записанные на гражданском автомобиле.

По ссылке ниже лежат раскадрованные видеоряды (множества файлов "Time=XXXX.jpg" в папках 1, 2, 3,..) простых проездов в простых условиях, синхронизированные с измерительными данными,
захватываемыми с CAN шины автомобиля (файл "Main_data.csv" в каждой папке):

https://www.dropbox.com/sh/puffx24cozjt37q/AADTc_W1LRTAeJ1yudVsxpUIa?dl=0

Каждая строка в .csv файле соответствует видеокадру, а столбцы содержат следующие значения по порядку {time, steering_angle (deg), speed (kph), throttle (%), brake (%), front_left_wheel_speed (kph), front_right_wheel_speed (kph), rear_left_wheel_speed (kph), rear_right_wheel_speed (kph), torque (relative), gear, rpm, light_on, hand_brake}
Раскадровка выступает в качестве обучающих данных (train_data), а данные с CAN - в качестве прецедентных (train_labels).

Основная задача - провести ряд экспериментов на TensorFlow, жонглируя архитектурой и параметрами нейросети, а также и данными, с целью определить возможности
существующих базовых нейросетевых технологий для работы в измерителом режиме (выходной слой activation='linear') при использовании данных, аналогичных представленным.

То есть мы предлагаем вам "потыкать" в TensorFlow, чтобы понять можно ли научить сравнительно простую сеть на представленных данных без предобработки или с минимальным её количеством
адекватно функционировать в дальнейшем на таких же данных в режиме определения (измерения) требуемого угла поворота руля (для начала) в зависимости от видеоряда.

Чтобы было легче начать, здесь лежат для этого 2 программки на Python, которые можно запросто сцепить в одну. Потребуются пакеты tkinter, natsort, numpy, PIL, pandas, tensorflow и др. - см. импорты.
1-ая (PrepareData.py) загружает раскадровку, изменяет размер изображений для удобства, собирает всё в общий сет и записывает его в отдельный файл numpy (train_data.npy). 2-ая (Train.py)
конструирует один из возможных вариантов нейросети на TensorFlow, загружает в неё раскадровку и данные с CAN (train_labels), запускает обучение и анализирует его результаты.
Меняйте количество слоев, количество нейронов, их фукнции активации, количество эпох, размеры входных данных, обработайте данные - делайте, в общем, всё, что подсказывает воображение.
Нужно понять, получится ли так или иначе обучиться, а потом пройти тесты на уже неизвестных сетах хоть с какими-то примечательными результатами (и проводите такие тесты, разумеется). 

Удачи!
