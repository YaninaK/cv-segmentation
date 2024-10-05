# Рекоммендации к коду ноутбука challenge notebook.ipynb

### Содержание

1. [Блок Look at the data (EDA), Label Refinement, Image Enhancement](https://github.com/YaninaK/cv-segmentation/tree/main/notebooks#1-%D0%B1%D0%BB%D0%BE%D0%BA-look-at-the-data-eda-label-refinement-image-enhancement)
2. [Блоки Preparing The Dataset, Dataset Class и Start Training](https://github.com/YaninaK/cv-segmentation/tree/main/notebooks#2-%D0%B1%D0%BB%D0%BE%D0%BA%D0%B8-preparing-the-dataset-dataset-class-%D0%B8-start-training)
3. [Блоки UNET Model, Attention U-Net, ResUNet, Various Losses for Training, Evalution Score](https://github.com/YaninaK/cv-segmentation/tree/main/notebooks#3-%D0%B1%D0%BB%D0%BE%D0%BA%D0%B8-unet-model-attention-u-net-resunet-various-losses-for-training-evalution-score-training)
4. [Блок Training](https://github.com/YaninaK/cv-segmentation/tree/main/notebooks#4-%D0%B1%D0%BB%D0%BE%D0%BA-training) 



## 1. Блок Look at the data (EDA), Label Refinement, Image Enhancement

Рекомендации реализованы в коде ноутбука 01_EDA.ipynb.

1. Блок Look at the data (EDA), Label Refinement & Image Enhancement лучше делать в отдельном ноутбуке и давать на него ссылку. 
2. Для воспроизводимости среды необходимые версии библиотек лучше зафиксировать, например, в файле requirements.txt:

```
imutils==0.5.4
torchsummary==1.5.1
...
```

  Тогда установка будет выглдеть так:

```
!pip install -r requirements.txt
```

3. Импорт лучше упорядочить по алфавиту с разделением на секции и по типам, например, как это делает библиотека [isort](https://pycqa.github.io/isort/).
  Это позволит избежать повторений как, например ```from tqdm import tqdm```, ```import random``` и т.п.

4. Лучше избегать избыточного кода, как, например, в ```display(y_train.head())``` - достаточно ```y_train.head()```.

5. Для генерации ```dict_wells_masks``` и ```dict_wells_images```, чтобы не дублировать код, лучше воспользоваться формулой:

```
from collections import Counter

def get_patches_per_well(patches_list):
  patch_names = [int(name_patch.split("_")[1]) for name_patch in patches_list]
  dict_wells_patches = Counter(patch_names)

  for key in sorted(dict_wells_patches.keys()):
    print(f'Well : {key} Number of patches : {dict_wells_patches[key]}')

  print(f'\nTotal number of wells : {len(dict_wells_patches)}')
  print(f'Total number of patches :', sum(dict_wells_patches.values()))

  return dict_wells_patches
```
6. Для того чтобы убедиться, что число patches у Mask well и Image well совпадают, лучше использовать ```assert```:

```
for y_key in sorted(dict_wells_masks.keys()):
  assert dict_wells_masks[y_key] == dict_wells_images[y_key]

```
7. ```dict(sorted(dict_wells_masks.items())).keys()``` - неудачная конструкция;
достаточно ```dict_wells_masks.keys()```

8. В коде

```
for i, key in enumerate(y_keys):
    print('Mask well :',key, 'Number of patches :', dict_wells_masks[key])
    if i == 15:
        break
    print('Image well :',list(image_keys)[i], 'Number of patches :', dict_wells_images[list(image_keys)[i]],'\n')
```

  использование
```
if i == 15:
    break
```
  избыточно - у нас всего 15 скважин, цикл остановится и без break.
  Соответственно, избыточно использовать и ```enumerate```.

9. Код строк ниже избыточен:
```
print('Ratio of pixels with corrosion :', round(n_pixels*100/overall, ndigits= 3),'%')
print('Ratio of patches with corrosion :', round(n_patches*100/len(y_train), ndigits=3),'%')
```
  Лучше написать так:
```
print(f'Ratio of pixels with corrosion : {n_pixels*100/overall: .3f} %')
print(f'Ratio of patches with corrosion : {n_patches*100/len(y_train): .3f} %')
```

  Эти параметры логично проанализировать в разрезе скважин - Ratio of pixels with corrosion существенно отличается межуду скважинами - у скважин 9 и 10 информации разметки по коррозиии нет.

10. В ячейке Data analysis в Number of outliers речь о доле выбросов в процентах, а не о числе выбросов - лучше написать Ratio of outliers.

11. В ячейке ```Statistics on the train images``` дополнительную информацию дала бы статистика по квантилям:

```
with np.printoptions(precision=3, suppress=True):
    print(np.quantile(cleaned_img, [0.0025, 0.005, 0.25, 0.5, 0.75, 0.995]))
```

12. В ячейке, где генерируется Histogram of the pixels from train images имеет смысл убрать дублирование кода.

13. Пропуски есть только в 2 изображениях. Почти все пропуски приходятся на один снимок, его лучше удалить. В другом 11 пропусков можно заполнить медианной этого снимка.

14. Анализировать информацию по всем точкам ```flat_list_img_train``` не информативно. Лучше смотеть по кадрам.

15. Выбросы в одномерном пространстве: из 347 изображений с выбросами (с минимальным значением меньше -0.25):

    - у трети (119) поврежденны 1-2 пикселя;
    - почти у половины (159) из 1296 пикселей повреждены меньше 15;
    - только 40 изображений содержат больше 200 выбросов, из них 34 полностью повреждены - их можно удалить.
    
    У остальных выбросы можно заполнить медианным значением.

16. Выбросы в многомерном пространстве: для обнаружения выбросов в многомерном пространстве можно воспользоваться библиотекой pca, кластеризацией DBSCAN, есть множество других методов. Все модели/ анасамбли моделей для обнаружения аномалий нужно дополнительно настраивать. В обучающей выборке свыше 200 изображений - выбросы в многомерном пространстве, их лучше удалить.

17. Чтобы избежать data leakage, распределение данных между обучающей, валидационной и тестовой выборкой делаем по скважинам:
```
train = [1, 6, 7, 8, 10, 11, 12, 13, 14]
val = [3, 9]
test = [2, 4, 5, 15]
```
  При распределении скважин стараемся добиться более-менее сбалансированных характеристик.

18. Материалы о Label Refinement и Image Enhancement лучше поместить в EDA.

19. Для удобства чтения хорошо использовать подзаголовки, чтобы была видна структура материла. Этого легко добиться, используя разное число знаков ```#```, например: ```#```, ```##``` или ```###```, в зависимости от уровня подзаголовка.
 


## 2. Блоки Preparing The Dataset, Dataset Class и Start Training.

Рекомендации реализованы в коде ноутбука 02_Datasets,_dataloaders_transforms.ipynb.

1. До генерации датасета имеет смысл очистить данные от выбросов. Препроцессинг данных лучше сделать в виде отдельного модуля:

```
y_train, images_train = load_data()

preprocess = PreprocessData()
y_train, _  = preprocess.fit_transform(y_train, images_train)
```
  Код ```PreprocessData``` можно посмотреть [здесь](https://github.com/YaninaK/cv-segmentation/blob/main/src/cv_segmentation/data/preprocess.py).

  Выбор модели/ ансамбля моделей для обнаружения аномалий, настройка гиперпараметров - отдельный блок задач, встроенный в процесс построения ```data preprocessing pipeline```.

  Порог, после которого начиаются выбросы, другие гиперпараметры определяются, начиная с этапа EDA.

2. Лучше отказаться от дублирования данных обучающей выборки в папке ```data/interim/```. Достаточно разбить y_train на 3 файла: ```y_train.csv```, ```y_val.csv``` и ```y_test.csv```  и поместить их в папку ```data/interim/```.
Тестовая выборка, в дополнение к валидационной, потребуется для анализа ошибок моделей.

3. Чтобы избежать data leakage, данные на обучающую, валидационную и тестовую выборки разбиваем в разрезе скважин, чтобы снимки и разметка каждой скважины появлялись только в одном блоке. Отбор скважин для каждого блока делаем на этапе EDA:
```
split_y_to_train_val_test(y_train)
```
Код ```split_y_to_train_val_test``` можно посмотреть [здесь](https://github.com/YaninaK/cv-segmentation/blob/main/src/cv_segmentation/data/validation.py).

4. Трансформацию данных лучше делать отдельно от датасета и присоединять при его инициализации. Вместо 
```
transform = False
``` 
использовать, например, внешнюю функцию ```apply_numpy_transform```:
```
transform = apply_numpy_transform
```
Подходы к трансформации данных в обучающей и валидационной/ тестовой выборках отличаются:
  * В валидационной/ тестовой выборке зачастую достаточно трансформировать данные в torch.Tensor

Код внешнего модуля трансформации данных можно посмотреть [здесь](https://github.com/YaninaK/cv-segmentation/blob/main/src/cv_segmentation/features/transform.py). Код ```SegmentationDataset``` можно посмотреть [здесь](https://github.com/YaninaK/cv-segmentation/blob/main/src/cv_segmentation/features/get_dataset.py).

5. Идеи для аугментации данных можно почерпнуть в библитеке torchvision.transforms.v2

  Например:
  - Вращение в диапазоне от -15° и +15° - v2.RandomRotation
  - Обрезка - v2.RandomResizedCrop
  - Размытие до 1,75 пикселей - v2.GaussianBlur
  - Добавление шума до 10% пикселей
  - Регулировка яркости между -50% и +50% - v2.functional.adjust_brightness
  - Данные у нас находятся в диапазоне -0.25 до 0.47 - можно подумать об их нормализации.

  Важно проследить, чтобы маска обучающей выборки трансформировалась также как изображение.

6. Пример реализации ```PreprocessData```, ```SegmentationDataset``` и иллюстрация работы ```DataLoader``` в рамках ```data_preparation_pipeline``` - в коде этого ноутбука 02_Datasets,_dataloaders_transforms.ipynb.

  Код ```data_preparation_pipeline``` можно посмотреть [здесь](https://github.com/YaninaK/cv-segmentation/blob/main/src/cv_segmentation/models/train.py).



## 3. Блоки UNET Model, Attention U-Net, ResUNet, Various Losses for Training, Evalution Score, Training
Рекомендации из ноутбука 03_Models_losses_&_evalution score.ipynb.

1. Все три модели ```UNET Model```, ```Attention U-Net``` и ```ResUNet``` могут быть задействованы в сегментации изображений, но в базовом варианте задействована только ```UNET Model```, поэтому остальные модели лучше показывать в отдельном ноутбуке в качестве заметок для дальнейшей работы.

  * Если эксперименты проводились на всех трех моделях, модели лучше вывести в отдельные модули, которые бы импортировались в ноутбук, а в ноутбуке провести сравнительный анализ результатов экспериментов.

2. В базовом варианте модели функция потерь ```Binary Cross Entropy``` - ```torch.nn.BCELoss``` "из коробки".

  * Если эксперименты проводились c ```Dice Loss + BCE``` и ```Focal Loss```, эти функции потерь лучше вывести в отдельные модули, которые бы импортировались в ноутбук, а в ноутбуке провести сравнительный анализ результатов экспериментов.

3. В дальнейшем можно поэксперементировать с перспективными архитектурами моделей:
    * UNet++: A Nested U-Net Architecture for Medical Image Segmentation Zongwei Zhou et al., [Jul 2018](https://arxiv.org/abs/1807.10165)
    * AG-CUResNeSt: A Novel Method for Colon Polyp Segmentation. Sang et al. [Mar 2022](https://arxiv.org/abs/2105.00402)
    * Mask R-CNN. Kaiming He et al. [Jan 2018](https://arxiv.org/abs/1703.06870)
    * Vision Transformer (ViT) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale Alexey Dosovitskiy et al.[Jun 2021](https://arxiv.org/abs/2010.11929)
    * DeiT (data-efficient image transformers)
    * VGG16-U-Net

4. Dice Score (```dice_coeff```) рассчитывается некорректно. Корректный вариант расчета выше в этом ноутбуке. Формула работает для тензоров pytorch.

5. Из-за того что функция ```dice_coeff``` на входе и выходе не работает с ```torch.tensor```, ```y_pred``` и ```y_true``` приходится переводить в ```numpy``` и переходить на ```cpu```, что замедляет скорость обучения модели. Формула ```dice``` (выше), работающая с ```torch.tensor```, позволит этого избежать.

6. Если исходить из логики, что основная задача - обучение модели, а валидация - вспомогательная, вместо отдельной формулы для валидации модели, я бы предложила сделать отдельную формулу для одной эпохи обучения (```training Loop```) и использовать этот законченный блок при запуске каждой эпохи. 
  
7. Инициализацию модели лучше сделать в отдельной ячейке, чтобы можно было легко добавлять больше эпох к текущему запуску.

8. Для логирования метрик и значений функции потерь при обучении модели и валидации, чтобы потом выводить результаты на ```TensorBoard```, в pytorch предлагается ```torch.utils.tensorboard.SummaryWriter```.
  TensorBoard: набор инструментов для визуализации TensorFlow - [здесь](https://www.tensorflow.org/tensorboard?hl=ru) ссылка.

  Код запускается при инициализации модели:
```
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/corrosion_segmentation_trainer_{}'.format(timestamp))
```
  В конце каждой эпохи логируем показатели обучения и валидации модели:
```
writer.add_scalars(
  'Training vs. Validation Loss',
  { 'Training' : avg_loss, 'Validation' : avg_vloss },
  epoch_number + 1
)
writer.flush()
```
  Пример кода для запуска ```TensorBoard``` в ```Google Colab```:
```
%load_ext tensorboard
%tensorboard --logdir='/content/cv-segmentation/notebooks/runs'
```

9. Имеет смысл отслеживать и записывать лучшие версии модели:
```
best_vloss = 1_000_000.
if avg_vloss < best_vloss:
    best_vloss = avg_vloss
    model_path = 'model_{}_{}'.format(timestamp, epoch_number)
    torch.save(model.state_dict(), model_path)
```
10. Все рекомендации реалиизованы в сквозном примере в ноутбуке 04_Baseline_model.ipynb.



## 4. Блок Training

Сквозной пример реализации рекомендаций к коду ноутбука challenge notebook.ipynb

В ноутбуке 04_Baseline_model.ipynb реализован на сквозном примере обучения модели реализованы основные рекомендации к коду ноутбука challenge notebook.ipynb.

1. Далее необходимо обучить модель при различных гиперпараметрах и выбрать лучшие по валидационной выборке.
2. После настройки гиперпараметров следует протестировать модель на тестовой выборке и изучить, на каких экземплярах модель ошибается. Это может дать представление, как нужно трансформировать данные или изменить архитектуру модели, чтобы улучшить метрики.
3. Еще одно направление для исследований - проанализироать насколько качественная разметка. От разметки зависит и качесвтво обучения, и метрики на тестовой выборке. Если в разметке обнаружатся какие-то системные ошибки, которые можно нивелироать, есть шанс, что метрики модели улучшатся. Другой вариант - делать поправку на качество разметки - помечать в данных способы разметки/ разметчиков, как дополнительный признак.
4. Формулы, задейстованные в подготовке данных, обучении модели, инфененсе должны быть покрыты тестами. Так будет удобнее поддерживать модель на протяжении ее жизненного цикла. Обычно для этих целей используют библиотеки ```pytest``` - [здесь](https://docs.pytest.org/en/stable/) ссылка , ```unittest.mock``` - [здесь](https://docs.python.org/3/library/unittest.mock.html) ссылка. 
