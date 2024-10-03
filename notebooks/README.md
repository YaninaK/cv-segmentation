# Рекоммендации по коду ноутбука challenge notebook.ipynb

### Содержание

1. [Блок Look at the data (EDA)](https://github.com/YaninaK/cv-segmentation/tree/b1/notebooks#1-%D0%B1%D0%BB%D0%BE%D0%BA-look-at-the-data-eda)
2. [Блоки Preparing The Dataset, Dataset Class и Start Training](https://github.com/YaninaK/cv-segmentation/tree/b1/notebooks#2-%D0%B1%D0%BB%D0%BE%D0%BA%D0%B8-preparing-the-dataset-dataset-class-%D0%B8-start-training)
3. [Блоки UNET Model, Attention U-Net, ResUNet, Various Losses for Training, Evalution Score](https://github.com/YaninaK/cv-segmentation/tree/b1/notebooks#3-%D0%B1%D0%BB%D0%BE%D0%BA%D0%B8-unet-model-attention-u-net-resunet-various-losses-for-training-evalution-score)




## 1. Блок Look at the data (EDA)

Рекомендации реализованы в коде ноутбука 01_EDA.ipynb.

1. Блок Look at the data (EDA) лучше делать в отдельном ноутбуке и давать на него ссылку. 
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
  Выбор модели/ ансамбля моделей для обнаружения аномалий, настройка гиперпараметров - отдельный блок задач, встроенный в процесс построения ```data preprocessing pipeline```.

  Порог, после которого начиаются выбросы, другие гиперпараметры определяются, начиная с этапа EDA.

2. Лучше отказаться от дублирования данных обучающей выборки в папке ```data/interim/```. Достаточно разбить y_train на 3 файла: ```y_train.csv```, ```y_val.csv``` и ```y_test.csv```  и поместить их в папку ```data/interim/```.
Тестовая выборка, в дополнение к валидационной, потребуется для анализа ошибок моделей.

3. Чтобы избежать data leakage, данные на обучающую, валидационную и тестовую выборки разбиваем в разрезе скважин, чтобы снимки и разметка каждой скважины появлялись только в одном блоке. Отбор скважин для каждого блока делаем на этапе EDA:
```
split_y_to_train_val_test(y_train)
```
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

## 3. Блоки UNET Model, Attention U-Net, ResUNet, Various Losses for Training, Evalution Score.
Рекомендации реализованы в коде ноутбука 03_Models_losses_&_evalution score.ipynb.

1. Все три модели ```UNET Model```, ```Attention U-Net``` и ```ResUNet``` могут быть задействованы в сегментации изображений, но в базовом варианте задействована только ```UNET Model```, поэтому остальные модели лучше показывать в отдельном ноутбуке в качестве заметок для дальнейшей работы.

  * Если эксперименты проводились на всех трех моделях, модели лучше вывести в отдельные модули, которые бы импортировались в ноутбук, а в ноутбуке провести сравнительный анализ результатов экспериментов.

2. В базовом варианте модели функция потерь ```Binary Cross Entropy``` - ```torch.nn.BCELoss``` "из коробки".

  * Если эксперименты проводились c ```Dice Loss + BCE``` и ```Focal Loss```, эти функции потерь лучше вывести в отдельные модули, которые бы импортировались в ноутбук, а в ноутбуке провести сравнительный анализ результатов экспериментов.

3. Dice Score (```dice_coeff```) рассчитывается некорректно. Корректный вариант расчета для numpy - в коде ноутбука.

  * Dice в pytorch доступен "из коробки" - [здесь](https://torchmetrics.readthedocs.io/en/v0.10.0/classification/dice.html) ссылка. Лучше использовать этот вариант.

4. В дальнейшем можно поэксперементировать с перспективными архитектурами моделей:
    * UNet++: A Nested U-Net Architecture for Medical Image Segmentation Zongwei Zhou et al., [Jul 2018](https://arxiv.org/abs/1807.10165)
    * AG-CUResNeSt: A Novel Method for Colon Polyp Segmentation. Sang et al. [Mar 2022](https://arxiv.org/abs/2105.00402)
    * Mask R-CNN. Kaiming He et al. [Jan 2018](https://arxiv.org/abs/1703.06870)
    * Vision Transformer (ViT) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale Alexey Dosovitskiy et al.[Jun 2021](https://arxiv.org/abs/2010.11929)
    * DeiT (data-efficient image transformers)
    * VGG16-U-Net
  
