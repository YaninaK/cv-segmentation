# Рекоммендации по коду ноутбука challenge notebook.ipynb

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

11. В ячейке, где генерируется Histogram of the pixels from train images имеет смысл убрать дублирование кода.

12. Пропуски есть только в 2 изображениях. Почти все пропуски приходятся на ```well_13_patch_1060 ```, его лучше удалить.

13. Анализировать информацию по всем точкам ```flat_list_img_train``` не информативно. Лучше смотеть по кадрам и в разрезе скважин: 347 кадров с минимальным значением меньше -0.25 - выбросы - в пределах 60 кадров на одну скважину. Их имеет смысл убрать из обучающей выборки.

14. Чтобы избежать data leakage, распределение данных между обучающей, валидационной и тестовой выборкой делаем по скважинам:
```
train = [6, 7, 8, 9, 11, 12, 13, 14]
val = [3, 10, 15]
test = [1, 2, 4, 5]
```
При распределении скважин стараемся добиться более-менее сбалансированных характеристик.

15. Материалы о Label Refinement и Image Enhancement лучше поместить в EDA.

16. Для удобства чтения хорошо использовать подзаголовки, чтобы была видна структура материла. Этого легко добиться, используя разное число знаков ```#```, например: ```#```, ```##``` или ```###```, в зависимости от уровня подзаголовка.
 

## 2. Блоки Preparing The Dataset и Dataset Class

Рекомендации реализованы в коде ноутбука 02_Datasets,_dataloaders_transforms.ipynb.

1. До генерации датасета имеет смысл очистить данные от выбросов. Порог, после которого начиаются выбросы, определеяется на этапе EDA:
```
csv_file="../data/raw/y_train.csv"
root_dir="/content/cv-segmentation/data/raw/images/"
y_train, images_train = load_data(csv_file, root_dir)
y_train, _ = clean_outliers(y_train, images_train)
```
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

6. Пример реализации SegmentationDataset и иллюстрация работы DataLoader в рамках data_preparation_pipeline - в коде этого ноутбука 02_Datasets,_dataloaders_transforms.ipynb.
