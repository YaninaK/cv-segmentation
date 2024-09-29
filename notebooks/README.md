# Рекоммендации по коду ноутбука challenge notebook.ipynb

## 1. Блок Look at the data (EDA)

Рекомендации реализованы в коде ноутбука 01_EDA.ipynb.

1. Блок Look at the data (EDA) лучше делать в отдельном ноутбуке и давать на него ссылку. 
2. Для воспроизводимости среды необходимые версии библиотек лучше зафиксировать, например, в файле requirements.txt.
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
10. В ячейке Data analysis в Number of outliers речь о доле выбросов в процентах, а не о числе выбросов - лучше написать Ratio of outliers.
11. В ячейке, где генерируется Histogram of the pixels from train images имеет смысл убрать дублирование кода.
12. Анализировать информацию по всем точкам ```flat_list_img_train``` не информативно. Лучше смотеть по кадрам: 347 кадров с минимальным значением меньше -0.25 - выбросы. Их имеет смысл убрать из обучающей выборки.


