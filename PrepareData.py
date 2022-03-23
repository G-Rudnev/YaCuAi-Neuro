import os
import natsort
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

path2Images = filedialog.askdirectory() + '/'   #выбор папки, содержащей видеокадры
filenames = natsort.natsorted(os.listdir(path2Images))  #имена файлов кадров

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 

#Изменение размера исходного изображения необходимо для ускорения и удобства конструирования слоев сети. 
#Сейчас предложен результирующий размер 500х150, но можно, а, скорее, даже нужно, выбрать и установить любой.
def resize_image(input_image_path, size = (500, 150)):
    np_image = np.array(Image.open(input_image_path).resize(size)).astype('int').reshape(3, size[0] * size[1])
    return np_image

train_data = np.zeros((len(filenames), 75000))

for i in range(len(filenames)): #изменение размеров 
    try:
        train_data[i, :] = resize_image(input_image_path = path2Images + filenames[i])
        print(i, ' - row appended')
    except:
        continue

# сохранение всех кадров в файл numpy
dataFileName = filedialog.asksaveasfilename(initialdir = '.', initialfile='train_data.npy', filetypes=[('npy', '*.npy')])
np.save(dataFileName, train_data)
#new_num_arr = np.load('data.npy')