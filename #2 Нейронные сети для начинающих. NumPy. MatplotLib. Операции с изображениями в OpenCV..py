#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


# vim: set ai et ts=4 sw=4:

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)

def sigmoid(alpha):
    return 1 / ( 1 + np.exp(- alpha * x) )

def main():
    dpi = 80
    fig = plt.figure(dpi = dpi, figsize = (512 / dpi, 384 / dpi) )

    plt.plot(x, sigmoid(0.5), 'ro-')
    plt.plot(x, sigmoid(1.0), 'go-')
    plt.plot(x, sigmoid(2.0), 'bo-')

    plt.legend(['A = 0.5', 'A = 1.0', 'A = 2.0'], loc = 'upper left')

    fig.savefig('sigmoid.png')
    
main()


# In[4]:


# Построим график прямой используя MatplotLib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])


# In[5]:


import numpy as np

# Независимая (x) и зависимая (y) переменные
x = np.linspace(0, 10, 50)
y = x

# Построение графика
plt.title("Линейная зависимость y = x") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
plt.grid()      # включение отображение сетки
plt.plot(x, y)  # построение графика


# In[6]:


# Построение графика
plt.title("Линейная зависимость y = x") # заголовок
plt.xlabel("x") # ось абсцисс
plt.ylabel("y") # ось ординат
plt.grid()      # включение отображение сетки
plt.plot(x, y, "r--")  # построение графика


# In[7]:


# Линейная зависимость
x = np.linspace(0, 10, 50)
y1 = x

# Квадратичная зависимость
y2 = [i**2 for i in x]

# Построение графика
plt.title("Зависимости: y1 = x, y2 = x^2") # заголовок
plt.xlabel("x")         # ось абсцисс
plt.ylabel("y1, y2")    # ось ординат
plt.grid()              # включение отображение сетки
plt.plot(x, y1, x, y2)  # построение графика


# In[8]:


# Линейная зависимость
x = np.linspace(0, 10, 50)
y1 = x

# Квадратичная зависимость
y2 = [i**2 for i in x]

# Построение графиков
plt.figure(figsize=(9, 9))

plt.subplot(2, 1, 1)
plt.plot(x, y1)               # построение графика
plt.title("Зависимости: y1 = x, y2 = x^2") # заголовок
plt.ylabel("y1", fontsize=14) # ось ординат
plt.grid(True)                # включение отображение сетки
plt.subplot(2, 1, 2)
plt.plot(x, y2)               # построение графика
plt.xlabel("x", fontsize=14)  # ось абсцисс
plt.ylabel("y2", fontsize=14) # ось ординат
plt.grid(True)                # включение отображение сетки


# In[11]:


fruits = ["apple", "peach", "orange", "bannana", "melon"]
counts = [34, 25, 43, 31, 17]
plt.bar(fruits, counts)
plt.title("FRUETS")
plt.xlabel("Fruit")
plt.ylabel("Count")


# In[12]:


import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import numpy as np
x = np.linspace(0, 10, 10)
y1 = 4*x
y2 = [i**2 for i in x]

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_title("Графики зависимостей: y1=4*x, y2=x^2", fontsize=16)
ax.set_xlabel("x", fontsize=14)        
ax.set_ylabel("y1, y2", fontsize=14)
ax.grid(which="major", linewidth=1.2)
ax.grid(which="minor", linestyle="--", color="gray", linewidth=0.5)
ax.scatter(x, y1, c="red", label="y1 = 4*x")
ax.plot(x, y2, label="y2 = x^2")
ax.legend()

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=10, width=2)
ax.tick_params(which='minor', length=5, width=1)

plt.show()


# In[ ]:


# Импорт
import numpy as np
import cv2

# Позвоните в камеру компьютера, 0: первая основная камера
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Преобразование цветового пространства
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Отображение изображения
    cv2.imshow('frame', frame)
    cv2.imshow('gray',gray)
    
    # Конец, клавиша q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закройте вызывающую программу камеры и закройте все окна изображений
cap.release()
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np

def viewImage(image):
    cv2.namedWindow(‘Display’, cv2.WINDOW_NORMAL)
    cv2.imshow(‘Display’, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayscale_17_levels (image):
    high = 255
    while(1): 
        low = high — 15
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray, col_to_be_changed_low,col_to_be_changed_high)
        gray[curr_mask > 0] = (high)
        high -= 15
        if(low == 0 ):
            break
            
image = cv2.imread(‘./path/to/image’)
viewImage(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayscale_17_levels(gray)
viewImage(gray)


# In[ ]:


def get_area_of_each_gray_level(im):
    ## преобразование изображения к оттенкам серого (обязательно делается до оконтуривания) 
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    output = []
    high = 255
    first = True
    while(1):
        low = high — 15
        if(first == False):
 
        # Делаем значения выше уровня серого черными. Так они не будут обнаруживаться 
        to_be_black_again_low = np.array([high])
        to_be_black_again_high = np.array([255])
        curr_mask = cv2.inRange(image, to_be_black_again_low, 
        to_be_black_again_high)
        image[curr_mask > 0] = (0)
 
        # Делаем значения этого уровня белыми. Так мы рассчитаем их площадь
        ret, threshold = cv2.threshold(image, low, 255, 0)
        contours, hirerchy = cv2.findContours(threshold, 
        cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if(len(contours) > 0):
            output.append([cv2.contourArea(contours[0])])
            cv2.drawContours(im, contours, -1, (0,0,255), 3)
            high -= 15
            first = False
        if(low == 0 ):
            break
    return output


# In[ ]:


image = cv2.imread(‘./path/to/image’)
print(get_area_of_each_gray_level(image))
viewImage(image)


# In[ ]:


import cv2
import numpy as np

def viewImage(image):
    cv2.namedWindow(‘Display’, cv2.WINDOW_NORMAL)
    cv2.imshow(‘Display’, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# Получаем HSV-представление для зеленого цвета 
green = np.uint8([[[0, 255, 0 ]]])
green_hsv = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print( green_hsv)


# In[ ]:


image = cv2.imread(‘./path/to/image.jpg’)

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
viewImage(hsv_img) # 1

green_low = np.array([45 , 100, 50] )
green_high = np.array([75, 255, 255])
curr_mask = cv2.inRange(hsv_img, green_low, green_high)
hsv_img[curr_mask > 0] = ([75,255,200])
viewImage(hsv_img) # 2

# Преобразование HSV-изображения к оттенкам серого для дальнейшего оконтуривания
RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
viewImage(gray) # 3

ret, threshold = cv2.threshold(gray, 90, 255, 0)
viewImage(threshold) # 4

contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
viewImage(image) # 5


# In[ ]:


def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)
    while (i < total_contours ):
        area = cv2.contourArea(contours[i])
        if(area > largest_area):
            largest_area = area
            largest_contour_index = i
        i+=1
    
    return largest_area, largest_contour_index

    # Чтобы получить центр контура 
    cnt = contours[13]
    M = cv2.moments(cnt)
    cX = int(M[“m10”] / M[“m00”])
    cY = int(M[“m01”] / M[“m00”])
    largest_area, largest_contour_index = findGreatesContour(contours)
    print(largest_area)
    print(largest_contour_index)
    print(len(contours))
    print(cX)
    print(cY)

