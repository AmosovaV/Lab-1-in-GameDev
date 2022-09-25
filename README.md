# Анализ данных и искусственный интеллект
Отчет по лабораторной работе #1 выполнила:
- Амосова Варвара Ивановна
- РИ-210940

Отметка о выполнении заданий:

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello World на Python и Unity.
Ход работы:
- Код программы на Python:

```py
print('Hello World')
```
- Запуск программы:

![Вывод питон](https://user-images.githubusercontent.com/114309754/192134344-ad5a06f4-f15d-4fbd-8950-4343d4341d2f.png)

- Схранение на диск:

![Сохранение ](https://user-images.githubusercontent.com/114309754/192134361-d35de74d-69eb-4822-84fc-90db0020eb8b.png)
![Диск](https://user-images.githubusercontent.com/114309754/192134366-f003caa4-cf37-4ff3-b1f5-8a7ff4a5f10e.png)
![Диск 2](https://user-images.githubusercontent.com/114309754/192134370-76677301-d399-46c4-88de-cac446f6b6a1.png)

- Код программы на Unity:

```py
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class lab : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
         Debug.Log("Hello world!");
    }
}
```
- Вывод сообщения на консоль:

![Unity](https://user-images.githubusercontent.com/114309754/192134448-d6d2d0a6-2689-4d0d-985b-97ba44443540.png)

## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задачи по теме лабораторной работы.

- Код программы при 1 итерации:

```py
#Import the required modules, numpy for calculation, and Matplotlib for drawing
!pip install numpy
!pip install matplotlib
import numpy as mss
import matplotlib.pyplot as plt
%matplotlib inline

#define data, and change list to array
x = [3, 21, 22, 34, 54, 34, 55, 67, 89,99]
x = mss.array(x)

y = [2, 22, 24, 65, 79, 82, 55, 130, 150, 199]
y = mss.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

#The basic linear regression model is wx+b, and since this is a two-dimensional space, the model is ax+b
def model(a, b, x):
    return a*x + b

#The most commonly used loss function of linear regresstion model is the loss function of mean, variance difference
def loss_function(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    return (0.5/num)*(mss.square(prediction - y)).sum()

#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize (a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    #Update the values of A and B by finding the partial derivatives of the loss function on a and b
    da = (1.0/num)*((prediction - y)*x).sum()
    db = (1.0/num)*((prediction - y).sum())
    a = a - Lr*da
    b = b - Lr*db
    return a, b

#iterated function, return a and b
def iterate (a, b, x, y, times):
    for i in range (times):
        a, b = optimize(a, b, x, y)
    return a, b

#Initialize parameters and display
a = mss.random.rand(1)
print (a)
b = mss.random.rand(1)
print(b)
Lr = 0.000001

#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a, b = iterate(a, b, x, y, 1)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```
- Результат работы программы при 1 итерации:
![1 итерация](https://user-images.githubusercontent.com/114309754/192134832-64c65669-fe76-4484-94e3-fb2162c6e590.png)

- Результат работы программы при 2, 3, 4, 5, 10000 итерациях соответственно:

![2 итерации](https://user-images.githubusercontent.com/114309754/192134858-f296eb77-ab26-4fed-bbde-ec5dab65db8d.png)
![3 итерации](https://user-images.githubusercontent.com/114309754/192134865-9c121afb-8e64-48a3-8d0f-d8a378acd3ac.png)
![4 итерации](https://user-images.githubusercontent.com/114309754/192134869-9e624c8b-fec3-4966-a730-5a31f2b861dd.png)
![5 итераций](https://user-images.githubusercontent.com/114309754/192134873-18cb946a-35fb-4e1f-8f0c-8b1b1e40da6e.png)
![10000 итераций](https://user-images.githubusercontent.com/114309754/192134879-9c560bea-acc6-42c1-9844-c29732248fbe.png)



## Задание 3
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

- На примере данного кода рассматривается реализация линейной регрессии, которая включает в себя функцию потерь (прямое распределение). В таком случае, величина loss показывает разницу между ожидаемым и реальным результатом, поэтому данный параметр должен стремиться к нулю при изменении исходных данных.

- Результат работы программы при меньшем значении параметра:

![Маленький loss](https://user-images.githubusercontent.com/114309754/192135112-39f670c7-c615-440f-9d65-b973ea7c41cc.png)

- Результат работы программы при большем значении loss:

![Большой loss](https://user-images.githubusercontent.com/114309754/192135066-14492ccc-682a-4af0-840c-ce6dc5b214e8.png)


### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

- Параметр Lr показывает шаг измерения в направлении, которое приводит к наибольшему снижению значения функции потерь. При большом значении параметра результат неточен, мы можем не получить фактический минимум функции. 


## Выводы

Абзац умных слов о том, что было сделано и что было узнано.
