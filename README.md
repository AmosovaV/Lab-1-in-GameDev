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
- Демонстрация работы программ и сохранения на диск прикриплена ниже.


## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задачи по теме лабораторной работы.

- Демонстрация работы программ прикреплена ниже.

```py

mew mew meww
mew
mew
lovemewing
mew mew

```

## Задание 3
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

- На примере данного кода рассматривается реализация линейной регрессии, которая включает в себя функцию потерь (прямое распределение). В таком случае, величина loss показывает разницу между ожидаемым и реальным результатом, поэтому данный параметр должен стремиться к нулю при изменении исходных данных.

Демонстрация работы программ прикреплена ниже.

### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

- Параметр Lr показывает шаг измерения в направлении, которое приводит к наибольшему снижению значения функции потерь. При большом значении параметра результат неточен, мы можем не получить фактический минимум функции. 


## Выводы

Абзац умных слов о том, что было сделано и что было узнано.
