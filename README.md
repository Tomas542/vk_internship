# Здравствуйте 👋

Данная работа подразумевает работу с датасетами, похожими на наши, но с возможно отличающимися данными широты и долготы. В случае изменения размерностей (количество колонок) у фичей, необходимо переделать переменные и скейлер.
Для начала установите все необходимые библиотеки с помощью:
```
pip install -r requirements.txt
```
Итак по поводу работы, у нас есть:
## exploration.ipynb
Файл содержит в себе всю работу по исследованию датасета, а также по подбору лучшей модели и гиперпараметров для неё.

## solution.py
Файл содержит в себе предобработку датасетов train и test и последующий расчёт score для test с сохранением результатам в <b>./datasets/submission.csv</b>

## datasets
Папка <b>datasets</b> содержит в себе:
1. features.csv - содержит наши полезные фичи
2. submission_sample.csv - пример submission 
3. submission.csv - сгенерированный submission на задачу
4. test.csv - тестовая выборка
5. train.csv - обучающая выборка

## config
Папка <b>config</b> содержит в себе:
1. lgb.conf - содержит гиперпараметры модели, которые мы подобрали для задачи в exploration.ipynb
2. scaler.pkl - StandardScaler для фич
