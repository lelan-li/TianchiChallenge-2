# Tianchi

## 肺部原始图，肺部腐蚀图，肺部气管图
<img src="fig/1.png" height="200" alt="肺部原始图"/><img src="fig/2.png" height="200" alt="肺部原始图"/><img src="fig/3.png" height="200" alt="肺部原始图"/>
![1](fig/1.png '肺部原始图' =200) ![2](fig/2.png "肺部腐蚀图") ![3](fig/3.png "肺部气管图")

## 肺部2D图，肺部3D图，肺部阈值图
![21](fig/21.gif =200*) ![22](fig/22.gif) ![23](fig/23.gif)

## Step 1
`python ./prepare/main.py`

## Step 2
```
python ./1_train/main.py
python ./1_train/check.py
```

## Step 3
```
python ./1_test/main.py
python ./1_test/check.py
```
## Step 4
`python ./1_test/save_csv.py`

## Step 5
```
python ./2_train/create_data.py
python ./2_train/check.py
```

Created by lining@2017/9/30
