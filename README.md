# Tianchi

## 背景
### 肺部原始图，肺部腐蚀图，肺部气管图
<img src="fig/1.png" height="200"/> <img src="fig/2.png" height="200"/> <img src="fig/3.png" height="200"/>

### 肺部2D图，肺部3D图，肺部阈值图
<img src="fig/21.gif" height="200"/> <img src="fig/22.gif" height="200"/> <img src="fig/23.gif" height="200"/>

### 真肿瘤
<img src="fig/31.gif" height="200"/> <img src="fig/32.gif" height="200"/> <img src="fig/33.gif" height="200"/>

### 假肿瘤
<img src="fig/41.gif" height="200"/> <img src="fig/42.gif" height="200"/> <img src="fig/43.gif" height="200"/>

## Run
### Step 1
`python ./prepare/main.py`

### Step 2
```
python ./1_train/main.py
python ./1_train/check.py
```

### Step 3
```
python ./1_test/main.py
python ./1_test/check.py
```
### Step 4
`python ./1_test/save_csv.py`

### Step 5
```
python ./2_train/create_data.py
python ./2_train/check.py
```

Created by lining@2017/9/30
