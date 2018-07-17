# 天池医疗AI大赛：肺部结节智能诊断[第二赛季]

## 背景
### 肺部原始图，肺部腐蚀图，肺部气管图
<img src="fig/1.png" height="200"/> <img src="fig/2.png" height="200"/> <img src="fig/3.png" height="200"/>

### 肺部2D图，肺部3D图，肺部阈值图
<img src="fig/21.gif" height="200"/> <img src="fig/22.gif" height="200"/> <img src="fig/23.gif" height="200"/>

## 任务
### 真肿瘤
<img src="fig/31.gif" height="150"/> <img src="fig/32.gif" height="150"/> <img src="fig/33.gif" height="150"/>

### 假肿瘤
<img src="fig/41.gif" height="150"/> <img src="fig/42.gif" height="150"/> <img src="fig/43.gif" height="150"/>

### 絮状肿瘤，小肿瘤，肺壁肿瘤
<img src="fig/51.gif" height="150"/> <img src="fig/52.gif" height="150"/> <img src="fig/53.gif" height="150"/>

## Run
```
python ./prepare/main.py

python ./1_train/main.py
python ./1_train/check.py

python ./1_test/main.py
python ./1_test/check.py

python ./1_test/save_csv.py

python ./2_train/create_data.py
python ./2_train/check.py
```

Created by lining@2017/9/30
