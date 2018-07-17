# Tianchi

![肺部原始图](fig/1.png'肺部原始图')|![2](fig/2.png'肺部腐蚀图')|![3](fig/3.png'肺部气管图')
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
