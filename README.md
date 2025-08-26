this project is my first trying to using yolo to solve a problem,some part of the content I borrowed from AI
这是我第一个作品，部分内容借鉴ai
这个脚本基于yolov8-seg ，构建了一个可视化的对密集排列的太阳能板进行分割并判断是内边还是外边
我在代码中使用了绝对路径，请在使用时将文章中的绝对路径替换为你自己的
本代码存在因系统问题（windows）导致的格式错误和转化bug，当遇到无法正确识别文件格式的情况时，请使用我提供的脚本对路径转化
请检查你的数据集格式，保持和下方的一致，否则会出现未知错误
yolov8_data_fixed_debug/
├─ images/                # 复制后的图片文件
│  ├─ train/              # 训练集图片（来自TRAIN_ROOT/images）
│  └─ val/                # 验证集图片（来自VAL_ROOT/images）
├─ labels/                # 转换后的YOLO格式标签文件
│  ├─ train/              # 训练集标签（.txt格式）
│  └─ val/                # 验证集标签（.txt格式）
└─ data.yaml              # YOLO训练所需的数据集配置文件

使用步骤：
1.
