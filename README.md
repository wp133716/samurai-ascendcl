# samurai-ascendcl
samurai-ascendcl是一个基于 AscendCL 的视觉跟踪推理项目,是samurai/sam2的AscendCL C++部署版本

#### onnx 模型下载
onnx模型可以在另一个项目 [samurai-onnx](https://github.com/wp133716/samurai-onnx) 中获取

#### om 模型转换
onnx模型转换为om模型的命令可以在 om_model/atc命令.md 文件中找到

#### 环境配置
- 安装Ascend-cann-toolkit_8.*，配置环境变量
- 安装OpenCV 4.10

#### 编译项目
``` bash
mkdir build
cd build
cmake ..
make -j8
```
#### 运行项目
``` bash
./sam2_tracker_acl
```


