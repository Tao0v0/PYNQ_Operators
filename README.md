# PYNQ_Operators

- 使用model1实现模型的搭建，每一层通过调用m = model(backend=software()) 实现实例化， 再通过m.add实现模型层构建，实现了仅软件版的模型推理。矩阵乘法接口在backend.py文件

- 算子层具体实现再pynn/layer.py代码中，可自行修改

- 测试用例cmnext是针对 B C H W输入， B N C输出的模型。加入了layernorm和Attention模块


