# ADLLM Prompt Optimizer 😄

目前是v0版本，把框架搭建好了，可以运行起来，不过整体效果依然不理想，输入的Optimizer的prompt依然需要继续优化

- **prompt_optimizer.py**：主文件运行这个文件即可进行优化。
- **data_processor**: 预处理图片将图片中的故障标注完毕。
- **generate_json**: 为每个异常图片寻找最相似的正常图片作为参考。

在运行prompt_optimizer之前要先运行后边两个程序。🎉🎉🎉
