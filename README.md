# SAR-Sidelobe-suppression1
unet网络，输入修改为孪生双输入。
x1为原图、x2为谱变形的图、目标值为去除旁瓣的图。
使用均分损失和结构性损失。
不知道能不能发论文...需要创新点
![input_x1](https://github.com/user-attachments/assets/f51a8faa-d76e-4f72-80d5-a4580e10b07d)
x1
![input_x2](https://github.com/user-attachments/assets/b8876b58-8fda-4aa8-a958-6a0c57c45da3)
x2
![predicted_clean_image](https://github.com/user-attachments/assets/14a0a085-e336-40f5-abf9-01bd971300e2)
pre
![ground_truth](https://github.com/user-attachments/assets/244c4864-8b5a-4abe-8cd8-0479d866177a)
gt
