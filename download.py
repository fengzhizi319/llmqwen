# 模型下载
from modelscope import snapshot_download

# # 下载第一个模型
# model_dir1 = snapshot_download('inferencerlabs/Qwen3.8-27B-MTP-MLX')
# print(f'模型1下载完成: {model_dir1}')
#
# # 下载第二个模型
# model_dir2 = snapshot_download('lmstudio-community/Qwen3.8-27B-MLX-8bit')
# print(f'模型2下载完成: {model_dir2}')

model_dir = snapshot_download('Qwen/Qwen3.8-27B')

print(f'模型3下载完成: {model_dir}')