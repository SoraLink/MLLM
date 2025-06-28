from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="llava-hf/llava-1.5-13b-hf",
    local_dir="llava-13b",
    local_dir_use_symlinks=False  # 确保是复制真实文件，不是符号链接
)