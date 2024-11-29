
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
srun --cpus-per-task=20 --partition=A100short --gres=gpu:1 --pty bash
