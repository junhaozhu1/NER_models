"""训练单个模型的便捷脚本"""
import sys
import subprocess

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python train_single.py <model_name>")
        print("示例: python train_single.py BiLSTM-CRF")
        sys.exit(1)

    model_name = sys.argv[1]
    cmd = ["python", "main.py", "--models", model_name]

    # 添加其他参数
    if len(sys.argv) > 2:
        cmd.extend(sys.argv[2:])

    subprocess.run(cmd)
