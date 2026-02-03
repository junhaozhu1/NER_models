"""快速评估所有已训练模型"""
import subprocess

subprocess.run(["python", "main.py", "--evaluate-only"])
