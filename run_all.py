
import subprocess, sys

subprocess.check_call([sys.executable, "src/train_classical.py"])
print("Classical pipeline finished.")
print("Run deep learning separately with:")
print("  python src/train_deep.py")
print("Then launch:")
print("  streamlit run app.py")
