import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from entrenar import model, history
import matplotlib.pyplot as plt

# Graficar evolución de precisión
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()