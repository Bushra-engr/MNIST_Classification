# 🔢 MNIST Digit Recognizer

A beautiful Streamlit web application that allows users to draw handwritten digits and get real-time predictions using a Machine Learning model trained on the MNIST dataset.



## ✨ Features

- 🎨 **Interactive Drawing Canvas** - Draw digits with your mouse
- ⚡ **Real-time Predictions** - Get instant predictions as you draw
- 📊 **Probability Chart** - Visual bar chart showing confidence for all digits (0-9)
- 🔍 **Debug View** - See the preprocessed 28x28 image that the model sees
- 🌙 **Beautiful Dark Theme** - Modern gradient dark UI design
- 🗑️ **Clear Canvas** - Reset and start fresh
 
## 🧠 Model

The application uses a **Logistic Regression** model trained on the MNIST dataset:
- Training samples: 56,000
- Test samples: 14,000
- Test Accuracy: ~92%

## 📁 Project Structure

```
MNIST/
├── app.py                              # Streamlit application
├── mnist.py                            # Model training script
├── mnist_logistic_regression_model.pkl # Trained model
├── MNIST.ipynb                         # Jupyter notebook
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── .gitignore                          # Git ignore file
```

## 💡 Tips for Best Results

1. **Draw thick strokes** - The model works better with bold lines
2. **Center your digit** - Keep it in the middle of the canvas
3. **Draw one digit at a time** - Clear before drawing a new digit

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - Web framework
- [Plotly](https://plotly.com/) - Interactive charts
- [scikit-learn](https://scikit-learn.org/) - Machine Learning
- [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) - Drawing component

