# Manuscript-Year-Prediction

This project presents a deep learning-based approach to predict the year of origin of historical handwritten manuscripts. The system is designed to automate what was once a manual and expert-driven process in the fields of archive science, paleography, and digital humanities.

By leveraging state-of-the-art CNN architectures, this project offers a scalable and accurate alternative to manual manuscript dating.

---

## 📌 Overview

Historical manuscript dating is essential for cultural preservation and scholarly research but has traditionally required expert manual analysis. This project uses deep learning to predict the year a manuscript was written based solely on its image.

We trained and compared the performance of several CNN architectures:

- ✅ **ResNet50** *(Best accuracy: 99%)*
- ResNet101
- MobileNetV2
- InceptionV3
- DenseNet121

---

## 🧠 Features

- Predicts the year of historical manuscript images
- Uses pre-trained CNNs with transfer learning
- Automated preprocessing and classification pipeline
- Evaluated on a real-world dataset with strong performance
- Implemented in Jupyter Notebook and Google Colab

---

## 🧰 Tech Stack

- Python 3.10
- TensorFlow 2.15
- Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab / Spyder

---

## 🧪 Dataset

The dataset was compiled from publicly available sources including the Internet Archive and English Library archives. Manuscripts span from the 13th to the 20th century, covering a variety of handwriting styles.

| Year  | Manuscripts | Pages |
|-------|-------------|-------|
| 1290  | 1           | 48    |
| 1364  | 2           | 108   |
| 1492  | 2           | 106   |
| 1700  | 2           | 524   |
| 1937  | 1           | 116   |

Images were preprocessed via resizing (224×224), normalization, and data augmentation (flipping, rotation).

---

## 🚀 Workflow

```text
1. Image Input
2. Preprocessing (resizing, normalization, augmentation)
3. Feature Extraction (CNN-based)
4. Classification/Regression (year prediction)
5. Output: Predicted year
````

---

## 📈 Model Performance

| Model       | Accuracy |
| ----------- | -------- |
| ResNet50    | 99%      |
| ResNet101   | 98%      |
| MobileNetV2 | 97%      |
| InceptionV3 | 94%      |
| DenseNet121 | 36%      |

---

## 📂 Recommended Folder Structure

```
manuscript-year-prediction/
├── manuscript_prediction.ipynb
├── model_resnet50.h5
├── dataset/
│   └── sample_images/
├── images/
│   └── architecture_diagram.png
├── README.md
└── requirements.txt
```

---

## 🧾 Key Contributions

* 📌 A modified ResNet50 model tailored for manuscript year regression
* 📌 Comparative benchmarking of 5 CNN models
* 📌 Dataset creation and preprocessing pipeline
* 📌 Full experimental setup and analysis
* 📌 Contribution to digital humanities and cultural heritage preservation

---

## 📈 Results Summary

ResNet50 outperformed all other models, achieving **99% accuracy** and showing strong generalization to unseen manuscript images. It effectively captured features like ink density, handwriting curvature, and paper texture.

---

## 👨‍💻 Authors

* **D. Lalith Kumar**
* N. Jaya Sravan Kumar
* Dhatri Gullapalli

---

## 📜 License

This project is developed for academic and educational purposes. Please cite the work or contact the authors for reuse in commercial applications.

---

## 📌 Future Enhancements

* 📈 Add more diverse and multilingual manuscript data
* 🧠 Incorporate multimodal learning (text + image)
* 🌐 Deploy as an interactive web app
* 📊 Improve interpretability with visualization tools like Grad-CAM

---

Thank you for checking out this project! ⭐️
