Live at https://cnn-pokemonpredict.streamlit.app

# CNN Image Classification - Pokémon Dataset

This project applies a Convolutional Neural Network (CNN) to classify Pokémon images based on their types. The trained model is deployed as a **Streamlit web app** on Streamlit Community Cloud.

## 📂 Project Structure
- `CNN_ImageClassification_PokemonDataset.ipynb` - Jupyter Notebook for training the CNN model.
- `app.py` - Streamlit application script for model deployment.
- `data/` - Directory containing Pokémon images and dataset files.
- `model/` - Folder storing trained CNN model weights.
- `README.md` - Project documentation.

## 📥 Dataset
The Pokémon dataset consists of images and corresponding labels (Type1 and Type2) fetched from a Pokémon classification dataset.

- **Source:** Pokémon Dataset (Community Dataset)
- **Features:**
  - Pokémon Name
  - Primary Type (Type1)
  - Secondary Type (Type2) (may be missing for some Pokémon)
  - Corresponding image

## 📦 Dependencies
Make sure you have the required libraries installed:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn streamlit PIL
```

## 🚀 Running the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/JebinAbraham/CNN-Image-Classification.git
   ```
2. Navigate to the project folder:
   ```bash
   cd CNN-Image-Classification
   ```
3. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `CNN_ImageClassification_PokemonDataset.ipynb` and run all cells sequentially.

## 📊 Steps in Analysis
1. **Data Loading**:
   - Extract Pokémon dataset and images.
   - Handle missing values in the dataset.
2. **Image Preprocessing**:
   - Resize images to a uniform shape (128x128 pixels).
   - Normalize pixel values to [0,1] range.
   - One-hot encode Pokémon types.
3. **CNN Model Training**:
   - Define a Convolutional Neural Network (CNN) using TensorFlow/Keras.
   - Train the model on Pokémon images.
   - Evaluate the model using accuracy and loss metrics.
4. **Model Deployment**:
   - Save the trained model as an `.h5` file.
   - Create a **Streamlit** app to classify user-uploaded Pokémon images.
   - Deploy the app on **Streamlit Community Cloud**.

## 🎮 Running the Streamlit App
To launch the web app locally:
```bash
streamlit run app.py
```

## 🌐 Deployment on Streamlit Community Cloud
1. Push your code to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Deploy your Streamlit app by connecting your GitHub repository.
4. Your app will be accessible via a public URL.

## 📈 Results & Observations
- The CNN model achieves a good accuracy in classifying Pokémon types based on images.
- Improvements such as data augmentation, deeper CNN architectures, and transfer learning can enhance accuracy.

## 🤝 Contributions
Feel free to contribute by opening an issue or submitting a pull request!

## 📜 License
This project is licensed under the MIT License.
