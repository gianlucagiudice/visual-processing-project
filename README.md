
<img src="https://www.unimib.it/sites/default/files/styles/paragrafo/public/logoist_3.jpg?itok=MDsuZlyU" width = "150" align='center'/>

# Visual Information Processing Project: GABAR

GABAR (Gender & Age Based Actor Retrieval) is a computer vision project realized on the occasion of the final exam of the Visual Information Processing and Management course held at the University of Milano-Bicocca.
Given an image with one or more faces, we aim to detect them and estimate their gender and age. With these information we will propose an actor that resembles the selected person in the image. We have made a telegram bot as an interface to allow user interaction.
A detailed explanation of the entire project is available at `visual-processing-project/GABAR.pdf` or <a href='https://drive.google.com/file/d/1_nAELZex7rRXohmTF3I7mttFcYidHANV/view?usp=sharing'>here</a>

## Project structure
- `model/`: Containing all the models used by the telegram bot. The pretrained models can be downloaded <a href='https://drive.google.com/drive/folders/1WWQ28Kq_KDDU7QVQblZXd4_FCDLofMRs?usp=sharing'>here</a>
- `src/`: Project code
  - `models/`
  - `detection/`
  - `retrieval/`
  - `telegram_bot/`
- `notebooks/`: Utility notebook
  - `metadata_conversion_matlab2pandas.ipynb`: Convert metadata from matlab format to pandas format.
  - `Gender_Age_Prediction_Using_VGGFace.ipynb`: Performs the actual training of the finetuned VGG-Face model

- `dataset` : Containing all the datasets used for training the models. Some of the datasets are currently not inside the folder because of Github storage limitations. You can find them in the following links:
  - <a href='http://vis-www.cs.umass.edu/fddb/'>FDDB</a> 
  - <a href='https://susanqq.github.io/UTKFace/'>UTKFaces</a> (already inside the folder)
  - <a href='http://www.vision.caltech.edu/Image_Datasets/Caltech256/'>Caltech256</a> 
  - The dataset used for similar actor retrieval was scraped by us and can be found entirely in the `dataset/Retrieval` folder

## How to use:

### Install requirements
```
pip install -r requirements.txt
```

### Instructions
- Generate a new telegram bot using BotFather
- Insert the bot token inside `src/config.py` in the variable TELEGRAM_BOT_TOKEN
- Download from <a href='https://drive.google.com/drive/folders/1WWQ28Kq_KDDU7QVQblZXd4_FCDLofMRs?usp=sharing'>here</a> the models and add them inside `visual-processing-project/model`
- Run `src/telegram_bot/bot.py`

## Authors

#### Gianluca Giudice - Computer Science Student @ University of Milano-Bicocca
  * g.giudice2@campus.unimib.it
  * [GitHub](https://github.com/gianlucagiudice)

#### Cogo Luca - Computer Science Student @ University of Milano-Bicocca
  * l.cogo@campus.unimib.it
  * [GitHub](https://github.com/LucaCogo)

#### Tonelli Lidia Lucrezia - Computer Science Student @ University of Milano-Bicocca
  * l.tonelli@campus.unimib.it
  * [GitHub](https://github.com/lutonelli)





