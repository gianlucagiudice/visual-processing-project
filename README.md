# Visual Information Processing Project

## Project structure
- `src/`: Project code
  - `models/`
  - `detection/`
  - `retrieval/`
  - `telegram_bot/`
- `notebooks/`: Utility notebook
  - `metadata_conversion_matlab2pandas.ipynb`: Convert metadata from matlab format to pandas format.

- `dataset` : Containing all the datasets used for training the models


## How to use:

### Install requirements
```
pip install -r requirements.txt
```

### Instructions
- Generate a new telegram bot using BotFather
- Insert the bot token inside 'src/config.py' in the variable TELEGRAM_BOT_TOKEN
- Download from <a href='https://drive.google.com/drive/folders/19HhDo2A6lWS1jTW4HlZMC3NFG6i2duyS?usp=sharing'>here</a> the 'model' folder and add it inside 'visual-processing-project/'
- Run `src/telegram_bot/bot.py`
