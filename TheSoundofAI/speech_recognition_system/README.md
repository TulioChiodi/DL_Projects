## Project

Project made following Valerio Velardo from The Sound of AI guide:
https://www.youtube.com/playlist?list=PL-wATfeyAMNpCRQkKgtOZU_ykXc63oyzp


## Download dataset and prepare dataset:

download dataset:
```
wget -P dataset/ http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
```

untar dataset:
```
tar -zxf dataset/speech_commands_v0.01.tar.gz -C dataset/
```

remove .tar.gz file and move _background_noise_ folder
```
rm dataset/speech_commands_v0.01.tar.gz && mv dataset/_background_noise_ ./noise 
```