
# [OTUS Deep Learning Course](https://otus.ru/lessons/dl-basic/)

## Final project: "Telegram bot for recognizing diseases of greenhouse plants" 

MobileNetv3 model is used. The dataset is a set of photographs of tomato leaves. There are 7 classes in the dataset: 6 types of diseases and one class with leaves of healthy plants. There are approximately 300 photographs per class.  
The dataset was composed of several datasets: [Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf), [Mendeley data](https://data.mendeley.com/datasets/ngdgg79rzb/1) and the rest of the pictures I just googled in google images)

When preparing the dataset, all photos were resized to 224x224 (because MobileNet was previously trained on ImageNet, and there the size of the pictures is just 224x224) and augmentation was made (rotations to different angles and reflections).  

During the training process, it turned out that after 30 epochs, the accuracy on the validation set is 95%, which is quite enough.  

**Tomato_leaf_diseases.zip** - dataset,  
**mobilenetv3_large_100_best.pth** - model weight,  
**otus-deep_learning-final_project.ipynb** - notebook with training and inference  
**vegecheck-bot.py** - telegram bot code
