import random
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from label_studio.ml import LabelStudioMLBase
from label_studio.ml.utils import get_single_tag_keys, get_choice, is_skipped
import tensorflow as tf
from tensorflow import image
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_size = 224
def transform(img):
    img = image.resize(img, (img_size, img_size))
    img = tf.convert_to_tensor(img)
    img = image.per_image_standardization(img)
    return img


class ImageClassifierDataset():
    
    def __init__(self, img_urls, img_class):
        self.images = []
        self.labels = []
        
        self.classes = list(set(img_class))
        self.class_to_label = {c:i for i, c in enumerate(self.classes)}
        self.img_size = 224
    
        for url, classes in zip(img_urls, img_class):
            img = plt.imread(url)
            transform_img = self.transform(img)
            self.images.append(transform_img)
            label = self.class_to_label[classes]
            self.labels.append(label)
    
    def transform(self, img):
        # image preprocessing: resize, normalization
        img = image.resize(img, (self.img_size, self.img_size))
        #img = image.central_crop()
        img = tf.convert_to_tensor(img)
        img = image.per_image_standaradization(img) #normalization
        
    def get_item(self, index):
        #rturn image and label for given index
        return self.images[index], self.labels[index]
    
    def __len__(self):
        #return number of images in teh Dataset Object
        return len(self.images)
        
class ImageClassifier(object):
    # A simplw wrapper for pre-trained Resnet model
    def __init__(self, n_classes):
        self.model = ResNet50(weights='imagenet')
        
    def save(self, path):
        self.model.save_weights(os.path.join(path,'model.ckpt'))
        
    def load(self, path):
        self.model = ResNet50(weights='imagenet')
        self.model.load_weights(path)
        
    def predict(self, img_urls):
        results = []
        for url in img_urls:
            img = plt.imread(url)
            img = transform(img)
            pred = self.model.predict(img)
            results.append(pred)
            
        return results
        
    def train(self, X, y, datagenerator, num_epochs=25):
        # Assume dataloader have inputs and labels feature
        since = time.time()
        
        self.model.train()
        self.model.compile(optimizer='adam',
                               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                               metrics=['accuracy'])
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-'*10)
            batches = 0
            for x_batch, y_batch in datagenerator.flow(X, y, batch_size=32):
                self.model.fit(x_batch, y_batch)
                batches += 1
                if batches >= len(X)/32:
                    break
                


class animalClassifier(LabelStudioMLBase):

    def __init__(self,  freeze_extractor = False, **kwargs):
        # don't forget to call base class constructor
        super(animalClassifier, self).__init__(**kwargs)
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'Choices', 'Image'
            )
        
        if self.train_output:
            self.classes = self.train_output['classes']
            self.model = ImageClassifier(len(self.classes))
            self.model.load(self.train_output['model_path']) # save the checkpoint of pretrained weight
        else:
            self.model = ImageClassifier(len(self.classes))
            
    def reset_model(self):
        self.model = ImageClassifier(len(self.classes))
        
    def predict(self, tasks, **kwargs):
        """This is where inference happens: model returns the list of predictions based on input list of tasks"""
        #collect input data
        img_urls = [task['data'][self.value] for task in tasks]
        logits = self.model.predict(img_urls)
        
        predict_label_ind = np.argmax(logits, axis = 1)  # predict label index of all classes
        predict_score = logits[np.arange(len(predict_label_ind)), predict_label_ind] # probability of the predict class
        
        predict_result = []
        for label_ind, prob in zip(predict_label_ind, predict_score):
            predict_label = self.labels[label_ind]
        
            predict_result.append({
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'value': {
                        'choices': [predict_label] 
                    }
                }],
                'score': prob
            })
        return predict_result

    def fit(self, completions, workdir=None, batch_size=32, n_epochs=10, **kwargs):
        """This is where training happens: train your model given list of completions, then returns dict with created links and resources"""
        img_urls, img_classes = [], []
        print('Collecting completions...')
        
        for completion in completions:
            if is_skipped(completion):
                continue
            img_urls.append(completion['data'][self.value])
            img_classes.append(get_choice(completion))
            
        print('Creating dataset...')
        data = ImageClassifierDataset(img_urls, img_classes)
        datagen = ImageDataGenerator(
            featurewise_center = True,
            featurewise_std_normalization=True
            )
        
        print('Training model...')
        self.reset_model()
        self.model.train(data.images, data.labels, datagen)
        
        print('Save model...')
        model_path = os.path.join(workdir, 'model.ckpt')
        self.model.save(workdir)
        
        return {'model_path':model_path, 'classes':data.labels}
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    