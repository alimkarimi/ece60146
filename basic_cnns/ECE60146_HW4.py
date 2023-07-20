#!/usr/bin/env python
# coding: utf-8

# In[65]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import PIL
from PIL import Image
import os
import torchvision
from torchvision import transforms as tvt
import torch
from torch.nn import parallel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import random
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import pathlib
import random


# In[66]:


#dataDir='..'
dataType='train2014'
annFile='annotations/instances_{}.json'.format(dataType)
# initialize COCO api for instance annotations
coco=COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


# In[68]:


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['airplane','bus','cat', 'dog', 'pizza']);
print(catIds)
imgIds = coco.getImgIds(catIds=5); #GET CAT IDS FOR GIVEN CLASS NAMES
#print(imgIds)
imgIds = coco.getImgIds(imgIds = [imgIds[2]])


# In[22]:


catNms=['airplane','bus','cat', 'dog', 'pizza']
catIds = coco.getCatIds(catNms=catNms); #get a list category ids for each category name 
###Create training data set ###
for n, cat_id in enumerate(catIds): ##for each category id
    print('in category id:,',cat_id)
    imgIds = coco.getImgIds(catIds = cat_id) #for each category id, get a list of img ids
    print(imgIds[:10]) #printing first 10 image ids
    pathlib.Path('train_orig/' + catNms[n]).mkdir(parents=True, exist_ok=True) #create a path to store 
#     training data for the current category
    coco.download(tarDir = 'train_orig/' + catNms[n], imgIds = imgIds[0:1500]) #download first 1500 image ids 
#     into the specified directory
    d = os.listdir('train_orig/' + catNms[n]) #create list of files in the created directory. This list has 1500 
#     jpg file names
#     print('list of files in d', d)
    for img in d: #iterate through list of downloaded images. Resize them to 64 x 64. 
        temp_img = Image.open('train/' + catNms[n] + '/' + img) #open image
        temp_img = temp_img.resize((64,64)) #resize
        temp_img.save(fp = 'train/' + catNms[n] + '/' + img) #overwrite image with the 64 x 64 version
        ## save function parameters:
#         fp – A filename (string), pathlib.Path object or file object.
#         format – Optional format override. If omitted, the format to use is determined from the 
#         ##filename extension.
#         If a file object was used instead of a filename, this parameter should always be used.

    

    
   


# In[51]:


### Plot images from each category ###
catNms=['airplane','bus','cat', 'dog', 'pizza']
fig, ax = plt.subplots(5,3, figsize=(16, 16))
for n in range(0,5):
    temp_dir = os.listdir('train/' + catNms[n])
    #open image
    for i in range(6,9):
        temp_img = Image.open('train/' + catNms[n] + '/' + temp_dir[i]) #get the i-th image from the directory. 
#         Here, we just grab the 0th, 1st, and 2nd.
        #convert to numpy array for plotting
        temp_np_arr = np.array(temp_img)
        ax[n,i-6].imshow(temp_np_arr)
        ax[n,i-6].set_title(catNms[n] + ' example ' + str(i-5) )

plt.savefig('example_training_images.jpg')
        


# In[24]:


###Create validation dataset ###
catNms=['airplane','bus','cat', 'dog', 'pizza']
catIds = coco.getCatIds(catNms=catNms); #get a list category ids for each category name 
for n, cat_id in enumerate(catIds): ##for each category id
    print('in category id:,',cat_id)
    imgIds = coco.getImgIds(catIds = cat_id) #for each category id, get a list of img ids
    print(imgIds[:10]) #printing first 10 image ids
    pathlib.Path('val_orig/' + catNms[n]).mkdir(parents=True, exist_ok=True) #create a path to store training data for the current category
    coco.download(tarDir = 'val_orig/' + catNms[n], imgIds = imgIds[1500:2000]) #download first 1500 image ids
#     into the specified directory
    d = os.listdir('val_orig/' + catNms[n]) #create list of files in the created directory. This list has 1500 
#     jpg file names
#     print('list of files in d', d)
    for img in d: #iterate through list of downloaded images. Resize them to 64 x 64. 
        temp_img = Image.open('val/' + catNms[n] + '/' + img) #open image
        temp_img = temp_img.resize((64,64)) #resize
        temp_img.save(fp = 'val/' + catNms[n] + '/' + img) #overwrite image with the 64 x 64 version
        ### save function parameters:
        #fp – A filename (string), pathlib.Path object or file object.
        #format – Optional format override. If omitted, the format to use is determined from the 
        ###filename extension.
        #If a file object was used instead of a filename, this parameter should always be used.


# In[39]:


### Create data_loader ### 
root_train = 'train/'
root_val = 'val/'
catNms=['airplane','bus','cat', 'dog', 'pizza']

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, catNms):
        super(MyDataset).__init__()
        self.root = {} #dictionary for main directory which holds all the images of a category
        self.filenames = {} #dictionary for filenames of a given category
        for cat in catNms:
            self.root[cat] = root + cat + '/'
        for cat in catNms:
        #create list of image files in each category that can be opened by __getitem__
            self.filenames[cat] = os.listdir(self.root[cat])
        
        self.rand_max = len(os.listdir(self.root[catNms[0]])) - 1 #number of files in directory

        self.mapping = {0 : 'airplane',
                        1: 'bus',
                        2: 'cat',
                        3: 'dog',
                        4: 'pizza'} #makes it easy to convert between index and name of a category.
        
        self.one_hot_encoding = {0: torch.tensor(np.array([1, 0, 0, 0, 0])),
                                1: torch.tensor(np.array([0, 1, 0, 0, 0])),
                                2: torch.tensor(np.array([0, 0, 1, 0, 0])),
                                3: torch.tensor(np.array([0, 0, 0, 1, 0])),
                                4: torch.tensor(np.array([0, 0, 0, 0, 1]))} #one hot encode each category. 

        
        self.to_Tensor_and_Norm = tvt.Compose([tvt.ToTensor(),tvt.Resize((64,64)) , 
                                        tvt.Normalize([0], [1]) ]) #normalize and resize in case the resize op 
#         wasn't done. Note that resizing here may not have any impact as the resizing was done previously.   
    def __len__(self):
        count = 0
        for cat in catNms:
            temp_num = os.listdir(self.root[cat])
            count = count + len(temp_num)
        return count #return count. Will be 2500 if the root=val/ and 7500 if root=train/

    def __getitem__(self, index):
        file_index = index % self.rand_max + 1
        class_index = index % 5
 
        img_file = self.filenames[self.mapping[class_index]]
        
        try:
            item = Image.open(self.root[self.mapping[class_index]] + img_file[file_index])
        except IndexError: #for debugging
            print('these are the indices for the line above when shape is correct', class_index , file_index)
            
        np_img = np.array(item)
        shape = np_img.shape
        while shape != (64, 64 ,3): #handle if the image from COCO is grayscale. 
            #print('found a grayscale image, fetching an RGB!')
            another_rand = random.randint(0,self.rand_max)  #generate another rand num
            #print('another_rand is', another_rand)
            try:
                item = Image.open(self.root[self.mapping[class_index]] + img_file[another_rand])
            except IndexError: #for debugging
                print('these are the indices for the line above when shape is incorrect', another_rand , class_index)            
            np_img = np.array(item)
            shape = np_img.shape

        img = self.to_Tensor_and_Norm(item)
        class_label = self.one_hot_encoding[class_index].type(torch.FloatTensor) #convert to Float 
        return img, class_label


my_train_dataset = MyDataset(root_train, catNms)
print(len(my_train_dataset))
index = 3
print(my_train_dataset[index][0].shape, my_train_dataset[index][1])
my_val_dataset = MyDataset(root_val, catNms)
print(len(my_val_dataset))
print(my_val_dataset[index][0].shape, my_val_dataset[index][1])


# In[40]:


# Use MyDataset class in PyTorches DataLoader functionality
my_train_dataloader = torch.utils.data.DataLoader(my_train_dataset, batch_size=12, num_workers = 4, drop_last=False)
my_val_dataloader = torch.utils.data.DataLoader(my_val_dataset, batch_size = 12, num_workers = 4, drop_last = False)
for n, batch in enumerate(my_train_dataloader):
#     #Note: each batch is a list of length 2. The first is a pytorch tensor B x C x H x W and the 
#     #second is a pytorch tensor of length B with the associated class labels of each image in the 
#     #first item of the list!
    print('batch is', n)


# In[95]:


# Example of target with class indices
loss = nn.CrossEntropyLoss()
inp = torch.randn(3, 5, requires_grad=True)
print(inp)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(inp, target)
output.backward()
print(target)


# In[41]:


###Create CNN ####
### Use valid mode - i.e, no padding

class HW4Net(nn.Module):
    def __init__(self):
        super(HW4Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(6272,64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[42]:


net1 = HW4Net()
loss_running_list_net1 = []
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net1.parameters(), lr = 1e-3, betas = (0.9, 0.99))
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(my_train_dataloader):
        inputs, labels = data
        optimizer.zero_grad() #Sets gradients of all model parameters to zero. We want to compute fresh gradients
        #based on the new forward run. 
        outputs = net1(inputs)
        loss = criterion(outputs, labels) #compute cross-entropy loss
        loss.backward() #compute derivative of loss wrt each gradient. 
        optimizer.step() #takes a step on hyperplane based on derivatives
        running_loss += loss.item() 
        if (i+1) % 100 == 0:
            print("[epoch: %d, batch: %5d] loss: %3f" % (epoch + 1, i + 1, running_loss / 100))
            loss_running_list_net1.append(running_loss/100)
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for n, data in enumerate(my_val_dataloader):
            images, labels = data
            outputs = net1(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) #add to total's total
            for n, i in enumerate(labels):
                temp = np.array(i) #temp holds the one hot encoded label
                idx = np.argmax(temp) #get the argmax of the encoded label - will be a value between 0 and 4.
                #print(idx)
                if idx == predicted[n]: #if the predicted value and label match
                    correct = correct + 1 #add to correct total

    print('Accuracy of the network on the val images: %d %%' % (
        100 * correct / total))
            


# In[76]:





# In[62]:


### Test performance of CNN 1 on val data ###
correct = 0
total = 0
y_pred = []
y_label = []
mapping = { 0: 'airplane',
            1: 'bus',
            2: 'cat',
            3: 'dog',
            4: 'pizza'}


with torch.no_grad():
    for n, data in enumerate(my_val_dataloader):
        images, labels = data

        outputs = net1(images)

        _, predicted = torch.max(outputs.data, 1) 

        total += labels.size(0) #add to total count of ground truth images so we can calculate total accuracy
        #print("total images in val set", total)
        for n, i in enumerate(labels):
            temp = np.array(i) #arrays are one hot encoded, we need to convert it into a human readable label for
            #display in the confusion matrix
            label_arg = np.argmax(temp) #get the argument of the one hot encoding
            y_label.append(mapping[label_arg]) #apply the argument to the mapping dictionary above. For example
            # if the argument is 3, then, that corresponds to a label of dog in the mapping dictionary
            t = int(np.array(predicted[n])) #get integer representation of prediction from network (will 
            #be an int from 0 to 4. 
            y_pred.append(mapping[t]) #append the predicted output of this label to the prediction list, but, 
            #via the mapping dictionary definition so that the y_pred list is human readable. 

            if label_arg == predicted[n]:
                correct = correct + 1 #add to total count of correct predictions so we can calculate total accuracy
            

print('Accuracy of the network on the val images: %d %%' % (
    100 * correct / total))
from sklearn.metrics import confusion_matrix

y_true = y_label
y_pred = y_pred
confusion_matrix=confusion_matrix(y_true, y_pred, labels = [ "airplane", "bus", "cat", "dog", "pizza"])
disp = ConfusionMatrixDisplay(confusion_matrix, display_labels = [ "airplane", "bus", "cat", "dog", "pizza"])
disp.plot()
disp.ax_.set_title("Confusion Matrix for CNN 1")
plt.show()
plt.savefig('CM_CNN1')


# In[44]:


### Create Net 2 ###
class HW4Net2(nn.Module):
    def __init__(self):
        super(HW4Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(8192,64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[45]:


### RUN TRAINING FOR NET 2 - THE NETWORK WITH PADDING ###
net2 = HW4Net2()
loss_running_list_net2 = []
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net2.parameters(), lr = 1e-3, betas = (0.9, 0.99))
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(my_train_dataloader):
        inputs, labels = data
        optimizer.zero_grad() #Sets gradients of all model parameters to zero. We want to compute fresh gradients
        #based on the new forward run. 
        outputs = net2(inputs)
        loss = criterion(outputs, labels) #compute cross-entropy loss
        loss.backward() #compute derivative of loss wrt each gradient. 
        optimizer.step() #takes a step on hyperplane based on derivatives
        running_loss += loss.item() 
        if (i+1) % 100 == 0:
            print("[epoch: %d, batch: %5d] loss: %3f" % (epoch + 1, i + 1, running_loss / 100))
            loss_running_list_net2.append(running_loss/100)
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for n, data in enumerate(my_val_dataloader):
            images, labels = data
            outputs = net2(images)


            _, predicted = torch.max(outputs.data, 1)
            if n < 1:
                print("this is _", _)
                print("this is predicted", predicted)
#             #print("outputs.data:", outputs.data)
#             if n < 1:
#                 #print(_)
#                 print("predicted", predicted)
            total += labels.size(0)
            #print("total images in val set", total)
            for n, i in enumerate(labels):
                #print(i)
                temp = np.array(i)
                #print(predicted[n])
                idx = np.argmax(temp)
                #print(idx)
                if idx == predicted[n]:
                    correct = correct + 1
                    #print('something is correct!!!')
            #correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the val images: %d %%' % (
        100 * correct / total))


# In[61]:


### RUN VAL WITH TORCH.NO_GRAD FOR NET 2 - THE NETWORK WITH PADDING ###
### GENERATE CONFUSION MATRIX ####

correct = 0
total = 0
y_pred = []
y_label = []
mapping = { 0: 'airplane',
            1: 'bus',
            2: 'cat',
            3: 'dog',
            4: 'pizza'}


with torch.no_grad():
    for n, data in enumerate(my_val_dataloader):
        images, labels = data

        outputs = net2(images)

        _, predicted = torch.max(outputs.data, 1) 

        total += labels.size(0) #add to total count of ground truth images so we can calculate total accuracy
        #print("total images in val set", total)
        for n, i in enumerate(labels):
            temp = np.array(i) #arrays are one hot encoded, we need to convert it into a human readable label for
            #display in the confusion matrix
            label_arg = np.argmax(temp) #get the argument of the one hot encoding
            y_label.append(mapping[label_arg]) #apply the argument to the mapping dictionary above. For example
            # if the argument is 3, then, that corresponds to a label of dog in the mapping dictionary
            t = int(np.array(predicted[n])) #get integer representation of prediction from network (will 
            #be an int from 0 to 4. 
            y_pred.append(mapping[t]) #append the predicted output of this label to the prediction list, but, 
            #via the mapping dictionary definition so that the y_pred list is human readable. 

            if label_arg == predicted[n]:
                correct = correct + 1 #add to total count of correct predictions so we can calculate total accuracy
            

print('Accuracy of the network on the val images: %d %%' % (
    100 * correct / total))
from sklearn.metrics import confusion_matrix

y_true = y_label
y_pred = y_pred
confusion_matrix=confusion_matrix(y_true, y_pred, labels = [ "airplane", "bus", "cat", "dog", "pizza"])
disp = ConfusionMatrixDisplay(confusion_matrix, display_labels = [ "airplane", "bus", "cat", "dog", "pizza"])
disp.plot()
disp.ax_.set_title("Confusion Matrix for CNN 2")
plt.show()
plt.savefig('CM_CNN2')


# In[47]:


### CNN TASK 3 ###
### CREATE 10 CONV LAYERS BETWEEN CONV LAYERS IN NET1/2 ###
### 32 IN, 32 OUT, KER 3, PAD 1 ###
### FORWARD SHOULD HAVE AN ACTIVATION FUNCTION BEFORE GOING TO NEXT LAYER ###
### Create Net 3 ###
class HW4Net3(nn.Module):
    def __init__(self):
        super(HW4Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv11 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.fc1 = nn.Linear(2048,64)
        self.fc2 = nn.Linear(64, 5)
    
    def forward(self, x): #we are passing in a torch.float32 into the network with a shape 12, 3, 64, 64
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))              
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# In[48]:


net3 = HW4Net3()
loss_running_list_net3 = []
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net3.parameters(), lr = 1e-3, betas = (0.9, 0.99))
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(my_train_dataloader):
        inputs, labels = data
        optimizer.zero_grad() #Sets gradients of all model parameters to zero. We want to compute fresh gradients
        #based on the new forward run. 
        outputs = net3(inputs)
        loss = criterion(outputs, labels) #compute cross-entropy loss
        loss.backward() #compute derivative of loss wrt each gradient. 
        optimizer.step() #takes a step on hyperplane based on derivatives
        running_loss += loss.item() 
        if (i+1) % 100 == 0:
            print("[epoch: %d, batch: %5d] loss: %3f" % (epoch + 1, i + 1, running_loss / 100))
            loss_running_list_net3.append(running_loss/100)
            running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for n, data in enumerate(my_val_dataloader):
            images, labels = data
            outputs = net3(images)


            _, predicted = torch.max(outputs.data, 1)
            if n < 1:
                print("this is _", _)
                print("this is predicted", predicted)
#             #print("outputs.data:", outputs.data)
#             if n < 1:
#                 #print(_)
#                 print("predicted", predicted)
            total += labels.size(0)
            #print("total images in val set", total)
            for n, i in enumerate(labels):
                #print(i)
                temp = np.array(i)
                #print(predicted[n])
                idx = np.argmax(temp)
                #print(idx)
                if idx == predicted[n]:
                    correct = correct + 1
                    #print('something is correct!!!')
            #correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the val images: %d %%' % (
        100 * correct / total))


# In[64]:


### RUN VAL WITH TORCH.NO_GRAD FOR NET 3 - THE NETWORK WITH PADDING AND MORE LAYERS ###
### GENERATE CONFUSION MATRIX ####

correct = 0
total = 0
y_pred = []
y_label = []
mapping = { 0: 'airplane',
            1: 'bus',
            2: 'cat',
            3: 'dog',
            4: 'pizza'}


with torch.no_grad():
    for n, data in enumerate(my_val_dataloader):
        images, labels = data

        outputs = net3(images)

        _, predicted = torch.max(outputs.data, 1) 

        total += labels.size(0) #add to total count of ground truth images so we can calculate total accuracy
        #print("total images in val set", total)
        for n, i in enumerate(labels):
            temp = np.array(i) #arrays are one hot encoded, we need to convert it into a human readable label for
            #display in the confusion matrix
            label_arg = np.argmax(temp) #get the argument of the one hot encoding
            y_label.append(mapping[label_arg]) #apply the argument to the mapping dictionary above. For example
            # if the argument is 3, then, that corresponds to a label of dog in the mapping dictionary
            t = int(np.array(predicted[n])) #get integer representation of prediction from network (will 
            #be an int from 0 to 4. 
            y_pred.append(mapping[t]) #append the predicted output of this label to the prediction list, but, 
            #via the mapping dictionary definition so that the y_pred list is human readable. 

            if label_arg == predicted[n]:
                correct = correct + 1 #add to total count of correct predictions so we can calculate total accuracy
            

print('Accuracy of the network on the val images: %d %%' % (
    100 * correct / total))
from sklearn.metrics import confusion_matrix

y_true = y_label
y_pred = y_pred
confusion_matrix=confusion_matrix(y_true, y_pred, labels = [ "airplane", "bus", "cat", "dog", "pizza"])
disp = ConfusionMatrixDisplay(confusion_matrix, display_labels = [ "airplane", "bus", "cat", "dog", "pizza"])
disp.plot()
disp.ax_.set_title("Confusion Matrix for CNN 3")
plt.show()
plt.savefig('CM_CNN3')


# In[59]:


### Plot training loss for CNN1 and CNN2 ###
    
plt.plot(loss_running_list_net1, label = 'Net1')
plt.plot(loss_running_list_net2, label = 'Net2')
plt.plot(loss_running_list_net3, label = 'Net3')
plt.xlabel('iteration * 100')
plt.ylabel('loss')
plt.title('Training loss for 3 CNNs over 10 Epochs')
plt.legend()
plt.savefig('Training_loss.jpg')

