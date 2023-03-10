#!/usr/bin/env python
# coding: utf-8

# In[193]:


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
import skimage
import cv2
from torchvision import ops


# In[2]:


print(os.getcwd())
class_list = ['bus', 'cat', 'pizza']
root = "/Users/alim/Documents/ECE60146/hw4/"
os.chdir(root)
dataType='train2014'
annFile= root + 'annotations/instances_{}.json'.format(dataType)

### Mapping from COCO label to Class indices ###
coco_labels_inverse = {}
coco = COCO(annFile)


# In[3]:


### PREPROCESSING TO GENERATE IMAGE UNIQUE IDS ###
catIds = coco.getCatIds(catNms = class_list)
imgIds = coco.getImgIds(catIds = catIds)
imgIds_list = []
for catId in catIds:
    imgIds_list.append(coco.getImgIds(catIds = catId))
concatenated_list = []
for x in imgIds_list:
    for y in x:
        concatenated_list.append(y)

np_li = np.array(concatenated_list)
unique_imgIds = np.unique(np_li) #make the list unique - there might be duplicate image ids

unique_imgIds = list(unique_imgIds) #convert numpy array back to list

for n, i in enumerate(unique_imgIds):
    unique_imgIds[n] = int(i) #convert each element of list to integer, so it can easily be interpretted by 
    #coco api



# In[4]:


### CODE TO DOWNLOAD THE CORRECT IMAGES AND GET THE CORRECT ANNOTATIONS (I.E THE DOMINANT OBJECT)
count = 0
count_true = 0
data_dict = {}
os.chdir("/Users/alim/Documents/ECE60146/hw5/")
for n, imgId in enumerate(unique_imgIds):
        count += 1
        annId = coco.getAnnIds(imgId, catIds=catIds, iscrowd=False)
        #now that we have the annotation id, we want to check if there is a dominant object. In order to get 
        #the details of that annotation id, we need to load the annotation using that id:
        annId_metadata = coco.loadAnns(annId)
        temp_max_bbox_area = 0 #need a variable to figure out which bbox is the largest, as there can be
        #multiple bboxes per imgId even in a given category! 
        max_bbox_area = 0
        found_valid_bbox = False
        key = -1 #temp value for key that doesn't exist
        for ann in annId_metadata:
            len_ann = len(annId_metadata)
            
            #print(ann['bbox'], "is the bbox for img_id", ann['image_id'])
            [x, y, w, h] = ann['bbox']
            if ((w * h) <= 40000):
                continue #do not consider looking at annotation details for that annId
            else: 
                #figure out which annotation for that image id is the largest:
                temp_max_bbox_area = w * h
                if temp_max_bbox_area > max_bbox_area: #if w*h is GT than the current largest bbox area, then
                    #make max_bbox_area the w*h that was just computed. Store the metadata of that ann in 
                    #dictionary.
                    found_valid_bbox = True
                    max_bbox_area = temp_max_bbox_area
                    #print("imgId being loaded", imgId)
                    #print(type(imgId))
                    imgId_metadata = coco.loadImgs(imgId)
                    #print("imgId_metadata", imgId_metadata, "for imgId", imgId)
                    #print(imgId_metadata)
                    ###ORGANIZATION OF data_dict:
                    #key is file_name
                    #value is a list of elements.
                    #element 0: annotation id
                    #element 1: bbox
                    #element 2: category id
                    #element 3: img_id coco_url
                    #element 4: img_id
                    #element 5: img height
                    #element 6: img width
                    #print("imgId", imgId)
                    key = imgId_metadata[0]['file_name']
                    data_dict[key] = [ann['id'], ann['bbox'], ann['category_id'], imgId_metadata[0]['coco_url'],
                                       imgId_metadata[0]['id'], imgId_metadata[0]['height'], 
                                     imgId_metadata[0]['width']]

        if (key != -1):
            coco.download(tarDir = 'train_orig/', imgIds = [data_dict[key][4]])
        if (found_valid_bbox == True): #check if we found all the expected images
            count_true += 1


            
print(count) # checked 7799 images
print(count_true) # 4856 valid bboxes found
print(len(data_dict), "is the len") #4856 image id keys in dictionary
fig, ax = plt.subplots(1, 6, figsize = (15, 15))
d = os.listdir('train_orig/') #create list of files in the created directory. This list has 1500 
print("d is", d, "length is", len(d))
d.remove(".DS_Store")
print("updated list d", d)
for num, i in enumerate(d[:6]):
    
    if i[-3:] == "jpg":
        PIL_img = Image.open('train_orig/' + i)
        np_img = np.array(PIL_img)
        np_img = np.uint8(np_img)
        [x, y, w, h] = (data_dict[i][1])
        #[x, y, w, h] = data_dict[i]
        np_img = cv2.rectangle(np_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        ax[num].imshow(np_img)
### Confirmed that dictionary is working properly and we can retrieve images and annotations from it! 
    
    


# In[5]:


### PREPROCESSING FOR VAL DATA
### PREPROCESSING TO GENERATE IMAGE IDS ###
### GET VAL DATA ####
print(os.getcwd())
class_list = ['bus', 'cat', 'pizza']
root = "/Users/alim/Documents/ECE60146/hw4/"
os.chdir(root)
dataType='val2014'
annFile= root + 'annotations/instances_{}.json'.format(dataType)

### Mapping from COCO label to Class indices ###
coco_labels_inverse = {}
coco = COCO(annFile)
catIds = coco.getCatIds(catNms = class_list)
print(catIds)
imgIds = coco.getImgIds(catIds = catIds)
imgIds_list = []
for catId in catIds:
    imgIds_list.append(coco.getImgIds(catIds = catId))
concatenated_list = []
for x in imgIds_list:
    for y in x:
        concatenated_list.append(y)
print(len(concatenated_list))

np_li = np.array(concatenated_list)
unique_imgIds = np.unique(np_li) #make the list unique - there might be duplicate image ids

unique_imgIds = list(unique_imgIds) #convert numpy array back to list

for n, i in enumerate(unique_imgIds):
    unique_imgIds[n] = int(i) #convert each element of list to integer, so it can easily be interpretted by coco api
print(len(unique_imgIds))


# In[6]:


### CREATE DICTIONARY FOR VALIDATION DATA AND DOWNLOAD TO DISK ###
count = 0
count_true = 0
data_dict_val = {}
os.chdir("/Users/alim/Documents/ECE60146/hw5/")
for n, imgId in enumerate(unique_imgIds):
        count += 1
        annId = coco.getAnnIds(imgId, catIds=catIds, iscrowd=False)
        #now that we have the annotation id, we want to check if there is a dominant object. In order to get 
        #the details of that annotation id, we need to load the annotation using that id:
        annId_metadata = coco.loadAnns(annId)
        temp_max_bbox_area = 0 #need a variable to figure out which bbox is the largest, as there can be
        #multiple bboxes per imgId even in a given category! 
        max_bbox_area = 0
        found_valid_bbox = False
        key = -1 #temp value for key that doesn't exist
        for ann in annId_metadata:
            len_ann = len(annId_metadata)
            
            #print(ann['bbox'], "is the bbox for img_id", ann['image_id'])
            [x, y, w, h] = ann['bbox']
            if ((w * h) <= 40000):
                continue #do not consider looking at annotation details for that annId
            else: 
                #figure out which annotation for that image id is the largest:
                temp_max_bbox_area = w * h
                if temp_max_bbox_area > max_bbox_area: #if w*h is GT than the current largest bbox area, then
                    #make max_bbox_area the w*h that was just computed. Store the metadata of that ann in 
                    #dictionary.
                    found_valid_bbox = True
                    max_bbox_area = temp_max_bbox_area
                    #print("imgId being loaded", imgId)
                    #print(type(imgId))
                    imgId_metadata = coco.loadImgs(imgId)
                    #print("imgId_metadata", imgId_metadata, "for imgId", imgId)
                    #print(imgId_metadata)
                    ###ORGANIZATION OF data_dict:
                    #key is file_name
                    #value is a list of elements.
                    #element 0: annotation id
                    #element 1: bbox (x coord, y coord), (x+w coord, y+h coord)
                    #element 2: category id
                    #element 3: img_id coco_url
                    #element 4: img_id
                    #element 5: img height
                    #element 6: img width
                    #print("imgId", imgId)
                    key = imgId_metadata[0]['file_name']
                    data_dict_val[key] = [ann['id'], ann['bbox'], ann['category_id'], imgId_metadata[0]['coco_url'],
                                       imgId_metadata[0]['id'], imgId_metadata[0]['height'], 
                                          imgId_metadata[0]['width']]

        #once we have figured out the annId with the largest bbox for that specific image id, download the 
        #image!
        
        #download using that metadata
        pathlib.Path('val_orig/').mkdir(parents=True, exist_ok=True) #create a path to store 
#         print(imgId)
#         print(type(imgId))
        if (key != -1):
            coco.download(tarDir = 'val_orig/', imgIds = [data_dict_val[key][4]])
        #  training data for the current category
        
        #print("Did we find a valid bbox for image Id", imgId, "?", found_valid_bbox)
        if (found_valid_bbox == True):
            count_true += 1
            #print("the data dict entry for ", imgId, "is ", data_dict[imgId])
            #print("imgId_metadata for imgId", imgId, imgId_metadata)

            
print(count) # checked 7799 images
print(count_true) # 4856 valid bboxes found
print(len(data_dict_val), "is the len") #4856 image id keys in dictionary
fig, ax = plt.subplots(1, 6, figsize = (15, 15))
d = os.listdir('val_orig/') #create list of files in the created directory. This list has 1500 
print("d is", d, "length is", len(d))
for i in d:
    if (i == "DS_Store"):
        d.remove(".DS_Store")
#print("updated list d", d)
for num, i in enumerate(d[:6]):
    
    if i[-3:] == "jpg":
        PIL_img = Image.open('val_orig/' + i)
        np_img = np.array(PIL_img)
        np_img = np.uint8(np_img)
        [x, y, w, h] = (data_dict_val[i][1])
        #[x, y, w, h] = data_dict[i]
        np_img = cv2.rectangle(np_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        ax[num].imshow(np_img)
### Confirmed that dictionary is working properly and we can retrieve images and annotations from it!


# In[7]:


#print out images and annotations for above to make sure images and annotations pass a sanity check
#also, apply rescaling to ensure that the images and rescaled bbox look accurate! 
length_data = len(data_dict)
print(length_data)
fig, ax = plt.subplots(10, 2, figsize=(15,15))
count = 0
for idx, img in enumerate(data_dict):
    count += 1
    #print("img", img)
    #print("data_dict[img]", data_dict[img])

    I = io.imread(data_dict[img][3]) #this is a np array

    image_orig = np.uint8(I)
    image_resized = cv2.resize(image_orig, (256,256) )
    
    label = data_dict[img][2]
    [x, y, w, h] = data_dict[img][1]
#     print("x", x, "y", y, "w" , w, "h", h)
#     print("if the above is the origial coords for the bbox, then, we need to apply the scaling factor\n",
#     "to each parameter that defines the bbox!")
#     print("original image shape:", image_orig.shape)
#     print("shape[0] is the H of image, i.e, the y axis while shape[1] is the W of image, i.e the x-axis!")
    scaling_factor_x =  256.0 / image_orig.shape[1] 
    scaling_factor_y =  256.0 / image_orig.shape[0]
    #print("scaling_factor_x:", scaling_factor_x, "scaling_factor_y:", scaling_factor_y)
    
    
    image_orig = cv2.rectangle(image_orig, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    image_resized = cv2.rectangle(image_resized, (int(x * scaling_factor_x), int(y*scaling_factor_y)), 
                                (int((x+w)*scaling_factor_x), int((y+h)*scaling_factor_y)), (0,255,0), 2)
    ax[idx,1].imshow(image_resized)
    ax[idx,0].imshow(image_orig)
    ax[idx,0].set_title("cat id" + str(label))
    if count == 10:
        break

        
#Images and rescaled version look good! 
    


# In[513]:


#Resize images on disk. We are able to compute the adjusted bounding box on the fly, so, no need to save it
#to a dict.
print(os.getcwd())
count = 0
root = '/Users/alim/Documents/ECE60146/hw5/'
folders = ['train_resized/', 'val_resized/']
for folder in folders:
    print('folder:', folder)
    imgs = os.listdir(folder) #create list of files in the created directory. 
    #print("imgs is", imgs, "length is", len(imgs))
    for img in imgs:
        if (img == ".DS_Store"):
            imgs.remove(".DS_Store")
            print('removed .DS_Store')
            continue
        temp_img = Image.open(root + folder + img) #open image
        temp_img = temp_img.resize((256,256)) #resize
        temp_img.save(fp = folder + img) #overwrite image with the 64 x 64 version
        count += 1
print(count)   


# In[533]:


# Make a figure of a selection of images from dataset. 3 images from each of the classes, with annotation of 
#dominant object. 
fig, ax = plt.subplots(3,3 ,figsize = (10,10))
count_bus = 0
count_cat = 0
count_pizza = 0
root = '/Users/alim/Documents/ECE60146/hw5/'

for n, i in enumerate(data_dict):
        cat = data_dict[i][2]
        #print('cat is', cat)
        if (cat == 6) and (count_bus < 3):
            PIL_img = Image.open(root + 'train_resized/' + str(i))
            np_img = np.uint8(PIL_img)
            scaling_factor_x =  256.0 / data_dict[i][6] 
            scaling_factor_y =  256.0 / data_dict[i][5]
            [x, y, w, h] = data_dict[i][1]
            np_img = cv2.rectangle(np_img, (int(x * scaling_factor_x), int(y*scaling_factor_y)), 
                                (int((x+w)*scaling_factor_x), int((y+h)*scaling_factor_y)), (0,255,0), 2)
            np_img = cv2.putText(np_img, "bus", (int(x * scaling_factor_x), int(y * scaling_factor_y + 25)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255, 12),2)
            ax[0, count_bus].imshow(np_img)            
            

            count_bus = count_bus + 1

        if (cat == 17) and (count_cat < 3):
            PIL_img = Image.open(root + 'train_resized/' + str(i))
            np_img = np.uint8(PIL_img)
            scaling_factor_x =  256.0 / data_dict[i][6] 
            print(data_dict[i][6] )
            scaling_factor_y =  256.0 / data_dict[i][5]
            [x, y, w, h] = data_dict[i][1]
            
            np_img = cv2.rectangle(np_img, (int(x * scaling_factor_x), int(y*scaling_factor_y)), 
                                (int((x+w)*scaling_factor_x), int((y+h)*scaling_factor_y)), (0,255,0), 2)
            np_img = cv2.putText(np_img, "cat", (int(x * scaling_factor_x), int(y * scaling_factor_y + 25)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255, 12),2)
            ax[1, count_cat].imshow(np_img)
            count_cat += 1

        if (cat == 59) and (count_pizza < 3):
            PIL_img = Image.open(root + 'train_resized/' + str(i))
            np_img = np.uint8(PIL_img)
            
            scaling_factor_x =  256.0 / data_dict[i][6] 
            scaling_factor_y =  256.0 / data_dict[i][5]
            [x, y, w, h] = data_dict[i][1]
            np_img = cv2.rectangle(np_img, (int(x * scaling_factor_x), int(y*scaling_factor_y)), 
                                (int((x+w)*scaling_factor_x), int((y+h)*scaling_factor_y)), (0,255,0), 2)
            np_img = cv2.putText(np_img, "pizza", (int(x * scaling_factor_x), int(y * scaling_factor_y + 25)), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255, 12),2)
            ax[2, count_pizza].imshow(np_img)
            count_pizza += 1
        if ((count_pizza == 3) and (count_bus == 3) and (count_cat == 3)):
            break

       


# In[588]:


#Find non-color images, make them color images so that all tensor shapes are the same in the dataset! #
# Will be important when computing loss and so that the dataloader outputs what one expects #
# Also, important so that the network can process the images properly ##
root = '/Users/alim/Documents/ECE60146/hw5/'
folders = ['train_resized/', 'val_resized/']
imgs = os.listdir(folders[1])
for i in imgs:
    temp_arr = np.array(Image.open(root + folders[1] + i))
    if (temp_arr.shape != (256, 256, 3)):
        print("bad shape in", i)
        print(temp_arr.shape)
        color_image = skimage.color.gray2rgb(temp_arr)
        print(type(test))
        print(test.shape)
        print(str(folders[1] + i))
        color_image = Image.fromarray(color_image)
        color_image.save(fp = folders[1] + i) #save reshaped image to disk

#after running this code with folders[0] and folders[1] as the directories that are looped through, there
#shouldn't be anymore images that have a shape not 256,256,3! 


# In[514]:


## Create dataloader that returns image, label (i.e, either bus, cat, pizza), and bbox params
## bbox should be in the format [x1, y1, x2, y2], where x1, y1 are the top left corner of bbox
## and x2, y2 are the bottom right corner of the bbox. 
## Note that the coordinate values should reside in the range (0,1). So, we should rescale the values of the 
## bbox coords (i.e, normalize them)

root = '/Users/alim/Documents/ECE60146/hw5/'
folders = ['train_resized/', 'val_resized/']

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, folder):
        super(MyDataset).__init__()
        self.root = root
        self.folder = folder
        if (folder == 'train_resized/'):
            self.data_dict = data_dict
        if (folder == 'val_resized/'):
            self.data_dict = data_dict_val
        
        self.mapping = {6: 0, 17: 1, 59: 2}
        self.images = os.listdir(folder) #create list of files in the train or val directory. We will use
        # this list to get bbox params, read image files, etc. 
        for img in self.images:
            if (img == "DS_Store"):
                self.images.remove(".DS_Store") #handle case when image isn't an image. Just remove it from the 
                #image list. 
        
        self.to_Tensor_and_Norm = tvt.Compose([tvt.ToTensor(), tvt.Normalize([0], [1])])

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        #prepare image:
        PIL_img = Image.open(self.root + self.folder + self.images[index])
        torch_img = self.to_Tensor_and_Norm(PIL_img)
        
        #prepare class label
        class_label = self.data_dict[self.images[index]][2]
        class_label = self.mapping[class_label]
        class_label = torch.tensor(class_label)
        
        #prepare bounding box
        bbox = self.data_dict[self.images[index]][1] #not rescaled. has original values from coco.
        [x, y, w, h] = bbox #bbox is still in terms of original image dims. 
        bbox = np.array(bbox)
        img_orig_H = self.data_dict[self.images[index]][5] #H or y
        img_orig_W = self.data_dict[self.images[index]][6] #W or x
        #rescaling bbox so the input is [x1, y1, x2, y2]
        scale_factor_x = 256.0 / img_orig_W
        scale_factor_y = 256.0 / img_orig_H
        bbox[0] = x * scale_factor_x
        bbox[2] = w * scale_factor_x
        bbox[1] = y * scale_factor_y
        bbox[3] = h * scale_factor_y
        bbox[2] = bbox[2] + bbox[0]
        bbox[3] = bbox[3] + bbox[1]

        #normalize bbox to between 0 and 1:
        bbox = bbox / 256.0
        bbox = torch.tensor(bbox) #convert to tensor so it can be used in pytorch for computing loss. 

        return torch_img, class_label, bbox
        
my_train_dataset = MyDataset(root = root, folder = folders[0])
index = random.randint(0,200)
print(my_train_dataset[index][0].shape, my_train_dataset[index][1], my_train_dataset[index][2])

my_val_dataset = MyDataset(root = root, folder = folders[1])


my_train_dataloader = torch.utils.data.DataLoader(my_train_dataset, batch_size=20, num_workers = 0, drop_last=False)

my_val_dataloader = torch.utils.data.DataLoader(my_val_dataset, batch_size=20, num_workers = 0, drop_last=False)



# In[534]:


## NETWORK ##

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SkipBlock, self).__init__() #make sure inherited classes are instantiated
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.convo1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1) #creates instance of
        #Conv2d that preserves shape of input!
        self.convo2 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1) #preserve shape
        self.convo3 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1) #preserve shape
        self.bn1 = nn.BatchNorm2d(out_ch) #batch norm to improve performance
        self.bn2 = nn.BatchNorm2d(out_ch) #batch norm to improve performance
        self.bn3 = nn.BatchNorm2d(out_ch) #batch norm to improve performance
        
    def forward(self, x):
        identity = x #save input so we can add it to the output
        #run three convolutions that preserve the shape, does BN, and applies an activation function
        out = F.relu(self.bn1(self.convo1(x))) 
        out = F.relu(self.bn2(self.convo2(x)))
        out = F.relu(self.bn3(self.convo3(x)))
        # add input to output to give the network more of the original signal
        out = out + identity
        return out
    


class HW5Net(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=8, n_blocks = 4): #ngf = num of conv filters in first convo layer
    #n_blocks is number of ResNet blocks
        assert(n_blocks >=0)
        super(HW5Net, self).__init__()
        # first conv layer
        model = [nn.ReflectionPad2d(3), 
                 nn.Conv2d(input_nc, ngf, kernel_size = 7, padding = 0),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True)]
        #add downsampling layers
        n_downsampling = 4
        for i in range(n_downsampling):
            mult = 2 ** i
            model = model + [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size = 3, stride = 2, padding = 1),
                            nn.BatchNorm2d(ngf * mult * 2),
                            nn.ReLU(True)]
        #add own Skip blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model = model + [ResBlock(64, 64)] # do 4 skip connections with 64 in_ch and 64 out_ch
        self.model = nn.Sequential(*model) 
        ### Classification head ###
        
        class_head = [nn.Linear(16384, 3)] #classification task just takes the skip connection's output and 
        #applies linear function
        self.class_head = nn.Sequential(*class_head)
        ### bounding box regression 

        bbox_head_conv = [nn.Conv2d(64, 64, kernel_size = 3, padding = 1), nn.BatchNorm2d(64),
                          nn.ReLU(inplace = True), nn.Conv2d(64, 64, 3 , 1), 
                          nn.BatchNorm2d(64), nn.ReLU(inplace = True)]
        self.bbox_head_conv = nn.Sequential(*bbox_head_conv)
        bbox_head_fc = [nn.Linear(12544, 1024), nn.ReLU(inplace = True), nn.Linear(1024,512),
                        nn.ReLU(inplace = True),
                       nn.Linear(512,4), nn.Sigmoid()] #sigmoid at the end to keep values between 0-1
        self.bbox_head_fc = nn.Sequential(*bbox_head_fc)
#         for i in range(n_blocks):
    def forward(self, x):
        #skip blocks:
        ft = self.model(x)
        #classification task:
        ft_r = ft.view(ft.shape[0], -1) #reshape output of skipblock to vector, then convert it to 
        # a vector (from original shape)
        cls = self.class_head(ft_r) #run classification task, return the 3 node output
        #bbox regression task
        bbox_temp = self.bbox_head_conv(ft)
        bbox_temp = bbox_temp.view(bbox_temp.shape[0], -1)
        bbox = self.bbox_head_fc(bbox_temp) #return the bounding box. 
        return cls, bbox.double()
        

        


# In[517]:


#TRAINING LOOP FOR MSE LOSS
test_net = HW5Net(input_nc = 3, output_nc = 3, ngf = 4, n_blocks=4)
loss_test_net = []
loss_test_net_MSE = []
criterion_CE = nn.CrossEntropyLoss()
criterion_MSE = nn.MSELoss()
optimizer = torch.optim.Adam(test_net.parameters(), lr = 1e-3, betas = (0.9, 0.99))
epochs = 4
num_layers = len(list(test_net.parameters()))

print(num_layers)
for epoch in range(epochs):
    running_loss_CE = 0.0
    running_loss_MSE = 0.0
    for i, data in enumerate(my_train_dataloader):
        print("in train dataloader iteration", i)
        inputs, class_label, bbox_label = data
        bbox_label = ops.box_convert(bbox_label, in_fmt = 'xyxy', out_fmt = 'cxcywh')
        optimizer.zero_grad() #Sets gradients of all model parameters to zero. We want to compute fresh gradients
        #based on the new forward run. 
        outputs = test_net(inputs) #output[0] is cls and output[1] is bbox
        bbox_outputs_converted = ops.box_convert(outputs[1], in_fmt = 'cxcywh', out_fmt = 'xyxy')
        bbox_label = ops.box_convert(bbox_label, in_fmt = 'cxcywh', out_fmt = 'xyxy')
        #print("shape of cls:", outputs[0].shape, "shape of bbox:", outputs[1].shape)
        loss_CE = criterion_CE(outputs[0], class_label) #compute cross-entropy loss
        #print("For computing CE loss, outputs[0] is", outputs[0], "and class_label is", class_label)
        loss_CE.backward(retain_graph = True)
        loss_MSE = criterion_MSE(bbox_outputs_converted, bbox_label)
        #print("bbox output from network:", outputs[1], "label:" ,bbox_label)

        loss_MSE.backward()
        #loss.backward() #compute derivative of loss wrt each gradient. 
        optimizer.step() #takes a step on hyperplane based on derivatives
        running_loss_CE += loss_CE.item()
        running_loss_MSE += loss_MSE.item()
        if (i+1) % 10 == 0:
            print("[epoch: %d, batch: %5d] loss: %3f" % (epoch + 1, i + 1, running_loss_CE + running_loss_MSE / 10))
            loss_test_net.append(running_loss_CE + running_loss_MSE/10)
            loss_test_net_MSE.append(running_loss_MSE/10)
            print('running_loss_MSE', running_loss_MSE/10)
            running_loss_CE = 0.0
            running_loss_MSE = 0.0


# In[518]:


#EVALUATION LOOP FOR OBJECT DETECTION MODEL TRAINED WITH MSE LOSS
correct = 0
total = 0
y_true = []
y_pred = []
mapping = { 0: 'bus',
            1: 'cat',
            2: 'pizza'}
iou_holder = torch.tensor(0)
with torch.no_grad():
    for n, data in enumerate(my_val_dataloader):
        print("in val dataloader iteration", n)
        #print("STARTING EVAL CODE")
        images, class_labels, bbox_label = data
        bbox_label = ops.box_convert(bbox_label, in_fmt = 'xyxy', out_fmt = 'cxcywh')
        cls_prediction, bbox_prediction = test_net(images)
        bbox_prediction = ops.box_convert(bbox_prediction, in_fmt = 'cxcywh', out_fmt = 'xyxy')
        bbox_label = ops.box_convert(bbox_label, in_fmt = 'cxcywh', out_fmt = 'xyxy')
        
        _, predicted = torch.max(cls_prediction.data, 1) # this returns max (_) and max index (predicted)

        total += class_labels.size(0) #add to total's total. This is the denominator for the prediction acc.
        for k, i in enumerate(class_labels): #compute number of correct predictions
            temp = np.array(i) #temp holds the one hot encoded label
            y_true.append(mapping[int(i)])
            y_pred.append(mapping[int(predicted[k])])
            #print("i is ", i)
            idx = np.argmax(temp) #get the argmax of the encoded label - will be a value between 0 and 4.
            if temp == np.array(predicted[k]): #if the predicted value and label match
                correct = correct + 1 #add to correct total
        #do IOU for entire batch:
        IOU = ops.box_iou(bbox_label, bbox_prediction)
        IOU = np.sum(np.diag(IOU))
        iou_holder = iou_holder + IOU



print('Accuracy of the network on the val images: %d %%' % (
    100 * correct / total))
print('mean IOU is ', iou_holder/total)


# In[541]:


model_total_params = sum(p.numel() for p in test_net.parameters() if p.requires_grad)
print("Total Parameters:" ,model_total_params)
num_layers = len(list(test_net.parameters()))
print("Total Layers:", num_layers)


# In[519]:


#CREATE CONFUSION MATRIX FOR OBJECT DETECTION MODEL USING MSE LOSS
from sklearn.metrics import confusion_matrix

y_true = y_true
y_pred = y_pred
confusion_matrix=confusion_matrix(y_true, y_pred, labels = [ "bus", "cat", "pizza"])
disp = ConfusionMatrixDisplay(confusion_matrix, display_labels = [  "bus", "cat", "pizza"])
disp.plot()
disp.ax_.set_title("Confusion Matrix for Object Detection Network with MSE")
plt.show()


# In[520]:


#TRAINING LOOP FOR MODEL USING COMPLETE IOU_LOSS LOSS
test_net_ciou = HW5Net(input_nc = 3, output_nc = 3, ngf = 4, n_blocks=4)
total_loss_test_net_using_ciou_loss = []
loss_test_net_ciou = []
criterion_CE = nn.CrossEntropyLoss()
#criterion_MSE = nn.MSELoss()
optimizer = torch.optim.Adam(test_net_ciou.parameters(), lr = 1e-3, betas = (0.9, 0.99))
epochs = 4
num_layers = len(list(test_net_ciou.parameters()))

print(num_layers)
for epoch in range(epochs):
    running_loss_CE = 0.0
    running_loss_CIOU = 0.0
    for i, data in enumerate(my_train_dataloader):
        print("in train dataloader iteration", i)
        inputs, class_label, bbox_label = data
        bbox_label = ops.box_convert(bbox_label, in_fmt = 'xyxy', out_fmt = 'cxcywh') #(boxes: Tensor, in_fmt: str, out_fmt: str)
        """
        ‘xyxy’: boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right. This is the format that torchvision utilities expect.
        ‘xywh’ : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.
        ‘cxcywh’ : boxes are represented via centre, width and height, cx, cy being center of box, w, h being width and height.
        """
        optimizer.zero_grad() #Sets gradients of all model parameters to zero. We want to compute fresh gradients
        #based on the new forward run. 
        outputs = test_net_ciou(inputs) #output[0] is cls and output[1] is bbox
        #print("shape of cls:", outputs[0].shape, "shape of bbox:", outputs[1].shape)
        
        loss_CE = criterion_CE(outputs[0], class_label) #compute cross-entropy loss
        #print("For computing CE loss, outputs[0] is", outputs[0], "and class_label is", class_label)
        loss_CE.backward(retain_graph = True)

        outputs_converted = ops.box_convert(outputs[1], in_fmt = 'cxcywh', out_fmt = 'xyxy')
        bbox_label = ops.box_convert(bbox_label, in_fmt = 'cxcywh', out_fmt = 'xyxy')

        loss_CIOU = ops.complete_box_iou_loss(boxes1 = outputs_converted, boxes2 = bbox_label, reduction = 'mean')

        loss_CIOU.backward()
        #loss.backward() #compute derivative of loss wrt each gradient. 
        optimizer.step() #takes a step on hyperplane based on derivatives
        running_loss_CE += loss_CE.item()
        running_loss_CIOU += loss_CIOU.item()
        if (i+1) % 10 == 0:
            print("[epoch: %d, batch: %5d] loss: %3f" % (epoch + 1, i + 1, running_loss_CE + running_loss_CIOU / 10))
            total_loss_test_net_using_ciou_loss.append(running_loss_CE + running_loss_CIOU/10)
            loss_test_net_ciou.append(running_loss_CIOU/10)
            print('running_loss_ciou', running_loss_CIOU/10)
            running_loss_CE = 0.0
            running_loss_CIOU = 0.0
            


# In[521]:


#EVALUATION LOOP FOR OBJECT DETECTION MODEL TRAINED WITH COMPLETE IOU LOSS
correct = 0
total = 0
y_true = []
y_pred = []
mapping = { 0: 'bus',
            1: 'cat',
            2: 'pizza'}
iou_holder = torch.tensor(0)
with torch.no_grad():
    for n, data in enumerate(my_val_dataloader):
        print("in val dataloader iteration", n)
        #print("STARTING EVAL CODE")
        images, class_labels, bbox_label = data
        bbox_label = ops.box_convert(bbox_label, in_fmt = 'xyxy', out_fmt = 'cxcywh')
        cls_prediction, bbox_prediction = test_net_ciou(images)
        bbox_prediction = ops.box_convert(bbox_prediction, in_fmt = 'cxcywh', out_fmt = 'xyxy')
        bbox_label = ops.box_convert(bbox_label, in_fmt = 'cxcywh', out_fmt = 'xyxy')
        _, predicted = torch.max(cls_prediction.data, 1) # this returns max (_) and max index (predicted)

        total += class_labels.size(0) #add to total's total. This is the denominator for the prediction acc.
        for k, i in enumerate(class_labels): #compute number of correct predictions
            temp = np.array(i) #temp holds the one hot encoded label
            y_true.append(mapping[int(i)])
            y_pred.append(mapping[int(predicted[k])])
            #print("i is ", i)
            idx = np.argmax(temp) #get the argmax of the encoded label - will be a value between 0 and 4.
            if temp == np.array(predicted[k]): #if the predicted value and label match
                correct = correct + 1 #add to correct total

        IOU = ops.box_iou(bbox_label, bbox_prediction)
        print('bbox is', bbox_label)
        print('bbox_pred is', bbox_prediction)
        IOU = np.sum(np.diag(IOU))
        
        print('IOU for iteration', n, 'is ', IOU)
        iou_holder = iou_holder + IOU



print('Accuracy of the network on the val images: %d %%' % (
    100 * correct / total))
print('mean IOU is ', iou_holder/total)


# In[523]:


#CREATE CONFUSION MATRIX FOR OBJECT DETECTION MODEL USING COMPLETE IOU LOSS
from sklearn.metrics import confusion_matrix

y_true = y_true
y_pred = y_pred
confusion_matrix=confusion_matrix(y_true, y_pred, labels = [ "bus", "cat", "pizza"])
disp = ConfusionMatrixDisplay(confusion_matrix, display_labels = [  "bus", "cat", "pizza"])
disp.plot()
disp.ax_.set_title("Confusion Matrix for Object Detection Network with CIOU")
plt.show()


# In[540]:


#compute total num of model parameters
param_hold = []

for i in list(test_net.parameters()):
    shape = i.shape
    temp = 0
    if len(shape) == 1:
        param_hold.append(shape)
    else:
        shape_hold = []
        for s in shape:
            shape_hold.append(s)
        for num, val in enumerate(shape_hold):
            if num == 0:
                temp = val
            else:
                temp = temp * val
        param_hold.append(temp)

        
print(param_hold)
total = 0
for i in param_hold:
    i = np.array(i)
    total = total + i
print(total)
    


# In[597]:


my_val_dataloader = torch.utils.data.DataLoader(my_val_dataset, batch_size=20, num_workers = 0, drop_last=False, 
                                               shuffle = True)

fig, ax = plt.subplots(3,3 ,figsize = (10,10))
count_bus = 0
count_cat = 0
count_pizza = 0

with torch.no_grad():
    for n, data in enumerate(my_val_dataloader):
        if n < 1:
            #print("STARTING EVAL CODE")
            images, class_labels, bbox_label = data
            bbox_label = ops.box_convert(bbox_label, in_fmt = 'xyxy', out_fmt = 'cxcywh')
            cls_prediction, bbox_prediction = test_net_ciou(images)
            bbox_prediction = ops.box_convert(bbox_prediction, in_fmt = 'cxcywh', out_fmt = 'xyxy')
            bbox_label = ops.box_convert(bbox_label, in_fmt = 'cxcywh', out_fmt = 'xyxy')
            _, predicted = torch.max(cls_prediction.data, 1) # this returns max (_) and max index (predicted)
#             print(class_labels)
#             print(bbox_label)
            #rescale bbox_labels:
            bbox_label = bbox_label * 256
            bbox_prediction = bbox_prediction * 256
#             print("bbox_label:", bbox_label)
#             print("bbox_pred:", bbox_prediction)
            

            for j, labels in enumerate(class_labels):
                if (labels == 0) and (count_bus < 3):
                    PIL_img = tvt.ToPILImage()(images[j])
                    np_img = np.uint8(PIL_img)
                    #reformat GT annotation
#                     bbox_label[j][2] = bbox_label[j][2] - bbox_label[j][0]
#                     bbox_label[j][3] = bbox_label[j][3] - bbox_label[j][1]
                    [x, y, x2, y2] = bbox_label[j]
                    [x_pred, y_pred, x2_pred, y2_pred] = bbox_prediction[j]

                    np_img = cv2.rectangle(np_img, (int(x), int(y)), 
                                        (int(x2), int(y2)), (0,255,0), 2)
                    np_img = cv2.rectangle(np_img, (int(x_pred), int(y_pred)), (int(x2_pred), 
                                        int(y2_pred)), (0, 0, 255), 2)
                    np_img = cv2.putText(np_img, "GT: bus", (int(x ), int(y + 25)), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255, 12),2)
                    np_img = cv2.putText(np_img, "prediction:" + mapping[int(predicted[j])], (int(x_pred), int(y_pred + 25)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    ax[0, count_bus].imshow(np_img)            
                    count_bus = count_bus + 1
                if (labels == 1) and (count_cat < 3):
                    PIL_img = tvt.ToPILImage()(images[j])
                    np_img = np.uint8(PIL_img)
                    #reformat GT annotation
#                     bbox_label[j][2] = bbox_label[j][2] - bbox_label[j][0]
#                     bbox_label[j][3] = bbox_label[j][3] - bbox_label[j][1]
                    [x, y, x2, y2] = bbox_label[j]
                    [x_pred, y_pred, x2_pred, y2_pred] = bbox_prediction[j]

                    np_img = cv2.rectangle(np_img, (int(x), int(y)), 
                                        (int(x2), int(y2)), (0,255,0), 2)
                    np_img = cv2.rectangle(np_img, (int(x_pred), int(y_pred)), (int(x2_pred), 
                                        int(y2_pred)), (0, 0, 255), 2)
                    np_img = cv2.putText(np_img, "GT: cat", (int(x ), int(y + 25)), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255, 12),2)
                    np_img = cv2.putText(np_img, "prediction:" + mapping[int(predicted[j])], (int(x_pred), int(y_pred + 25)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    ax[1, count_cat].imshow(np_img)            
                    count_cat = count_cat + 1
                if (labels == 2) and (count_pizza < 3):
                    PIL_img = tvt.ToPILImage()(images[j])
                    np_img = np.uint8(PIL_img)
                    #reformat GT annotation
#                     bbox_label[j][2] = bbox_label[j][2] - bbox_label[j][0]
#                     bbox_label[j][3] = bbox_label[j][3] - bbox_label[j][1]
                    [x, y, x2, y2] = bbox_label[j]
                    [x_pred, y_pred, x2_pred, y2_pred] = bbox_prediction[j]

                    np_img = cv2.rectangle(np_img, (int(x), int(y)), 
                                        (int(x2), int(y2)), (0,255,0), 2)
                    np_img = cv2.rectangle(np_img, (int(x_pred), int(y_pred)), (int( x2_pred), 
                                        int(y2_pred)), (0, 0, 255), 2)
                    np_img = cv2.putText(np_img, "GT: pizza", (int(x ), int(y + 25)), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255, 12),2)
                    
                    np_img = cv2.putText(np_img, "prediction:" + mapping[int(predicted[j])], (int(x_pred), int(y_pred + 25)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    ax[2, count_pizza].imshow(np_img)            
                    count_pizza = count_pizza + 1
                    
            
        if n > 1:
            break


# In[542]:


saved_loss_test_net_MSE = loss_test_net_MSE #these values were from one full epoch of training. 
saved_loss_test_net_CIOU = loss_test_net_ciou
plt.plot(saved_loss_test_net_MSE, label = "MSE")
plt.plot(saved_loss_test_net_CIOU, label = 'CIOU')
plt.title('Loss Curves for Different Loss Functions')
plt.legend()
plt.xlabel('Iterations * 10 (batch size = 20). 4 Epochs in Total')
plt.ylabel('Loss')


# In[ ]:




