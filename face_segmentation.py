import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms
from torchsummary import summary

import numpy as np

import matplotlib.pyplot as plt
import cv2

cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")

print('torch version {}\ntorchvision version {}'.format(torch.__version__, torchvision.__version__))
print('CUDA available? {} Device is {}'.format(cuda_available, device))
if cuda_available:
    print('Number of available CUDA devices {}\n'.format(torch.cuda.device_count()))

# ======================== hyper-params ================================================================================

"""This section is for hyper-param definition"""

# Path to where the dataset is stored locally
data_sets_folder = '../../data-sets/face_segmentation/V2/'

# We create a new experiment folder with a unique date and time identifier
# It will contain all the generated segmentation images
date_time_now = datetime.now().strftime("%d-%m-%Y_%H%M")
save_img_folder = '../../generated_face_seg/experiment_{}/'.format(date_time_now)
if not os.path.exists(save_img_folder):
    os.makedirs(save_img_folder)

batch_size = 4
validation_batch_size = 1

learning_rate = 1e-2
validation_split = .1

max_training_epochs = 100
img_resize_factor = 256

random_seed = 42
shuffle_dataset = True

training_losses = []
validation_losses = []

# Predefined set of color coding's for transformation from RGB to class name (hair,skin,etc)
red_background = np.array([255, 0, 0])
yellow_face = np.array([255, 255, 0])
brown_hair = np.array([127, 0, 0])
cayn_nose = np.array([0, 255, 255])
blue_eyes = np.array([0, 0, 255])
green_mouth = np.array([0, 255, 0])


# ======================== Util functions ==============================================================================


def plot_losses():
    plt.plot(training_losses, label='training loss')
    plt.plot(validation_losses, label='validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('loss value')

    plt.savefig(save_img_folder + "train_validation_loss_graphs" + '.png')
    plt.close()


def display_images(imgs_arr, plot_name):
    """This method receives an array of images, creates a figure containing all
        and saves the figure to the experiment folder as plot_name"""
    labels = ['img', 'gt_segm', 'pred_segm']
    fig, axarr = plt.subplots(1, len(imgs_arr))
    for i, img in enumerate(imgs_arr):
        axarr[i].imshow(imgs_arr[i])
        axarr[i].set_title(labels[i])
        axarr[i].axis('off')
    fig.tight_layout(pad=1.0)
    fig.suptitle(plot_name, fontsize=10)
    plt.savefig(save_img_folder + plot_name + '.png')
    plt.close(fig)


def visualize_training(epoch):
    """This method is called after each training epoch to visualize model current state
        loads images from the validation set, uses the model to predict segmentation,
        and create and stores the ground-truth and predicted segmentation as a figure"""

    # face_img_arr = []
    # segm_label_arr = []
    # pred_segmented_img_arr = []
    # for i in range(3):

    # Get an image and segmentation label from the validation set
    face_img, segm_label = next(iter(validation_loader))

    # Get predicated segmentation from the model, and convert it to rgb for visualization
    face_img = face_img.to(device)

    model.eval()
    pred_segmented_img = model(face_img)
    model.train()

    pred_segmented_img = torch.squeeze(pred_segmented_img)
    predicted_class = np.argmax(pred_segmented_img.detach().cpu().numpy(), axis=0)
    pred_segmented_img = convert_seg_one_hot_to_rgb(predicted_class)

    # Convert face image back to numpy and normalzie to range [0 255]
    face_img = (torch.squeeze(face_img)).permute(1, 2, 0).cpu().numpy()
    face_img = (255 * (face_img - np.min(face_img)) / np.ptp(face_img)).astype(int)

    # Convert segmentation label to contain one channel with class indices for each pixel
    segm_label = torch.squeeze(segm_label)
    segm_label_class = np.argmax(segm_label.detach().cpu().numpy(), axis=0)
    segm_label = convert_seg_one_hot_to_rgb(segm_label_class)

    # face_img_arr.append(face_img)
    # segm_label_arr.append(segm_label)
    # pred_segmented_img_arr.append(pred_segmented_img)

    display_images(imgs_arr=[face_img, segm_label, pred_segmented_img], plot_name="Training_epoch_{}".format(epoch))
    # display_images(imgs_arr=[face_img_arr, segm_label_arr, pred_segmented_img_arr], plot_name="Training_epoch_{}".format(epoch))


def convert_segm_rgb_to_one_hot(segmentation_label):
    """This method is used to convert a segmentation images of 3 channels (RGB)
    to a 6 channels (segmentation classes) as a one-hot encoding for each class (color)
    input is an image of shape WxHx3 out put os WxHx6"""
    red_background_out = (segmentation_label == red_background).all(2)
    yellow_face_out = (segmentation_label == yellow_face).all(2)
    brown_hair_out = (segmentation_label == brown_hair).all(2)
    cayn_nose_out = (segmentation_label == cayn_nose).all(2)
    blue_eyes_out = (segmentation_label == blue_eyes).all(2)
    green_mouth_out = (segmentation_label == green_mouth).all(2)

    one_hot_segment_label = np.stack(
        [red_background_out,
         yellow_face_out, brown_hair_out, cayn_nose_out, blue_eyes_out, green_mouth_out],
        axis=2, out=None)

    return one_hot_segment_label.astype(int)


def convert_seg_one_hot_to_rgb(segm_label_one_hot):
    """This method converts the class encoding representation of a segmentation image back to RGB format for display"""
    segmentation_label_rgb = np.zeros(shape=(img_resize_factor, img_resize_factor, 3), dtype=int)

    # 6 Classes are: background-red-0, skin-yellow-1, hair-brown-2, nose-cayn-3, eyes-blue-4, mouth-green-5
    red_background_indexes = np.where(segm_label_one_hot[:, :] == 0)
    yellow_face_indexes = np.where(segm_label_one_hot[:, :] == 1)
    brown_hair_indexes = np.where(segm_label_one_hot[:, :] == 2)
    cayn_nose_indexes = np.where(segm_label_one_hot[:, :] == 3)
    blue_eyes_indexes = np.where(segm_label_one_hot[:, :] == 4)
    green_mouth_indexes = np.where(segm_label_one_hot[:, :] == 5)

    segmentation_label_rgb[red_background_indexes[0], red_background_indexes[1]] = red_background
    segmentation_label_rgb[yellow_face_indexes[0], yellow_face_indexes[1]] = yellow_face
    segmentation_label_rgb[brown_hair_indexes[0], brown_hair_indexes[1]] = brown_hair
    segmentation_label_rgb[cayn_nose_indexes[0], cayn_nose_indexes[1]] = cayn_nose
    segmentation_label_rgb[blue_eyes_indexes[0], blue_eyes_indexes[1]] = blue_eyes
    segmentation_label_rgb[green_mouth_indexes[0], green_mouth_indexes[1]] = green_mouth

    return segmentation_label_rgb


# ======================== Custom Dataset Definition ===================================================================


class FaceSegmentationDataset(Dataset):
    """Face part segmentation dataset"""

    def __init__(self, data_dir, label_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the facial images.
            label_dir (string) : Directory with all the face segmentation images (the label)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.data_dir)[idx]

        face_img = cv2.imread(os.path.join(self.data_dir, img_name))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

        segmentation_label = cv2.imread(os.path.join(self.label_dir, img_name))
        segmentation_label = cv2.cvtColor(segmentation_label, cv2.COLOR_RGB2BGR)
        segmentation_label = cv2.resize(segmentation_label, dsize=(img_resize_factor, img_resize_factor),
                                        interpolation=cv2.INTER_CUBIC)
        segmentation_label = convert_segm_rgb_to_one_hot(segmentation_label)

        if self.transform:
            face_img = self.transform(face_img)
            segmentation_label = torch.from_numpy(segmentation_label).permute(2, 0, 1)

        return face_img, segmentation_label


# ========================= Model Architecture =========================================================================

class FaceSegmentationModel(nn.Module):
    def __init__(self):
        super(FaceSegmentationModel, self).__init__()

        # Encoder

        # Two high-res conv
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=7, stride=1, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=7, stride=1, padding=3)
        self.batch_norm2 = nn.BatchNorm2d(12)

        # then pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Two medium-res conv
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.batch_norm3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.batch_norm4 = nn.BatchNorm2d(48)
        # Then pooling again

        # Two low-res convs
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(192)

        self.conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(192)

        # Two more
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.batch_norm7 = nn.BatchNorm2d(768)

        self.conv8 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.batch_norm8 = nn.BatchNorm2d(768)

        self.dropout = nn.Dropout()

        # Decoder
        self.upsample1 = nn.ConvTranspose2d(in_channels=768, out_channels=192, kernel_size=2, stride=2, padding=0)
        self.batch_norm_up1 = nn.BatchNorm2d(192)

        self.upsample2 = nn.ConvTranspose2d(in_channels=384, out_channels=48, kernel_size=2, stride=2, padding=0)
        self.batch_norm_up2 = nn.BatchNorm2d(48)

        self.upsample3 = nn.ConvTranspose2d(in_channels=96, out_channels=12, kernel_size=2, stride=2, padding=0)
        self.batch_norm_up3 = nn.BatchNorm2d(12)

        self.upsample4 = nn.ConvTranspose2d(in_channels=24, out_channels=6, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = F.relu(x)

        # x = self.conv2(x)
        # x = self.batch_norm2(x)
        # x = self.pool(x)
        # x = F.relu(x)

        x1 = x

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.pool(x)
        x = F.relu(x)

        x = self.dropout(x)

        # x = self.conv4(x)
        # x = self.batch_norm4(x)
        # x = self.pool(x)
        # x = F.relu(x)

        x2 = x

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.pool(x)
        x = F.relu(x)

        # x = self.conv6(x)
        # x = self.batch_norm6(x)
        # x = self.pool(x)
        # x = F.relu(x)

        x3 = x

        x = self.conv7(x)
        x = self.batch_norm7(x)
        x = self.pool(x)
        x = F.relu(x)

        # x = self.conv8(x)
        # x = self.batch_norm8(x)
        # x = self.pool(x)
        x = self.dropout(x)
        # x = F.relu(x)


        # Decoder
        x = self.upsample1(x)
        x = self.batch_norm_up1(x)
        x = F.leaky_relu(x)

        x = torch.cat((x, x3), dim=1)

        x = self.upsample2(x)
        x = self.batch_norm_up2(x)
        x = F.leaky_relu(x)

        x = torch.cat((x, x2), dim=1)

        x = self.upsample3(x)
        x = self.batch_norm_up3(x)
        x = F.leaky_relu(x)

        x = torch.cat((x, x1), dim=1)
        x = self.upsample4(x)
        return x


class FaceSegmentationModel_submitted(nn.Module):
    def __init__(self):
        super(FaceSegmentationModel_submitted, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.upsample1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.batch_norm_up1 = nn.BatchNorm2d(32)
        self.upsample2 = nn.ConvTranspose2d(in_channels=32, out_channels=6, kernel_size=6, stride=2, padding=0)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Decoder
        x = self.upsample1(x)
        x = self.batch_norm_up1(x)
        x = F.leaky_relu(x)

        x = self.upsample2(x)
        return x

class FaceSegmentationModel_no_down_sample(nn.Module):
    def __init__(self):
        super(FaceSegmentationModel_no_down_sample, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1,
                               padding=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1,
                               padding=2)
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1,
                               padding=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1,
                               padding=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1,
                               padding=2)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=5, stride=1,
                               padding=2)

        # self.upsample1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0)
        # self.batch_norm_up1 = nn.BatchNorm2d(32)
        #
        # self.upsample2 = nn.ConvTranspose2d(in_channels=32, out_channels=6, kernel_size=6, stride=2, padding=0)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        # x = self.batch_norm1(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv2(x)
        # x = self.batch_norm2(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv3(x)
        # x = self.batch_norm2(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv4(x)
        # x = self.batch_norm2(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv5(x)
        # x = self.batch_norm2(x)
        x = F.relu(x)
        # x = self.pool(x)

        x = self.conv6(x)
        # x = self.batch_norm2(x)
        x = F.relu(x)
        # x = self.pool(x)
        return x


class FaceSegmentationModel_deep(nn.Module):
    def __init__(self):
        super(FaceSegmentationModel_deep, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1,
                               padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2,
                               padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2,
                               padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2,
                               padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)

        self.upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=6, stride=2, padding=0)
        # self.batch_norm_up1 = nn.BatchNorm2d(46)
        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=6, stride=2, padding=0)
        self.upsample3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2, padding=0)
        self.upsample4 = nn.ConvTranspose2d(in_channels=32, out_channels=6, kernel_size=6, stride=2, padding=0)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        # Decoder
        x = self.upsample1(x)
        x = F.leaky_relu(x)

        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        # x_up = torch.cat((x_up, x), dim=1)
        # x_up = self.upsample3(x_up)
        # x_up = self.conv_end(x_up)
        # x = self.upsample3(x)
        return x


class FaceSegmentationModel_with_skip(nn.Module):
    def __init__(self):
        super(FaceSegmentationModel_with_skip, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1,
                               padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=2,
                               padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1,
                               padding=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1,
                               padding=1)

        self.upsample1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # self.batch_norm_up1 = nn.BatchNorm2d(46)

        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=6, kernel_size=10, stride=2, padding=0)

        self.upsample3 = nn.ConvTranspose2d(in_channels=38, out_channels=6, kernel_size=10, stride=2, padding=0)

        # self.conv_end = nn.Conv2d(in_channels=38, out_channels=6, kernel_size=5, stride=1,padding=1)

        # self.upsample3 = nn.ConvTranspose2d(in_channels=32, out_channels=6, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = self.pool(x)

        x2 = F.relu(self.conv3(x1))
        x2 = F.relu(self.conv4(x2))
        x3 = self.pool(x2)

        # Decoder
        x_up = self.upsample1(x3)
        x_up = torch.cat((x_up, x2), dim=1)
        # x = self.batch_norm_up1(x)
        x_up = F.leaky_relu(x_up)

        x_up = self.upsample2(x_up)
        x_up = torch.cat((x_up, x), dim=1)
        x_up = self.upsample3(x_up)

        # x_up = self.conv_end(x_up)
        # x = self.upsample3(x)
        x_up = F.sigmoid(x_up)
        return x_up


# ======================== Analyze Dataset Section =====================================================================

""" IMPORTANT !!! This section was ran only once before we apply the transofrm to detrmine 
which value to use in the transform """


def analyze_dataset():
    transform_init = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(size=(img_resize_factor, img_resize_factor)),
         transforms.ToTensor(),
         ])

    face_seg_dataset_init = FaceSegmentationDataset(data_dir=data_sets_folder + 'Train_RGB/',
                                                    label_dir=data_sets_folder + 'Train_Labels',
                                                    transform=transform_init)

    # designed to explore the images scale is we are dealing with a variable image size dataset
    # to try and best determine what transformation we need to apply to our data
    avg_height = avg_width = 0
    smallest_height = smallest_width = np.inf
    for img, label in face_seg_dataset_init:
        img_height = img.shape[0]
        img_width = img.shape[1]
        avg_height += img_height
        avg_width += img_width

        if img_height < smallest_height: smallest_height = img_height
        if img_width < smallest_width: smallest_width = img_width

    train_size = len(face_seg_dataset_init)
    print("Image avarge height %d and avarge width %d" % (avg_height / train_size, avg_width / train_size))

    # Calculate the images population mean and std for better transform
    dataloader = torch.utils.data.DataLoader(face_seg_dataset_init, batch_size=100, shuffle=False, num_workers=4)
    face_imgs, segm_label = next(iter(dataloader))
    numpy_image = face_imgs.numpy()
    pop_mean = np.mean(numpy_image, axis=(0, 2, 3))
    pop_std0 = np.std(numpy_image, axis=(0, 2, 3))
    print("Images population mean values are {}  std values are {}".format(pop_mean, pop_std0))


# ====================== Load Dataset Section ==========================================================================

# Transform loaded images, resize them to a symmetrical form and then normalize their pixel values
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(size=(img_resize_factor, img_resize_factor)),
     transforms.ToTensor(),
     transforms.Normalize([0.4283, 0.335, 0.275], [0.240, 0.219, 0.216])
     # Mean and STD Values calc by Analyze_dataset()
     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #image net values
     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # default values
     ])

face_seg_dataset = FaceSegmentationDataset(data_dir=data_sets_folder + 'Train_RGB/',
                                           label_dir=data_sets_folder + 'Train_Labels', transform=transform)

# Creating data indices and shuffle them, then creates raining and validation splits:
indices = list(range(len(face_seg_dataset)))
split = int(np.floor(validation_split * len(face_seg_dataset)))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Define data random samplers for train and validation
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# Define dataloaders for both train and validation that will create batches of images using the random samplers
train_loader = DataLoader(face_seg_dataset, batch_size=batch_size,
                          sampler=train_sampler)

validation_loader = DataLoader(face_seg_dataset, batch_size=validation_batch_size,
                               sampler=valid_sampler)


# ========================= Model Training and Validation section ======================================================


def validate_model(epoch):
    """This method is for model validation, runs over all validation set images
    and calculate the avg segmentation classification loss
    after that is visualize one sample from the validation set to visually reflect training progress"""

    val_loss = 0.0
    for batch_idx, (face_imgs, segm_labels) in enumerate(validation_loader):
        # if cuda avilable
        face_imgs = face_imgs.to(device)
        segm_labels = segm_labels.to(device)

        model.eval()
        outputs = model(face_imgs)
        model.train()
        segm_labels = torch.argmax(segm_labels, dim=1)
        _loss = criterion(outputs, segm_labels)
        val_loss += _loss.item() * validation_batch_size

    avg_valid_loss = val_loss / len(valid_sampler)
    print("Validation after Epoch {} avg loss is {}".format(epoch, avg_valid_loss))
    validation_losses.append(avg_valid_loss)
    visualize_training(epoch)


def train_model():
    """This method is for model training, loads randomly created batchs of images
    predict per-pixel classification and calculate loss
    it omits statistics at the end of each training epoch"""

    for epoch in range(max_training_epochs):
        epoch_loss_sum = 0.0
        for batch_idx, (face_imgs, segm_labels) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # if cuda_available:
            face_imgs = face_imgs.to(device)
            segm_labels = segm_labels.to(device)

            # forward + backward + optimize
            outputs = model(face_imgs)
            labels = torch.argmax(segm_labels, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * batch_size

        avg_epoch_loss = epoch_loss_sum / len(train_sampler)
        print("Epoch {} avg loss is {}".format(epoch, avg_epoch_loss))
        training_losses.append(avg_epoch_loss)
        validate_model(epoch)
    plot_losses()


# ========================= Main Section ===============================================================================

# Create model
model = FaceSegmentationModel().to(device)
# print(model)
summary(model, input_size=(3, img_resize_factor, img_resize_factor))

# Loss definition
criterion = torch.nn.CrossEntropyLoss().to(device)

# Optimization
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# analyze_dataset() # this was called in the beginning of development to asses the dataset
train_model()

# visalize_final_model_validation()