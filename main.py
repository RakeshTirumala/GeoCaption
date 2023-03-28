import torch

print(f'PyTorch version: {torch.__version__}')
print('*' * 10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')

#%%
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# %%
from datasets import load_dataset
import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from torchvision.models import densenet201

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# LOADING THE DATASET
# print("Loading dataset...")
# rsicDataset = load_dataset("arampacha/rsicd")
# print(rsicDataset)
#
# %%
# ARRANGING THE DATA
# print("STARTING ARRANGEMENT...")
# dataImages = rsicDataset["train"]["image"] + rsicDataset["test"]["image"] + rsicDataset["valid"]["image"]
# dataCaptions = rsicDataset["train"]["captions"] + rsicDataset["test"]["captions"] + rsicDataset["valid"]["captions"]
# dataFilename = rsicDataset["train"]["filename"] + rsicDataset["test"]["filename"] + rsicDataset["valid"]["filename"]
# for idx in range(len(dataCaptions)):
#     dataCaptions[idx] = dataCaptions[idx][1]
# print("Images length: ", len(dataImages))
# print("Captions length: ", len(dataCaptions))
#

# #%%
# minL = 202
# for sml in dataCaptions:
#     if len(sml)<minL: minL=len(sml)
# print(minL)
#%%
import pickle
# pickle.dump(dataImages, open("dataImages.pkl", "wb"))
# pickle.dump(dataCaptions, open("dataCaptions.pkl", "wb"))
# pickle.dump(dataFilename, open("dataFilename.pkl", "wb"))

#%%
dataImages = pickle.load(open("dataImages.pkl", "rb"))
dataCaptions = pickle.load(open("dataCaptions.pkl", "rb"))
dataFilename = pickle.load(open("dataFilename.pkl", "rb"))
dataImages = dataImages[:3087]
dataCaptions = dataCaptions[:3087]
dataFilename = dataFilename[:3087]
print("Loaded data!!!")

# %%
from collections import Counter
import re

# text preprocessing
print("TEXT PREPROCESSING...")
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")


#
def preprocessingText(dataCaptions, maxLengthOfCaption):
    for index in range(len(dataCaptions)):
        caption = dataCaptions[index]
        caption = caption.lower()
        caption = "<start> " + caption + " <end>"
        dataCaptions[index] = caption
        maxLengthOfCaption = max(maxLengthOfCaption, len(caption))
    return dataCaptions, maxLengthOfCaption


def text_preprocessing(caption):
    tokenizedCaption = tokenizer(caption)
    return tokenizedCaption


def buildVocabulary(captionsList, minFreq=1):
    wordFreq = Counter()
    for caption in captionsList:
        tokenizedCaption = text_preprocessing(caption)
        wordFreq.update(tokenizedCaption)

    words = [w for w in wordFreq.keys()]
    wordMap = {k: v + 1 for v, k in enumerate(words)}
    wordMap['<unk'] = len(wordMap) + 1
    wordMap['<start>'] = len(wordMap) + 1
    wordMap['<end>'] = len(wordMap) + 1
    wordMap['<pad>'] = 0

    return wordMap


maxLengthOfCaption = 0
dataCaptions, maxLengthOfCaption = preprocessingText(dataCaptions, maxLengthOfCaption)
wordMap = buildVocabulary(dataCaptions, 5)
print("COMPLETED!!!")
print("dataCaptions[0]", dataCaptions[0])

# %%
# import pickle
#
# pickle.dump(wordMap, open('wordMapMin5.pkl', 'wb'))
# %%
import pickle

wordMap = pickle.load(open('wordMapMin5.pkl', 'rb'))
print("WordMap loaded!!!")

# %%
# print(wordMap['motorways'])

# %%
from torchvision.models import DenseNet201_Weights

numOfDataImages = len(dataImages)
splitIndex = round(0.70 * numOfDataImages)
trainingSetImages = dataImages[:splitIndex]
testingSetImages = dataImages[splitIndex:]
print("Done!!")
#%%
# densenet201Model = densenet201(weights=DenseNet201_Weights.DEFAULT)
# # densenet201Model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
# # densenet201Model.classifier = nn.Identity()
#
# densenet201Model = torch.nn.Sequential(*list(densenet201Model.children())[:-1])
# densenet201Model.eval()
# print("Done!!")

# %%
# Define a transform to preprocess the input images
# from PIL import JpegImagePlugin
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
#
# # Load and preprocess the input Images
# images = [transform(imgObj) for imgObj in dataImages]
# images = torch.stack(images)
# print("Preprocessing input images completed!!!")


# %%
# from tqdm import tqdm
# import pickle
#
# # Pass the input images through the model and extract the features
# allImagesFeatureMap = {}
# count = 0
# densenet201Model.to(device)
# with torch.no_grad():
#     for image in tqdm(images):
#         image = image.to(device)
#         feature = densenet201Model(image.unsqueeze(0))
#         feature = np.array(feature.squeeze().tolist())
#         allImagesFeatureMap[count] = feature
#         count += 1
#         if len(allImagesFeatureMap) == 1092:
#             pickle.dump(allImagesFeatureMap, open("P:/Folder Bulbasaur/MajorProject_RSIC/allImagesFeatureMap" + str(
#                 count // len(allImagesFeatureMap)) + ".pkl", "wb"))
#             allImagesFeatureMap = {}
#         else:
#             continue
#
# print("Completed!!!->Saved Features!!!")

# %%
import pickle

# allImagesFeatureMap = pickle.load(open("allImagesFeatureMap.pkl", "rb"))
# print("Loaded!!!")
fetchedFeaturesMap = {}
for mc in range(1, 4):
    with open("P:/Folder Bulbasaur/MajorProject_RSIC/allImagesFeatureMap" + str(mc) + ".pkl", "rb") as f:
        features = pickle.load(f)
        fetchedFeaturesMap.update(features)


print("Completed fetching maps...")

# %%
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import re


class CustomDataset(Dataset):
    def __init__(self, X, y, tokenizer, dataFilename, vocab_size, max_length, features, wordMap):
        self.X = X.copy()
        self.y = y.copy()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.dataFilename = dataFilename
        self.wordMap = wordMap

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # image = self.X[index]
        key = index
        feature = torch.tensor(self.features[key])

        caption = self.y[index]
        print("Caption:", caption)
        words = re.findall(r"[\w']+|[.,!?;]", caption[7:-5])
        words.insert(0, '<start>')
        words.insert(len(words), '<end>')
        encoded = []
        for word in words:
            try:
                encoded.append(wordMap[word])
            except:
                encoded.append(wordMap['<unk'])

        rl = self.max_length - len(encoded)
        # print("rl:", rl)
        for _ in range(rl):
            encoded.append(0)
        input_ids = encoded
        # print("input_ids:", input_ids)
        # attention_mask = encoded['attention_mask'][0]
        one_hot_encoded = one_hot(torch.tensor(input_ids), num_classes=self.vocab_size+1)
        # print("one_hot_encoded:", one_hot_encoded[0])
        in_seq = torch.tensor(input_ids)
        out_seq = one_hot_encoded

        X1 = feature
        X2 = in_seq
        y = out_seq

        print(np.array(X1).shape, np.array(X2).shape, np.array(y).shape)

        return (X1, X2), y


class CustomDataGenerator(DataLoader):
    def __init__(self, X, y, batch_size, tokenizer, dataFilename, vocab_size, max_length, features, wordMap,
                 shuffle=True):
        dataset = CustomDataset(X, y, tokenizer, dataFilename, vocab_size, max_length, features, wordMap)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def __len__(self):
        print("len of dataset:", len(self.dataset))
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        for batch in super().__iter__():
            (X1, X2), y = batch
            yield (X1, X2), y

    # def collate_fn(self, batch):
    #     X1, X2 = zip(*[x[0] for x in batch])
    #     y = [x[1] for x in batch]
    #     X2 = pad_sequence(X2, batch_first=True, padding_value=maxLengthOfCaption)
    #     # return (torch.stack(X1), X2), torch.tensor(y).view(-1, 1).float()
    #     return (torch.stack(X1), X2), torch.tensor(y).view(-1, 1)

print("Completed!!!-Generator-")
# %%
from transformers import AutoTokenizer

train_generator = CustomDataGenerator(X=trainingSetImages,
                                      y=dataCaptions[:splitIndex],
                                      batch_size=32,
                                      tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'),
                                      dataFilename=dataFilename[:splitIndex],
                                      vocab_size=len(wordMap),
                                      max_length=maxLengthOfCaption,
                                      features=fetchedFeaturesMap,
                                      wordMap=wordMap)

validation_generator = CustomDataGenerator(X=testingSetImages,
                                           y=dataCaptions[splitIndex:],
                                           batch_size=32,
                                           tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'),
                                           dataFilename=dataFilename[splitIndex:],
                                           vocab_size=len(wordMap),
                                           max_length=maxLengthOfCaption,
                                           features=fetchedFeaturesMap,
                                           wordMap=wordMap)

print("Completed!!!--Loading from generator---")

# %%

import torch
import torch.nn as nn
import torch.optim as optim


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, max_length, batch_size):
        super(CaptionModel, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.img_features = nn.Sequential(nn.Flatten(),
                                          nn.Linear(1920*7*7, 1024),
                                          nn.ReLU())
        self.embedding = nn.Embedding(self.vocab_size+1, 1024)
        # self.sentence_features = nn.Sequential(
        #                             nn.Linear(self.max_length, self.vocab_size+1))

        self.gru = nn.GRU(input_size=1209,hidden_size=1024, num_layers=4,
                          bias=True, batch_first=False, dropout=0.5, bidirectional=False)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, self.max_length)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.max_length, self.vocab_size+1)

    def forward(self, input1, input2):

        img_features = self.img_features(input1)
        # x = self.embedding(input2)
        # x = x.view(x.size(0), -1)
        # sentence_features = self.sentence_features(input2)
        print("loaded img_features")
        sentence_features = self.embedding(input2)
        print("Loaded sentence_features")
        sentence_features = torch.argmax(sentence_features, dim=-1)

        print("ImgF:", img_features.shape, "SF:", sentence_features.shape)
        merged = torch.cat((img_features, sentence_features), dim=1)
        print("concating completed")
        sentence_features, _ = self.gru(merged)
        print("GRU layer completed!")
        print("SF:", sentence_features.to('cpu').detach().numpy().shape)
        x = self.dropout1(sentence_features)
        print("Dropout1")
        print("x shape:", x.shape, "img_features shape:", img_features.shape)

        x = torch.add(x, img_features)
        print("Completed torch.add")
        x = self.fc1(x)
        print("Completed fc1")
        x = nn.ReLU()(x)
        print("Completed Relu")
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # print("Completed fc2")
        x = nn.Softmax(dim=1)(x)
        print("Completed softmax")

        return x


vocab_size = len(wordMap)
caption_model = CaptionModel(vocab_size, maxLengthOfCaption, batch_size=32).to(device=device)

model_optimizer = optim.Adam(caption_model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
criterion = nn.CrossEntropyLoss().to(device=device)


# Print the summary of the model
print(caption_model)
print("Done!!!--Caption model summary--")
#%%
# from torch_lr_finder import LRFinder
# lr_finder = LRFinder(caption_model, model_optimizer, criterion, device="cuda")
# lr_finder.range_test(train_generator, end_lr=1, num_iter=100)
# lr_finder.plot()


# %%
# Train the model
#
num_epochs = 1
train_lossL = []
valid_lossL = []
train_accL = []
valid_accL = []
print("Training...")
for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_acc= []
    print("epoch:",epoch)
    caption_model.train()
    print("TrainGene:")
    for batch_idx, ((X1, X2), y) in enumerate(train_generator):
        print("Inside training...")
        print("Batch index:", batch_idx, "train_generator len:", len(train_generator))
        model_optimizer.zero_grad()
        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        # print("X1:",  X1.to('cpu').detach().numpy().shape)
        # print("X2:",  X2.to('cpu').detach().numpy().shape)
        # print("y:",   y.to('cpu').detach().numpy().shape)
        outputs = caption_model(X1.to(torch.float32), X2.to(torch.long))
        print(outputs)
        print("Loss...")

        y = torch.argmax(y, dim=-1).to(torch.float32)
        # y = torch.squeeze(y, dim=1)

        try:
            print("y:", y.shape, "output:", outputs.to('cpu').detach().numpy().shape)
        except:
            print("Error!")

        loss = criterion(outputs, y)

        print("got the loss!")
        loss.backward()
        print("<--backword")
        model_optimizer.step()
        train_loss += loss.item()

        # Calculate training accuracy
        predicted_labels = torch.argmax(outputs)
        train_correct = torch.sum(predicted_labels == y)
        print("train_correct:", train_correct)
        # accuracy = train_correct / y.size(0)
        # # train_acc += accuracy.item()
        # print("Accuracy:", accuracy.item())

    train_accuracy = train_correct / train_total
    train_accL.append(train_accuracy)
    # print(train_accL)

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(X1), len(train_generator.dataset),
               100. * batch_idx / len(train_generator), loss.item()))
    train_loss /= len(train_generator)
    train_lossL.append(train_loss)
    print("Train Loss len:", len(train_lossL))

    # Evaluate the model
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_acc = []

    caption_model.eval()
    with torch.no_grad():
        print("TestGene:")
        for batch_idx, ((X1, X2), y) in enumerate(validation_generator):
            print("Inside validation")
            print("Batch index:", batch_idx)
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)

            outputs = caption_model(X1.to(torch.float32), X2.to(torch.long))
            print("Loss...")

            y = torch.argmax(y, dim=-1).to(torch.float32)
            print(y.shape)

            loss = criterion(outputs, y)
            val_loss += loss.item()

            # Calculate validation accuracy

            predicted_labels = torch.argmax(outputs)
            val_correct = torch.sum(predicted_labels == y)
            print("val_correct:", val_correct)
            accuracy = val_correct / y.size(0)
            # val_acc += accuracy.item()
            print("vAccuracy:", accuracy.item())

    # val_accuracy = val_correct / val_total
            valid_accL.append(accuracy.item())
    # print(valid_accL)

    val_loss /= len(validation_generator)
    valid_lossL.append(val_loss)
    print("Valid Loss len:", len(valid_lossL))

    print('Epoch: {}\tTraining Loss: {:.2f}\tValidation Loss: {:.2f}\tTraining Accuracy:{:.2f} \tValidation Accuracy:{:.2f}'.format(
        epoch, train_loss, val_loss, train_accL[-1], valid_accL[-1]))

    print("Saving the model and other stats")
    torch.save(caption_model.state_dict(), 'ImageCaptioningModel.pth')

    pickle.dump(train_accL,  open("train_accuracy.pkl", "wb"))
    pickle.dump(valid_accL,  open("valid_accuracy.pkl", "wb"))
    pickle.dump(train_lossL, open("train_loss.pkl", "wb"))
    pickle.dump(valid_lossL, open("valid_loss.pkl", "wb"))

    print("moving to the next epoch...")

print("Model training completed!!!")
#%%
# torch.save(caption_model.state_dict(), 'ImageCaptioningModel.pth')
#
# pickle.dump(train_accL,  open("train_accuracy.pkl", "wb"))
# pickle.dump(valid_accL,  open("valid_accuracy.pkl", "wb"))
# pickle.dump(train_lossL, open("train_loss.pkl", "wb"))
# pickle.dump(valid_lossL, open("valid_loss.pkl", "wb"))
#
# print("Saved the model and other stats...")

#%%
# saved_model = CaptionModel(vocab_size, maxLengthOfCaption, batch_size=8)
# saved_model.load_state_dict(torch.load('ImageCaptioningModel.pth'))

#%%
import matplotlib.pyplot as plt
import pickle
import torch
# training_accuracy = pickle.load(open("train_accuracy.pkl", "rb"))
# valid_accuracy = pickle.load(open("valid_accuracy.pkl", "rb"))
# print(training_accuracy)
training_loss = pickle.load(open("train_loss.pkl", "rb"))
