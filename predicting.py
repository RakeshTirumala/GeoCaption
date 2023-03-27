import numpy as np
import torch
import torch.nn as nn
import pickle
from torchvision import transforms
from PIL import Image
from torchvision.models import DenseNet201_Weights
from torchvision.models import densenet201
from torch.nn.functional import one_hot

class CaptionModel(nn.Module):
    def __init__(self, vocab_size, max_length, batch_size):
        super(CaptionModel, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.img_features = nn.Sequential(nn.Flatten(),
                                          nn.Linear(1920*7*7, 256),
                                          nn.ReLU())
        self.embedding = nn.Embedding(self.vocab_size+1, 256)
        # self.sentence_features = nn.Sequential(
        #                             nn.Linear(self.max_length, self.vocab_size+1))

        self.gru = nn.GRU(input_size=256,hidden_size=256, num_layers=10,
                          bias=True, batch_first=False, dropout=0.5, bidirectional=False)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256,256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256,202)

    def forward(self, input1, input2):

        img_features = self.img_features(input1)
        # x = self.embedding(input2)
        # x = x.view(x.size(0), -1)
        # sentence_features = self.sentence_features(input2)
        sentence_features = self.embedding(input2)
        print("ImgF:", img_features.shape, "SF:", sentence_features.shape)
        merged = torch.cat((img_features.unsqueeze(1), sentence_features), dim=1)
        sentence_features, _ = self.gru(merged)

        x = self.dropout1(sentence_features)
        img_features = self.img_features(input1).unsqueeze(1).expand(-1, x.size(1), -1)
        print("x shape:", x.shape, "img_features shape:", img_features.shape)

        x = torch.add(x, img_features)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)

        return x

def texts_to_sequences(seq, wordMap):
    encoded = []
    for word in seq:
        try:
            encoded.append(wordMap[word])
        except:
            encoded.append(wordMap['<unk'])

    rl = 202 - len(encoded)
    return encoded, rl
def pad_sequences(seq, rl):
    for _ in range(rl):
        seq.append(0)
    input_ids = seq
    one_hot_encoded = one_hot(torch.tensor(input_ids), num_classes=vocab_size + 1)
    out_seq = one_hot_encoded[0].unsqueeze(0)
    return out_seq

def idx_to_word(y_pred, wordMap):
    for word, val in wordMap.items():
        if val == y_pred:
            return word
        else:
            continue
    return None

wordMap = pickle.load(open('wordMapMin5.pkl', 'rb'))
print("WordMap loaded!!!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

vocab_size = len(wordMap)
saved_model = CaptionModel(vocab_size,202, 1)
saved_model.load_state_dict(torch.load('ImageCaptioningModel.pth'))

# set the model to evaluation mode
saved_model.eval()

# create a new DataLoader for your test image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_image_path = "P:/Folder Bulbasaur/MAJOR_PROJECT/NWPU-RESISC45/NWPU-RESISC45/airplane/airplane_016.jpg"

image = Image.open(test_image_path)
image = transform(image)
image = image

densenet201Model = densenet201(weights=DenseNet201_Weights.DEFAULT)
densenet201Model = torch.nn.Sequential(*list(densenet201Model.children())[:-1])
densenet201Model.eval()

feature = densenet201Model(image.unsqueeze(0))
feature = torch.tensor(np.array(feature.tolist()))
print(np.array(feature).shape)

in_text = "<start>"
for i in range(202):
    sequence, rl = texts_to_sequences([in_text], wordMap)
    sequence = pad_sequences(sequence, rl)
    y_pred = saved_model(feature.to(torch.float32), sequence.to(torch.long))
    # print(y_pred)
    y_pred = y_pred.detach().numpy()

    # word = idx_to_word(y_pred, wordMap)

#     if word==None:
#         break
#
#     in_text += " " + word
#     print(in_text)
#
#     if word == "<end>":
#         break
#
# print(in_text)

# _, predicted = torch.max(output.data, 1)
# print("Prediction:", predicted)

