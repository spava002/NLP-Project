import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import os

# Dataset file paths
train_file_path = "trainingData/english_dataset.tsv"
test_file_path = "trainingData/hasoc2019_en_test-2919.tsv"

# Hyperparameters
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUM_EPOCHS = 6

# Create a checkpoints directory if doesnt exsit already to hold model data 
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a new BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Create optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# If a checkpoint file exists, load that instead of starting from new
checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)]
if checkpoint_files:
    # Get the path of the checkpoint file
    print("Loading checkpoint...")
    checkpoint = torch.load('checkpoints/model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")
else:
    print("No checkpoint file found. Starting from anew.")

# Define dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['task_2']

        # Tokenize text and convert labels to integers
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        if label == "HATE":
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)

        return input_ids, attention_mask, label

# Reading TSV files
train_data = pd.read_csv(train_file_path, sep='\t')
test_data = pd.read_csv(test_file_path, sep='\t')

# Create new training/testing datasets and loaders
train_dataset = HateSpeechDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = HateSpeechDataset(test_data, tokenizer, MAX_SEQ_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Creating optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# **ONLY UNCOMMENT TO TRAIN MODEL**

# # Training of model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.train()
# for epoch in range(NUM_EPOCHS):
#     total_loss = 0
#     for input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
#         input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
    
#     # Save a model checkpoint after each epoch
#     checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
#     torch.save({
#         'epoch': epoch+1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': total_loss / len(train_loader),
#     }, checkpoint_path)
#     print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader)}")

# # Evaluation of model accurracy
# model.eval()
# total_correct = 0
# total_samples = 0
# with torch.no_grad():
#     for input_ids, attention_mask, labels in tqdm(test_loader, desc="Testing"):
#         input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         _, predicted = torch.max(outputs.logits, 1)
#         total_correct += (predicted == labels).sum().item()
#         total_samples += labels.size(0)

# accuracy = total_correct / total_samples
# print(f"Accuracy: {accuracy}")

def predict_hate_speech(text):
    # Tokenizing the received text
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)

    # Model makes a prediction based on input
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=1)

    # Retrieve the assigned label and confidence score of input text
    predicted_label_index = torch.argmax(logits, dim=1).item()
    predicted_label = ['NOT', 'HATE'][predicted_label_index]
    confidence_score = probabilities[0][predicted_label_index].item() * 100

    return predicted_label, confidence_score

# For testing predictions
input_text = "Hello, am I a form of hate speech?"
prediction, confidence = predict_hate_speech(input_text)
print(f"Prediction for hate speech:\n{input_text}\nIt is: {prediction} with{confidence: .2f}% confidence.")