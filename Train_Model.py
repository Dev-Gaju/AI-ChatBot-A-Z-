import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from Model import ChatDataset, tags, X_train, NeuralNet, all_words


# Hyperparameter
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size,
                        shuffle =True, num_workers=2)

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch +1) % 100 == 0:
        # f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}'
        print("epoch {}/{}, loss={:.4f}.".format(epoch+1,num_epochs,loss.item()))

# print(f'Final loss, loss={loss.item():.4f}')
print("Final Loss, loss{:.4f}".format(loss.item()))

data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hiddent_size": hidden_size,
    "all_words":all_words,
    "tags": tags
}
FILE = "model.pth"
torch.save(data, FILE)

print("Traning complete. file saved to",FILE)