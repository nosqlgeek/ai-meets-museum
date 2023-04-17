from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv('./imageClassification/test_input.csv')

# Split the dataset into features (X) and labels (y)
filename = data['filename']
label = data['label']

# Perform a stratified split
train_filename, test_filename, train_label, test_label = train_test_split(filename, label, test_size=0.2, stratify=label, random_state=25)

train = pd.DataFrame({'filename': train_filename, 'label': train_label})
test = pd.DataFrame({'filename': test_filename, 'label': test_label})

# Save the splited datasets
train.to_csv('./imageClassification/dataset/train.csv', index=False)
test.to_csv('./imageClassification/dataset/test.csv', index=False)