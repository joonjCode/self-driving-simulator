from utils import *
from sklearn.model_selection import train_test_split

# Step 1
path = 'myData'
data = importDataInfo(path)

# Step 2
data = balanceData(data, display=False)

# Step 3
imagesPath, steering = loadData(path, data)
# print(imagesPath[0], steering[0])


# Step 4 : Split data
x_train, x_test, y_train, y_test = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Total Training images: ', len(x_train))
print('Total Test images: ', len(x_test))


# Step 5 : Augmentation
