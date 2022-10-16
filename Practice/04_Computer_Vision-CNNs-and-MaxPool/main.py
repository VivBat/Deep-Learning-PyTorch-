import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


##  trying out nn.conv2d


random_tensor = torch.randn([1, 1, 3, 3])
random_tensor_0 = random_tensor[0]

print(f"Input to the conv2d: {random_tensor_0}")

model_test_convd2d = nn.Conv2d(in_channels=1,
                               out_channels=2,
                               kernel_size=2,
                               stride=1,
                               padding=0)



print(f"Output of a conv2d: {model_test_convd2d(random_tensor_0)}")

# print(model_test.state_dict())


##  trying out nn.MaxPool2d

model_test_maxPool = nn.MaxPool2d(kernel_size=2,
                                  stride=1)
print(f"Output of a max pool: {model_test_maxPool(random_tensor_0)}")



# A CNN model

class CNN_model_V01(nn.Module):
    """
    Replicates the TinyVGG architecture from (https://poloclub.github.io/cnn-explainer/)
    """
    def __init__(self, input_shape, output_shape, neurons):
        super().__init__()
        self.conv2d_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=neurons,
                      kernel_size=(3,3),
                      stride=1,
                      padding=0
                      ),
            nn.ReLU()
        )
        self.conv2d_block2 = nn.Sequential(
            nn.Conv2d(in_channels=neurons,
                      out_channels=neurons,
                      kernel_size=(3,3),
                      stride=1,
                      padding=0
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.classification_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=neurons*12*12,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv2d_block1(x)
        print(f"Output shape of conv2d_block1: {x.shape} ")
        x = self.conv2d_block2(x)
        print(f"Output shape of conv2d_block2: {x.shape} ")
        x = self.classification_layer(x)
        print(print(f"Output shape of classification_layer: {x.shape} "))
        return x


# a random dataset like tensor to play with the model
random_dataset_like_tensor = torch.randn([32, 3, 28, 28])

print(f"Shape of sample random tensor: {random_dataset_like_tensor.shape}")

first_image_like = random_dataset_like_tensor[0]

print(first_image_like.shape)

input_shape = first_image_like.shape[0]
NEURONS = 10
OUTPUT_SHAPE = 10

model = CNN_model_V01(input_shape=input_shape,
                      output_shape=OUTPUT_SHAPE,
                      neurons=NEURONS)

# try:
model(first_image_like)
# except RuntimeError as error:
#     print("passing image through the cnn didnt work...check what is wrong with it.")
#     print(f"-------------------{error}-------------------------------")

