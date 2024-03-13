import torch
from torch.functional import Tensor
import torch.nn as nn

""" This script defines the network.
"""

# If you are using PyTorch 2.1 and have extra time to play around, you could try to use the PyTorch JIT feature to compile the model by uncommenting the torch.compile decorator. :)
# Please refer to https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html for more details.
# @torch.compile
class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        # define conv1
        self.start_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters*4, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        ### YOUR CODE HERE
        super(batch_norm_relu_layer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        # self.relu = nn.functional.relu()
        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        out = nn.functional.relu(self.bn1(inputs))
        return out
        ### YOUR CODE HERE

class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        self.conv1 = nn.Conv2d(first_num_filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        # self.relu = nn.functional.relu()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.shortcut = nn.Sequential()

        ### YOUR CODE HERE
        if projection_shortcut is not None:
            self.conv1 = nn.Conv2d(first_num_filters, filters, kernel_size=3, stride=strides, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(filters)
            # self.relu = nn.functional.relu()
            self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(filters)
            self.shortcut = nn.Sequential(
                nn.Conv2d(first_num_filters, filters ,kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(filters)
            )
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        out = nn.functional.relu(self.bn1(self.conv1(inputs)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = out + self.shortcut(inputs)
        return out
        ### YOUR CODE HERE6

class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv1 = nn.Conv2d(filters, (int)(filters / 4), kernel_size=1, stride=1, bias=False)
        # self.relu = nn.functional.relu()
        self.bn2 = nn.BatchNorm2d(int(filters/4))
        self.conv2 = nn.Conv2d((int)(filters / 4), (int)(filters / 4), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(filters/4))
        self.conv3 = nn.Conv2d((int)(filters / 4), filters, kernel_size=1, stride=1, bias=False)
        self.shortcut = nn.Sequential()
        ### YOUR CODE HERE

        if projection_shortcut is not None:
            self.bn1 = nn.BatchNorm2d(first_num_filters)

            
            ### YOUR CODE HERE
        
            ### YOUR CODE HERE

            self.conv1 = nn.Conv2d(first_num_filters, (int)(filters / 4), kernel_size=1, stride=strides, bias=False)
            # self.relu = nn.functional.relu()
            self.bn2 = nn.BatchNorm2d(int(filters/4))
            self.conv2 = nn.Conv2d((int)(filters / 4), (int)(filters / 4), kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(int(filters/4))
            self.conv3 = nn.Conv2d((int)(filters / 4), filters, kernel_size=1, stride=1, bias=False)
            self.shortcut = nn.Sequential(
                nn.Conv2d(first_num_filters, filters ,kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(filters)
            )
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        out = nn.functional.relu(self.bn1(inputs))
        out = self.conv1(out)
        out = nn.functional.relu(self.bn2(out))
        out = self.conv2(out)
        out = nn.functional.relu(self.bn3(out))
        out = self.conv3(out)
        out = out + self.shortcut(inputs)
        return out
        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        filters_in = first_num_filters
        filters_out = filters * 4 if block_fn is bottleneck_block else filters
        ### END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        if block_fn is standard_block:
            layers = []
            for i in range(resnet_size):
                projection_shortcut = None
                if filters_out != filters_in:
                    if filters_out == 2*first_num_filters:
                        filters_in = first_num_filters
                        projection_shortcut = 'yes'
                        layers.append(standard_block(filters_out,projection_shortcut,strides,filters_in))
                        filters_in = filters_out
                    if filters_out == 4*first_num_filters:
                        filters_in = 2*first_num_filters
                        projection_shortcut = 'yes'
                        layers.append(standard_block(filters_out,projection_shortcut,strides,filters_in))
                        filters_in = filters_out
                    # print('filters_out',filters_out,'\n','filters_in',filters_in)

                    # filters_in = filters_out
                layers.append(standard_block(filters_out,projection_shortcut,strides,filters_out))

        #Now we do it for Bottleneck...This is going wrong somewhere man...WTF
        if block_fn is bottleneck_block:
            layers = []
            for i in range(resnet_size):
                projection_shortcut = None
                if filters_out != filters_in:
                    if filters_out == 4*first_num_filters:

                        # print(first_num_filters)
                        filters_in = first_num_filters

                        # print("check 1 - g")
                        projection_shortcut = 'yes'
                        layers.append(bottleneck_block(filters_out,projection_shortcut,strides,filters_in))
                        filters_in = filters_out
                    if filters_out == 2*4*first_num_filters:
                        filters_in = 4*first_num_filters
                        projection_shortcut = 'yes'
                        
                        ### YOUR CODE HERE
        
                        ### YOUR CODE HERE

                        layers.append(bottleneck_block(filters_out,projection_shortcut,strides,filters_in))
                        filters_in = filters_out
                    if filters_out == 4*4*first_num_filters:
                        filters_in = 2*4*first_num_filters
                        projection_shortcut = 'yes'
                        layers.append(bottleneck_block(filters_out,projection_shortcut,strides,filters_in))
                        filters_in = filters_out
                    # print('filters_out',filters_out,'\n','filters_in',filters_in)
                else:
                    # filters_in = filters_out
                    layers.append(bottleneck_block(filters_out,projection_shortcut,strides,filters_out))

        self.layer = nn.Sequential(*layers)
        
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        out = self.layer(inputs)
        return out
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        if (resnet_version == 2):
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        if (resnet_version == 1):
            filters = filters/4
        

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(filters) , num_classes)
        ### END CODE HERE
        
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        out = self.avg_pool(inputs)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
        
        ### END CODE HERE