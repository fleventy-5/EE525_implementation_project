import torch
import torch.nn.functional as F

def cross_product(x,y):
    r1 = x[0][0][0]
    g1 = x[0][0][1]
    b1 = x[1][0][1]
    r2 = x[1][0][0]
    g2 = x[2][0][0]
    b2 = x[2][0][1]

    r_channel = r1@y[0][0]+r2@y[0][3]
    g_channel =  g1@y[0][1] +g2@y[0][4]
    b_channel =  b1@y[0][2]+ b2@y[0][5]

    # print(r_channel.shape)
    # print(b_channel.shape)
    # print(g_channel.shape)

    combined_image = torch.stack((r_channel, g_channel, b_channel), dim=0)

    return combined_image.unsqueeze(0)

# cross_product(ans,mfn_out) 

# img_fuse = cross_product(ans,mfn_out)
# img_fuse.shape

# def MeanFilter(image, filter_size):
#     # create an empty array with same size as input image
#     output = np.zeros(image.shape, np.uint8)

#     # creat an empty variable
#     result = 0

#     # deal with filter size = 3x3
#     if filter_size == 9:
#         for j in range(1, image.shape[0]-1):
#             for i in range(1, image.shape[1]-1):
#                 for y in range(-1, 2):
#                     for x in range(-1, 2):
#                         result = result + image[j+y, i+x]
#                 output[j][i] = int(result / filter_size)
#                 result = 0
    
#     return torch.from_numpy(output)

# # den_inp = MeanFilter(img_fuse,3)

# # den_inp = den_inp.to(torch.float)
# # den_inp.shape

def arithmetic_mean_filter(image, kernel_size=3):
    # Define a kernel for the arithmetic mean filter for each channel
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    
    # Separate the RGB channels
    red_channel = image[:, 0:1, :, :]
    green_channel = image[:, 1:2, :, :]
    blue_channel = image[:, 2:3, :, :]

    # Apply the filter to each channel using convolution
    red_filtered = F.conv2d(red_channel, kernel, padding=kernel_size // 2)
    green_filtered = F.conv2d(green_channel, kernel, padding=kernel_size // 2)
    blue_filtered = F.conv2d(blue_channel, kernel, padding=kernel_size // 2)

    # Combine the filtered channels back into an RGB image
    filtered_image = torch.cat((red_filtered, green_filtered, blue_filtered), dim=1)

    return filtered_image

def preprocess(img):
    red= img[:,0,:,:]
    green= img[:,1,:,:]
    blue= img[:,2,:,:]

    # blank_canvas = torch.zeros_like(red)
    # RG = torch.stack([red,green,blank_canvas],dim=0).unsqueeze(0)
    # RB = torch.stack([red,blank_canvas,blue],dim=0).unsqueeze(0)
    # GB = torch.stack([blank_canvas,green,blue],dim=0).unsqueeze(0)
    
    RG = torch.stack([red,green],dim=1)
    RB = torch.stack([red,blue],dim=1)
    GB = torch.stack([green,blue],dim=1)
    #print(RG.shape)

    return RG,RB,GB

# for i in data_train:
#     preprocess(i)
