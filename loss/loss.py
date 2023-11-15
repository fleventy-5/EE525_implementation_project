import torch
import torch.nn.functional as F
# from ..utils.utils import arithmetic_mean_filter

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


def color_relevance_loss(im_pre,im_nor):
        
        # Extract channels P and Q from enhanced and normal images
        tot_loss =0
        for enhanced_images, normal_images in zip(im_pre, im_nor):
                
                P_enhanced = enhanced_images[ :,0, :, :]  # Assuming channel 0 for P, adjust as needed
                Q_enhanced = enhanced_images[:,1, :, :]  # Assuming channel 1 for Q, adjust as needed

                P_normal = normal_images[:,0, :, :]  # Assuming channel 0 for P, adjust as needed
                Q_normal = normal_images[ :,1, :, :]  # Assuming channel 1 for Q, adjust as needed

                # Reshape P and Q to be 2D matrices
                N = P_enhanced.size(1) * P_enhanced.size(2)  # N = H x W
                P_enhanced = P_enhanced.view(-1, N)
                Q_enhanced = Q_enhanced.view(-1, N)
                P_normal = P_normal.view(-1, N)
                Q_normal = Q_normal.view(-1, N)

                # Compute the channel relevance map X for both normal and enhanced images
                X_enhanced = calculate_channel_relevance(P_enhanced, Q_enhanced)
                X_normal =calculate_channel_relevance(P_normal, Q_normal)

                # Compute the loss as the absolute difference between X in normal and enhanced images
                loss = torch.abs(X_normal - X_enhanced)
                tot_loss+=loss

        return tot_loss

def calculate_channel_relevance(P, Q):
        # Compute the correlation matrix X
        X = torch.mm(P, Q.t())

        # Apply softmax to calculate the channel relevance map
        X = F.softmax(X, dim=-1)

        return X
    



class MFN_loss:
    def __init__(self,w_r=1,w_c=0.1,w_p=5,w_s=3):
        self.w_r = w_r
        self.w_c = w_c
        self.w_p = w_p
        self.w_s = w_s
        

    def reconstruction_loss(self,img_pre,img_nor):
        return torch.norm(torch.sum(torch.abs(img_pre - img_nor)),p=1)
    
    
    def spatial_consistency_loss(self,enhanced_images, normal_images, K=4):
        """Calculates the spatial consistency loss.

        Args:
            enhanced_images: A PyTorch tensor containing the enhanced images.
            normal_images: A PyTorch tensor containing the normal images.
            K: The number of local regions.

        Returns:
            The spatial consistency loss.
        """

        # Calculate the average intensity value of each local region in the enhanced and normal images.
        enhanced_images_regions = torch.split(enhanced_images, K, dim=2)
        normal_images_regions = torch.split(normal_images, K, dim=2)
        enhanced_images_regions_mean = torch.mean(torch.cat(enhanced_images_regions, dim=0), dim=0)
        normal_images_regions_mean = torch.mean(torch.cat(normal_images_regions, dim=0), dim=0)

        # Calculate the squared difference between the average intensity values of adjacent regions.
        squared_difference = torch.pow(enhanced_images_regions_mean - normal_images_regions_mean, 2)

        # Sum the squared difference and normalize it by the number of local regions.
        loss = torch.sum(squared_difference) / K

        return loss
    
    def smoothness_loss(self, enhanced_image, normal_image):
        # Compute gradients for the enhanced and normal images
        enhanced_gradients = self.compute_gradients(enhanced_image)
        normal_gradients = self.compute_gradients(normal_image)

        # Compute the absolute difference between the gradients
        absolute_diff = torch.abs(enhanced_gradients - normal_gradients)

        # Sum the differences across color channels (R, G, B)
        lmfns = torch.mean(absolute_diff, dim=1)

        # Calculate the overall smoothness loss
        smoothness_loss = torch.mean(lmfns)

        return smoothness_loss

    def compute_gradients(self, image):
        # Compute gradients using Sobel filters
        sobel_filter_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(image.device)
        sobel_filter_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).to(image.device)

        gradient_x = F.conv2d(image, sobel_filter_x)
        gradient_y = F.conv2d(image, sobel_filter_y)

        # Combine the horizontal and vertical gradients
        gradients = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

        return gradients
    
    def smoothness_loss(self,img_pre,img_nor):
        tot_sum = 0
        for i in range(3):
            tot_sum+=self.smoothness_loss(img_pre[0:, i, :, :].unsqueeze(0),img_nor[:, i, :, :].unsqueeze(0))
        return tot_sum

    def get_mfn_loss(self,img_pre,img_nor):
        total_loss = self.w_s*self.smoothness_loss(img_pre,img_nor)+ self.w_r*self.reconstruction_loss(img_pre,img_nor)+ self.w_p*self.spatial_consistency_loss(img_pre,img_nor)

        return total_loss


class DEN_loss:
    def __init__(self) -> None:
        pass           

    def get_den_loss(self, intermediate_image, original_image, kernel_size=3):
        # Compute Inor (original image minus arithmetic mean filter result)
        
        mean_filtered_image = arithmetic_mean_filter(original_image, kernel_size)
        Inor = original_image - mean_filtered_image

        #Ipre =intermediate_image-arithmetic_mean_filter(intermediate_image,kernel_size) #intermediate_image-
        Ipre = intermediate_image

        # Compute LDEN
        LDEN = Ipre- Inor

        l1_norm = torch.norm(LDEN, p=1)

        return l1_norm