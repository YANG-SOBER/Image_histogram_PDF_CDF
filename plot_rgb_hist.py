import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_rgb_img(img):
    fig, axs = plt.subplots(nrows=1, ncols = 3, figsize=(36, 8))
    for i, ax in zip(range(3), axs):
        temp = np.zeros(img.shape, dtype='uint8')
        temp[:, :, i] = img[:, :, i]
        ax.imshow(temp)
        ax.set_axis_off()
    plt.show()

# occurrences computation
def compute_h(img_single):
    h = [0] * 256
    for i in range(img_single.shape[0]):
        for j in range(img_single.shape[1]):
            h[img_single[i, j]] += 1

    return np.array(h)

def compute_prob_h(h):
    prob_h = [0] * 256
    for i in range(len(h)):
        prob_h[i] = h[i] / num_pixels

    return np.array(prob_h)

def compute_cumul_h(h):
    cumul_h = [0] * 256
    for i in range(255):
        cumul_h[i + 1] = cumul_h[i] + h[i + 1]

    return np.array(cumul_h)

def compute_cumul_prob_h(cumul_h):
    cumul_prob_h = [0] * 256
    for i in range(len(cumul_h)):
        cumul_prob_h[i] = cumul_h[i] / num_pixels

    return np.array(cumul_prob_h)

def produce_x_axis():
    x = [0] * 256
    for i in range(256):
        x[i] = i

    return x

def plot_img_hist(img):
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(60 , 8))
    x = produce_x_axis()
    color=['red', 'green', 'blue']

    axs[0].imshow(img)
    axs[0].set_title('Milan Cathedral')
    axs[0].set_axis_off()

    for i in range(3):
        temp = np.zeros(img.shape, dtype='uint8')
        temp[:, :, i] = img[:, :, i] # r, g, b channel respectively
        h = compute_h(temp[:, :, i])
        axs[1].plot(x, h, color[i], label=color[i].capitalize() + ' Channel')
        axs[1].set_title('Image Histogram')
        axs[1].set_xlabel('Intensity Values', fontsize=8)
        axs[1].set_ylabel('Occurrences', fontsize=8)
        axs[1].legend()

        prob_h = compute_prob_h(h)
        axs[2].plot(x, prob_h, color[i], label=color[i].capitalize() + ' Channel')
        axs[2].set_title('Empirical PDF')
        axs[2].set_xlabel('Intensity Values', fontsize=8)
        axs[2].set_ylabel('Probability', fontsize=8)
        axs[2].legend()

        cumul_h = compute_cumul_h(h)
        axs[3].plot(x, cumul_h, color[i], label=color[i].capitalize() + ' Channel')
        axs[3].set_title('Image Cumulative Histogram')
        axs[3].set_xlabel('Intensity Values', fontsize=8)
        axs[3].set_ylabel('Cumulative Occurrences', fontsize=8)
        axs[3].legend()

        cumul_prob_h = compute_cumul_prob_h(cumul_h)
        axs[4].plot(x, cumul_prob_h, color[i], label=color[i].capitalize() + ' Channel')
        axs[4].set_title('Empirical CDF')
        axs[4].set_xlabel('Intensity Values', fontsize=8)
        axs[4].set_ylabel('Cumulative Probability', fontsize=8)
        axs[4].legend()
    fig.savefig('rgb_hist.jpg')
    plt.show()

if __name__ == '__main__':
    img_path = 'milan.jpeg'
    img = np.array(Image.open(img_path), dtype='uint8')
    num_pixels = img.shape[0] * img.shape[1]
    plot_rgb_img(img)
    plot_img_hist(img)
