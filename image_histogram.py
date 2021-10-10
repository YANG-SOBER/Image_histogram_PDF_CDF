import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(img_rgb):

    return img_rgb[..., :3] @ [0.2989, 0.5870, 0.1140]

def produce_x_axis():

    x = [0] * 256

    for i in range(256):
        x[i] = i

    return x

def produce_bins():

    bins = [0] * 257

    for i in range(257):
        bins[i] = i

    return bins

def hist_computation(img_gray):
    h = [0] * 256
    height, width = img_gray.shape
    for i in range(height):
        for j in range(width):
            h[img_gray[i, j]] += 1 # O(N), N is the number of pixels

    return np.array(h)

def cumulative_hist_computation(h):
    cumul_h = [0] * 256
    for i in range(h.shape[0]-1):
        '''
        cumul_h[0] = h[0] # h[0] = h[0]
        cumul_h[1] = h[0] + h[1] # h[1] += h[0]
        cumul_h[2] = h[0] + h[1] + h[2] # h[2] += h[1]
        ...
        # h[255] += h[254]
        '''
        cumul_h[i+1] = cumul_h[i] + h[i+1]

    return np.array(cumul_h)

def probability_density_computation(img_gray, h):
    N = img_gray.reshape(-1).shape[0] #img_gray.shape[0] * img_gray.shape[1]
    prob_h = [0.] * 256
    for i in range(h.shape[0]):
        prob_h[i] = h[i] / N

    return prob_h

def plot_hist_both(img_gray, h):
    print(f'''
    h.shape = {h.shape}

    h = {h}
    ''')
    fig = plt.figure(figsize=(48, 8))
    x = produce_x_axis()

    fig.add_subplot(151)
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    fig.add_subplot(152)
    plt.plot(x, h, color='green')
    plt.title('Image Pixel Distribution (self)')
    plt.xlabel('Intensity Values')
    plt.ylabel('Occurrences')

    fig.add_subplot(153)
    n, bins, patches = plt.hist(img_gray.reshape(-1), bins=256, range=(0, 256), color='red')
    print(f'''
    img_gray = \n{img_gray.reshape(-1)}\n
    n.shape = \n{n.shape}\n
    n =\n{n}\n
    bins =\n{bins}\n
    n.dtype = {n.dtype}
    ''')
    plt.title('Image Histogram (matplotlib)')
    plt.xlabel('Intensity Values')
    plt.ylabel('Occurrences')

    fig.add_subplot(154)
    plt.plot(x, n, color='red')
    plt.title('Image Pixel Distribution (matplotlib)')
    plt.xlabel('Intensity Values')
    plt.ylabel('Occurrences')


    fig.add_subplot(155)
    bins_pro = produce_bins()
    n2, bins2 = np.histogram(img_gray.reshape(-1), bins=bins_pro)
    plt.plot(x, n2, color='blue')
    plt.title('Image Pixel Distribution (numpy)')
    plt.xlabel('Intensity Values')
    plt.ylabel('Occurrences')
    fig.savefig('img_dist_hist.jpg')
    plt.show()

def plot_cumulative_probability_hist(img_gray, cumul_h, prob_h, prob_cumul_h):
    fig = plt.figure(figsize=(48, 8))
    x = produce_x_axis()

    fig.add_subplot(141)
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')


    fig.add_subplot(142)
    plt.plot(x, cumul_h, color='green')
    plt.title('Image Cumulative Histgram')
    plt.xlabel('Intensity Values')
    plt.ylabel('Cumulative Occurrences')

    fig.add_subplot(143)
    plt.plot(x, prob_h, color='blue')
    plt.title('Image Empirical Probability Density Func')
    plt.xlabel('Intensity Values')
    plt.ylabel('Empirical PDF')

    fig.add_subplot(144)
    plt.plot(x, prob_cumul_h, color='red')
    plt.title('Image Empirical Cumulative Density Func')
    plt.xlabel('Intensity Values')
    plt.ylabel('Empirical CDF')
    fig.savefig('img_cumul_prob_hist.jpg')
    plt.show()

def plot_bars_both(img_gray, h):
    h = hist_computation(img_gray)
    print(f'''
    h.shape = {h.shape}

    h = {h}
    ''')
    fig = plt.figure(figsize=(48, 8))
    x = produce_x_axis()

    fig.add_subplot(151)
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    fig.add_subplot(152)
    plt.bar(x, h, color='green', align='edge')
    plt.title('Image Pixel Distribution (self)')
    plt.xlabel('Intensity Values')
    plt.ylabel('Occurrences')

    fig.add_subplot(153)
    n, bins, patches = plt.hist(img_gray.reshape(-1), bins=256, range=(0, 256), color='red')
    print(f'''
    img_gray = \n{img_gray.reshape(-1)}\n
    n.shape = \n{n.shape}\n
    n =\n{n}\n
    bins =\n{bins}\n
    n.dtype = {n.dtype}
    ''')
    plt.title('Image Histogram (matplotlib)')
    plt.xlabel('Intensity Values')
    plt.ylabel('Occurrences')

    fig.add_subplot(154)
    plt.bar(x, n, color='red', align='edge')
    plt.title('Image Pixel Distribution (matplotlib)')
    plt.xlabel('Intensity Values')
    plt.ylabel('Occurrences')

    fig.add_subplot(155)
    bins_pro = produce_bins()
    n2, bins2 = np.histogram(img_gray.reshape(-1), bins=bins_pro)
    plt.bar(x, n2, color='blue', align='edge')
    plt.title('Image Pixel Distribution (numpy)')
    plt.xlabel('Intensity Values')
    plt.ylabel('Occurrences')
    fig.savefig('img_dist_hist.jpg')
    plt.show()

if __name__=='__main__':

    img = mpimg.imread('milan.jpeg')
    img_gray = np.uint8(rgb2gray(img))

    h = hist_computation(img_gray)
    plot_hist_both(img_gray, h)

    cumul_h = cumulative_hist_computation(h)
    prob_h = probability_density_computation(img_gray, h)
    prob_cumul_h = probability_density_computation(img_gray, cumul_h)

    plot_cumulative_probability_hist(img_gray, cumul_h, prob_h, prob_cumul_h)
    plot_bars_both(img_gray, h)
