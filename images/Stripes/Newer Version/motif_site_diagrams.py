# %%
import numpy as np
import matplotlib.pyplot as plt
import random
import PIL
import PIL.Image

# %%
def get_random_color(seed = None, brightness_range = list(range(256))):
    if seed is None:
        random.seed(random.random())
    else:
        random.seed(seed)
    # return random.randrange(0, 2**24)
    return [random.choice(brightness_range), random.choice(brightness_range), random.choice(brightness_range)]

# %%
colors = [get_random_color(i + 100) for i in range(20)]

# %%
lum = lambda x: np.sqrt(0.2126 * x[0]**2 + 0.587 * x[1]**2 + 0.114 * x[2]**2)/255

# %%
lums = [lum(x) for x in colors]

# %%
rand_site_colors = np.array([colors], np.uint8)
rand_site_colors = np.repeat(rand_site_colors, 400, 0)
rand_site_colors = np.repeat(rand_site_colors, 100, 1)

# %%
img = PIL.Image.fromarray(rand_site_colors, mode='RGB')
img.save("site_image.png")
plt.imshow(img)
for i, color in enumerate(colors):
    plt.text(i*100+50, 200, str(i), color = 'black' if lum(color) > 0.5 else 'white', fontsize = 15, horizontalalignment = 'center')

# %%
random.seed(0)
random.shuffle(colors)
rand_kin_colors = np.array([[random.choice(colors) for _ in range(500)]], np.uint8)
rand_kin_colors = np.repeat(rand_kin_colors, 400, 0)
rand_kin_colors = np.repeat(rand_kin_colors, 50, 1)

# %%
img = Image.fromarray(rand_kin_colors, mode='RGB')
img.save("kin_image.png")
plt.imshow(img)


