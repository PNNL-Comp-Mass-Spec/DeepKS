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
colors = [get_random_color(i + 100, brightness_range=list(range(200 - 16, 200 + 16))) for i in range(20)]

# %%
random.seed(0)
random.shuffle(colors)
rand_site_colors = np.array([[random.choice(colors) for _ in range(15)]], np.uint8)
rand_site_colors = np.repeat(rand_site_colors, 400, 0)
rand_site_colors = np.repeat(rand_site_colors, 100, 1)

# %%
img = Image.fromarray(rand_site_colors, mode='RGB')
img.save("site_image.png")
plt.imshow(img)

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


