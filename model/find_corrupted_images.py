from PIL import Image
import os
import matplotlib.pyplot as plt



#Findinf corrupted images
image_dir = "downloaded_images/dataset/train/calcite"
corrupted_images = []

for filename in os.listdir(image_dir):
    try:
        with Image.open(os.path.join(image_dir, filename)) as img:
            #img.verify()
            img.resize(size=[224,224])
           
    except (SyntaxError) as e:
        print(f'Bad file: {filename}')
        plt.imshow(img)
        corrupted_images.append(filename)

print(f'Found {len(corrupted_images)} corrupted images.')