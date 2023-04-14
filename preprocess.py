from pathlib import Path
import os
import pandas as pd
from PIL import Image
import os
from glob import glob
from shutil import rmtree

path = Path("test_data_smaller")
images_dir = Path(path/'images/images')
df = pd.read_csv(path/'artists.csv')

#check distribution of paintings
artists_df = df[['name', 'paintings']].groupby(['name'], as_index = False).sum()
names = artists_df.sort_values('paintings', ascending = False)[:50]

#remove spaces from names
images_dir = Path(path/'images/images')
artists = names['name'].str.replace(' ', '_').values

#delete albrecht duerer, because of problems with folder name
pattern = os.path.join(Path(images_dir), "Albrecht_*")
for item in glob(pattern):
    if not os.path.isdir(item):
        continue
    rmtree(item)

# needed for first run
painting_list = []
for artist in artists:
    folder = Path(images_dir/artist)
    for subdir, dirs, files in os.walk(images_dir):
        # to-do delete folder with albrecht duerer images (problems because of ue)
        for file in files:
            img_path = os.path.join(subdir, file)
            img = Image.open(img_path)
            img = img.resize((224, 224))
            img.save(img_path)

