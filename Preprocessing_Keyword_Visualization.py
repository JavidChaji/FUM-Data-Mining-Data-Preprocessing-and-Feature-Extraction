from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator



extracted_key_tokens_csv = pd.read_csv('./four_extracted_key_tokens.csv')

zero_point_two_UOSA_Phase0_csv = pd.read_csv('./zero_point_one_UOSA_Phase0.csv')

zero_point_two_UOSA_Phase0_csv['Outcome_Description'] = extracted_key_tokens_csv['Outcome_Description']


zero_point_two_UOSA_Phase0_csv.to_csv('./zero_point_two_UOSA_Phase0_visualization.csv', index=False)


dataset_csv = " ".join(review.replace("'"," ") for review in zero_point_two_UOSA_Phase0_csv['Outcome_Description'] if type(review) != float)

# Generate a word cloud image
mask = np.array(Image.open("img/mask.jpg"))
wordcloud = WordCloud(background_color="white", max_words=100, width=4000, height=3000, mask=mask).generate(dataset_csv)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[7,7])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
plt.savefig("img/wordcloud.png", format="png")
plt.show()
