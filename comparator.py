import pandas as pd 
from custom_similarity.similarity import get_similarity_from_memory, get_similarity_from_files, get_similarity
from PIL import Image

r1 = Image.open('1.jpg').convert('RGB')
r2 = Image.open('2.jpg').convert('RGB')
r3 = '3.jpg'
r4 = '4.jpg'


sim1 = get_similarity_from_memory(r1, r2)
sim2 = get_similarity_from_files(r3, r4)
sim3 = get_similarity(r1, r4)

print(sim1)
print(sim2)
print(sim3)

print('Done!')










