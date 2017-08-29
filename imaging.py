

#run as imaging.py <datafile> <fileStarter>
#datafile is "centers" for the clusters
#fileStarter is how the image files should begin - ie "10_clust"

from PIL import Image
import sys

file_starter = sys.argv[2]

#open file
text_file = open(sys.argv[1])

#create list of lines
data = text_file.readlines()

#close the file
text_file.close()

#put data in float form instead of string
for line in range(len(data)):
  new = []
  for val in data[line].split(" "):
    if val != "\n":
      new.append(float(val))

  data[line] = new




#create a new image
img = Image.new("L", (8, 8))

for cluster in range(len(data)):
  file_string = file_starter + "_" + str(cluster) + ".png"
  for x in range(8):
    for y in range(8):
      val = (data[cluster][(x * 8) + y])/16
      val = round(val * 255)
      img.putpixel((x, y), val)
  img.save(file_string)


