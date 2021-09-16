from os import listdir
from PIL import Image
countbad = 0
countgood = 0
genres = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
for genre in genres:
    for filename in listdir('/media/misiek/Dane/fma_spectrograms/validation/' + genre + '/'):
        if filename.endswith('.png'):
            countgood += 1
            try:
                img = Image.open('/media/misiek/Dane/fma_spectrograms/validation/' + genre + '/'+filename) # open the image file
                img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print('Bad file:', genre + '/' + filename) # print out the names of corrupt files
                countbad += 1
print(countgood)
print(countbad)