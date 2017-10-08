import moxel

model = moxel.Model('moxel/awesome:latest', where='localhost')
image = moxel.space.Image.from_file('download.jpg')
output = model.predict(image=image)
print(output['nude'])
