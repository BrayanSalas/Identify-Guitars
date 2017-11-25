import os, sys
from PIL import Image
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = sys.argv[1]
ruta = sys.argv[1]
abrirImagen = Image.open(''+ruta)
abrirImagen.show()

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    # Codigo para imprimir el resultado de la predición con su puntuación
    i = 0
    auxPuntuacion = 0
    nombre = []
    puntuacion = []
    #Almacena el nombre de la carpeta en human_string y la puntuacion obtenida en score, luego estas son almacenadas en arreglos
    #para su comparacion
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        nombre.append(human_string)
        puntuacion.append(score)
        i = i + 1
    #Comparacion de puntuaciones segun la predicción
    for i in range(0, 4):
        if(puntuacion[i] > auxPuntuacion):
            auxPuntuacion = puntuacion[i]
            auxNombre = nombre[i]
    #Imprime la marca 
    print("La guitarra es de marca: "+auxNombre.capitalize())
    mostrar = input("Desea mostrar los porcentajes de las demas marcas? y/n")
    if(mostrar == "y"):
        for i in range(0,4):
            print('%s (Puntuacion = %.5f)' % (nombre[i].capitalize(), puntuacion[i]))