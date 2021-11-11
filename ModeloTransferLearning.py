import tensorflow as tf

class ModeloTransferLearning:
    def __init__(self, size_image, especifico=None):
        sz = size_image
        
        Imagenet = {
            "VGG16":tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=sz),
            "VGG19":tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=sz),
            "MobileNetV2":tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=sz),
            "ResNet50":tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=sz),
            "ResNet152V2":tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=sz),
            "Xception":tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=sz),
            "InceptionV3":tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=sz),
            "InceptionResNetV2":tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=sz)
        }
        
        Descongelar = {
            "VGG16":3,
            "VGG19":3,
            "MobileNetV2":4,
            "ResNet50":4,
            "ResNet152V2":4,
            "Xception":3,
            "InceptionV3":17,
            "InceptionResNetV2":3
        }
        
        self.TransferLearning = dict()
        for key, conv_base in Imagenet.items():            
            if especifico is not None:
                self.TransferLearning[especifico] = self.descongelar_capas(Imagenet[especifico], Descongelar[especifico])
                break
                
            self.TransferLearning[key] = self.descongelar_capas(Imagenet[key], Descongelar[key])
                
    def getModels(self):
        return self.TransferLearning            
        
    def descongelar_capas(self, model, capas_descongelar):
        model.trainable = True # congelo todas las capas del modelo
        for layer in model.layers[:-capas_descongelar]: # descongelo las 5 ultimas capas
            layer.trainable = False
        return model