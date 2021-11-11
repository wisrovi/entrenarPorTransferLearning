#callback para mejorar los entrenamientos
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback, CSVLogger, TensorBoard  

class CallbacksRNA:
    llamadas_RNA = list()
    
    def addParadaTemprana(self, epocasSinCambios=2):
        #EarlyStopping, detener el entrenamiento una vez que su pérdida comienza a aumentar
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=epocasSinCambios, #argumento de patience representa el número de épocas antes de detenerse una vez que su pérdida comienza a aumentar (deja de mejorar). 
            min_delta=0,  #es un umbral para cuantificar una pérdida en alguna época como mejora o no. Si la diferencia de pérdida es inferior a min_delta , se cuantifica como no mejora. Es mejor dejarlo como 0 ya que estamos interesados ​​en cuando la pérdida empeora.
            mode='auto',
            restore_best_weights=True)
        
        self.llamadas_RNA.append(early_stop)
        
    def addLogCSV(self, filename):
        logger = CSVLogger(
            filename, separator=',', append=False
        )
        self.llamadas_RNA.append(logger)
        
    def addReducirValorAprendizaje(self, epocasSinCambios=2):
        #ReduceLROnPlateau, que si el entrenamiento no mejora tras unos epochs específicos, reduce el valor de learning rate del modelo
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1, 
            patience=5, 
            mode='min')
        self.llamadas_RNA.append(reduce_lr)
        
    def addGuardadoPorEpocas(self, path="/"):        
        #Para algunos casos es importante saber cual entrenamiento fue mejor, 
        #este callback guarda el modelo tras cada epoca completada con el fin de si luego se desea un registro de pesos para cada epoca
        #Se ha usado este callback para poder optener el mejor modelo de pesos, sobretodo en la red neuronal creada desde cero
        #siendo de gran utilidad para determinar el como ir modificando los layer hasta obtener el mejor modelo
        mcp_save = ModelCheckpoint(
            path + 'model-{accuracy:03f}-{val_accuracy:03f}.h5', 
            monitor='val_loss',
            save_best_only=True, 
            mode='min')
        self.llamadas_RNA.append(mcp_save)
        
    def addTensorBoard(self, path="/logs"):
        tensorboard_callback = TensorBoard(log_dir=path, update_freq=1, histogram_freq=1)
        self.llamadas_RNA.append(tensorboard_callback)
        
    def getCallbacks(self):
        return self.llamadas_RNA
        