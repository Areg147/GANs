import tensorflow as tf

class Preprocessor:
    def __init__(self,root_images_path:str,image_shape:list,batch_size:int):
        self.root_images_path = root_images_path
        self.image_shape = image_shape
        self.batch_size = batch_size
        

    def parse_image(self,file_path):
        """
        Read and preprocess an image from the specified file path.
        Args:
            file_path (str): Path to the image file.

        Returns:
            tf.Tensor: The preprocessed and resized image as a TensorFlow Tensor.
        """

        row_content = tf.io.read_file(file_path)
        image = tf.io.decode_jpeg(row_content,channels=3) # 3 chnngel image
        image = image/255
        image = image*2
        image = image-1
        
        converted_image = tf.image.convert_image_dtype(image,tf.float32)
        resized_image = tf.image.resize(converted_image,self.image_shape)
        return resized_image
    
    def fit_transform(self):
        
        list_ds = tf.data.Dataset.list_files(self.root_images_path + "/" + "*") #You can change a '/' to fit your working environment
        dataset = list_ds.map(self.parse_image)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset_batched = dataset.batch(self.batch_size,drop_remainder=True)
        return dataset_batched
