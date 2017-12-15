from keras import applications
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

OUTPUT_DIR = 'Z:/Master/Workspace/Python/Github/Keras_All/KerasWithComments/keras02/keras/output/ling/GenerateModelPng'
# == Applications ==
model = applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                               classes=1000)
plot_model(model, to_file=OUTPUT_DIR+'/Xception_model.png',show_shapes=True,show_layer_names=True,rankdir="TB")
model.summary()
model = applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                         classes=1000)
plot_model(model, to_file=OUTPUT_DIR+'/VGG16_model.png',show_shapes=True,show_layer_names=True,rankdir="TB")
model.summary()
model = applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                         classes=1000)
plot_model(model, to_file=OUTPUT_DIR+'/VGG19_model.png',show_shapes=True,show_layer_names=True,rankdir="TB")
model.summary()
model = applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                               classes=1000)
plot_model(model, to_file=OUTPUT_DIR+'/ResNet50_model.png',show_shapes=True,show_layer_names=True,rankdir="TB")
model.summary()
model = applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                                      pooling=None, classes=1000)
plot_model(model, to_file=OUTPUT_DIR+'/InceptionV3_model.png',show_shapes=True,show_layer_names=True,rankdir="TB")
model.summary()
model = applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True,
                                 weights='imagenet', input_tensor=None, pooling=None, classes=1000)
plot_model(model, to_file=OUTPUT_DIR+'/MobileNet_model.png',show_shapes=True,show_layer_names=True,rankdir="TB")
model.summary()

