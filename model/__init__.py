import logging
import model.simpleunet as simpleunets
import model.inet as inets
import model.unet as unets
import model.vgg16 as vgg16
import model.refine as refines

# import model.resunet as resunets

MODELS = []


def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a or 'MLP' in a or 'Vgg' in a])


add_models(simpleunets)
add_models(inets)
add_models(unets)
add_models(vgg16)
add_models(refines)
# add_models(resunets)


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = MODELS
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    logging.info(f'Invalid model index. You put {name}. Options are:')
    # Display a list of valid model names
    for model in all_models:
      logging.info('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass
