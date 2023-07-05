import numpy as np


def get_augmented_text(augment_name, params = None):
    '''
    Append a description of what image augmentation has been applied to the training data.
    The first sentence is the original text description, and the appended portion is the augmentation.
    e.g. for 90 rotation we could have something like, "A picture of an orange in a field of limes. Rotated by 90"
    
    Params:
        augment_name (str): The augmentation that was applied (name obtained from Albumentation class name).
        params (dict): Any params that are used in the augmentation (e.g. {'angle': 40.3232} for rotation).
    '''
    
    ### TODO:
    ### - Is this needed, or does a single option suffice?
    ###
    # Supply a few options for the augmentation description.
    #
    aug_text = {
        'HorizontalFlip': [' Horizontally flipped.', ' Applied horizontal flip.', ' Horizontal flip = True'],
        'VerticalFlip': [' Vertically flipped.', ' Applied vertical flip.', ' Vertical flip = True'],
        'RandomRotate90': [' Rotated by ', ' Applied a rotation of ', ' Rotation = '], 
        'RandomScale': [' Scaled by ', ' Applied a scaling of ', ' Scale = '],
        'ShiftScaleRotate': [[' Rotated by ', 'scaled by ', 'shifted x by ', 'shifted y by '], [' Rotation = ', 'scale = ', 'shift x = ', 'shift y = ']],
    }
    
    # Get the list of options for the augmentation text.
    #
    text_options = aug_text[augment_name]
    
    # Randomly select an entry in the possible variations of describing the augmentation.
    #
    idx = np.random.randint(0, len(text_options))
    text = text_options[idx]
    if params is None:
        return text
    
    
    ### TODO:
    ### - Clean this up to be more general instead of case-by-case handling.
    ###
    if augment_name == 'RandomRotate90': 
        params = list(params.values())[0] # integer value of number of rotations (e.g. 2 for 180 deg)
        params *= 90 # convert to a multiple of the degrees
        text = text + str(params)
        
    if augment_name == 'RandomScale': 
        params = list(params.values())
        params = round(params[0], 3)
        text = text + str(params)
        
    if augment_name == 'ShiftScaleRotate':
#         NOTE: params = {'angle': 4.707024976908613, 'scale': 1.0290787620541384, 'dx': 0.007608877060265515, 'dy': 0.01500264384975191}
        aug_vals = [round(params['angle'], 3), round(params['scale'], 3), round(params['dx'], 3), round(params['dy'], 3)]
        text = ', '.join(list(map(lambda x, y: x + str(round(y, 3)), text, aug_vals)))
        
    return text
