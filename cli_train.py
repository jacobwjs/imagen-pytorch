import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from imagen_pytorch import (
    Unet,
    Imagen,
    ImagenTrainer,
    ElucidatedImagenConfig,
    ImagenConfig
)
from imagen_pytorch.data import Dataset

import os
import numpy as np
import json
import einops
from pathlib import Path
import braceexpand
import webdataset as wds
import binascii
import ast
import base64
import io
from PIL import Image
import click



def bytes_to_buffer(x, shape, dtype = np.float32, to_tensor = True):
    x = np.frombuffer(x, dtype = dtype).reshape(shape)
    if to_tensor:
        x = torch.tensor(x)
    
    return x


def decode_text(x):
    return x.decode('utf-8')


def drop_metadata(x):
    # Remove all key value pairs not in "keep" keys.
    #
    keep = ["width", "height", "similarity", "punsafe", "pwatermark", "aesthetic", "url", "sha256"]
    metadata = json.loads(x)
    return {key: val for key, val in metadata.items() if key in keep}


def decode_image(x, to_pil = True):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    
    try:
        image_data = base64.b64decode(x, validate = True)
    except binascii.Error:
        image_data = x

    if to_pil: 
        image_data = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    return image_data


def load_json(x):
    return json.loads(x)


class Collator:
    def __init__(
        self,
        channels,
        image_size
    ):
        self.channels = channels
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])
        
    def __call__(self, batch):
        texts = []
        images = []
        for items in batch:
            try:
                keys, _urls, urls, images, metas, captions, embedding_dims, text_embeddings = items
                text_embeddings = [bytes_to_buffer(emb, dims) for emb, dims in zip(text_embeddings, embedding_dims)]
                images = [self.transform(image.convert(self.channels)) for image in images]
                images = torch.stack([image for image in images])
                images = images.to(memory_format=torch.contiguous_format).float()
            except Exception as e:
                print("ERROR (Collator): unable to extract batch")
                print(e)


        if len(text_embeddings) == 0:
            return None
        

        newbatch = []
        for idx in range(len(text_embeddings)):
            newbatch.append((images[idx], text_embeddings[idx]))

        return torch.utils.data.dataloader.default_collate(newbatch)


@click.command(help = 'Train the Imagen model')
@click.option('--config', default = './configs/default_config.json', help = 'Path to the Imagen model config')
@click.option('--unet', default = 1, help = 'Unet to train', type = click.IntRange(1, 3, False, True, True))
@click.option('--epoches', default = 1000, help = 'Amount of epoches to train for')
@click.option('--text', required = False, help = 'Text to sample between epoches', type=str)
@click.option('--valid', is_flag = False, flag_value=50, default = 0, help = 'Do validation between epoches', show_default = True)
def train(
    config,
    unet,
    epoches,
    text,
    valid
):
    

    # check config path
    
    config_path = Path(config)
    full_config_path = str(config_path.resolve())
    assert config_path.exists(), f'config not found at {full_config_path}'
    
    with open(config_path, 'r') as f:
        config_data = json.loads(f.read())

    # print(config_data)
    
    assert 'checkpoint_path' in config_data, 'checkpoint path not found in config'
    assert ('batch_size' in config_data['dataset']) or ('batch_size' in config_data['webdataset']) , 'A batch_size is required in the config file'
    
    model_path = Path(config_data['checkpoint_path'])
    full_model_path = str(model_path.resolve())
    
    # setup imagen config
    #
    imagen_config_klass = ElucidatedImagenConfig if config_data['type'] == 'elucidated' else ImagenConfig
    imagen = imagen_config_klass(**config_data['imagen']).create()

    
    trainer = ImagenTrainer(
        imagen = imagen,
        **config_data['trainer']
    )

    
    if torch.cuda.is_available():
        trainer = trainer.cuda()

    
    # Load in webdataset and send to trainer.
    #
    cache_dir = config_data['webdataset']['cache_dir']
    os.makedirs(cache_dir, exist_ok = True)
    input_urls = braceexpand.braceexpand(config_data['webdataset']['url'])
    batch_size = config_data['webdataset']['batch_size']
    dataset_total_length = config_data['webdataset']['total_length']
    dataset_nominal_length = dataset_total_length // batch_size
    dataset = (
        wds.WebDataset(input_urls, cache_dir=cache_dir, nodesplitter=wds.split_by_node)
        .shuffle(1000)
        .to_tuple("__key__", "__url__", "url", "image.jpg", "metadata.json", "caption.txt", "embedding_dims.json", "text_embedding.bytes")
        .map_tuple(None, None, None, decode_image, None, decode_text, load_json, None)
        .batched(batch_size)
    )
    dataset.with_length(dataset_nominal_length)
    
    image_size = config_data['imagen']['image_sizes'][unet-1]
    collate_fn = Collator(
        image_size = image_size,
        channels = "RGB"
    )
    
    dataloader = DataLoader(
        dataset,
        collate_fn = collate_fn,
        batch_size = 1, # Already handled in webdataset, as recommended
        num_workers = 4,
        shuffle = False # Already handled in webdataset
    )
    trainer.add_train_dataloader(dataloader)
    
    max_batch_size = config_data['gradient_accum_size']
    for epoch in range(epoches):
        
        loss = trainer.train_step(
            unet_number = unet,
            max_batch_size = max_batch_size
        )
        
        if epoch % 10 == 0:
            print(f'epoch: {epoch}, loss: {loss}')

    

if __name__ == "__main__":
    
    train()

    

    

