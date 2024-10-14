
from datasets import load_dataset
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    question_id = [item['question_id'] for item in batch]
    # image = [item['image'] for item in batch]
    question = [item['question'] for item in batch]
    answer = [item['answer'] for item in batch]
    image_url = [item['image_url'] for item in batch]
    height = [item['height'] for item in batch]
    width = [item['width'] for item in batch]

    return {
        'question_id': question_id,
        # 'image': image,
        'question': question,
        'answer': answer,
        'image_url': image_url,
        'height': height,
        'width':width
    }

class CoCoCaption2017():
    def __init__(self, dataset_name, cache_dir):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.cache_dir    = cache_dir
        self.dataset = load_dataset(self.dataset_name, cache_dir = self.cache_dir)

    def get_train_dataloader(self, batch_size = 32):
        return self.DataLoader(self.dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        
    def get_val_dataloader(self, batch_size = 32):
        return self.DataLoader(self.dataset['validation'], batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
    def get_test_dataloader(self, batch_size = 32):
        return self.DataLoader(self.dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


