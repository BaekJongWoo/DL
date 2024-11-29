class dataLoader():
    def __iter__(self):
        return datasetIterator(self)
    
    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self):
        raise NotImplementedError()

# for enumerate magic python function returns Iterator
class datasetIterator():
    def __init__(self, dataloader):
        self.index = 0
        self.dataloader = dataloader

    def __next__(self):
        if self.index < len(self.dataloader):
            item = self.dataloader[self.index]
            self.index += 1
            return item
        # end of iteration
        raise StopIteration