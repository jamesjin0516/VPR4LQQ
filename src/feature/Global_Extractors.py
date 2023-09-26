import sys
from os.path import join
sys.path.append(join(sys.path[0],'..'))
from third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor


class Global_Extractors():
    def __init__(self, **config):
        self.configs = config
        self.extractor = config['vpr']['global_extractor']
        self.extractor_list = {}

    def netvlad(self, content):
        self.extractor_list.update({'netvlad': NetVladFeatureExtractor
        (join(self.configs['root'], content['ckpt_path']), arch=content['arch'],
         num_clusters=content['num_clusters'],
         pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])})

    def vlad(self, contend):
        pass

    def bovw(self, contend):
        pass

    def run(self):
        for extractor, content in self.extractor.items():
            if content['use']:
                if extractor == 'netvlad':
                    self.netvlad(content)
                if extractor == 'vlad':
                    pass
                if extractor == 'bovw':
                    pass

        return self.extractor_list