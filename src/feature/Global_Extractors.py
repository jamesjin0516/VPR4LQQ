from os.path import join
from third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
from third_party.MixVPR.feature_extract import MixVPRFeatureExtractor
from third_party.AnyLoc.feature_extract import VLADDinoV2FeatureExtractor


class GlobalExtractors:

    model_classes = {"MixVPR": MixVPRFeatureExtractor, "AnyLoc": VLADDinoV2FeatureExtractor}

    def __init__(self, root, g_extr_conf, preprocess=False):
        """
        Creates a container for a list of vpr methods for inference.
        - root: prefix path for model parameters (same as in test_trained_model.py)
        - g_extr_conf: global extractor configurations; corresponding key under test_trained_model.yaml
        - preprocess: if false, only prepare netvlad. Because other methods aren't concerned with comparing
        with our trained netvlad's loss configuration.
        """
        self.models_objs = {}
        # Special setup for netvlad
        if "netvlad" in g_extr_conf and g_extr_conf["netvlad"]["use"]:
            model_configs = g_extr_conf["netvlad"]
            netvlad_model = NetVladFeatureExtractor(join(root, model_configs["ckpt_path"]), arch=model_configs['arch'],
                num_clusters=model_configs['num_clusters'], pooling=model_configs['pooling'], vladv2=model_configs['vladv2'],
                nocuda=model_configs['nocuda'])
            netvlad_model.model.eval()
            self.models_objs["pretrained_netvlad"] = netvlad_model
        # If run by testing_data.py or netvlad isn't used, consider remaining vpr methods
        if preprocess or "netvlad" not in g_extr_conf or not g_extr_conf["netvlad"]["use"]:
            for model in set(g_extr_conf.keys()).difference({"netvlad"}):
                if not g_extr_conf[model]["use"]: continue
                if model not in GlobalExtractors.model_classes:
                    print(f"GlobalExtractors doesn't have {model}'s implementation, skipped.")
                else:
                    self.models_objs[model] = GlobalExtractors.model_classes[model](root, g_extr_conf[model])

    def __call__(self, request_model, images):
        """
        Evaluate input images, assumed to be batched tensors already normalized, on the requested vpr model
        - request_model: string name of the desired vpr model
        """
        if request_model not in self.models_objs:
            print(f"GlobalExtractors wasn't initialized with {request_model}, skipped. options: {self.models}")
        else:
            return self.models_objs[request_model](images)
    
    def compare_trained(self):
        return "pretrained_netvlad" in self.models_objs

    @property
    def models(self):
        return list(self.models_objs.keys())

    def feature_length(self, model):
        return self.models_objs[model].feature_length