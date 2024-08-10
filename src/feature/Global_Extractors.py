from os.path import join
from third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
from third_party.MixVPR.feature_extract import MixVPRFeatureExtractor
from third_party.AnyLoc.feature_extract import VLADDinoV2FeatureExtractor
from third_party.salad.feature_extract import DINOV2SaladFeatureExtractor
# TODO: after allowing extractors to return encodings, I have to make sure this interface is consistent everywhere

class GlobalExtractors:

    model_classes = {"MixVPR": MixVPRFeatureExtractor, "AnyLoc": VLADDinoV2FeatureExtractor, "DinoV2Salad": DINOV2SaladFeatureExtractor}

    def __init__(self, root, g_extr_conf, pipeline=False):
        """
        Creates a container for a list of vpr methods for inference.
        - root: prefix path for model parameters (same as in test_trained_model.py)
        - g_extr_conf: global extractor configurations; corresponding key under test_trained_model.yaml and trainer_pitts250.yaml
        - pipeline: controls whether to use a trained model (assumes "model_best.pth") or the pretrained verison ("checkpoint.pth")
        """
        self.models_objs = {}
        for model_name, model_configs in {name: conf for name, conf in g_extr_conf.items() if conf["use"]}.items():
            if model_name == "NetVlad":
                # Special setup for netvlad
                netvlad_model = NetVladFeatureExtractor(join(root, model_configs["ckpt_path"]), type="pipeline" if pipeline else None,
                    arch=model_configs['arch'], num_clusters=model_configs['num_clusters'], pooling=model_configs['pooling'],
                    vladv2=model_configs['vladv2'], nocuda=model_configs['nocuda'])
                netvlad_model.model.eval()
                self.models_objs["NetVlad"] = netvlad_model
            elif model_name in GlobalExtractors.model_classes:
                self.models_objs[model_name] = GlobalExtractors.model_classes[model_name](root, model_configs, pipeline)
            else:
                print(f"GlobalExtractors doesn't have {model_name}'s implementation, skipped.")

    def __call__(self, request_model, images):
        """
        Evaluate input images, assumed to be batched tensors already normalized, on the requested vpr model
        - request_model: string name of the desired vpr model
        """
        if request_model not in self.models_objs:
            print(f"GlobalExtractors wasn't initialized with {request_model}, skipped. options: {self.models}")
        else:
            return self.models_objs[request_model](images)

    @property
    def models(self):
        return list(self.models_objs.keys())

    def feature_length(self, model):
        return self.models_objs[model].feature_length