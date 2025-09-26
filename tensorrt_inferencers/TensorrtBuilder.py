import yaml
import logging

from .NLP.tensorrtreranker import TensorRTReranker
from .NLP.tensorrtnli import TensorRTNLI
from .NLP.tensorrtembedding import TensorRTEmbedding

class TensorrtBuilder:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.config_path = config_path
        self.config = self.load_config(config_path)

        self.embedding_model_configs = self.config.get('embedding_models', {})
        self.reranking_model_configs = self.config.get('reranking_models', {})

        # add nli model configs
        self.nli_model_configs = self.config.get('nli_models', {})

        self.model_name = []
        self.model = {
            "embedding_models": [],
            "reranking_models": [],
            "nli_models": []
        }

        self._load_model()

    def load_config(self, path):
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
        
    def _load_model(self):

        # embedding models
        for key, config in self.embedding_model_configs.items():
            self.logger.info(f"[TensorrtBuilder] Loading model {key}")
            model = TensorRTEmbedding(
                engine_path=config['model_path'],
                tokenizer_path=config.get('tokenizer_path')
            )
            setattr(self, key, model)
            self.model_name.append(key)
            self.model['embedding_models'].append(key)

        # reranking models
        for key, config in self.reranking_model_configs.items():
            self.logger.info(f"[TensorrtBuilder] Loading model {key}")
            model = TensorRTReranker(
                engine_path=config['model_path'],
                tokenizer_path=config.get('tokenizer_path')
            )
            setattr(self, key, model)
            self.model_name.append(key)
            self.model['reranking_models'].append(key)

        # nli models
        for key, config in self.nli_model_configs.items():
            self.logger.info(f"[TensorrtBuilder] Loading model {key}")
            model = TensorRTNLI(
                engine_path=config['model_path'],
                tokenizer_path=config.get('tokenizer_path')
            )
            setattr(self, key, model)
            self.model_name.append(key)
            self.model['nli_models'].append(key)