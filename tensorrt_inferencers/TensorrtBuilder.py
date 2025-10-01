import yaml
import logging
from .NLP.cudart_version.tensorrtreranker import TensorRTReranker
from .NLP.cudart_version.tensorrtnli import TensorRTNLI
from .NLP.cudart_version.tensorrtembedding import TensorRTEmbedding

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
                tokenizer_path=config.get('tokenizer_path'),
                reuse_dynamic_buffer=config.get('reuse_dynamic_buffer', True)
            )
            setattr(self, key, model)
            self.model_name.append(key)
            self.model['embedding_models'].append(key)
            
            # initialize cuda graph for batch size
            cuda_graph_list = config.get('cuda_graph_list', [])
            for batch_size in cuda_graph_list:
                self.logger.info(f"[TensorrtBuilder] Building CUDA graph for model {key} with batch size {batch_size}")
                model.build_cuda_graph(batch_size=batch_size)

        # reranking models
        for key, config in self.reranking_model_configs.items():
            self.logger.info(f"[TensorrtBuilder] Loading model {key}")
            model = TensorRTReranker(
                engine_path=config['model_path'],
                tokenizer_path=config.get('tokenizer_path'),
                reuse_dynamic_buffer=config.get('reuse_dynamic_buffer', True)
            )
            setattr(self, key, model)
            self.model_name.append(key)
            self.model['reranking_models'].append(key)
            
            # initialize cuda graph for batch size
            cuda_graph_list = config.get('cuda_graph_list', [])
            for batch_size in cuda_graph_list:
                self.logger.info(f"[TensorrtBuilder] Building CUDA graph for model {key} with batch size {batch_size}")
                model.build_cuda_graph(batch_size=batch_size)

        # nli models
        for key, config in self.nli_model_configs.items():
            self.logger.info(f"[TensorrtBuilder] Loading model {key}")
            model = TensorRTNLI(
                engine_path=config['model_path'],
                tokenizer_path=config.get('tokenizer_path'),
                reuse_dynamic_buffer=config.get('reuse_dynamic_buffer', True)
            )
            setattr(self, key, model)
            self.model_name.append(key)
            self.model['nli_models'].append(key)
            
            # initialize cuda graph for batch size
            cuda_graph_list = config.get('cuda_graph_list', [])
            for batch_size in cuda_graph_list:
                self.logger.info(f"[TensorrtBuilder] Building CUDA graph for model {key} with batch size {batch_size}")
                model.build_cuda_graph(batch_size=batch_size)