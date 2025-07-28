from lightx2v.models.networks.wan.model import WanModel
from lightx2v.dist.wrappers.distrifusion.infer import DistriFusionWanTransformerInferWrapper
from lightx2v.dist.wrappers.distrifusion.weights import DistriFusionWanTransformerWeightsWrapper


class DistriFusionWanModelWrapper:
    def __init__(self, model: WanModel, config):
        self.model = model
        self.config = config
        
        self._wrap_transformer(self.model, self.config)

    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.model, name)
        
    def __delattr__(self, name: str):
        if name in self.__dict__:
            del self.__dict__[name]
        else:
            del self.model.__dict__[name]
            
    def _wrap_transformer(self, model, config):
        model.transformer_weights = DistriFusionWanTransformerWeightsWrapper(model.transformer_weights, config)
        model.transformer_infer = DistriFusionWanTransformerInferWrapper(model.transformer_infer, config)
        
    def infer(self, inputs, is_warmup=True):
        if self.config["cpu_offload"]:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()

        embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=True)
        x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out, is_warmup=is_warmup)
        noise_pred_cond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

        if self.config["feature_caching"] == "Tea":
            self.scheduler.cnt += 1
            if self.scheduler.cnt >= self.scheduler.num_steps:
                self.scheduler.cnt = 0
        self.scheduler.noise_pred = noise_pred_cond

        if self.config["enable_cfg"]:
            embed, grid_sizes, pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs, positive=False)
            x = self.transformer_infer.infer(self.transformer_weights, grid_sizes, embed, *pre_infer_out, is_warmup=is_warmup)
            noise_pred_uncond = self.post_infer.infer(self.post_weight, x, embed, grid_sizes)[0]

            if self.config["feature_caching"] == "Tea":
                self.scheduler.cnt += 1
                if self.scheduler.cnt >= self.scheduler.num_steps:
                    self.scheduler.cnt = 0

            self.scheduler.noise_pred = noise_pred_uncond + self.config.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

            if self.config["cpu_offload"]:
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()