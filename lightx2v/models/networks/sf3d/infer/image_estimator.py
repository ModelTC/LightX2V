# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.image_estimator import Any, Float, Normalize, Tensor, get_activation, nn, openclip_image, torch


def ClipBasedHeadEstimator_forward(self, cond_image: Float[Tensor, "B 1 H W 3"], sample: bool = True) -> dict[str, Any]:
    cond_image = nn.functional.interpolate(cond_image.flatten(0, 1).permute(0, 3, 1, 2).contiguous(), size=(224, 224), mode="bilinear", align_corners=False)
    cond_image = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(cond_image)
    image_features = openclip_image(self.visual_weights, cond_image)
    outputs = {}
    for head_dict in self.cfg.heads:
        head_name = head_dict.name
        shared_head, d1_h, d2_h = self.heads[head_name]
        shared_features = shared_head(image_features)
        # Distribution parameters/moments are scalar statistics, not GEMMs.
        d1, d2 = [head(shared_features).squeeze(-1).float() for head in [d1_h, d2_h]]
        if self.cfg.distribution == "normal":
            mean = d1
            var = d2
            if mean.shape[-1] == 1:
                outputs[head_name] = torch.distributions.Normal(mean + head_dict.output_bias, torch.nn.functional.softplus(var))
            else:
                outputs[head_name] = torch.distributions.MultivariateNormal(mean + head_dict.output_bias, torch.nn.functional.softplus(var).diag_embed())
        elif self.cfg.distribution == "beta":
            outputs[head_name] = torch.distributions.Beta(torch.nn.functional.softplus(d1 + head_dict.output_bias), torch.nn.functional.softplus(d2 + head_dict.output_bias))
        else:
            raise NotImplementedError
    if sample:
        for head_dict in self.cfg.heads:
            head_name = head_dict.name
            dist = outputs[head_name]
            if self.cfg.distribution_eval == "mean":
                out = dist.mean
            elif self.cfg.distribution_eval == "mode":
                out = dist.mode
            elif self.cfg.distribution_eval == "sample_mean":
                out = dist.sample([10]).mean(-1)
            else:
                out = dist.rsample() if self.training else dist.sample()
            outputs[head_name] = get_activation(head_dict.output_activation)(out)
            outputs[f"{head_name}_dist"] = dist
    for head in self.cfg.heads:
        if head.shape:
            if not sample:
                raise ValueError("Cannot reshape non-sampled probabilisitic outputs")
            outputs[head.name] = outputs[head.name].reshape(*head.shape)
        if head.add_to_decoder_features:
            outputs[f"decoder_{head.name}"] = outputs[head.name]
            del outputs[head.name]
    return outputs
